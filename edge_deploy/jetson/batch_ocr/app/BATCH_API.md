# C++调度应用与独立批入口

原单图服务不替换、不自动恢复，Extract不接入。C++正式应用复用既有有界队列／RAII信用及严格JSON：输入池2、GPU调度1、结果池2、独立事件接收1；Python仍负责两个算法worker与一个模型执行线程。池线程委派CPU算法，不将原OCR整体翻译成C++。

## 私有前端协议

仍用24B AOCR v1帧，request_id固定1（每条前端连接一请求）。父目录0700、socket0600、内核同UID检查；元信息≤4096B，不接受文件路径、pickle、Base64或任意命令。C++与backend的同session／同客户端PID四角色连接保持常驻，不每批重载模型。

```json
{"schema":"ocr.batch/1","op":"batch.submit","remaining_ms":300000,"images":[{"image_id":"doc1","image_type":3,"encoded_bytes":123}]}
```

载荷按images顺序拼接编码字节，1～4图，单图1～16MiB、总≤64MiB，ID唯一，type=1/2/3。批软时限最大600秒，单图最多300秒，模型阶段及收发仍受原上限约束；不得因resume／collect清理控制更新原页deadline。成功meta为`schema=ocr.batch/1,op=batch.reply,status=ok,count=N`，body是≤32MiB严格JSON对象`{"results":[...],"execution":{...}}`。结果按输入顺序，逐图有image_id、status及data或固定code；坏图可逐项失败，不能把部分结果标全成功。

`batch.health`只需schema、op、remaining_ms，载荷0；它不运行模型。backend断连／GPU故障使整个C++应用停止并保留失败，不自动重试原任务。

## 额度与所有权

- C++活动信用2×256MiB，image_id与backend页一一对应。只在拿到信用后提交，收到CPU finalizer之后的image.result才归还；backend的2页512MiB是同一逻辑信用的镜像，不再加一份512MiB当总量。断连后信用关闭、不再用于接纳新任务，后台仍自行保留输入到确认GPU安全结束。
- 输入64MiB是共享batch载荷，不按准备线程复制整批；每个提交帧的独立副本属于传输预算。输入／ready／结果队列分别数量4及64／64／32MiB，ready按实际引用Tensor总字节计权。持队列锁时不执行IPC／推理。
- 核心传输128MiB分为C++64MiB及Python后端64MiB，默认独立组件仍保留原128MiB；C++输入／结果复制分别取得信用，输入ACK及原始缓冲释放后才记submitted、结果原始帧释放后才记done。外部客户端自己的输入不算服务预算。
- 新HTTP是第三个进程，另给256MiB静态前端信用，覆盖24MiB上传、Base64解码、32MiB私有结果接收及Unicode／HTTP序列化副本；不是再给模型开一份预算，也不是立即分配256MiB。单批gate保留到WSGI发送完或断连关闭，非view返回就释放；忙时其它最多3个HTTP线程仅64KiB块丢弃，不解码。核心768MiB＋HTTP256MiB＝1GiB显式账本；JSON DOM、模型和库内部仍非逐字节RSS硬限，须实测资源门槛，不能把768MiB说成整个三进程服务预算。
- 结果信用64MiB，其中逐项8MiB、最多4项32MiB，另留32MiB批序列化；保留逐项信用至前端整帧发送成功。发送用字符串视图，不再复制一份整批vector。整帧被本地socket接受仍非客户端已经解析的确认。
- HTTP客户端在收到完整结果后再作一次无推理health握手，确认C++已走完发送及释放，才开始下一份完整HTTP序列化，避免两侧序列化副本叠加。常驻OCR诊断只保留最近4页，不按历史批次数无界增长。
- 模型／JSON DOM／库内部／内核缓冲不是该账本的逐字节RSS硬上限；CPU原算法独立副本也需端到端资源实测。每个socket只有受mutex串行化的帧所有者；通道锁不替代或移除原context锁。

## 独立HTTP接口（仅loopback 127.0.0.1:5052）

`GET /healthz`检查子进程与backend私有状态标记，另报busy；不做推理。`POST /OCRBatch`接收以下JSON，要求Content-Length及application/json，请求体≤24MiB：

```json
{"images":[{"imageId":"doc1","imageType":3,"imageBase64":"RAW_BASE64_WITHOUT_DATA_URL_PREFIX"}]}
```

字段严格匹配，拒绝重复JSON键／非法Unicode／非有限数值，图数1～4、单解码编码串≤16MiB；标准Base64，不接data URL或任意路径。上传绝对5秒、输入错误400、缺长度411、超限413、格式415、繁忙429、上传超时504、后端不健康503。忙时先有界丢弃仍在上传的body，避免早429重置TCP；不进入推理。正常JSON结果通过Gunicorn单次sendall的5秒超时限制，发送或关闭后才归还gate；健康检查由另一个HTTP线程响应。原/OCR与/Extract在新入口返回404。

`results`按输入ID返回，status=ok或partial；每个成功项的`data`复用原`/OCR`的`msg`格式（total_pages／pages、原图width／height、原图像素blocks及原Markdown）。通过原可信server源码的结果组装片段与原helper生成，不另复制一套坐标／标题规则；不执行原激活／扣次／Extract流程。内部pipeline data仍可在离线验证模式独立严格对照，不能把内部处理图坐标误当对外原图坐标。

Gunicorn只开一个worker＋4个HTTP线程，显式`worker-connections=8`限制已接收连接，listen backlog另为4；backlog不能替代连接上限，不能沿用默认1000。Supervisor只创建一组Python模型后端与C++应用；不按HTTP worker或图片复制模型。新运行目录在项目runtime下独占生成，正常退出清理自有目录；失败证据或GPU状态未知时保留，不SIGKILL／reset或自动重试推理。明确标记`jetson_dev_no_license`，不接授权、扣次或公网／FRP暴露。

## 构建与启动

```bash
# 已验收交付目录附带硬解版二进制，不必重新编译；不要重复启动模型。
bash app/start_batch.sh
```

Root CMake支持3.10，可只构建应用／协议／队列；`-DAUTOOCR_BUILD_JPEG=ON`同时构建硬解应用与探针，需要匹配SDK及V4L2头。构建在私有build目录生成SDK副本并用fuzz=0应用partial-MCU补丁、复核前后SHA，不修改系统SDK；末尾高度非完整MCU时走原indirect复制，避免补齐行覆盖最后有效行。`ctest --output-on-failure`运行传输／阶段／队列／颜色／批策略自检，颜色断言在Release也有效。

E162按用户用途调整启动策略：`start_batch.sh`在变量未设置时默认`AUTOOCR_BATCH_JPEG=1`和`AUTOOCR_BATCH_DET=1`，即硬解＋兼容det N2。显式0仍覆盖默认，空字符串不是合法开关。只改变官方启动入口的默认值，独立组件／探针的N1默认不变；不重建引擎或改模型精度。这是功能采用，不是吞吐优化结论，E161负收益仍保留。

硬解范围为偶数baseline420、≤8百万像素、尺寸≤8192；其它合法格式保留原软件路径，解码执行错误逐项失败而非偷偷重试。C++输出BGR字节直接交接，不用PNG/JPEG中转；Python逐页复用原预处理，保留JPEG惰性draft。输入池各自拥有codec实例，不共享decoder。

启动默认加载已验证N2引擎，`AUTOOCR_BATCH_DET=0`可关闭；仅同type、同实际输入shape的det合批，每次收集空等最多100ms且不越过页deadline，另一页尚未到det时可先执行其layout／cell／YOLO／增强前驱，不多开GPU线程。原10ms在实际IPC/resume及CPU准备下没有形成配对，失败保留，不能算batch性能。不同shape、尾项及收集超时走原N1，不改变像素凑batch、不因执行错误回退。N2先前raw-det及整链路负收益不改写，不从线程数推断加速。

## 使用与回退

独立交付目标为`/home/xchq/work/autoocr/overlays/batch-ocr-e161-r1`；不拷贝或改动原权重，引擎／宽rec及依赖仍从同一项目原位置只读加载。因此它是这块板子的增量部署，不是可直接搬到不同TensorRT／GPU版本的通用安装包。最终交付与计时是否通过，以plan最新状态及技术日志E161为准。

```bash
base=/home/xchq/work/autoocr
app="$base/overlays/batch-ocr-e161-r1/app"
# 板端没有curl，使用现有Python；不读取图片／不推理。
"$base/envs/business-py310/bin/python" -c 'import json; from urllib.request import build_opener,ProxyHandler; print(json.load(build_opener(ProxyHandler({})).open("http://127.0.0.1:5052/healthz", timeout=3))["status"])'
# 每次1～4图；输出计数／坐标边界检查，不打印正文，不调用Extract。
"$base/envs/business-py310/bin/python" "$app/smoke_batch.py" \
  --images "$base/testpic/你的图片.jpg" --image-type 3
```

服务未启动时才执行`bash "$app/start_batch.sh"`；前台Ctrl+C正常退出，或核对本实例Gunicorn master的PID／cmdline后仅向该PID发送SIGTERM，等待子进程安全收尾，不能`pkill python`或`kill -9`。自有runtime目录正常退出删除；失败／GPU安全状态未知时保留证据，不能手动删目录冒充正常回收。

正常停止当前实例后，`AUTOOCR_BATCH_DET=0 AUTOOCR_BATCH_JPEG=0 bash "$app/start_batch.sh"`显式回退软件解码／N1；只设置其中一个0可单独关闭对应功能。清除两个变量后启动（`env -u AUTOOCR_BATCH_DET -u AUTOOCR_BATCH_JPEG bash "$app/start_batch.sh"`）恢复默认硬解＋N2，而不是回到软件N1。这个回退只切新批入口，不启动原5051、不改systemd／FRP；不要同时启动两套模型。

Windows调用时可用本地SSH转发`ssh -N -L 15052:127.0.0.1:5052 -p SSH_PORT USER@JETSON_HOST`，再运行本机`smoke_batch.py --url http://127.0.0.1:15052 --images <本机图片路径>`。不要把服务改绑公网来方便测试；SSH密码只在终端输入，不写脚本。

`execution`含队列高水位、保留额度、实际det配对和硬解计数、逐图准入等待／墙钟及Base64耗时；health的recent_pages最多4页，含CPU预后处理及模型宿主调用。时间有嵌套，不能相加重复计费；N2整批宿主调用分别归因到两个slot，不可再逐图求和称为GPU总时。GPU宿主调用包含传输／同步，不是CUDA event纯kernel时间；硬解total包含decoder建立、颜色转换及释放。

验收是固定13张现有图片上的一致性／有限持续运行，不是GT准确率、24小时长稳或穷尽媒体模糊测试；故障生命周期由受控测试覆盖，未主动破坏真实GPU驱动注入故障。
