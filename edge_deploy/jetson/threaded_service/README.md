# E167：实际批OCR的REC owner线程

2026-10-31补充E168：[真实HTTP kernel并发已验证](../http_rec_trace/README.md)。同实际业务代码32页CUPTI窗口中四个不同REC owner的kernel确有共同区间，累计100.476ms，占已映射REC kernel活跃并集0.3733%；≥2路30.0165%、≥3路3.5308%，不是全程四路或4倍。临时root采样窗口不代表无采样占比，旧E167+15.06%不变。原E167已恢复master10656、5052 ready/空闲、4路/错误0，Extract不动；下面27246为E167原验收时点。两项诊断失败及严格数据校核见新报告，原生产源码未改。

2026-10-30，用户要求集成实际OCR服务，以testpic端到端吞吐选并发，/Extract不动。

## 最终结论（2026-10-30）

已在实际批OCR服务 `127.0.0.1:5052/OCRBatch` 部署 **4个REC owner线程**，不是只跑离线REC探针。原单图5051服务未启动，`/Extract`未改、未调用；新入口本来就不提供Extract。当前交付是本板增量overlay，不是独立可移植安装包，也未改systemd自启。

| 顺序/配置 | 正式页数 | 请求墙钟合计/s | 吞吐/页·s⁻¹ |
| --- | ---: | ---: | ---: |
| E162首基线 | 96 | 327.058613 | 0.293525 |
| 1个REC owner | 96 | 334.246840 | 0.287213 |
| 2个REC owner | 96 | 282.083639 | 0.340325 |
| 4个REC owner | 96 | 275.091184 | 0.348975 |
| E162尾基线 | 96 | 305.962880 | 0.313764 |

首尾基线同为96页，先平均耗时：`(327.058613+305.962880)/2=316.510746s`，对应 `0.303307页/s`。4路相对它**吞吐+15.0567%，时间减少13.0863%**，两者不是同一百分比。尾基线比首基线更快，存在运行漂移；不能只用首基线宣称+18.89%，也不是统计显著性或普遍性能保证。

按事先冻结的“最快有效候选2%以内选最少线程”，2路比4路吞吐低约2.48%，不满足门槛，故选4路。换成4路比2路快的分母口径是+2.54%。这是1/2/4有限矩阵的选择，不是Xavier NX硬件绝对上限；如果更重视内存，2路仍是合理取舍，但本轮没有擅改预定选择规则。

五组每组32页预热＋96页正式，共640次重复页处理（480正式）；部署后另32页严格回归、BUSY期间4页和交付启动器重启后4页，共680页。**独立图片仍只有16张**，普通type3/增强type1各一，共32个参考键。673张仅全量检查JPEG头，没有全部推理。每页完整公开data（包括文本、Markdown、分组、框等）的规范化SHA与E162参考严格相同，不只是框数相等；这不是GT正确率或CER。

计时起点在客户端读文件/SHA/Base64/JSON之前，终点在HTTP响应解析之后；各正式请求墙钟求和，含完整业务处理和loopback HTTP，不含模型加载、对照SHA比较、请求之间的空闲health等待。每请求4图，串行发送批请求，服务内并发；不是同时放开多个HTTP批，忙时仍429。所有组硬解/DET N2开关同为1；线程组128页累计实际硬解48/软件80/解码错误0，但本轮各组实际det配对均0，不能写成本轮证实了N2计算收益。首基线计数含额外4页历史参考检查，累计132页，与正式计时96页分开。

该+15.06%比较的是E162同步实现与新的stage/2＋owner线程完整方案，包含原有提前供给/CTC/异步交接及资源分工变化，**不是单独增加Python线程的因果收益**。本轮没采kernel时间线；4个owner/4条独立REC流、实际逻辑在途峰值4不等于证明在线始终4个kernel同时计算。E166合成探针的短暂16流重叠是另一份证据，不混入本轮结论。

## 冻结方案（以下方法已实际执行）

- 独立overlay，复用E164 stage/2、C++有界队列/额度/CPU超宽路由/CTC确认，只增加REC owner线程及显式启用开关；原E162/E164/原发布不覆盖。
- 每线程独占窄/宽FP32 context、nonblocking stream和共享互斥IO银行，同CUDA primary context；原张量/字典/预后处理/精度不变。主GPU线程只管理前驱和det，REC结果经既有有限完成槽返回，不复制完整业务模型。
- 先比较1/2/4线程，保持原240MiB固定IO/CPU路由slot5/768MiB核心+256MiB HTTP预算；不因上轮16路峰值擅扩协议或上线16线程。权重/context/workspace另计，至少2GiB模型加载余量。
- testpic当前673个JPEG，先全量头信息盘点；冻结原13图＋3张新增不同尺寸图，共16张×普通/增强=32页。不是全量673图验收。原13图须符合历史公开结果SHA；新增以本轮E162为对照，无GT。
- 统一默认硬解/DET N2、每HTTP最多4图、相同正反顺序；E162首尾对照、线程1/2/4各预热32页＋三轮96页。计时包含本地读文件/base64/HTTP往返/服务完整流程，不含加载；不拿纯REC微基准或旧17.71%当本轮结果。
- 每页公开data严格相同；失败/取消/退役/满队列/CPU路由/启动和安全关闭须先测。每批空闲后页/注册/活动/HTTP信用归零，线程/FD不持续增长，内存有余量；不放宽质量门槛换速度。
- 选择有效线程候选中聚合吞吐最快2%以内的最少线程，不挑单轮；若不胜E162仍报告真实差值。通过后独立loopback5052入口运行所选配置，保留原E162回退。失败或安全未知不强杀GPU/重置驱动。
- 不启用Graph，不新增特权画像，不改Extract/原发布/授权/系统/功耗/时钟/自启，不提交推送Git。最终更新计划和技术日志，留代码/报告SHA可复算证据。

## 实现边界

`app/serve_backend.py`只新增显式`AUTOOCR_REC_THREADS=1`选择，且要求`AUTOOCR_REC_SLOTS`为1/2/4；未启用时原路径不变。新独立overlay根目录`start_threaded.sh`读取`selection.json`，设置所选线程数后执行原批服务启动器，仍只有一个Gunicorn worker、一个模型后端进程，不启动多套完整OCR服务。

`multistream/threaded_rec.py`复用原`AsyncRuntime`和原裁剪/归一化/CTC。每路owner线程创建自己的窄/宽Session与Slot，窄宽互斥复用IO银行，使用原FP32引擎和profile。只有REC复制权重/context，其他模型仍原单套；原工厂rec会话保留，因此不能宣称零额外显存成本。原单owner rec IO银行先在其所有者线程安全关闭，避免与新银行重复占用固定IO预算。

调度链为：原C++ rec-ready共享队列 → stage/2选取逻辑槽 → GPU协调线程投递该owner的单项有界邮箱 → owner执行H2D/enqueue/本流等待/D2H并复制独立bytes → Future完成 → 原I/O所有者登记输出 → 原独立CTC worker消费 → `rec.consumed`归还逻辑信用。native context/stream/buffer不跨owner并发调用；只传不可变CPU字节，不把device指针交给其它线程。线程不绑定CUDA核心或SM。

前驱/DET仍由原GPU协调线程执行，非DET前驱保留原阶段屏障；不是把整套6模型都改成无限并发。输入CPU双worker、CTC、超宽CPU路由及C++输入/结果线程池继续保留。GPU宽32～2304，2305～3200仍原CPU路径，不补白、不缩窄、不在GPU执行错误时回退CPU。

owner失败后锁存，不重试推理；同步与关闭作为命令由资源原owner处理，全部安全同步确认后才释放，未知状态保留隔离资源。启动中途失败清理已启动owner；空闲owner必须清除上一项request/reply引用，避免注册表归还信用后仍隐藏持有张量。`owner_activation_r2.json`记录此修正在任何候选GPU运行前生效，旧版弱引用负对照失败、新版通过，所有正式候选采用同版。

## 已遇到的预检/观察脚本问题（不抹除）

1. testpic多数扩展名为大写JPG，最初只匹配小写导致漏图及空候选，停服前失败；改为大小写无关头盘点，673/673头有效，原13文件SHA全相同。
2. HTTP返回字段为`image_id`而非输入的`imageId`，已在任何HTTP基准执行前按真实契约纠正。
3. 首基线128页质量/资源通过后，正常停止旧四进程成功；采样`/proc`时读到短命ps子进程已退出，观察脚本失败。保留原控制器/报告，修复仅忽略已退出的非根子进程、根进程消失仍报错。新r3明确核无旧模型、无5052监听、原指纹/boot/unit与内存余量后续接，不重跑挑选更快首基线。`request/cycle/start`的AST与原r2一致，不改计时方法。
4. 第一份最终回归日志因缺少`AUTOOCR_IPC_PROBE`/`AUTOOCR_STAGE_PROBE`/`AUTOOCR_TEST_JPEG`跳过17＋1＋1项。R1原件保留，不称零跳过；随后补充脚本显式使用已有同版本C++探针和硬解开关，**113核心＋40 multistream＋29应用＋3额外真实四通道集成共185项全部通过、零跳过**。补测不重跑正式计时、不停服务、不运行GPU模型，测试前后线上REC计数不变。

## 资源与安全验收

- 5组资源门槛均通过：预热端点到正式结束，树内每进程FD增量≤4/线程增量≤2、树内RSS合计增长≤128MiB、MemAvailable≥1GiB、swap增长≤128MiB；每批空闲后页/注册/活动/HTTP信用归零，REC错误0、owner共享CUDA context。候选4路FD/线程均无端点增长、RSS合计增长62.09MiB，结束MemAvailable约2415MiB；不是24h不泄漏证明。
- 4路REC固定host+device IO为180,052,992B，加DET后239,035,392B，仍低于240MiB。固定IO不是权重/context/workspace总成本，1GiB显式额度也不是RSS或显存硬限。
- 本轮4路结束进程树RSS合计3660.41MiB（可能重复计共享映射，不能当独占物理内存）；全机NvMap约8350.92MiB，旧E162约5550.95MiB。NvMap是全机共享指标，不是服务专属VRAM，不能和RSS相加。REC独立engine/context的额外内存是主要代价之一。
- 无效结构/空图片/坏Base64三个400、真实有效批运行期间第二请求429、有效4页结果、交付启动器正常停启后4页严格回归通过。NER只工厂预热1次，Extract业务调用0；旧发布、E162/E164 overlay、引擎、boot和四unit状态/PID未改。
- 5项owner测试在Windows和Jetson通过；弱引用旧版失败/新版释放通过，旧`/proc`观察竞态负对照失败/修正版通过。真实GPU破坏性故障未注入；仅模拟故障/取消/退役测试，不把模拟写成驱动故障验证。

## 当前运行与测试

交付目录：`/home/xchq/work/autoocr/overlays/batch-ocr-threaded-e167-r1`。`selection.json`保存4，`start_threaded.sh`显式设置`AUTOOCR_REC_THREADS=1`、`AUTOOCR_REC_SLOTS=4`；硬解/DET N2仍沿用官方启动器默认1。验收后master为27246、Cpp为27293，**PID只代表验收时点，后续操作必须重新核对，不能照旧PID盲杀**。当前已经运行，不要重复启动。

```bash
base=/home/xchq/work/autoocr
service="$base/overlays/batch-ocr-threaded-e167-r1"
py="$base/envs/business-py310/bin/python"
# 只读健康检查；不推理、不调用Extract。
"$py" -c 'import json; from urllib.request import build_opener,ProxyHandler; h=json.load(build_opener(ProxyHandler({})).open("http://127.0.0.1:5052/healthz",timeout=5)); print(h["status"],h["busy"],h["models"]["rec"]["threaded"])'
# testpic中前4个jpg/JPG，一批多图，输出统计/边界、不打印正文。
pics=("$base/testpic/"*.[jJ][pP][gG])
"$py" "$service/app/smoke_batch.py" --images "${pics[@]:0:4}" --image-type 3
# 增强流程：将上一行的3改成1；每次仍最多4图。
```

正常停止时先核实当前master的cmdline/cwd属于该overlay且无忙请求，再仅给该master发SIGTERM、等待所有子进程安全退出；不`pkill python`/`kill -9`/reset。**仅服务完全停止后**启动：`bash "$service/start_threaded.sh"`。回退原E162也须先正常停止新实例，再`env -u AUTOOCR_REC_THREADS -u AUTOOCR_REC_SLOTS bash "$base/overlays/batch-ocr-e161-r1/app/start_batch.sh"`；不自动恢复5051/Extract，不同时加载两套模型。

Windows测试可沿用[原批接口说明](../batch_ocr/app/BATCH_API.md)的SSH本地转发和`smoke_batch.py`，无需改绑公网。模型/词典/profile/FP32精度都不变；本轮没有量化/剪枝/重建引擎，也没有把整套业务改成C++。

## 可复算证据

板端：`staging/threaded_e167_r1/`；Windows：`temp/threaded_e167_r1/`。只归档技术JSON/日志/实际源码，不带图片、正文、权重或凭据。

- 主包`technical_evidence_r1.tgz`：458,789B、124成员、81份生产源码（另固定核C++可执行SHA），SHA256 `6c2557308b544a115f9576932b38752fd61a180431ea416180e4f712ea5c529f`。Windows `verification_final.json`已核全部成员/实际部署源码与本机一致、五组每页SHA/资源/选择/首尾计算/边界重启/原文件未改。
- 补包`supplement_evidence_r1.tgz`：6,238B、4成员，SHA256 `34dff97349ce681c5aadf5a2f07bbf757d9a954d516a89b84910ac9eb0264d16`，已Windows逐成员/日志/本地脚本独立核验。记录19项跳过的纠正、185项零跳过完整日志及测试入口native SHA；主包不覆盖，计时与线上源码不变。
- 主测控制器r2/r3、owner r1/r2和激活时点均保留；`final_unit_tests.log`的原跳过结果仍可追溯。C++沿用E164二进制 `f4baeef9a403324f67e49e2cb05c219d51302f7aa28ae60a4aec71de8c228b1c`，本轮未重新编译/CMake/UBSan，不冒称新跑过。

E167本轮未验证：全673图、GT/CER、24h、实车、kernel时间线、功耗锁定或多批HTTP公平调度；E168后来另补真实HTTP代码的kernel时间线，不能冒充E167原计时窗口。换数据、并发或引擎后需重新做质量/内存/吞吐选择，不能套用15.06%。
