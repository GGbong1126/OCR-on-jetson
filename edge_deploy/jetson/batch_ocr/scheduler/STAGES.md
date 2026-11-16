# OCR阶段与资源契约（E155）

这是先于集成冻结的运行时设计及可测试约束，**不是已上线的OCR后端**。E154探针默认仍只允许probe；E155独立验证消息schema、有界队列／内存租约和合成线程协作。原服务、Extract、原模型及N1默认均不变，不新建第二套模型。

E156补充：[会话／资源生命周期](LIFECYCLE.md)新增session.ready／rpc.reply／model.failed及非成功image.result错误码，提供单I/O所有者的Python字节注册表与状态机。以下E155关于“尚未实现”的描述保留原阶段范围，最新核心范围及仍缺的真实算法／I/O／C++JSON接线以LIFECYCLE.md为准；不等于M0或M2完成。

E160契约补充：worker内部输出Tensor可为float32或int32，输入及model.ready仍仅FP32，不将整数检测数量错读为概率。Session保留image.submit的format供真实算法拒绝未适配raw BGR。rec阶段宽度32～3200与原_rec_run上限一致；GPU仍原32～1536及1537～2304，2305～3200由原外层OCRSession显式CPU路由，绝非GPU执行异常后回退。两端schema同步更新，不能混用旧的2304上限消费者；limits的4096只是更外层设计护栏，不表示原OCR会生成该宽度。完整业务适配范围见[OCR worker](OCR_WORKERS.md)，以下旧阶段说明按其时点理解。

## 执行归属

| 角色 | 数量 | 职责／约束 |
|---|---:|---|
| C++输入池 | 2 | 读取／字节校验、任务接纳、提交CPU准备阶段；不直接调用共享Python对象 |
| C++ GPU调度 | 1 | 消费model.ready，控制模型执行入口／兼容det组批；不是绑定GPU物理核心 |
| C++结果池 | 2 | 交付model.resume、收集／校验最终JSON、按图片ID归并；不改原业务规则 |
| C++事件接收 | 1 | 独占事件socket，按ID向队列分发；不执行模型，不与GPU调度竞争context |
| Python CPU算法worker | 2 | 复用原解码／缩图／模型前后处理、crop／CTC、版式归属等；两个C++池通过阶段任务委派算法，不宣称这些算法已转C++ |
| Python GPU执行 | 1 | 唯一模型执行线程，初始化／拥有TensorRT context和buffer；启动预热另计 |
| Python IPC I/O | 1 | 接受控制请求、派发事件／唤醒等待者；不得被等GPU的CPU任务占满 |

Python业务worker到达推理点后发布ready并等待结果；独立GPU线程不依赖这两个worker才可执行，resume由I/O路径唤醒，防止CPU池全在等GPU时形成死锁。原Ultralytics predictor等可变包装器不得直接跨任务共用；集成前需按每图上下文隔离或明确串行锁。此桥接尚未实现，不把并行调用原process当作线程安全证明。

模型输入／输出留在Python有界buffer注册表，C++调度传handle和规格，避免每个网络都把大张量跨进程来回复制。输入图像进入backend和最终JSON离开backend时才传载荷；不引入共享内存／FD传递／零拷贝。未来C++必须用成熟JSON解析器消费规格，目前仍未安装解析库或实现该消费者。

## 阶段消息

沿用E154的24B帧；JSON使用独立schema=`ocr.stage/1`，仅在显式选择stage校验器的新连接使用，旧probe默认拒绝。所有消息带session（32位小写hex），图片相关消息带ASCII image_id；call_id为1～2^63−1。session在backend重启／断连后失效，handle只在所属session内有效；真实注册表／状态机在M2实现。

| op／方向 | 关键字段／语义 | 二进制 |
|---|---|---|
| image.submit，C++→Python | image_id、image_type=1/2/3、format=encoded或bgr8；encoded无文件路径，bgr8必须shape=[H,W,3]／stride=W×3 | encoded≤16MiB；BGR≤1600万像素／各边8192；解码前仍须检查真实图像头，不能仅信元信息 |
| model.ready，Python→C++ | image_id／image_type、call_id、model、inputs（name／handle／dtype／shape／bytes）；仅layout/cell/yolo/enhance/det/rec，NER禁止 | 0，真实buffer仍在backend |
| model.run，C++→Python | call_ids长度1；实验det N2显式开启后可2，需不同图片、同type、同模型及完全相同输入规格 | 0 |
| model.done，Python→结果池 | call_ids及一一对应的output_bytes；实际结果仍在backend，需真实完成状态才可resume | 0 |
| model.resume，结果池→Python | call_id；只能消费已完成结果一次，不允许客户端提供伪造输出 | 0 |
| image.collect，结果池→Python | image_id；只取已完成结果，不重复执行模型 | 0；未来成功回复为≤8MiB UTF-8 JSON载荷 |
| image.cancel，C++→Python | image_id；协作取消，不杀GPU／重置context | 0 |
| image.result，Python→C++ | image_id及status=ok/error/cancelled；非ok不带正文 | ok为1～8MiB以内的UTF-8 JSON；错误／取消为0B |

模型ready的绑定名来自实机原会话：layout/cell为image、im_shape、scale_factor，det/rec为x，YOLO/增强为images。ready只描述原N1输入；det N2由两个ready组装，不修改各自张量，不新增拉伸／补白。rec的GPU宽路由仍32～1536／1537～2304，E160阶段契约覆盖原32～3200输入并保留范围外CPU规则；4096设计护栏不改变旧/OCR的3200预处理上限。GPU错误不得转CPU或重试N1。

公共字段还包括schema、session和remaining_ms（正整数，不超过单图deadline）；实际handler须按本地剩余时间与该阶段上限取更小值，不能因消息带了更大的值重置deadline。model.done只承诺成功完成事件，错误仍需独立错误envelope。成功／失败响应的完整envelope、C++成熟JSON消费、结果JSON业务结构、事件注册表和状态转移仍需联调；E155的schema校验不伪称实现了这些业务handler。每连接一个读写所有者，输入／GPU／结果通道分离，事件接收单独拥有连接，不让多个线程交错发一帧。

## 冻结预算

参数见[limits.json](limits.json)，是后续集成的初始上限，不是实测最优线程数／峰值：

- 总在途图片4，活动算法图片2。活动页在开始解码前**一次预留256MiB**，两页512MiB；一页的显式CPU图像、张量、IPC复制均计入这份预留。后续阶段超页预算则失败，不能持有半份额度再等另一份而自锁。
- 输入队列4项／64MiB，ready队列4项／64MiB（按引用的输入张量总字节计权，不只是小handle大小），结果队列4项／32MiB。队列出队释放的是队列占用，不释放页的全局预留。
- 另保留入口64MiB、传输128MiB、结果64MiB独立预算，总显式预留768MiB，避免活动页把控制／结果交付所需空间吃完。每个独立副本都要记账；共享同一buffer的handle不重复算真实分配，队列计权和内存预留不是相加的两份RAM。
- 本轮C++ ByteBudget／BoundedQueue只验证机制；跨进程每页记账、真实图像解码限制、注册表和原算法临时分配尚未接入，不声称768MiB已限制整套进程RSS。GPU权重／激活／workspace、ORT／Torch／Pillow内部内存、JSON DOM、内核socket缓冲及文件缓存均需后续单列测量。
- 入队等待5秒、控制RPC5秒、单模型执行等待120秒、单图总deadline300秒；均为可取消的软时限，GPU提交后不能靠socket超时证明kernel已取消。迟到结果必须按session／call_id丢弃并等执行安全结束后释放buffer。
- det N2默认关闭；显式实验模式等待同shape第二项最多10ms，尾项仍N1。当前N2负收益事实不因本轮线程协作演示改变。

## 生命周期与验收

停止接收→close队列唤醒阻塞者→按drain／cancel策略处理排队项→等活动任务安全结束→join线程→释放模型／socket。cancel排队项可归还租约，活动项的租约保持到真实结束；ByteBudget的Lease可安全晚于预算句柄析构，但队列对象必须活到其等待线程退出。禁止持队列锁执行推理／IPC或析构用户任务。

本轮先用纯合成任务验证数量／字节双背压、绝对超时、退出唤醒、drain／cancel、出队后仍占全局额度、租约只释放一次、双输入／单模拟执行／双结果的ID归还与顺序恢复。没有GPU调用、OCR输出或吞吐提升结论；M0/M1a/M2/M3完整状态按plan分别验收。
