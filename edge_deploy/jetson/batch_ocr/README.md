# 批量OCR改造：独立实机验证（E143～E164）

E164应用验收完成：[Plan2](../plan2.md)及[异步槽实现/启动/回退](multistream/README.md)。新独立候选已接1/2/4rec多Stream、共享有界crop队列、独立完成处理和C++ stage/2调度；原预后处理/宽度/精度不变。最终r19 HTTP三周期102页与边界/重启通过，原E162已恢复，候选未默认推广。r18软件N1/IPC三轮rec4 0.350007页秒、相对同步约+17.71%，不是HTTP加速或已实证kernel重叠；一次画像目标重叠为0，负结果保留。早前Event异常说法已用原始JSON纠正，是终端数值误读而非CUDA计时故障。下面是历史交付过程，以Plan2最新状态为准。

E162已交付：用户接受小幅负收益，官方启动默认改为硬解＋det N2，显式0可回退；原组件默认与推理算法不变，性能表不改写。29应用／UBSan29项及默认→软件回退→默认102页严格质量、PNG／坏图／边界／正常停启通过，默认周期实际N2配对3／5、硬解各14页。新master1423保持ready，93成员／60源码证据已取回复核，当前用法见[plan](../plan.md)。下面E161“不默认启用”为历史策略。

## E161：正式批应用与独立HTTP已交付

[当前计划／可运行测试命令](../plan.md)、[协议／开关／启动回退](app/BATCH_API.md)、[代码职责索引](app/DELIVERY.md)。M0～M3已按现有13图有限范围验收：C++2输入／单调度／2结果／事件接收与Python原算法／单执行入口组合，四通道IPC／信用／错误锁存及安全收尾；可选真实det N2、NVJPG硬解及末尾MCU项目补丁。三组合各104页公开msg严格一致、资源全过；HTTP三周期102页严格对照及3次PNG／坏图／边界／重启／回退通过。

最终111核心、28应用及UBSan28项、CTest5、JPEG诊断4项板端0跳过，1268成员／60源码证据已复核。新服务在独立`overlays/batch-ocr-e161-r1`、loopback5052运行，默认软件N1；N2与硬解无明显端到端收益，不默认启用。原5051仍停止／Extract与原文件不动，不改系统或自启；非GT、24小时稳定性或全链路零拷贝声明。以下各E段为历史状态，早期“raw BGR拒绝／未接HTTP”等不再代表当前实现。

## E160：六模型／原完整Python OCR worker小闭环

[契约／所有权／复跑](scheduler/OCR_WORKERS.md)、[适配器](ipc/ocr_worker_adapter.py)、[完整验证](ipc/verify_ocr_workers.py)。复用原load_business／process、原权重及预后处理，不把所有算法翻成C++，不接Extract／HTTP。

- 两CPU算法worker在原layout／YOLO／cell→可选增强→det→逐框rec／CTC→融合／Markdown流程中的每个模型调用处委派；单执行线程管理一套原模型及rec宽overlay，原CPU范围外路由保留。内部输出保留float32／int32，输入仅FP32，CPU消费为独立可写副本，不将NMS计数错读为概率或让torch写只读bytes。
- 每页隔离pipeline／OCR外壳及融合配置；YOLO predictor整体加锁、返回前深拷贝Results。proxy额外锁存页失败，旧业务吞异常不能被包装为成功；关闭不恢复直连GPU访问。Pillow头／尺寸校验在实际解码前，当前完整适配只接受encoded，raw BGR明确拒绝。
- 原两图×普通／增强4页与同一套模型串行参考完整data／全文／Markdown／框坐标／special／调用数严格一致，反序collect；候选layout／cell／YOLO／det各4，enhance18，rec199，合233次，NER业务0、候选CPU回退0。两CPU实际观察、单执行入口，关闭后活动／注册／传输／worker清零。两侧使用同一隔离适配壳和原process，非历史HTTP／Windows响应或GT准确率验收。
- rec阶段元信息两端覆盖原32～3200；GPU原32～1536及1537～2304不改。额外合成W2305实际CPU+1／GPU+0、输出[1,288,18385] finite；原factory NER预热1次及宽GPU预热另计。4页真实rec最大1326，未宣称这4页覆盖宽GPU候选路径或真实长行精度。
- 板端首版96项通过，补格式保存／整数输入拒绝／输入边界等后最终98项无跳过；26项×20轮（520检查）约15.301秒，最终全套约8.443秒、UBSan 14项约5.322秒，FD4→4／线程1→1。元信息352（27合法／325非法）＋15结果双端一致；Windows42通过／56项明确跳过，不安装新NumPy／Pillow补测，不将其写成板端验证。
- 原12项发布＋N2、宽overlay、boot及四unit状态／PID不变，无worker／socket／页临时目录残留；原OCR仍停止。模型加载后可用7529984KiB，仅端点，不是峰值／内存上限；没有速度／长稳结论。GCC构建的是协议消费者，不是重建模型；未改BSP、桌面、挂载、时钟或授权。
- 独立目录`/home/xchq/work/autoocr/staging/ocr_workers_e160_r1`；119成员技术归档及35份最终源码已取回复核。下一步正式C++双池／调度应用组合、更多样本／边界与统一额度，再推进M2／M3；硬解质量、批HTTP及性能资源仍待验收。

## E159：生产worker与真实det N1小闭环

[设计／生命周期／复跑](scheduler/WORKERS.md)、[桥接](ipc/worker_bridge.py)、[N1适配](ipc/det_worker_adapter.py)、[真图验证](ipc/verify_det_workers.py)。固定2CPU算法＋1GPU执行，I/O所有者独立推进Session；不是每个线程绑定GPU核心，也不复制完整OCR模型。

- bounded Job／Request／GPU完成槽和socketpair通知，CPU等GPU时不持锁；完成后等待model.resume，CPU消费完再释放Call。新增image.done，C++ --workers允许页完成事件与其它模型事件交错。
- 满事件队列保留完成结果、不重跑GPU；取消／超时／断连唤醒CPU，CPU finalizer和GPU安全同步都确认后再释放。同步也失败时隔离资源并拒绝成功close，只允许显式重试同步，不执行推理重试／reset／CPU回退。
- 最终88项全部执行、0跳过约7.900秒；17项新增×20轮（340次）及14项UBSan约5.153秒通过，FD4→4／线程1→1。Windows88项中40通过／48项Linux或编译专用跳过；未绕过Windows socketpair的TCP模拟限制。
- E153固定前两张不同图片：原det输入分别960×672／960×704，候选框45／49；概率图逐字节、原框、ID及反序collect与参考一致。软件解码＋原_det_run AST预后处理，2CPU实际观察、1GPU所有者、一份原engine／context，参考2＋候选2 GPU调用，后端错误0。transport高水位1662232B、最终注册／传输额度和worker线程清零；不是RSS／GPU峰值。
- r1有一个合成测试ID适配错误（不是模型错误），修正后86项通过；补线程启动失败／重复close锁存后88项通过。真实启动r4非私有／相对socket父路径、r5仍相对路径被安全检查拒绝，均未创建bridge或加载模型；r6规范化绝对路径并使用0700临时目录后通过，不改原安全检查或放宽规则。
- 独立目录`/home/xchq/work/autoocr/staging/workers_e159_r1`；历史版本／拒绝报告保留。原12项发布＋N2 SHA、boot及四个unit状态／PID不变，OCR仍停止，GDM／SSH／FRP运行；无socket／worker残留，不执行Extract／NER／HTTP、授权或新引擎构建。
- C++联调仍是消费者探针；其它模型／完整OCR、正式C++双池组合、跨进程额度、硬解质量／批HTTP／性能资源验收待做。M0／M1a／M2／M3不勾选完成，N2负收益不推广。

## E158：四连接组及单所有者非阻塞I/O

[协议／范围／复跑](scheduler/CHANNEL_IO.md)、[I/O核心](ipc/channel_io.py)、[测试](ipc/test_channel_io.py)。只用已有Python标准库、Session和固定C++JSON库，不安装依赖，不接实际OCR模型或启动HTTP。

- input／gpu／result三控制＋独立events共4条连接；同session与内核SO_PEERCRED客户端PID，角色唯一、绑定占ID1、控制从2分别递增。错误／重复绑定不替换合法角色；绑定未齐不执行业务，任一已绑定连接失败即关闭整组。
- 实机EpollSelector的单I/O所有者按连接每轮64KiB处理，保留半包／短写进度；事件队列最多4项、传输额度有界。5秒是绑定／一帧收发时限，不是空闲超时，不把模型等待误踢出。poll前后推进原图片绝对deadline。
- 发送整帧后才delivered，失败先关闭连接／丢弃待发视图再abandon_delivery；并非客户端解析确认或跨崩溃恰好一次。running及活动页仍需安全complete／CPU retire，close不会抢先销毁所有者；测试夹具对模拟执行的收尾不代表真实GPU已能自动恢复。
- C++ `--channels` 实际建立4连接并消费ready／done，推进两图9请求、提前collect得到STATE、最终反序收集，测试使用不同通道独立ID。原单连接与全部旧回归保留，新增双端绑定schema后291组元信息及15组结果一致。
- 板端GCC7.5／C++14编译，71项无跳过约7.146秒；UBSan下13项无跳过约4.639秒，14项新增测试重复20轮（280次用例）全部通过，FD4→4／线程1→1。本机71项中39通过／32项Linux或编译专用跳过。
- 覆盖同PID绑定、重复／旧会话／其它PID、部分绑定时限、慢半包不阻塞gpu控制、事件满队列不消耗ready状态、结果接收停滞时保留额度且其它通道继续、运行中断连安全收尾。I/O仍由调用者poll驱动，Harness中的Future／队列／合成driver只属测试，生产worker桥接未实现。

独立目录`staging/channels_e158_r1`，技术归档32成员已取回Windows `temp/channels_e158_r1`核验。原发布12项＋N2、OCR inactive／MainPID0／enabled及GDM／SSH／FRP状态不变，未调用或修改Extract。下一步是真实阶段／worker完成通知与finalizer、NumPy／GPU所有权及统一额度桥接，硬解质量仍未通过；不能把本轮称完整M0／M2／M3、GPU并行或吞吐收益。

## E157：C++严格JSON消费及阶段交接

[范围与复跑](scheduler/CPP_STAGE.md)、[C++校验器](ipc/stage_json.hpp)、[阶段客户端](ipc/stage_client_probe.cpp)、[跨语言对照／交接测试](ipc/test_cpp_stage.py)。板端无可用JSON开发头，因此项目内固定nlohmann/json v3.12.0单头／MIT，SHA核对官方发布；不装系统包、不手写通用JSON解析器，不升级BSP。

- C++实现E156全部阶段操作的字段／类型／ID／session／绑定／shape／字节乘积检查，解析器callback拒绝重复键，含转义后同名键。transport新增显式validator重载，元信息合法后才分配载荷，旧probe默认不变。
- 成功结果JSON最多8MiB、深度64；Python共享validate_result与Session.finish同步，覆盖Unicode代理项、非有限／溢出数字、重复键及大小／深度边界。只校验通用完整性，不是原OCR字段或GT精度验收，也未更改原HTTP协议。
- C++经同UID私有Unix socket固定session，消费反序ready里的call_id发run，核对done／resume，提前collect收到STATE，最后反序收集两图结果；共9请求。合成BGR每图6B、det输入12288B、模拟输出4B，无图像解码／模型／OCR文字。探针是单条受控连接、整体绝对2秒，不是正式三通道或算法等待预算。
- 板端GCC7.5／C++14 `-O2 -Wall -Wextra -Werror -pthread`编译，57项无跳过约6.167秒；267组元信息（20合法／247非法）、15组结果（4合法／11非法）两端判定一致，包含精确8MiB及超限、深度64／65。UBSan编译后12项相关测试无跳过约4.455秒，20轮真实阶段交接通过；父进程FD4→4／线程1→1。本机34项通过、23项Linux或编译专用跳过。
- 终验再收紧结果错误用例的JSON状态断言，不只依赖退出码；新测试版本复用相同SHA二进制，未重新编译或改功能，首版与r2源码／报告均保留。精确终验用时及技术归档见上级E157日志。
- 错session／应答帧ID／for_op、重复键、声明64MiB但元信息不允许载荷、无进展超时均拒绝；不记录原请求／结果正文，不把安全模拟失败当真实CUDA故障恢复。

独立目录`staging/json_ipc_e157_r1`。技术证据取回Windows `temp/json_ipc_e157_r1`；原12项发布＋N2未变，OCR停止、GDM／SSH／FRP运行，Extract不动。M0仍需正式连接绑定及真实事件映射，M2还需算法／NumPy／GPU所有权及跨进程额度；M1a硬解质量未过，N2负收益不推广。以下各节为历史验证阶段。

## E156：会话／注册表与取消、超时、断连收尾

[状态与错误协议](scheduler/LIFECYCLE.md)、[Session核心](ipc/stage_session.py)、[20项生命周期测试](ipc/test_stage_session.py)。沿用标准库及E155额度，不安装依赖、创建新模型或启动服务。代码只接合成bytes，不支持直接借用NumPy／CUDA对象，不宣称已改造原OCR算法。

- 新增session.ready／rpc.reply／model.failed白名单；非成功image.result必须带固定错误码，禁止异常／路径／正文出现在错误元信息。旧probe默认不放开、Extract／NER不在控制白名单内。
- 单I/O所有者管理有界Page／Call／Batch：4张在途、2张活动，输入／ready／结果字节受限，模型张量留在后端。按input／gpu／result通道各自单调request_id防重放，内部旧对象身份不能改写重用图片ID或新会话。
- 取消／超时／断连不抢先释放运行中输入，安全complete后还须CPU retire；resume也不是“CPU已不用输出”，需release_call。GPU错误锁存、不转CPU、不重试，不把失败同步当安全完成。发送中的结果需delivered或关闭连接后的abandon_delivery确认才释放。
- 实际AF_UNIX／0700父目录／0600 socket／同UID检查，验证握手、两图submit、ready／run／done／resume、反序collect与逐帧ID；模型完成事件为模拟，不是GPU。新核心没有创建CPU/GPU工作池或持续I/O循环，C++业务JSON解析仍未实现。
- 板端Python3.10最终45项无跳过，20项核心重复30轮（600次用例）通过；同进程重复窗口FD4→4／线程1→1。本机Python3.13共32项通过／13项Linux专用跳过。原C++组件源码／二进制指纹核对后复用，7组80任务自检再次通过，不重复GPU构建。
- r1的43项及30轮18项先通过；补两项用例发现旧Batch间接保留12288B合成输入、旧Page保留18B结果。板端旧版负对照两项稳定失败，r2在安全释放点清空Call输入输出及Page结果后通过，原源码／报告保留。不能只删字典项就假设所有引用已释放；外部直接持有的bytes别名仍由调用者负责。

独立目录`/home/xchq/work/autoocr/staging/lifecycle_e156_r1`保留src（首版）／r2（修正版）。在新的隔离源码根复跑（ipc／scheduler同级，不能覆盖证据）可沿用下面E155命令，unittest discover现应运行45项；只测生命周期可运行：

```bash
cd ipc
python -B -m unittest test_stage_session -v
```

真实算法接入前还需C++成熟JSON消费、实际连接组／事件循环、CPU等待者通知／finalizer、NumPy及GPU所有权桥接和跨进程额度，硬解质量问题仍在。M0／M1a／M2／M3未完成；原发布12项＋N2候选未变、OCR停止、GDM／SSH／FRP运行，未执行或修改Extract。技术证据在Windows `temp/lifecycle_e156_r1`，归档细节见技术日志E156。

## E155：阶段校验与有界队列／额度基础组件

[阶段／线程／资源契约](scheduler/STAGES.md)、[初始上限](scheduler/limits.json)、[Python元信息校验](ipc/stage_contract.py)、[C++队列／RAII额度](scheduler/bounded_queue.hpp)、[合成协作测试](scheduler/test_scheduler.cpp)。没有新增依赖；GCC7.5／C++14直接编译，Python使用business-py310。CMake目标仅静态提供，本板无CMake、未运行配置。

- 阶段schema必须显式选择；旧probe默认和Extract拒绝规则保留。C++仍未实现成熟JSON业务消费者，backend的注册表／状态机、错误envelope和OCR桥接尚未实现。
- 冻结2输入／1GPU调度／2结果及独立事件接收；Python2 CPU算法／1 GPU执行／1 I/O，模型张量留在后端、只传handle和规格。当前执行线程只做合成XOR，不是GPU或已接入OCR的线程池。
- 拟4张在途／2张活动，每活动页256MiB预约，另设入口／传输／结果预算，合计768MiB显式额度。队列按项数＋字节双限，任务出队仍持全局租约，取消排队项与活动项的释放时机不同。跨进程额度和库内部分配尚未受此机制约束，768MiB不是服务RSS上限。
- 板端原生7组检查重复50轮、每轮80任务全部通过；逻辑worker与活动字节最终0、FD4→4／最终线程1→1。UBSan通过，25项IPC/schema测试0跳过；Windows13项通过／12项Linux专用跳过。检查背压、过期deadline、退出唤醒、drain／cancel、租约移动／延后释放、乱序完成后按ID恢复。
- 首轮r1第三次通用断言失败；r2只增强诊断，30轮中3次在所有逻辑检查后立即/proc线程数为2而非1。r3增加逻辑worker计数和最多1秒的精确终态观察；50轮中6次初读2线程，最大1132µs内回到1，队列核心未改、最终门槛未放宽。没有据此确认已安装内核／glibc的确切退出顺序或显示驱动问题。

新隔离目录`staging/scheduler_e155_r1`保留src（r1失败）／r2（诊断）／r3（终验）及原始报告。复跑时使用新的隔离目录，不覆盖原二进制／证据，保持ipc与scheduler两个同级目录：

```bash
# 工作目录：新的隔离源码根，包含ipc/与scheduler/
source /home/xchq/work/autoocr/envs/business-py310/bin/activate
g++ -std=c++14 -O2 -Wall -Wextra -Werror -pthread scheduler/test_scheduler.cpp -o test_scheduler
./test_scheduler
g++ -std=c++14 -O2 -Wall -Wextra -Werror -pthread ipc/ipc_probe.cpp -o ipc_probe
AUTOOCR_IPC_PROBE="$PWD/ipc_probe" python -B -m unittest discover -s ipc -p 'test_*.py' -v
g++ -std=c++14 -O1 -g -Wall -Wextra -Werror -pthread -fsanitize=undefined -fno-sanitize-recover=undefined scheduler/test_scheduler.cpp -o test_scheduler_ubsan
./test_scheduler_ubsan
```

终态原12项发布指纹＋N2 SHA不变（纠正此前“13项原发布”的计数），OCR仍inactive／MainPID0／enabled，GDM／SSH／FRP保持原PID运行；所有本轮测试退出、socket残留0。技术归档54成员取回到Windows `temp/scheduler_e155_r1`并独立复核；仅技术源码／合成统计／失败诊断，无图片、OCR文字、模型或许可。M0／M1a／M2／M3未完成，没有新HTTP／OCR／Extract调用或速度结论。

## E154：IPC v1隔离传输小闭环

[帧格式与边界](ipc/PROTOCOL.md)、[Python校验／I/O](ipc/ipc_wire.py)、[C++传输](ipc/ipc_wire.hpp)、[合成探针](ipc/ipc_probe.cpp)、[测试](ipc/test_ipc_wire.py)。板端`staging/ipc_e154_r1`用GCC7.5／C++14／`-O2 -Wall -Wextra -Werror -pthread`直接编译，17项测试0跳过，测试窗口FD4→4／线程1→1；本机Windows6项纯逻辑通过、11项Linux专用跳过，不当作Windows IPC验收。代码无模型或HTTP导入，未调用Extract，不复制第二套七模型；原发布及N2候选SHA均不变，OCR仍停止、GDM／SSH／FRP运行。

必须使用已有业务Python3.10环境，不用系统Python3.6。新隔离目录中复跑示例（已存在的证据／二进制不要覆盖）：

```bash
source /home/xchq/work/autoocr/envs/business-py310/bin/activate
g++ -std=c++14 -O2 -Wall -Wextra -Werror -pthread ipc_probe.cpp -o ipc_probe
./ipc_probe --self-test
AUTOOCR_IPC_PROBE="$PWD/ipc_probe" python -B -m unittest test_ipc_wire -v
```

本板未安装CMake，`ipc/CMakeLists.txt`仅提供独立目标，未实测CMake配置。C++当前不解析JSON，只核对已知请求的原样回包；Python严格校验唯一probe操作。它不是OCR阶段后端，不提供业务乱序聚合、线程池或有界队列背压；完整M0／M2仍待实现。归档与原始技术JSON在Windows `temp/ipc_e154_r1/verified`，详细证据／限制见技术日志E154。

## E153：det N2独立验证终态

E153终态：`staging/det_batch_e153_r1`已构建独立FP32 N2候选，13/13 GPU数值／原框、3个profile边界和4种越界查询拒绝通过；补齐原服务库环境后，进一步在`business_cross_image_r3`按同image_type／shape配不同图片，10个N2组＋6个N1尾项的26/26文字／分组／Markdown／特殊字段及坐标一致，NER业务0、det／rec两侧CPU回退均0。本机／板端各12项测试通过。四进程ABBA已完成：每13图平均N1=6482.008ms、routed=6700.607ms，慢3.37%、预热全机NvMap多约1.30GiB；候选保留但不默认推广。M1b独立小闭环完成，不代表线程池／IPC／新批HTTP上线；OCR保持停止、GDM运行、原发布未变，任务均退出。详见[计划](../plan.md)和上级技术日志E153；以下E151保留历史状态。

E153完整业务离线验证需继承原服务的运行库和离线／CPU线程环境，再运行对应脚本；不能只激活venv。缺失项目CUDA库搜索路径会导致Myelin按无版本名查找cuBLASLt失败，见日志E153。首轮失败报告不覆盖；business_r2保留同图普通／增强配对的历史质量结果，business_cross_image_r3才是不同图片配对补测；两者均不是在线调度或GT精度验收。初版技术归档47成员、新版跨图片补充18成员，Windows副本位于temp/det_batch_e153_r1，SHA／原始样本已独立复核。

```bash
export LD_LIBRARY_PATH=/home/xchq/work/autoocr/lib/cuda10.2:/usr/local/cuda-10.2/targets/aarch64-linux/lib
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_AUTOINSTALL=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
unset PYTHONHOME
```

2026-10-29：E144已补NVJPG提交trace、RGB像素及HTTP配对验证；当前候选业务不一致，**未采用，原服务仍软件解码**。IPC方案见[plan.md](../plan.md)第3节；IPC代码、线程池、det batch和硬解正式接入均未完成。

当前补充（E150／E151）：断线前OCR／GDM按授权停止，Extract随共用服务暂不可用、代码未改。清理显示事件积压后继续M1b，**独立det N2的CPU门槛13/13已通过，GPU构建时SSH中断，当前构建／服务状态待重连核验，未主动切换原服务**；下面E143／E144的active状态是历史事实。硬解RGB候选仍不采用。

## E143已实测（保留当时结论）

- 原`autoocr-jetson-dev.service`仍active/enabled、健康ready，只监听127.0.0.1:5051；没有重启或替换发布。
- 两张既有图片各跑imageType=3和1，共4次OCR。响应哈希均与E129保存的原结果一致，NER调用增量为0。没有调用或修改Extract。
- 原发布源码、配置、七引擎／插件及已有overlay指纹前后不变。单次请求耗时不是性能基准。
- 用板端GCC7.5／C++14编译`jpeg_decode_probe.cpp`成功；动态链接现有Tegra `libnvjpeg.so`，使用匹配32.7.1头文件和`TEGRA_ACCELERATE`。
- 2张真实图、2个合成JPEG解码成功；另外7例覆盖渐进式／444／PNG／灰度、坏头、像素超限、字节超限，按预定状态拒绝。共9个合成用例。
- 本机和板端均通过新增6项Python单元测试。没有设备级硬件调用trace、RGB像素对照或硬解后的OCR效果证据，不能据此宣布全链路硬解已上线或有加速收益。

## 文件与范围

| 文件 | 作用 |
|---|---|
| [baseline_batch_ocr.py](../baseline_batch_ocr.py) | 固定原13图中的两图、普通／增强四项串行对照；仅保存统计／哈希，拒绝Extract请求 |
| [jpeg_decode_probe.cpp](jpeg_decode_probe.cpp) | 独立进程执行NvJPEGDecoder::decodeToBuffer；默认只检查几何，E144新增可选RGB24候选输出，不接HTTP |
| [check_jpeg_probe.py](../check_jpeg_probe.py) | 生成合成用例并测试既有两张真实图，保存技术报告，结束清理合成图片 |
| [compare_jpeg_rgb.py](../compare_jpeg_rgb.py) | 小内存RGB／像素诊断和设备trace统计，不加载OCR模型 |
| [compare_jpeg_ocr_http.py](../compare_jpeg_ocr_http.py) | 复用原服务的普通／增强配对；无损PNG只作质量测试传输，明确不调用Extract |
| [CMakeLists.txt](CMakeLists.txt) | 配置匹配SDK及V4L2路径；板端未安装CMake，本轮实际使用下述g++命令，CMake配置尚未实测 |

探针只接受8-bit、非渐进式、YCbCr 4:2:0 JPEG；字节上限16MiB，像素上限3200万。可选第二个路径参数输出RGB24候选，仅允许偶数尺寸，独占创建0600文件，不覆盖已有文件。退出码0为该模式检查通过，3为不在探针范围，2为输入／执行错误。退出码3不认证输入合法，也不是已经实现了软件回退；未来集成时由输入校验后选择原软件路径。这些是**探针限制**，没有改变原OCR接口。

SDK默认错误处理和损坏码流的完整性检查还需处理，探针保持进程隔离／测试超时；不直接复用为处理不可信上传的生产模块。

## 板端证据

目录：`/home/xchq/work/autoocr/staging/batch_ocr_e143_r1`。源码、程序、报告及SDK均在已授权项目挂载下，没有写入正式发布或系统SDK目录。

- `baseline_r1.json`：4项原服务对照及前后指纹。
- `baseline_e129_comparison_r1.json`：4项响应哈希与E129一致。
- `jpeg_probe_r1.json`：9个合成用例＋2张真实图。
- `audit_r1.json`：环境、依赖哈希、健康状态及未完成项。
- `evidence_manifest.json`：本轮初版构建／验证来源及报告哈希；不包含后续未执行的CMake路径修订。板端该修订另存`CMakeLists.r2.txt`，未覆盖初版。

Windows技术报告副本位于项目忽略目录`temp/batch_ocr_e143_r1`，不含图片像素或识别正文。

依赖只下载、校验、解压，没有安装系统包：

- NVIDIA `nvidia-l4t-jetson-multimedia-api_32.7.1-20220219090344_arm64.deb`，SHA256 `8f088ce087d009b67785df56076bcd589c9fafc486c284c0660afd9b1bf8c586`。
- Ubuntu `libv4l-dev_1.14.2-1_arm64.deb`，SHA256 `1a67e281ea8e1169dcc8ee84bec1a140f492dfbabb6b4c290917cea132c8f4d0`；仅使用其头文件，链接板端已有runtime。

本轮实际编译命令（复跑请使用新隔离目录，勿覆盖已保存证据）：

```bash
MM="$PWD/sdk/extracted/usr/src/jetson_multimedia_api"
COMMON="$MM/samples/common/classes"
g++ -std=c++14 -O2 -Wall -Wextra -DTEGRA_ACCELERATE \
  -I"$MM/include" -I"$MM/include/libjpeg-8b" -I"$PWD/sdk/v4l2/usr/include" \
  jpeg_decode_probe.cpp "$COMMON/NvJpegDecoder.cpp" "$COMMON/NvBuffer.cpp" \
  "$COMMON/NvElement.cpp" "$COMMON/NvElementProfiler.cpp" "$COMMON/NvLogging.cpp" \
  -L/usr/lib/aarch64-linux-gnu/tegra -Wl,-rpath,/usr/lib/aarch64-linux-gnu/tegra \
  -lnvjpeg -l:libv4l2.so.0 -lpthread -o jpeg_decode_probe
```

## 必须继续验证

实际输出为YUV420M三平面，而非RGB。513×257合成图的U/V为256×128；2049×2989真实图为1024×1494，2118×3033真实图为1059×1516，均涉及奇数边界向下取整。不能忽略边缘、直接拉伸色度平面，或直接使用有限范围视频YUV公式声称与JPEG RGB相同。

E144已继续下面的独立实验，不覆盖E143原始记录。奇数边界在验证前不接硬解主链，保持原软件路径；其它JPEG变体／合法格式也保留原软件路径。M1a通过后才组合硬解；M1b可先独立沿用原软件解码验证batch，不同时更换多个变量。

## E144：硬件提交／RGB／HTTP配对

- 新隔离目录：`/home/xchq/work/autoocr/staging/batch_ocr_e144_r1`，旧E143全部清单成员复核未变。原服务MainPID3022／worker3069未重启，原发布／引擎／插件／overlay／libnvjpeg未改变。
- 将Ubuntu匹配包`strace_4.21-1ubuntu1_arm64.deb`仅项目内解压，未安装系统包。只跟踪自己启动的合成图解码子进程，不attach现有服务；只记录open／close／raw ioctl，无图片读写数据。
- 新探针trace：`/dev/nvhost-nvjpg`打开2次、成功ioctl 3次，其中1次`CHANNEL_SUBMIT`（低32位`0xc080481a`，对照板端内核UAPI的命令26）。证明该次测试确实提交NVJPG硬件任务，不代表所有编码格式都硬解、GPU利用率或加速倍数。
- RGB候选使用明确的三平面／stride、近邻色度展开和JPEG全范围转换假设；3个预先固定纯色色块与Pillow RGB完全一致。奇数RGB输出拒绝、已有文件保护、0600权限及精确字节长度检查通过。
- 两张额外真实偶数图2318×3410／2038×2998在8百万像素预算内，原RGB MAE为0.02551／0.41830，最大差29／33；原缩图／JPEG90后MAE为0.05422／0.88239，最大差12／18。均为诊断，不直接等同识别准确率或自动通过门槛。
- 另选E126已验证样本中的两张偶数图1918×2902／2158×3074，各做普通／增强，共4组、8次原服务OCR。候选RGB经无损PNG传递，并逐字节验证PNG往返不改变RGB；此法只为避免第二套模型驻留的质量对照，**不是部署IPC或性能方案**。
- 配对结果：全响应严格一致0/4；顺序文本序列和Markdown各2/4；坐标0/4；原图尺寸4/4。暂未区分其它两组的拆并框／空白／字符差异，无GT，不能说候选一定识别更差；但按预先固定规则，当前候选不采用，不用失败后改门槛掩盖差异。
- 8次HTTP模型调用增量：det8、rec245、layout8、cell4、YOLO8、enhance30、NER0；无新增CPU回退／错误，Extract调用0。原服务继续软件解码，硬解功能没有接入原接口。
- 板端可用内存约590～610MiB，测试每项前要求384MiB，像素差分按128行分块；未加载额外模型、停止服务或清理其它进程。CMake仍未实测，实际沿用GCC7.5直编；本机／板端各11项相关单测通过，旧9合成＋2真图几何回归也通过。

板端报告：`geometry_r1.json`、`rgb_r1_plan.json`／`rgb_r1.json`、`http_quality_r1_plan.json`／`http_quality_r1.json`、`audit_r1.json`及`evidence_manifest.json`；设备trace为`device_trace_r2.log`，输入只是合成纯色图。真实RGB／PNG临时数据已清理，报告不存识别正文。Windows技术副本位于忽略的`temp/batch_ocr_e144_r1`。

候选复跑入口（应另建目录／报告，保留失败证据）：

```bash
python compare_jpeg_rgb.py --probe ./jpeg_decode_probe \
  --trace device_trace_r2.log --output rgb_new.json
python compare_jpeg_ocr_http.py --probe ./jpeg_decode_probe --output http_quality_new.json
```

后续优先定位色度／IDCT／原预处理及文字序列差异，硬解暂不推广；可独立推进原软件解码的det batch验证。构建新引擎前先审查内存，若需停原服务（Extract也会暂不可用）须确认窗口，不能直接增加一套模型或擅自停服务。硬解仍为计划要求，不因本轮候选失败就删除。

接口参考：[NVIDIA NvJPEGDecoder](https://docs.nvidia.com/jetson/l4t-multimedia/classNvJPEGDecoder.html)。该网页标记32.7.4；本轮编译依据是已校验的32.7.1实包，不以网页版本代替实机ABI。

## E151：原软件解码的独立det batch2（CPU通过，构建待核验）

隔离目录`/home/xchq/work/autoocr/staging/det_batch_e151_r1`，保留原ONNX与N1引擎，只输出新候选。原ONNX动态N、结构检查通过且无Reshape／Flatten；实际CPU执行13图后，数值／原框13/13一致，5个双图组重复／反序正确，3个尾项走N1。没有用改shape标签代替执行验证。

| 文件 | 范围 |
|---|---|
| [det_batch2.py](../det_batch2.py) | 固定13图／指纹、原源码det预后处理、CPU门槛、独立FP32 N2构建及逐图GPU检查 |
| [check_det_batch_boundaries.py](../check_det_batch_boundaries.py) | min／opt／max合成输入实际执行、重复／反序和仅查询的范围外拒绝 |
| [evaluate_det_batch2.py](../evaluate_det_batch2.py) | 26组普通／增强原OCR质量回放，保留宽rec；NER业务0，不调用Extract／HTTP，不把预计算det后的页耗时当作加速 |
| [benchmark_det_batch2.py](../benchmark_det_batch2.py) | 质量通过后4独立进程ABBA raw-det计时、输出哈希检查及端点资源，不是完整OCR基准 |
| [test_det_batch2.py](../test_det_batch2.py) | 分组／尾项／错误／所有权／ID拆批及逐页回放契约测试 |

构建参数与验收规则在[plan.md](../plan.md)第4节预先冻结，源文件SHA／原发布指纹与报告门槛在每阶段重新核对。本机／板端各8项测试通过。CPU门槛已过不等于GPU／业务／性能通过；GPU构建过程中SSH中断，尚无构建完成证据或候选执行／业务／计时结果，不认定构建已失败或板子已重启。先检查旧PID／工件／指纹及资源，不能直接再启动一份；CPU原始JSON尚未取回本机，benchmark最新参数补传也待核对。详情见技术日志E151。不为得到正收益改原门槛，不修改原发布、授权、计数或启动设置。
