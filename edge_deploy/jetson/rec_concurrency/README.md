# E166：REC 实际 GPU 并发定位

2026-10-30。用户要求持续完成隔离测试，允许本项目所需正常停5052和自有进程画像；测试后恢复原E162，/Extract不动。新授权不沿用E164的一次画像授权，不修改系统性能分析权限。

## 结果（本轮有限观测）

**已实现四路以上真实REC kernel同时执行；最多观测到16路，而不是只看到16项在途。** 这是固定原shape、设备常驻输入的隔离诊断，不是当前HTTP服务已经默认启用16路，更不是16倍加速或硬件绝对上限。

| 原始输入宽W（N=1,C=3,H=48） | 单宿主常驻enqueue峰值 | 独立owner线程持续提交峰值 |
|---|---:|---:|
| 32 | 3 | 16 |
| 160 | 5 | 16 |
| 1287 | 5 | 7 |
| 2081 | 5 | 8 |

旧Slot提交W160对照峰值1。线程W160的>=16路共同kernel时间累计约1443.845us（1.444ms），W32仅约1.6us，W2081的>=8路约1.152us；**峰值很短，不是整网始终16路同时运行**。每次模型调用含许多kernel，部分低资源kernel可重叠，不代表主要卷积都能16路驻留。这里只测试最多16逻辑路，长宽输入的观测峰值也不是其普遍上限。

常驻输入路径把检查/shape准备/输入输出搬运移出重复enqueue，较旧Slot路径出现重叠，支持宿主供给和交接路径会影响重叠机会；没有把这些因素逐项消融。线程版还改为各worker独立连续enqueue→本流等待，不再每轮等整组完成，**不能把5→16唯一归因为增加CPU线程**。同一个CUDA primary context内各线程独占不同TRT context/stream/buffer，经CUPTI context ID检查，未去锁并发复用同一context。

普通9组（slot160、resident四宽、Graph四宽）和线程四宽均完成1/2/4/8/16路正反矩阵，输出与该原FP32引擎参考有限/allclose/CTC索引一致并安全关闭。9个有效REC画像（5普通＋4线程）另行采集，统计不同流kernel的共同物理时间区间；不把整次推理/Event/copy区间当kernel重叠。

## 方法与范围

- 只复用原FP32/noTF32窄/宽REC engine、原TRT8.2.1/CUDA10.2，不重建模型，不改生产源码和启动默认值。
- 独立合成FP32输入，batch=1，固定宽32/160/1287/2081；同shape不补白。用于调度/数值诊断，不是GT、真实图片质量或HTTP吞吐。
- 每独立进程最多16个独立engine/context/nonblocking stream/固定buffer，逐项分配且要求至少3GiB MemAvailable；容量不足明确停止扩大，不OOM试探。
- 输入预装、每context预热3次。`resident`只enqueue已有设备输入，全部提交后统一收集；`graph`只捕获固定shape推理、复用原buffer；`slot`为已有submit/collect对照。Graph不支持保留原错误，不悄悄回退。
- 单owner各进程1/2/4/8/16路，正/反两轮，每档9轮中前3轮预热、后6轮计时，最后逐槽验证有限/allclose/CTC索引。线程版每context启动预热，每档每worker12次连续调用、正反两轮，每档最后逐worker验证；计时含线程推进，不含验证。两种运行方式、同步粒度和计时范围不同，不当纯线程数的单变量速度比较。普通小基准不带profiler，不与历史整链路17.71%比较。
- 另用项目内nvprof仅记录自有候选GPU时间线，性能与画像分开。需要时增加同CUDA context、各自TRT context的提交线程，不能并发复用context/buffer。
- 解析同一时刻正在执行kernel的不同REC Stream数，报告峰值、持续时间及>=2/3/4/8/16的时间；整推理区间重叠、copy重叠和未完成Event不算kernel并发。报告是本引擎/shape/测试上限内观测，不冒充GPU所有负载的硬上限。
- 保留所有失败和各版本。未知GPU同步状态时隔离保留资源，不杀GPU进程/重置驱动。最终恢复原E162、默认硬解N2、REC取消、四页金标一致；原发布/Extract/boot/unit/系统配置指纹不变。

## 失败、修正与恢复

- Graph普通四宽都运行成功，但首个nvprof Graph32进程signal139（wrapper15），未返回closed。控制器立即停止后续GPU和恢复，不重复同Graph画像；保留原失败及frozen_r1实际源码。没有堆栈级定因，不断言Graph模型本身不可用，也没有Graph kernel并发结论。
- 该子进程已退出，无残留模型进程，NvMap归还574156KiB原基线；新空CUDA context初始化/同步均0且boot/units不变后，r3显式续接线程测试。没有kill GPU/reset/改驱动权限。R1原报告保留unsafe历史，最终由control_threads_r1.json证明完成和恢复。
- 非REC低资源PTX正对照初次cuModuleLoadData709、无kernel且安全关闭；positive_initialized.py只增加cudaFree(NULL)显式初始化runtime primary，同PTX已成功，不释放实际allocation、不reset。控制峰值不计作REC路数。为了在无服务干扰的窗口修正对照，身份核对后曾短暂SIGSTOP仅等待root子进程的CPU控制器；GPU任务自然完成后补有限对照，确认closed后SIGCONT继续最终恢复。
- R1 scripts包/frozen_r1和r3续接实际脚本分别冻结/核哈希；r2只上传未解包。4分析单测Windows/板端通过，未重跑历史完整服务单测或CMake，生产代码/Cpp未变。

原E162已恢复master23702，5052 ready/空闲、默认硬解N2、REC取消，四页普通/增强金标严格一致；/Extract未调用未改，原发布/系统/启动配置未变。候选未默认推广。最终审计、SHA归档与Windows独立复算以postflight_r1.json和temp/rec_concurrency_e166_r1/verification_final.json为准；不再启动新GPU测试。

## Windows独立复核与性能解释

技术包`technical_evidence_r1.tgz`：3,899,499B、91成员，SHA256 **`93c5d2ed6040feb0fba4b03fbe65a7e732f866c5b51a3bde8646dade8d36890f`**。本机已验证外层/每成员SHA和大小、安全路径/类型、R1与r3实际脚本哈希、78份生产源码逐字节一致、板端候选Cpp固定SHA、原E162实际Cpp/启动默认/四页金标/ready空闲、全部有效原始CSV和性能聚合。`verification_final.json`状态`archive_members_sources_script_versions_timelines_and_restoration_verified`。

首个本机验证把installation的第79项板端可执行文件当成Windows源码而失败；修正仅对确切`app/autoocr_batch_app`验证已知f4...64位SHA，其余78项仍全部要求本机存在且字节匹配，不泛化忽略缺失。旧失败未影响测量/原始报告。

另独立选取峰值共同区间的中点，直接扫描原始CSV，无依赖并发计数器：W32和W160分别确有16个不同REC流kernel覆盖同一点；W160该例共同区间36.033us，累计>=16为1443.845us，约占本窗口已映射REC kernel活跃时间并集0.1313%（不是全机GPU利用率）。

同一线程版微基准（全部16套资源固定驻留，1/2/4/8/16仅改变活跃前N路）的聚合结果：

| W | 1路 次/s | 2路 | 4路 | 8路 | 16路 |
|---|---:|---:|---:|---:|---:|
| 32 | 88.717 | 118.683 | 106.300 | 97.529 | 98.115 |
| 160 | 67.846 | 79.613 | 72.982 | 69.570 | 66.984 |
| 1287 | 10.547 | 10.078 | 10.012 | 9.994 | 9.316 |
| 2081 | 6.488 | 6.348 | 6.320 | 6.303 | 5.873 |

这解释“可同时计算路数”不等于“最快配置”：短/典型宽本轮线程版2路最快，长/宽1路最快；不是普遍最优选择规则。单owner resident W160则4路76.526次/s、16路70.733次/s；Graph W32本轮8路142.505次/s，不能因普通速度结果宣称Graph kernel并发（画像失败）。这些都是固定合成crop、不同调度/计时口径的有限微基准，未锁频/功耗、无GT/实图整体质量/HTTP负载，不能换算到历史17.71%或直接推广默认配置。
