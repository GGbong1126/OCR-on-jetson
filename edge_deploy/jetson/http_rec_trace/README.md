# E168：真实批OCR的四owner REC kernel验证

2026-10-31。**已验证：复用实际E167代码的真实HTTP/OCRBatch采样窗口中，四个不同REC owner确实存在GPU kernel共同执行区间；不是持续四路。** Windows已取回原始CSV独立复算，当前E167普通用户服务已恢复master10656、5052 ready/空闲，4owner存活/错误0，/Extract未改未调用。PID只代表验收时点，后续操作必须重新核实。

用户新授权本轮正常停止5052、仅项目进程必要sudo采样、完成后恢复当前E167四路；不是沿用E164单次权限。临时诊断没有覆盖生产代码，也不是直接附加到旧master27246或恢复后的10656进程上；核心后端/REC方法/调度/Cpp/HTTP业务原文件复用，32页输出严格相同。R1初始化失败已安全关闭/恢复E16710110并4页一致，未开始画像；R2使用新目录`http_rec_trace_e168_r2`，不覆盖R1。

## 结果与怎么理解

Xavier NX/Tegra194、Ubuntu18.04/L4T32.7.1、CUDA10.2/TRT8.2.1+localcp310，同原FP32引擎、4owner、硬解/DET N2开关1，原testpic16独立图×普通/增强32页，4图/批。映射同context1内8条窄/宽流，按4owner合并；共有272,991条已映射REC kernel记录，128,598条拷贝活动不计计算并发。

| 至少同时有多少个不同REC owner的kernel | 累计共同区间/s | 占已映射REC kernel活跃并集 |
| --- | ---: | ---: |
| 2 | 8.079694 | 30.0165% |
| 3 | 0.950394 | 3.5308% |
| 4 | 0.100476 | 0.3733% |

这些是“至少”口径、相互包含，不能求和。分母为已映射REC kernel活跃区间的**并集26.917541s**，不是各kernel时长之和、32页HTTP耗时或整个GPU利用率。大部分REC活跃时间只有1～2路；四路重叠确实出现，但占比很小，不能称四个REC全网全程齐步计算、更不能称4倍提速。没有新吞吐实验或17.71%/15.06%的因果分解，旧E167吞吐结果不变。

一组直接来自原始CSV的四路见证（同context1，均为各owner的narrow流，时间为GPU trace原始us）：

| owner | stream | kernel起点/us | kernel终点/us |
| --- | ---: | ---: | ---: |
| 0 | 340 | 170462809.828 | 170462871.621 |
| 1 | 394 | 170462809.572 | 170462818.916 |
| 2 | 448 | 170462554.462 | 170462815.492 |
| 3 | 502 | 170462768.675 | 170462827.908 |

四者交集`[170462809.828,170462815.492]us`，长**5.664us**，超过预声明0.1us阈值。这是一个见证，不是最长四路区间。0/2/3为`trt_volta_scudnn_128x32_relu_medium_nn_v1`，1为`cask_trt::computeOffsetsKernel`；包含卷积与REC辅助kernel，不假称每一条都是卷积或每个时钟四路都在乘加。不是只统计CPU enqueue/线程存活/逻辑在途。

## 失败、恢复与证据完整性

1. R1 factory初始化失败，原桥接吞掉栈。发现诊断`profile.py`遮蔽标准库，纯CPU `import cProfile`负对照明确失败；R2改名`sample_service.py`并仅在factory异常额外记栈，不改成功路径。不能因旧栈缺失而断言唯一根因。R1源码/失败报告仍在板端，关键原件也入本次归档。第一次本机R2测试还受整个旧rec_concurrency目录的PYTHONPATH影响；改为按确切文件加载区间分析器，之后本机/板端7项全过。043aeec…包只在本机，实际R2b包为e1c671291929d6b5c1fff91ebaf757c9ff4fd16b84e5ec8c39397980afc531e6。
2. R2的32页采样和全部SHA检查已经完成，idle停止采样时ProfilerStop排空超过观察器10秒限时，原`trace/profile.json`仍为failed/TimeoutError、`control.json`仍为sample_failed_e167_restored、外层return1，**没有改成“脚本全通过”**。
3. 后续`gate.json=stopped`、后端`status=stopped/phase=stopped/close_error=false`，原生nvprof退出0、Gunicorn/Cpp退出0，110,296,712B CSV完整落盘；未发现dropped/权限拒绝/进程信号错误，正常关闭后恢复E167。仅Unified Memory profiling不支持提示不影响此kernel记录。新`finalize.py`只接受这个确切Stop观察超时，并核完整32页、停止确认、原生0、安全关闭和真实恢复，另写`reconciliation.json`；没有放宽别的错误、缺页或未知GPU清理，没有重跑GPU挑结果。
4. R2普通控制32＋预热32＋采样32＋恢复4=100重复页，完整公开data SHA全与E167金标相同；另R1控制32＋恢复4=36页，合计136次重复处理，独立图仍16张。恢复由原`start_threaded.sh`在普通用户xchq下完成，REC_THREADS/SLOTS=1/4、硬解/N2开关1/1、NER仅factory1，信用全归零。82项交付指纹、原发布/旧E162/宽REC/N2及boot/四unit状态与PID不变；原5051 inactive，Extract业务0，无系统/自启/FRP/许可/Git变更。
5. 本机/板端7项分析与方法检查通过；本轮未重跑E167的185项全套/CMake/UBSan，不将旧测试冒充新跑。当前所有采样器/控制器/收尾器都已结束，不能再次启动采样或同时加载另一套模型。

板端`staging/http_rec_trace_e168_r2/technical_evidence_r1.tgz`（R2实验的第1份归档），Windows`temp/http_rec_trace_e168_r2/`。归档**11,476,513B、129成员、81份交付源码＋固定C++可执行SHA**，SHA256：

`ebd17c66e2abd8c1c476739520175c405c2564cd40eef051bbff3b5a9ff3c4e7`

`verify_capture.py`已独立核外层/逐成员、实际源码与Windows逐字节、冻结脚本、100页SHA、两次失败原件/特定超时的校核条件、流映射/原始CSV重算/四路见证交集及恢复；结果`temp/http_rec_trace_e168_r2/verification_final.json`。原严格成功版`collect.py`/`verify_download.py`未执行，不将其断言放宽或覆盖冻结脚本。证据只含技术元数据/代码/日志，不含图片、正文、权重或凭据。

## 结论边界

这是临时root诊断/CUPTI窗口中的实际业务代码证据，**不是普通用户、无采样运行时四路占比也一定0.373%**的证明。未映射kernel可能含其它模型或库内部流；分析器字段`other_model_kernel_count`只作“未映射kernel”理解，不据此给102,117条记录逐模型归因，所有比例只用已明确映射的REC流。未做SM逐周期指令采样/核心绑定、GT/CER、全673图或24h。本轮不新增优化；持续提高四路重叠/吞吐需要另做供给与算子占用实验，不能直接增加线程数保证收益。

以下保留冻结方法，实际R1/R2结果和失败状态以以上记录为准。

## 方法与范围

1. 普通用户核对E16782项交付SHA、原发布/宽REC/N2/旧E162、boot/systemd与实际master身份；先在未修改的线上服务跑原16图×type3/1共32页HTTP控制，严格核对全部公开data SHA。
2. 正常停止这一组模型，禁止旁加载第二组。独立`staging/http_rec_trace_e168_r2`运行临时诊断服务：原E167的`serve_backend.py`、C++二进制、`serve_http.create_app`全部原文件，Gunicorn同单worker/4线程/相同限制；仍真实HTTP/OCRBatch，硬解/DET N2开关1、REC owner4。
3. 临时`backend_trace.py`仅在原Resource创建后读取窄/宽流的CUPTI ID，继承的`Resource.run`/同步/关闭、模型/profile/FP32/输入/预后处理/调度完全不替换。仅idle状态从主I/O线程调用ProfilerStart/Stop；未更改运行默认、不覆盖生产代码。
4. 诊断后端从启动时由本项目nvprof包装。HTTP入口复用原create_app与私有交换，只外置生命周期管理以知道真实后端PID；不把nvprof父PID冒充后端。诊断进程限本项目/root，HOME/cache/tmp在新目录隔离；不采全用户/全系统，不改驱动权限/BSP/时钟/桌面/自启/FRP。
5. 独立诊断预热32页，idle开采样后同16图×两type32页、4图/批。原黄金SHA严格相同；再idle停采样、正常逐一停Gunicorn/C++/后端并等待nvprofflush，GPU安全未知则保留不强杀、不reset、不加载新模型。
6. 安全结束后普通用户用原`start_threaded.sh`真正恢复E1674路，普通/增强4页严格对照和健康/信用/开关/原指纹检查；优先恢复再CPU分析。

## 判据

将8条窄/宽流映射到4个owner，仅保留同context内这些流的真实kernel行；memcpy、其它模型与其它context排除。复用E166已测试的区间扫描，按不同owner去重，报告>0.1us共同区间峰值、累计≥2/3/4路时间及REC kernel活跃并集占比，保留原CSV和峰值见证行。

有四路共同区间才能说本窗口观测到四路实际计算；不能由owner/alive/逻辑在途4、吞吐+15.06%或E166合成16流推断。本次是带采样开销/临时root诊断窗口，不是新吞吐基准；没有重叠时只称本窗口未观测，不断言普通运行绝不并行。独立图片仍16张，无GT/24h/全673图结论。

`test_analysis.py`覆盖四owner、窄宽去重、拷贝/其它context排除、短交叠/相接、不完整或被拒trace，以及确认包装器没有覆盖原推理方法。旧`rec_concurrency/analyze.py`打包原样复用，按源SHA核验。
