# Xavier NX板端迁移

> 历史迁移记录。当前源码交付入口、E167/E168结论和运行依赖以[仓库README](../../README.md)为准。下文的“当前”、PID及服务状态均指相应实验发生时；被引用的私有技术日志、计划和原始报告未纳入公开源码仓库。

2026-10-06 E126：`staging/rec_soak_e126_r1`完成45图两路径的有界持续运行，90预热＋270正式共360次业务全部一致。预热后约17.06分钟，全局NvMap逐页最大增长0.3125MiB、RSS39.19MiB；19564次页内GPU识别、宽引擎120次，七模块CPU回退均0。进程退出后NvMap较空闲基线仅多0.3125MiB，原r2/可选包/BSP/挂载不改、HTTP仍关闭。47成员归档与本机复算通过；末尾摘要打印缺少导入已在独立`post_archive_fix`修补，原始报告不变。非小时级/并发/GT或真实超2304CPU驻留验收，详见上级TECHNICAL_LOG E126。

2026-10-06 E125：使用既有testpic扩大回归，目录673图中预先选取原13图之外的32图。原版/宽适配各64次普通＋增强业务、共128次执行，64/64结果一致；3745个识别输入/CTC全部对应，26次CPU→宽GPU，其余3719次原GPU输出字节不变。累计45个不同图片文件已验证，非673图全量；本批最大识别宽1970，没有2304附近/更宽真实输入证据。77项本机/33项板端测试、45成员技术归档已核验；stage为`staging/rec_corpus_e125_r1`，原r2和E124可选包不变、HTTP关闭。本轮扩大覆盖，非新测速/GT/长稳结论；下一步继续已有图片的持续运行，详见上级TECHNICAL_LOG E125。

2026-10-06 E124：已部署独立可选包`overlays/rec-wide-e124-r1`，不覆盖原r2；开启后原32～1536和新1537～2304两GPU引擎共同覆盖rec，范围外仍原CPU、不改输入、不在推理错误时回退。三个新进程关闭26页→开启130页→关闭26页，182/182一致；四边界及实际超范围回退通过。预热后104页约6.01分钟内NvMap快照无正增长、RSS最大增长21.625MiB，有限持续运行门槛通过，非瞬时峰值或小时级压测。68项本机/37项板端测试、45成员技术归档复核完成；原发布/系统/挂载不改，HTTP仍关闭。配置选择、无HTTP加载检查及重启回滚见[REC_WIDE_ADAPTER.md](REC_WIDE_ADAPTER.md)，详细结果见TECHNICAL_LOG E124。

2026-10-06 E123：独立`staging/rec_wide_e123_r1`已完成rec FP32宽profile（1537/1696/2304），构建457.298秒、87.47MiB。4真实/5边界数值与CTC、26/26业务质量全部一致；四进程ABBA的208正式/104预热页均通过，整批均值3826.103→3348.871ms、3804.453→3344.348ms（耗时降低12.47%/12.09%），慢页两路径约减半。预热全局NvMap额外265.3～265.6MiB，低于384MiB门槛，非服务峰值。48项本机/17项板端终验、47成员归档复算通过；原七模型/配置/插件/BSP/挂载不变、HTTP仍关闭。状态`wide_pipeline_gate_passed_not_promoted`：尚未接入原r2，下一步可回滚业务适配、持续资源和更宽回退验证，不自动激活或启动服务。详见上级TECHNICAL_LOG E123。

2026-10-06 E122：`staging/rec_fallback_e122_r1`完成78页原七模型诊断，3426次GPU rec/12次CPU rec、零错误/业务漂移。仅同图普通/增强两路径产生4个超宽原张量（1622/1695/1697/2081），实际profile上限1536；后预热CPU每次2.52～3.33秒，每慢页两次共约5.17/5.95秒，占54.0%～58.5%。初次CPU会话创建0.361秒单独记录；其余24案例无回退。31项本机/7项板端终验、30成员归档核验通过，原r2模型/配置/插件/BSP/挂载不变、HTTP仍关闭。下一步独立FP32宽引擎，只接原超宽张量并保留原范围外回退；尚未构建候选或测速新方案，详见上级TECHNICAL_LOG E122。

2026-10-06 E121：`staging/ner_pipeline_e121_r1`完成完整离线OCR＋NER四进程反序对照，208正式/104预热业务处理全部结果一致；两轮整页均值3826.974→3749.757ms（快2.02%）、3748.784→3783.103ms（慢0.92%），未达门槛且方向不稳，保留原r2发布。NER子调用在本批快约10%，不等于整页收益；本实验`run_ner=True`，不能当作`/OCR`接口基准。预热NvMap增量少5.3～5.9MiB、RSS快照少337～350MiB，非峰值资源验收。41项本机/12项板端测试、29成员归档与复算通过，服务仍disabled/inactive；下一步先诊断慢页rec超宽CPU回退，不直接继续NER half。详见上级TECHNICAL_LOG E121。

2026-10-06 E120：`staging/ner_overhead_e120_r2`完成原NER/完整FP32插件NER的四进程ABBA对照，240正式/336总调用无漂移和错误；两轮均值142.773→120.120ms、142.820→119.503ms，降低15.87%/16.33%，预热全局NvMap增量少5.63MiB。仅六个固定真实输入，长输入收益显著、部分短输入变慢，不代表整页收益。r1无GPU错误定位为业务加载后的`CUDA_VISIBLE_DEVICES=-1`被新进程继承，r2在测试内恢复启动环境解决，失败证据保留；不修改服务环境或原引擎。29项本机/12项板端测试、40成员归档及复算通过。性能门槛已过但未推广，原r2部署/HTTP关闭状态不变；下一步先做完整业务端到端反序验证，详见上级TECHNICAL_LOG E120。

2026-10-06 E119：`staging/ner_protected_e119_r2`完成完整FP32 NER结构对照，直接复用E117/E118两个已验证插件，25Norm＋12注意力全部可见FP32。构建54.570秒、461.955MiB；6真实/40边界数值及标签通过，13图×两路径26/26完整业务一致、无额外CPU回退。r1因文档SHA多抄一字符在准备阶段停止，保留记录并已纠正，不是模型故障。56项本机/17项板端相关测试、29成员技术归档及独立复算完成。尚未测性能、未量化或替换r2发布；下一步原NER/新FP32结构同板开销对照，详见上级TECHNICAL_LOG E119。

2026-10-06 E118：`staging/ner_softmax_e118_r1`已完成FP32注意力局部插件。保持原score/8、有限-FLT_MAX掩码、Softmax语义，72组CPU原图真实特征＋32组边界检查全部通过；208调用0错误，实际FP32插件层可见。全遮挡4组保持原均匀分布，未改阈值或语义。仅局部验证，未整网/业务/性能验收，原r2与HTTP关闭状态不变。39项本机/11项板端相关测试、24成员技术归档及独立复算通过；下一步E117/E118接回完整FP32 NER结构对照，详见上级TECHNICAL_LOG E118。

2026-10-06 E117：独立`staging/ner_norm_stable_e117_r1`使用移位中心化FP32 LayerNorm插件，原epsilon/权重不变，Inspector明确显示FP32输入/输出。150组原特征对原CPU与FP64均通过；30组合成压力输入对FP64全通过，旧3组低方差MAE约改善9.5万～21万倍（非速度收益）。旧CPU相等门槛仍有15组合成差异，原E116失败不改写。540插件调用、0错误、重复一致；仅局部稳定性通过，未整网/实体/性能验收，原r2及HTTP关闭状态保持。28项本机/11项板端相关测试和27成员技术归档/独立复算通过，详见上级TECHNICAL_LOG E116补记与E117。

2026-10-05 E115：`staging/ner_mixed_e115_r1`独立NER候选构建227.455秒、465.051MiB；声明只将72个编码器投影设half，其余敏感计算FP32。6实输入、3长度边界、6补齐掩码输入有限且标签一致；完整13图×两路径共52次业务，26/26实体跨度/标签/阈值成员关系及正文/字段一致，无额外CPU回退。但实际Inspector仅2条记录，主体为Myelin，未分别暴露目标和保护节点，不能确认内部精度保护。最终`not_accepted`指精度证据不足，不是已观察到业务退化；benchmark0、不换r2。28项本机/19项板端相关测试、29成员技术归档及独立复算通过，原七模型/BSP/挂载不变、HTTP仍关闭。后续先做可核验的敏感算子边界小样；详见上级TECHNICAL_LOG E115。

2026-10-05 E114：在`staging/cell_partition_e114_r2`完成完整cell三段组合（229节点FP32前段→7节点FP32/mixed块→959节点FP32后段），18采样实际FP32。13图×两路径×三分支78次离线业务，分段FP32全一致；mixed仅18/26联合通过（18组cell正例中10组通过，另8组负对照），8组分组/3组全文/2组NER变化。匹配诊断说明顺序和新增框能影响业务，即使对应框偏移小、逐行OCR及数字符号全一致。质量失败，不benchmark、不换r2。36项本机/24项板端相关测试及41成员技术归档/独立复算通过；原模型/BSP/挂载不变、HTTP仍关闭。cell保留FP32，下一步NER，详见上级TECHNICAL_LOG E114。

2026-10-05 E113：单元格三卷积小块完成FP32/局部FP16构建、6实输入对照、24次微引擎执行/0错误。原两输出均消费、1195节点均为输出祖先；mixed实际只三Conv半精度，SiLU/外部输出FP32，没有额外half计算。FP32对CPU前缀容差6/6，mixed对GPU FP32容差0/6，末层最大误差0.135～0.193；均输出有限、重复一致。尚无完整cell候选、业务验收或测速，原r2/HTTP关闭状态不变。21项本机/12项板端相关测试、26成员归档已核验；现有NVMe `staging/cell_probe_e113_r2`保存证据并保留r1审计脚本失败记录，详见上级TECHNICAL_LOG E113。

2026-10-05 E112完成：E111版式两输出FP32候选独立有效框＋业务验收26/26通过，同板240次计时两轮均值367.5965→356.1492ms、371.6583→357.8115ms，改善3.114%/3.726%。预热单会话NvMap增量764.754→718.371MiB，减少46.383MiB，非完整服务峰值。未达预定5%速度门槛，保留独立备选，不切换r2、不进行整页性能验收。历史原表数值失败仍保留，非GT精度；23项本机/10项板端相关测试、39成员技术归档及独立复算通过。证据在`staging/layout_semantic_e112_r1`；原发布/系统/挂载不变、HTTP关闭，下一步cell实际消费输出及敏感边界，再NER。详见上级TECHNICAL_LOG E112。

2026-10-05 E111：独立版式FP32候选移除业务未读取的48MB掩码输出及20节点。真实执行格式核验通过；原检测表逐元素检查未过，但六组有效框全部匹配，13图×两路径26组完整业务诊断的文字/字段/NER一致、rec回退两支均4。旧数值失败保留，未benchmark、未改发布。证据在`staging/layout_outputs_e111_r1`，39项本机/30项板端测试及32成员归档已核验；后续独立做有效框＋业务验收与性能对照，详见上级TECHNICAL_LOG E111。

2026-09-24 E87：七模型默认TensorRT GPU发布及离线验收完成，目录`/home/xchq/work/autoocr/releases/jetson-r2`；systemd已指向r2，仍disabled/inactive。按用户要求不执行激活、不启动HTTP，不改原授权校验。以下E85/E86描述及激活命令保留历史含义，不作为当前执行步骤。

## E87 交付结果

### 当前能力摘要（E89）

已完成Xavier NX上的七模型默认TensorRT FP32部署及原业务离线链路验证：OCR检测/识别、版式、单元格、YOLO、NER、增强均有板端引擎；版式/单元格通过ONNX图改写和GridSample CUDA插件适配TRT8.2。保留CPU预后处理及超范围OCR回退，不是全流程纯GPU。

r2发布、哈希校验、真实执行统计、旧r1回退配置和systemd配置已具备；最新已归档审计的服务仍disabled/inactive，激活和HTTP验收尚未执行。正式benchmark、长稳/并发及GT准确率不在已完成结论内。用户本轮反馈手动测试完毕但未提供新报告，以下验收数字仍是E87原报告，本轮未再次SSH或推理。

### 优化范围（E91）

当前r2仍是七模型FP32/noTF32部署，未将量化候选切入发布，也没有剪枝/稀疏化、QAT或蒸馏。已完成TensorRT引擎接入、ONNX常量/形状折叠与兼容改写、自定义GridSample CUDA插件、每session的engine/context及按容量复用I/O设备buffer；这些属于推理工程优化/兼容适配，不等于模型压缩，也未逐项证明独立加速收益。E92新增的INT8 PTQ独立实验见下节。

WSL中做过FP16 NaN排查、保守混合精度和固定context/补白分桶实验，但未作为当前Jetson默认方案。多图补白出现文本变化，未推广。详情见技术日志E91；下述内存数值属于FP32配置，不是量化后结果。

### 版式选择性FP16（E110，实际精度边界不合格，不采用）

2026-10-05在现有NVMe `staging/layout_hotspot_e110_r3`完成Conv.91/92/93候选构建，前缀六组FP32范围安全、三个卷积half及后续float边界真实存在，但Inspector还出现额外FP16 Myelin融合子图，以及im_shape/scale_factor转半精度。18个采样插件仍FP32，不足以证明整条敏感路径FP32；格式门槛失败，未进入候选完整业务或benchmark，原发布不变。

构建1737.414秒，202.289→187.188MiB只是engine文件缩小7.47%，不能写成服务内存/速度收益。六组输出有限、重复一致也不是准确率验收。26项本轮测试两端通过、34成员技术归档保留所有失败记录；当前HTTP仍关闭、无激活/系统/盘变更。下一步优先评估FP32的未消费输出分支，详见技术日志E110与[模型路线](../MODEL_OPTIMIZATION_PLAN.md)。

### 增强FP32双块调度（E109，收益不足，不采用）

2026-10-05完成同权重静态batch2候选，原512分块两两组合，奇数末块仍走原batch1；不改重叠、顺序、拼接和阈值。27tile容差检查及13增强图＋1普通对照全部通过，无额外CPU回退，文字/字段/数字符号/几何一致。候选与证据保存在现有NVMe `staging/enhance_batch2_e109_r2`，旧未完成r1目录保留。

| 同9tile序列均值，每配置20轮 | 原batch1 | batch2＋原尾块 | 改善 |
|---|---:|---:|---:|
| r1，原→候选 | 903.8401ms | 887.0668ms | 1.8558% |
| r2，候选→原 | 910.9585ms | 891.6212ms | 2.1227% |

合计80序列/720tile/560实际推理调用，含组装和同步拷贝、不含分块拼接及完整OCR；不是HTTP速度或GT精度。双引擎额外预热NvMap约360MiB，非服务峰值。速度未达预定5%门槛，不采用、不扩大batch4；原七模型/插件/配置/BSP/挂载不变、HTTP仍关闭。本机35项/板端21项测试、41成员技术归档及汇总复算通过。

下一项为版式选择性精度优化，尚未实施；详见技术日志E109与[模型路线](../MODEL_OPTIMIZATION_PLAN.md)。

### 增强FP32工作区（E108，业务一致但无加速，不采用）

2026-09-29在新盘`staging/enhance_fp32_e108_r1`完成同ONNX/权重/FP32/noTF32、batch1、原512分块的工作区256→1024MiB候选。构建792.33秒，27个原tile全部通过数值容差；13增强图＋1普通对照14/14文字、字段、数字符号与框几何一致，未增加CPU回退。没有改分块、拼接或阈值。

| 增强同步整调用均值，各180次 | 原版 | 候选 | 变化 |
|---|---:|---:|---:|
| r1 | 99.1149ms | 99.4175ms | 慢0.305% |
| r2，反序 | 99.5861ms | 99.6849ms | 慢0.099% |

共720正式调用，无有效提速；预热NvMap增量原271.83MiB、候选274.54MiB。工作区预算不是实际常驻内存，快照不是服务峰值。候选未过预定5%收益门槛，**不替换、不声称整页加速**。19项测试、35成员两端技术归档/汇总重算通过；原七模型/插件/配置/BSP/挂载不变，HTTP/激活仍不启用。

下一步从FP32 batch2原分块调度入手，奇数末块走原batch1，先核对逐块及完整业务再测速；此方案尚未实施。详见技术日志E108和[模型路线](../MODEL_OPTIMIZATION_PLAN.md)。

### 检测整网混合精度（E107，业务一致但性能未通过，不采用）

2026-09-29在现有NVMe的`staging/det_cudnn_e107_r1`完成低误差cuDNN候选接回整网：仅Conv.87计算/输出FP16，显式恢复FP32后连接Conv.88/91，其余浮点计算保持FP32。构建约49.52分钟，实际格式及6个原始输入＋3个形状边界核验通过；构建耗时不代表推理耗时。

13张既有图片×普通/增强26组对照的全文、字段、数字符号与框几何全部一致，无新增CPU回退。概率逐元素容差仍未全部通过，一致性不是GT准确率。

| 检测同步整调用均值，各60次 | 原FP32 | 候选 | 变化 |
|---|---:|---:|---:|
| r1 | 490.0447ms | 485.7736ms | 快0.87% |
| r2，反序 | 488.4847ms | 488.8126ms | 慢0.07% |

两轮均未达到预定5%改善门槛，**候选不进入发布，检测保留原FP32**。预热NvMap增量从771.80降至743.89MiB（约少27.91MiB），是单会话快照而非服务峰值；没有稳定提速，未继续整页/长稳测试。下一步转向增强FP32卷积/分块调度，尚未实施该新方案。

21项本机/板端测试通过，39成员技术证据两端校验、汇总重算完成。七模型/插件/配置/BSP/挂载不变，实验进程退出、HTTP仍关闭；未激活、未提交Git。完整配置和边界见技术日志E107。

### 检测热点实现筛选（E106，历史单层阶段，整网结果见E107）

新盘`staging/det_error_e106_r1`完成同源六特征的误差拆分：输入/权重舍入MAE约0.005～0.006，输出舍入约0.010～0.013，原FP16实现残差约0.50～0.59。排除旧implementation后，新微型引擎实际使用CudnnConvolution、FP16输入/输出和FP32外部输出。

相对GPU FP32，MAE降为0.01229～0.01470、最大误差0.938～1.079；六真实输入可重复、三个形状边界可执行，候选15调用/0错误。但allclose仍0/6通过，不能称无损，也不是40倍提速。未构建完整det候选、未测最终文字/字段或性能、未替换七模型发布。下一步先验证整网两分支精度边界与完整业务，质量通过后才测速。

8项本机/板端测试通过；24成员技术证据已两端校验归档。原发布、BSP及挂载不变，HTTP/激活仍不启用，详见技术日志E106。

### 检测单层精度边界（E105，可执行但未通过精度/业务验收）

2026-09-28在新盘`staging/det_boundary_e105_r2`完成三组对照：单层FP32可构建；FP16计算直接FP32输出仍无法满足约束；FP16计算/输出后Identity转FP32可构建。第三组6个真实特征和3个min/opt/max合成形状均可执行、输出有限，Inspector确认FP16卷积格式和FP32外部输出；每个成功引擎在补充诊断中15调用/0错误。

但候选相对GPU FP32的MAE0.497～0.586、最大绝对误差44.04～62.18，不能因无NaN而认定无损。CPU/GPU FP32原容差下20/15974400元素不通过，独立CPU卷积与原CPU图完全相同、204个FP64位置诊断与舍入差异相符；原失败门槛未放宽，补充诊断明确不算质量验收。

本轮未构建完整det候选、未做最终文字/字段对照或benchmark、未替换发布。下一步先区分输入/权重FP16舍入与计算实现误差，再筛选实现；不能仅凭tactic名断言内部累加精度。10项本机/板端测试通过，r1工具问题及r2全部技术证据两端校验归档，详见技术日志E105。

### 检测热点局部精度试验（E104，首轮构建失败，未替换发布）

2026-09-28在现有NVMe的`staging/det_hotspot_e104_r1`完成原FP32范围诊断：3张原开发图×普通/增强共6个实际det输入。热点Conv.87输入/输出最大绝对值61.50/3181.18，但Conv.91/139输出分别达到107621.55/1270364.25，不能整段直接改FP16。这是带探针的ORT原FP32图范围，不是TensorRT内部累加精度证明。

首个严格候选保持原动态profile和256MiB工作区，只请求Conv.87的计算为FP16、输出仍FP32，其余浮点计算/输出FP32。TensorRT8.2经过947.232秒构建后报告该层没有满足精度约束的实现，OBEY按要求终止。没有候选engine，也没有候选业务/性能结果；不将信息级工作区提示归因为OOM，不通过放宽约束掩盖失败。

新增构建、范围检查、验收及审计工具，18项本机/板端测试通过。验收/benchmark工具已准备，但本候选并未执行这些阶段。下一步先做单层FP16输出→后续恢复FP32的可行性试验，通过后再考虑整网重建；不是已实现加速。完整配置、风险与证据见技术日志E104。

### 按模型选择优化路线（E103，诊断完成，短profile候选不采用）

2026-09-28已完成七模型两轮反序分层诊断：14份报告，640次正常计时＋320次独立Profiler计时，原输入/同engine输出核验通过。正常Session.run与Profiler计时分开；使用同张既有图片的普通/增强代表输入，不是全量GT或整页性能。

据此建立[各模型优化路线](../MODEL_OPTIMIZATION_PLAN.md)：识别/增强/检测重点是卷积；layout/cell名称含Grid的回调仅约0.75%/1.05%，先不重写插件；NER大块ForeignNode不能据此拆解内部算子；YOLO每页成本很小，降低优先级。

首个候选为保持原张量/精度的rec短宽FP32引擎（32/128/256）＋原宽引擎路由，不padding、不截断、不改变超宽CPU策略。240个短crop序列相同，13图×普通/增强26/26完整业务文字/字段一致；但测速失败：

| Session.run均值，各1000调用 | 原FP32 | 短宽FP32路由 | 变化 |
|---|---:|---:|---:|
| r1 | 23.1075ms | 26.5417ms | 慢14.86% |
| r2，反序 | 23.0389ms | 26.5024ms | 慢15.03% |

额外预热NvMap166.02MiB，快照不是峰值/服务总内存。质量一致不等于性能通过，也不是GT准确率；**候选不采用，原发布保持不变**。下一步优先检测热点的受控局部混合精度，然后增强FP32卷积/小batch，再版式、cell、NER；这些候选尚未实施。21项本机/板端测试通过，技术报告与代码在现有NVMe及本机已做哈希归档，见技术日志E103。BSP/挂载/许可不变，HTTP仍inactive/disabled，无激活或Git推送。

### FP32识别运行时诊断（E102，未发现稳定提速，不改发布）

2026-09-27在现有新盘`staging/rec_runtime_e102_r1`完成同一FP32引擎的两轮反序测试：第一张原校准图的50个真实张量、40种宽度、范围32～1287，原输入/精度/字典/CTC保持不变。每轮分别测原实现和“相邻shape未变时跳过setter”。

| Session.run均值，各1000调用 | 原实现 | 跳过重复设置 | 延迟下降 |
|---|---:|---:|---:|
| r1 | 23.2449ms | 23.2158ms | +0.1253% |
| r2，反序 | 23.1328ms | 23.1578ms | -0.1082% |

- 没有稳定收益，不加入发布；不是HTTP/整套OCR加速。另4640次带宿主分段计时的诊断输出SHA全部相同，原序列预热/计时后逐输入检查亦通过，未做候选13图完整业务或GT验收。
- 两个同输入宽度按相同比例比较，交替相对固定重复多约3.08/3.04ms，没有复现此前WSL约130ms切换成本。预热后全部诊断显式malloc/free为0。原序列940/1000次变宽，setter均值约0.14ms，同步execute_v2宿主区间约21.5ms，后者不是纯enqueue/kernel计时。
- 当前TRT8.2单profile引擎第二context active profile为-1；本地SDK明确动态context需独立、未被其他存活context占用的profile。因此更正E101方案前提：不能直接照搬WSL10.7共享池。需要另建多profile候选或其他设计后才能评估，不代表所有TRT版本都不支持共享。未强行运行无效context，也未构建多engine缓存。
- 2/4/8/16槽位的精确shape冷LRU模拟只命中4/5/7/10次（50调用）；只是模拟，不称GPU实测。四次单rec预热NvMap增量均217.613MiB，与原方案相同，不是服务峰值或独占显存。

9项新测试本机/板端通过。技术归档`temp/jetson_jp461_r1/rec_runtime_e102_complete`已做两端哈希、19个成员和报告/源码一致性检查；原七模型/插件/配置未变，服务inactive/disabled，无激活/HTTP。下一步先做FP32 rec逐层耗时诊断再决定profile/tactic候选，尚未实施，不放宽数字/符号要求。完整口径见[技术日志E102](../TECHNICAL_LOG.md)。

### 图像增强FP16试验（E101，未通过业务一致性，未发布）

2026-09-26在现有NVMe的`staging/enhance_e101_fp16_r1`完成增强候选构建和真实业务对比。原MobileNetV3_Unet、固定`1×3×512×512`、batch1、RGB分块/阈值/拼接/JPEG流程不改，只替换增强引擎；其余六模型仍FP32。FP16 enabled＋FP32 fallback，TF32/INT8关闭、外部FP32 I/O，workspace256MiB/avgTiming8；不是INT8量化或剪枝。

- 构建约29.1分钟，引擎84.772→35.236MiB（文件缩小58.43%）。27个同输入真实tile输出均有限、后端错误0；Inspector249层中246层含FP16格式，不是纯FP16计算占比。
- 13张增强图：4张业务结果一致，9张正文/Markdown变化，3张数字或符号序列变化，5张NER变化。标题、分类号、页码字符串13/13相同仍不足以通过正文/NER验收；没有GT，不能将4/13称为准确率。
- 另1组普通路径对照通过，两边增强调用均为0。所有增强配对tile输入哈希/顺序一致，共每分支114个tile。按用户既有要求，框坐标和special内部元数据仅作诊断，文字/字段字符串仍严格比较；一组仅内部坐标变化的结果通过，因此9组失败并非纯框差异。
- 首个符号变化组按FP32→FP16→FP16→FP32复测，各自输出哈希/业务结果稳定，但跨后端仍出现相同符号替换。

**候选不启用**：原增强和识别继续FP32，未运行失败候选的benchmark，没有可报告的速度/显存收益。原七引擎、插件和deployment哈希不变，HTTP仍inactive/disabled，没有激活或计数操作。

工具：`build_enhance_fp16.py`、`optimize_enhance_fp16.py`、`replay_enhance_fp16.py`；9项新测试本机和板端通过。本机技术归档为`temp/jetson_jp461_r1/enhance_e101_complete`，已核对SHA/源码/报告引用及统计；不含图片、正文、张量、引擎权重或许可。详见[技术日志E101](../TECHNICAL_LOG.md)。

下一步建议评估FP32识别的动态宽度开销和共享引擎、有界多上下文复用，必须保持原输入/解码并补内存上限、业务回归；不是重复实现已有单context和容量buffer复用。本轮尚未实施这项运行时优化，其他模型仍可独立评估低精度，不能套用本次结论。

### OCR识别混合精度区域试验（E100，未通过业务一致性，未发布）

2026-09-26延续E99，已在新盘项目暂存目录完成两种显式FP32区域候选。固定原ONNX哈希、原输入/字典/CTC/profile，TRT8.2开启FP16和OBEY精度约束，关闭TF32/INT8，workspace512MiB、avgTiming8。不按失败图片或宽度加例外，不改变现有七模型发布。

| 候选策略 | 原失败符号探针 | 后续结果 |
| --- | --- | --- |
| 首次ReduceMean及后段FP32，前段允许FP16 | 仍丢失符号 | 不进入完整回归/benchmark |
| 前段FP32，后段允许FP16 | 恢复且重复稳定 | 13图×普通/增强22/26严格文字一致，4组失败 |

后者4处CTC差异在相同输入上均可重复，其中有数字替换、数字增加、数字丢失，另有一个其他字符丢失；1组同时影响NER。不是框坐标或允许的空格变化，因此**不能替换FP32**。没有GT，22/26也不是准确率；旧失败样本恢复并不等于整体通过。所有测试使用原板端testpic，不输出私人正文。

独立目录为`staging/rec_mixed_e100_tail_r1`和`staging/rec_mixed_e100_prefix_r1`，工具为`rec_mixed_precision.py`、`diagnose_rec_mixed.py`，6项新测试本机/板端通过。两份技术归档已校验保存在本机`temp/jetson_jp461_r1/rec_mixed_e100_{tail,prefix}_complete`；构建参数、差异类别、哈希与证据限制见[技术日志E100](../TECHNICAL_LOG.md)。本轮未benchmark两失败候选，不能报告部署速度/内存收益，FP16混合精度也不是INT8量化或剪枝。

原七个FP32引擎、插件、deployment哈希不变，HTTP仍disabled/inactive，未进入激活或计数流程。识别继续保留FP32；下一轮可优先尝试固定512输入的增强模型FP16，并以增强后的OCR/数字符号/NER一致性验收。本轮尚未实施增强候选。

### OCR识别FP16实验（E99，未通过业务一致性，未发布）

2026-09-26用户允许在新挂载盘现有项目目录保存实验。项目`/home/xchq/work/autoocr`现在映射到NVMe，约110GiB可用；本轮只使用现有文件系统，不分区/格式化/调整挂载。旧E97约460MiB可用是此前eMMC时点，不是当前项目空间。

先在3张原校准分组图片上测普通/增强路径：OCR识别每页约1.12～1.38s（48～52次调用，40～46种形状），是本组最主要模型耗时；增强约0.80～0.83s、检测约0.42～0.46s。随后只为rec构建FP16 enabled＋FP32 fallback候选，保持原始输入、字典、CTC和超profile策略。FP16不是INT8量化，也未进行剪枝。

| 识别模块配置 | FP32部署基线 | FP16候选 |
| --- | ---: | ---: |
| 引擎文件 MiB | 86.648 | 42.359 |
| 两轮原Session.run均值 ms | 23.308 / 23.208 | 16.024 / 16.103 |
| 两轮p95 ms | 53.418 / 53.281 | 27.331 / 27.281 |
| 预热NvMap扣加载前增量 MiB | 217.613 | 129.676 |
| engine声明的context内存 MiB | 43.543 | 22.054 |

计时为同一张校准图50个原始rec张量，按原顺序预热3轮/测10轮，每配置500次、正反顺序各一轮，合计2000次。包含输入检查、H2D、执行、同步、D2H和输出有限值检查；不含图片预后处理，不是HTTP或整套OCR benchmark。输入哈希一致，错误0；两轮平均延迟下降31.25%/30.61%。GPU频率端点均1109.25MHz，但未锁频/连续采样。原workspace256MiB、候选512MiB，verbosity也不同，属于候选配置比较，不是纯精度单变量消融。

内存快照是在同进程完成取样并关闭七模型GPU会话后，重新加载单rec所产生的整机NvMap增量；仍可能含共享运行库状态，不是冷启动独占显存/峰值或七模型服务总占用。Jetson是共享内存；引擎缩小51.11%不等于整机内存减半。

**质量门槛未通过**：3图普通/增强共303个同输入文字框字符序列一致（含10个共同空文本），但5张留出图×两条完整业务路径仅9/10组一致。差异组正文/Markdown变化；NER及原业务标题/分类号/页码字符串仍一致。复测发现一个`1×3×48×45`输入，FP32输出1个符号、FP16为空；两者各自重复原始输出逐元素一致，差异可复现。没有GT，不能断言哪个更正确，但不满足现行回归约束，不能以9/10当作90%准确率或放宽符号要求。

工具：`profile_business_models.py`、`optimize_rec_fp16.py`、`diagnose_rec_difference.py`。候选/报告位于板端`staging/other_models_e99_r1`，本机技术证据在`temp/jetson_jp461_r1/other_e99_complete`；只归档统计/日志/源码，不含图像/正文/权重/许可。8项新单测本机和板端通过，另3项隐私安全差异统计检查本机通过。

下一步是独立验证保留敏感算子FP32的混合精度候选，不能靠为这个失败宽度加例外来掩盖回归。若根据这5张图继续调参，它们成为开发回归集，最终验收需要新的独立样本。增强、检测、NER、版式/单元格本轮仅做原版耗时诊断，尚未新增低精度候选。原七模型FP32发布/插件/配置哈希不变，服务仍disabled/inactive，无激活/HTTP/系统或存储调整。

### 量化性能诊断与双会话原型（E97，未发布）

2026-09-26已补建与INT8相同workspace512MiB/profile/DETAILED的FP32对照，5个真实输入数值检查通过。原部署构建avgTiming=8，Python构建默认=1，原先不止workspace一个差异；扩大workspace不保证更快。

单YOLO、batch1、5图原张量按原顺序循环，每组预热25/正式100调用、正反顺序两轮，无Profiler的实测：

| 配置 | 两轮调用均值 ms | 预热NvMap扣空载增量 MiB |
| --- | --- | ---: |
| 原部署FP32，单会话 | 16.351 / 14.689 | 547.375 |
| 匹配参数FP32，单会话 | 25.724 / 26.224 | 537.504 |
| E92 INT8，单会话 | 23.441 / 24.347 | 516.102 |
| 原部署FP32，双固定尺寸会话 | 12.890 / 13.763 | 668.355 |
| 匹配参数FP32，双固定尺寸会话 | 12.002 / 14.809 | 648.738 |
| E92 INT8，双固定尺寸会话 | 11.043 / 9.387 | 607.809 |

双会话原型保留640×448/512各一个engine/context，不补白、不改变输入，计时前后与各自同engine单会话输出均逐元素相等。它体现运行时复用与量化的组合，而非纯INT8收益；额外内存不能忽略，也尚未做共享engine。这里只验证两种尺寸，不是通用服务缓存实现。

独立IProfiler诊断中，匹配FP32/INT8重格式化层数56→87、累计均值0.965→1.881ms；卷积类耗时下降。交替尺寸与固定尺寸差异明显，不能仅凭Myelin警告归因。未锁频，端点GPU频率有变化；这些是小样本观察，不是稳定生产收益、GT准确率或整套服务内存/速度。

**当前部署不切换**：E92的检测结构差异仍存在，本轮没有新增完整业务回归或HTTP验收。下一步先做共享engine、有界精确shape缓存与完整业务回归，再决定INT8是否值得启用。七模型发布、授权和NVMe保持不变。

工具：`diagnose_yolo_int8.py`（build/check/benchmark/profile/pooled），8项本机/板端测试通过。26份技术测量报告已校验归档到`temp/jetson_jp461_r1/quant_e97_complete`；板端在`staging/quant_yolo_e97_r1`。可只读查看`pooled_int8_mixed_r1.json`、`pooled_int8_mixed_r2.json`和`profile_int8_mixed_r1.json`。现有命令拒绝覆盖报告；E97时点根盘仅余约460MiB，E99项目现已映射到NVMe，允许写入指定实验目录但不擅自清理旧证据。完整参数/限制见[技术日志E97](../TECHNICAL_LOG.md)。

### YOLO INT8 PTQ首轮（E92，未发布）

2026-09-24已在板端完成真实校准/构建/执行/回归/性能测量。候选在`staging/quant_yolo_e92_r1`，EntropyCalibrator2、8张校准/5张留出、INT8＋FP32回退、FP16/TF32关闭；326个Inspector层记录中172个具有INT8格式，不是全模型纯INT8。

| 项目 | FP32基线 | INT8候选 |
| --- | --- | --- |
| engine文件 | 26.843 MiB | 10.338 MiB（-61.49%） |
| 单YOLO调用均值，两轮 | 16.597 / 14.517 ms | 19.140 / 19.714 ms |
| 单YOLO进程预热设备分配增量 | 547.375 MiB | 516.102 MiB |

调用测量含H2D/执行/同步/D2H/检查，batch1，5张留出图每轮预热25调用、测量100调用，正反顺序独立进程；不是整套服务耗时。原基线workspace256MiB、候选512MiB，属于配置比较而非精度单变量消融。设备分配是NvMap扣空载快照，不是峰值/应用独占显存；整套七模型量化内存未测。

普通＋增强共10组：全文、逐行文字、Markdown、NER、原业务标题/分类号/页码最终字符串均一致，但8组有检测框结构/编号变化（原严格结构门槛只有2组通过）。这既不是“准确率下降80%”，也不是“量化无损”；无GT且只有5张留出图。补充诊断保留在独立报告，没有放宽原门槛。

**当前候选不替换r2**：文件变小，但同步调用慢15.32%～35.80%，仅节省约31.27MiB单模块进程设备分配。HTTP仍关闭，授权流程未执行。下一步先统一构建参数、逐层profile并扩大代表性校准/标注集，再考虑敏感层保留FP32等方案。全部失败、参数、哈希与解释见[技术日志E92](../TECHNICAL_LOG.md)。

工具是`quantize_yolo_ptq.py`，流程`prepare → build → inspect → evaluate → benchmark`。每个新实验必须使用独立staging目录；各步拒绝覆盖已有报告。TRT8.2动态engine的Inspector需在执行后导出，本轮原build失败报告已保留，由独立inspection证明同一engine可执行。不要删除旧报告或重新在r1目录构建。

已完成实验可只读查看（SSH登录板端后）：

```bash
cd /home/xchq/work/autoocr
envs/business-py310/bin/python -m json.tool staging/quant_yolo_e92_r1/inspection.json
envs/business-py310/bin/python -m json.tool staging/quant_yolo_e92_r1/evaluation_details.json
envs/business-py310/bin/python -m json.tool staging/quant_yolo_e92_r1/benchmark_candidate_r2.json
```

Windows技术证据归档：`temp/jetson_jp461_r1/quant_e92_complete`，不含测试图/正文/权重。候选engine只保留板端；不要将Jetson TRT8.2 engine直接用于WSL TRT10.7。

### 内存实测（E90，单进程离线）

2026-09-24经SSH测量r2七模型FP32/batch1，加载预热后同一张2170×3032图片分别普通/增强处理各一次，100ms采样，不加载CPU参考layout/cell。`NvMapMemUsed`扣空载基线的设备分配增量约3.316GiB，样本峰值3.326GiB；tegrastats整机RAM扣基线增量约5.480GiB，样本峰值5.590GiB。进程RSS峰值另为3.090GiB，不能与前两者相加。GPU错误、OCR CPU回退和Swap使用均0。

Jetson CPU/GPU共享DRAM；NvMap是整机设备统计而非应用专属显存，差值只是近似归因。以上不是HTTP/并发/全数据最大占用，也不是额外独立显存需求。退出后设备分配已回到基线，服务仍inactive/disabled。完整口径与证据见[技术日志E90](../TECHNICAL_LOG.md)，本机报告`temp/jetson_jp461_r1/memory_e90/memory_probe_r2.json`；本轮未启动HTTP或激活。

### 手动测试入口（E88，离线、不激活）

先在Windows终端执行`ssh USER@jetson.local`。登录板端后整段执行以下Bash命令，先测试testpic按文件名排序后的第一张，不是在Windows PowerShell直接执行：

```bash
cd /home/xchq/work/autoocr
export LD_LIBRARY_PATH="$PWD/lib/cuda10.2:/usr/local/cuda-10.2/targets/aarch64-linux/lib"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

report="reports/manual_type3_$(date +%Y%m%d_%H%M%S_%N).json"
envs/business-py310/bin/python tools/compare_pipeline_offline.py \
  --release releases/jetson-r2 \
  --images testpic \
  --image-type 3 \
  --limit 1 \
  --output "$report"
```

通过时末尾应为`offline_pipeline_consistent_not_http_or_gt_validated`，且images=passed=1。若报错，保留完整报错及新报告，不覆盖历史结果或修改release哈希。

- 全部图片：删除`--limit 1`，重新生成report变量后执行。自选图片：`--images`接图片目录，不接单个文件；建议用一个新的板端测试目录。
- 增强路径：改成`--image-type 1`，输出名可改manual_type1前缀；仍用新时间戳。type3跳过实际图像增强，但模型初始化阶段有增强预热。
- 同图两次原process比较，仅layout/cell在CPU/GPU间切换，其他五个模型固定GPU。比纯GPU单次调用更慢，首次还包含加载/预热，不能当正式benchmark。
- JSON中的gpu_execution记录每模块backend/calls/errors和OCR的cpu_fallback；回归模式下cpu_reference_execution出现CPU调用是预期行为。严格文本一致不代表GT准确率，boxes_exactly_equal=false也不自动否定文字门槛。
- 当前工具不保存识别正文、图片或可视化框；若要查看这些实际业务输出，需要另外增加离线导出工具。不要把它当作已有HTTP/GUI演示入口。
- HTTP仍关闭，此时curl5051或访问浏览器不能作为本轮测试方式；本说明不新增激活流程，也不启动服务。

| 模块 | 默认网络前向 |
| --- | --- |
| OCR检测、识别 | 原TRT8.2 FP32；超profile原张量显式回退ORT CPU |
| 版式、单元格 | 新TRT8.2 FP32＋共享GridSample CUDA插件 |
| YOLO、NER、图像增强 | 原TRT8.2 FP32 |

CPU仍承担预后处理；Python/PyTorch宿主张量不代表网络在CPU执行。版式固定batch1/800×800、cell固定batch1/640×640，保留原im_shape/scale_factor输入值。FP32/noTF32，没有引入FP16或重训。

- 13张板端testpic：35个模块案例通过（含9个真实表格裁剪），GPU无错误。
- 原process普通路径13/13：两侧各634个业务框、536行OCR文字，文本、Markdown、特殊字段及NER严格一致；累计4次超宽rec CPU回退，无错误。
- 增强路径3/3：各140个业务框、127行OCR文字，文本/字段严格一致，增强55次GPU调用（含1次预热），本组OCR回退0。
- 对照只替换layout/cell，其他五模型固定GPU；不是GT准确率、Windows整套CPU对照或正式性能benchmark。原始浮点输出及Box字典不完全相等，差异已保留在报告。

验收凭据：`releases/jetson-r2/edge_deploy/jetson/ACCEPTANCE.json`；原始证据位于项目`reports/compare_layout_cell_business_r1.json`、`pipeline_r2_type3_r1.json`和`pipeline_r2_type1_r1.json`。引擎、插件、源码和适配器均绑定哈希。

Windows已取回完整技术报告至`temp/jetson_jp461_r1/e87_complete/`并核对压缩包SHA256；不包含测试图片或识别正文。设备最终剩余约1.05GB空间，后续新建引擎/下载依赖前需继续检查容量，不能改用未授权NVMe。

原`jetson-r1`目录和unit文件保留，可回退配置；不要就地覆盖任一release的模型/插件，后续修改应新建版本并重跑回归。当前unit不启动是预期状态，不是GPU加载失败。HTTP验收尚未执行。

实现工具：`lower_deformable.py`、`grid_sample_plugin.cu`、`build_grid_plugin.py`、`check_grid_plugin.py`、`build_deformable.py`、`compare_deformable.py`、`compare_pipeline_offline.py`、`stage_gpu_release.py`、`accept_gpu_release.py`。测试/转换脚本默认拒绝覆盖已有输出。

用户目标：把已验证的WSL完整server业务迁移到Jetson Tegra194 / Xavier NX，并尽可能使用GPU。当前使用用户指定的`ssh USER@jetson.local`（已与旧IP主机公钥核对一致）。项目根目录按用户要求为`/home/xchq/work/autoocr`，已有work/frp保留不动。

2026-09-24 E86：激活按用户要求暂不推进，不移除现有校验。测试图片已由用户放到板端`testpic`，SSH核实共13张。版式/单元格TensorRT化需要图改写及GridSample CUDA插件：两图各18个GridSample，版式另有23个LayerNormalization。当前是已核查的实施路线，尚未实现/验证两图GPU。详细算子和验证安排见技术日志E86；以下E85激活命令只作历史/后续参考，当前不执行。

2026-09-24 E85：独立Python3.10.21环境、ARM64业务依赖和本地TensorRT8.2 cp310绑定已安装。完整server模型已在单进程加载并通过合成预热：det、rec、YOLO、NER、增强五项使用TensorRT GPU；版式和单元格保留原ORT CPU。unit已安装并通过systemd语法验证，但保持disabled/inactive，5051尚未监听：等待首次激活和真实测试材料传输许可，不能称为完整上线。过程及失败证据见[技术复盘](../TECHNICAL_LOG.md) E85。

## E85 当前部署结构与剩余步骤

```text
/home/xchq/work/autoocr/
  releases/jetson-r1/          已校验业务源码、原权重、原ONNX
    pth/tensorrt/              板端生成的5个FP32引擎（硬链接，不重复占空间）
    edge_deploy/jetson/        完整业务适配、deployment.json、启动/健康检查
  envs/python310/              独立ARM64 Python，不替换系统Python3.6
  envs/business-py310/         应用venv
  lib/cuda10.2/                指向现有cuBLAS/Lt的项目内无版本链接
  runtime/                    本板独立日志/许可/计数目录，禁止复制WSL许可
  staging/                    原模型、离线包、派生ONNX及构建引擎
  reports/                    数值对照、构建、合成模型检查
```

GPU矩阵：det高宽32～960（原预处理尺寸为32倍数）、rec宽32～1536、YOLO高宽64～640（原letterbox/stride64）、NER序列2～512、增强512×512/batch1。保留FP32/noTF32及原预后处理。OCR超profile原张量回退CPU并计数；GPU运行错误保持失败并使请求失败，不静默切换CPU。`healthz`明确显示每个模块后端及预热/调用/错误；诊断计数不是许可余额。

已完成：det复用板端既有实图张量数值检查；rec四种合成宽度、YOLO四种合成形状、NER四种合成长度（含mask）、增强合成输入均CPU/GPU allclose；完整7模块单进程合成预热无错误。这些不等于实图HTTP/字段准确率或benchmark。

首次激活必须沿用原流程；尚未执行。用户确认后，在板端终端使用：

```bash
cd /home/xchq/work/autoocr
export AUTOOCR_SOURCE_ROOT="$PWD/releases/jetson-r1"
export AUTOOCR_RUNTIME_DIR="$PWD/runtime"
export LD_LIBRARY_PATH="$PWD/lib/cuda10.2:/usr/local/cuda-10.2/targets/aarch64-linux/lib"
umask 077
envs/business-py310/bin/python releases/jetson-r1/edge_deploy/jetson/serve_jetson.py --activate
```

设备申请码仅交给授权管理员，不写入报告或Git；输入激活码为隐藏输入。代码仍调用原验证/保存/计数初始化函数，不复用WSL许可，不重置已有余额。激活后再启用服务、通过SSH转发执行OCR→Extract实图对照、验证正常重启；这些操作尚未完成，不应提前标记ready。

计划访问方式为板端仅监听127.0.0.1:5051，Windows执行`ssh -L 15051:127.0.0.1:5051 USER@jetson.local`后访问`http://127.0.0.1:15051/healthz`。当前转发已建立，但服务未启动；没有开放公网/局域网裸HTTP。

下面为E81～E84历史记录，旧版本/空间/固定尺寸说明只适用于对应阶段。

## 实测环境

| 项目 | 2026-09-23实测 |
|---|---|
| 系统 | aarch64、Ubuntu18.04.6、L4T32.7.1、Python3.6.9 |
| 计算组件 | E81已新增CUDA10.2运行库/cuDNN8.2.1/TRT8.2.1，Python绑定8.2.1.9可导入；未装完整Toolkit、PyTorch、ORT或Docker GPU runtime |
| 内存与系统盘 | 约16GB内存；E84构建/归档后eMMC根分区可用约5.8G |
| NVMe | 119.2G、未挂载；未检测到签名不代表无数据，禁止写入/分区/格式化 |
| 网络/时间 | E83已通过jetson.local连接；板端已有DHCP默认路由、DNS可用、日期正常，本轮未修改这些设置 |
| 主机 | WSL22.04在D盘，另有WSL24.04；物理D盘约7.15GiB、C盘约35.71GiB可用 |
| USB | usbipd5.3只见CH340(COM5)等外设，未见NVIDIA Recovery设备 |

## 当前路线：不升级系统，离线运行库已安装

[JetPack4.6.1](https://developer.nvidia.com/embedded/jetpack-sdk-461)对应CUDA10.2/cuDNN8.2.1/TensorRT8.2.1。基础运行库已经离线安装，但不等于完整业务兼容；Python3.6不能直接运行WSL的Python3.10环境与固定依赖。

1. 使用jetson.local，后续文件放work/autoocr；旧暂存目录已经在构建结束后迁入staging/jp461-r1。电脑下载、SSH传包的离线方案也保留，不需要开启Windows网络共享。
2. 后续Python依赖、模型、engine和缓存继续做版本及容量审查。发现系统/BSP升级或删除即停止。根分区剩余约5.8G，不装整个JetPack元包或大型容器，不自动清理文件，不动NVMe。
3. 在隔离目录实现旧版TRT/CUDA适配并验证OCR，按需选择可用ARM64/Python环境。命名张量/v3执行接口、CUDA12库名、Python新API、ONNX算子支持均须审查，不只是换requirements版本号。
4. 逐一适配版式、单元格、YOLO、NER、增强，最后接回完整业务并与Windows/WSL对照。最终目标仍是完整服务和尽可能GPU执行，兼容性/容量不足应明确报告，不宣称全部模块必定能GPU运行。

先前升级路线已取消：不再要求Recovery USB、载板刷机配置、Ubuntu20.04刷机主机或50–60GB刷机工作区。相关调查保留在日志E78供历史复盘；应用部署仍需另行核算真实可用空间。当前不执行刷机、BSP升级或do-release-upgrade。

## E81离线产物与恢复入口

- 本机：`temp/jetson_jp461_r1/`，包含官方仓库元数据、10个固定版本deb、bundle.json、SHA256SUMS，共约395MiB软件包。不要提交这些包。
- 板端：`/home/xchq/work/autoocr/staging/jp461-r1/`，包含元数据/包、安装审查、日志、4个原ONNX和新检测engine；由旧`/home/xchq/autoocr-stage-jp461-r1/`原地重命名迁入，旧路径已不存在。仅用户暂存目录，不是业务发布目录。
- 签名链：板端已有NVIDIA OTA公钥 → Release.gpg → Release中的索引SHA256 → 索引中的deb SHA256。仅SHA匹配不能代替签名；`prepare_trt82_bundle.py`不自动验证GPG，也不自动安装。
- apt模拟为10新增/0升级/0删除；实际apt本地路径处理出错，未解包。随后dpkg dry-run通过，再用同组包dpkg安装成功；无依赖补装或BSP升级。详细记录见E81。
- E83已同步`smoke_trt82_v2.py`，真实GPU实算通过，见smoke_trt82_r2.log；旧失败证据不删除。
- E83已执行`models/probe_onnx_trt82.py`，OCR det/rec解析通过，layout/cell失败，见parse_onnx_r1.json/log；不是全部模型可部署。

安装、GPU与解析完整证据已取回Windows的`temp/jetson_jp461_r1/e83_complete/`；E84构建日志、迁移记录和真实图片检测输出在`e84_complete/`，含Windows后处理框对照报告。接下来验证识别引擎与剩余模块兼容性，再接完整业务。单独探测脚本不替代完整HTTP、业务一致性和合法激活验收。

## 当前板端目录与检测复验

```text
/home/xchq/work/autoocr/
  staging/jp461-r1/       原始模型、匹配运行库包、构建引擎及历史日志
  tools/                 板端独立验证工具
  cases/det_case_r1/      真实图片预处理张量、Windows CPU参考（不入Git）
  reports/               迁移和GPU验证证据
```

当前检测engine仅支持`[1,3,960,704]`，不能当作任意尺寸业务引擎。板端Python3.6环境可复验如下，输出必须采用未存在的新目录：

```bash
cd /home/xchq/work/autoocr
export LD_LIBRARY_PATH="/usr/local/cuda-10.2/targets/aarch64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3 tools/infer_tensor_trt82.py \
  --engine staging/jp461-r1/engines/ocr_det_fp32_960x704_r1.engine \
  --case cases/det_case_r1 \
  --output reports/det_tensor_trt82_r2
```

E84原报告为`reports/det_tensor_trt82_r1/comparison.json`：MAE约3.0033e-8、最大绝对误差约4.2677e-5、allclose通过、0.1阈值mask差异为0。三次调用不是正式benchmark；此工具不识别文字，不执行板端图片预/后处理或HTTP业务。

## 先读取实际环境

Windows PowerShell可以将只读探测脚本通过SSH stdin发送给板端Python；在本机提示时输入SSH密码，不将密码写进命令、脚本或聊天。此命令不会在板子上安装文件，不读取序列号/网卡地址/许可/计数。

```powershell
Get-Content -Raw -Encoding UTF8 /path/to/OCR-on-jetson\edge_deploy\jetson\preflight_jetson.py |
  ssh USER@jetson.local python3 -
```

依赖版本必须根据报告选择；不要直接执行WSL的install_gpu_dependencies.py或套用其systemd unit。

## 迁移边界

1. 用户已取消JetPack升级，E78的同意不再作为授权。保留当前OS/BSP/内核/驱动，不通过安装依赖间接升级；明确禁止分区/格式化NVMe，不擅自改功耗/锁频或停已有无关服务。
2. 采用新的板端工作目录和独立环境。先确认ARM64/Python/JetPack/CUDA/cuDNN/TensorRT及可用内存/磁盘；WSL的x86_64 venv、cu126 wheel和engine不复制用于执行。
3. 复用经过清单/哈希校验的源码与模型，排除license.dat、余额/已用码、私人日志及测试文档。板端硬件授权单独按原合法流程办理，不复制WSL激活状态或绕过授权。
4. 原工具含Python3.10/Linux x86_64/CUDA12/TensorRT10.7固定校验，以及Path.is_relative_to等Python版本要求，需做单独板端适配；先确认实际版本，不能仅删除校验来“兼容”。
5. 板端按自身TensorRT重建OCR engine，先FP32与原输入/预后处理对照；各模块逐一验证实际GPU执行、有限输出和内存余量，再接完整HTTP。ORT/PyTorch需要JetPack匹配的ARM64构建，不把通用aarch64或桌面CUDA wheel视作Jetson兼容保证。
6. 先loopback、单worker；经SSH转发联调。同图结果与现有WSL/Windows基准比较，通过后再配置板端独立服务及回退。首次板端激活、sudo安装需求或资源冲突出现时明确报告，不修改许可算法或已有系统来强行部署。

依据：[TensorRT兼容性说明](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/developer-guide/index.html)说明默认序列化引擎依赖构建平台/GPU/TRT版本；[PyTorch Jetson兼容表](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html)记录JetPack对应的历史构建，不能据此推断现有WSL依赖或全部模型已兼容。
