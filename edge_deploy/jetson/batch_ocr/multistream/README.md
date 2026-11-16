# Plan 2：有界多crop与det／rec异步槽

2026-10-30，E164应用已交付并验收：1/2/4槽探针、内部26页、四组正式计时、硬解N2组合26页及最终r19 HTTP三周期102页严格回归/边界/正常停启均通过。独立候选选rec4；按用户要求已恢复原E162，候选没有默认推广。详细证据与归档状态以Windows仓库的[plan2.md](../../plan2.md)和[技术日志](../../../TECHNICAL_LOG.md)为准（这些相对链接针对源码仓库布局）。

## 实现分工

| 文件 | 职责 |
|---|---|
| `trt_async.py` | GPU宿主所有者、独立engine/context/profile0/非默认Stream/Event；启动预分配页锁定host/device IO，`submit/ready/collect`分离 |
| `runtime.py` | 原六模型适配；保留同步API、原det/rec阈值/裁剪/预处理/CTC，AST拆分原rec函数，不另写一套识别算法 |
| `bridge.py` | 两页面CPU生产者、一GPU所有者、一独立CTC完成worker、一范围外CPU路由worker；有界预约、fan-out/fan-in和安全收尾 |
| `stage2.py` | 显式stage/2能力、group/crop/box/route/slot、多个crop Call和CPU消费确认；复用stage/1页面/传输逻辑，旧默认不变 |
| `../app/batch_client.hpp` | C++同一8项/16MiB rec-ready兼容槽扫描；CPU槽忙不挡GPU；det及前驱有限优先推进，结果按图片ID还原 |
| `probe_async.py` | 实际1/2/4rec与det隔离探针，原shape合成FP32张量，不是GT或吞吐benchmark |
| `gpu_trace.py`、`inspect_trace.py` | 可选自有进程时间线、CUPTI stream ID和严格CSV分析；缺权限/无kernel不能当成功，普通服务和计时不用此路径 |

每crop始终batch1；不补固定宽度、不截到2304。GPU窄路由W32～1536、宽路由1537～2304，2305～3200保留原CPU模型，CPU执行不占唯一GPU宿主线程。其它模型先经过有限阶段屏障，不能宣称全链路恒定五路kernel并行。

同一个逻辑rec槽的窄/宽context各自独立，但互斥复用一套固定IO银行；det N1/N2也互斥复用。`collect(as_bytes=True)`在槽仍拥有页锁定存储时复制一次独立bytes，不经大NumPy副本中转。GPU完成通知不等于槽信用可复用：C++须等独立CTC消费后的`rec.consumed`。

## 预算与失败

核心768MiB总信用不增加：active512MiB重新分为2×128MiB页面、240MiB固定IO池上限、16MiB准备暂态；其余ingress64MiB、transport128MiB、result64MiB不变，HTTP另256MiB。准备先预约4倍目标输入大小以覆盖原NumPy预处理暂态，入队后降为实际输入bytes；8项/16MiB等待子限额和跨语言镜像不是第二份实体张量。

权重、context/workspace、第三方分配另测，**这不是1GiB整进程RSS硬上限**。候选页预约128MiB低于E162的256MiB，超额会明确LIMIT；只验证现有13图，不承诺所有最大尺寸场景。单tensor仍64MiB。

队列满则生产者等条件变量，不丢crop；producer结束且本组所有任务有安全终态/消费确认后才返回。模型失败服务级锁存，不重试推理、不错误回退CPU；内部CTC异常可导致本批/候选服务失败，不承诺其它页继续成功。未提交取消可回收，已经提交须Event/安全同步确认；同步未知保留隔离资源，不能SIGKILL/reset。先join全部新worker，再关闭wakeup/selector。

## 检查与已知边界

- 单元/集成覆盖满空、零crop、80个CPU范围外crop、乱序、旧ID、重复、deadline、错误/取消/退出、原AST语义、共享IO和bytes隔离。
- 确定性负对照：CPU槽阻塞时，旧C++代码会挡住GPU，新兼容扫描必须推进；保留失败日志，不靠放宽断言通过。
- r19再补启动边界：辅助Thread.start失败先安全收尾已有worker，未启动线程不join。旧r18两项负对照失败、新35项/UBSan35通过；仅构造/关闭改变，计时段AST与r18相同，性能仍标r18，不冒充r19重测。
- 最终板端核心113/候选35/应用29/UBSan候选35均0跳过；Cpp全工程/CTest5沿用r16相同输入构建。Windows候选24通过/11项Linux或编译条件跳过，不冒充板端完整集成。
- 硬解N2组合实际4次配对、10页硬解/16页软件、解码错误0；固定IO239,035,392B，未超过240MiB。最终HTTP默认硬解N2→软件N1→默认重启各34页（rec始终4），另PNG/坏图/400/413/504/429/404验证及正常清理；随后真正回退原E162并四页严格一致，不把HTTP中间关det/jpeg当整套旧E162回退。
- 真实CUDA探针rec1/2/4分别最多2/3/5个未完成Event（含det）；完整13图×type3/1的双/四槽公开数据及调用严格一致。数量/Stream配置或Event在途都不能单独证明kernel重叠/吞吐提升。
- 普通用户nvprof/CUPTI元信息被权限拒绝；已获准且仅执行一次sudo自有进程画像，四页严格一致/安全退出，工具仅项目解压、系统权限不改。63,818条映射活动中四个窄rec流均有kernel，但本次det-rec/rec-rec/copy-compute跨流重叠均0，不能称已证实kernel并行，也不能据此断言无画像运行必然串行。
- 同四页普通控制12.028823秒/画像17.347338秒（观察差+44.21%），只是单次两进程诊断，UID/cache/调度/先后不同，不是纯因果开销或性能benchmark，不修正正式数字。**纠正早前终端误读**：s1 rec Event累计实际104.838秒，候选总wall364.368秒，并非1004.838秒；原始JSON一直正确，没有据此认定Event异常的依据，额外9次idle校准也通过。Event区间可含排队/宿主提交间隙，不当纯kernel时间，CPU/队列wall与GPU Event不可相加，最后26页快照不冒充三轮平均。
- 不改`/Extract`、原发布、Windows授权/计数、BSP/时钟/桌面/FRP/挂载/自启，无新模型依赖或引擎重建。

范围外CPU推理不在GPU宿主线程的调用栈中，但旧前驱屏障仍等待在途rec（含CPU路由）收尾，不保证任意前驱/CPU/GPU阶段都并行。基类`threads_alive`仅计2页面CPU＋GPU，完整退出看`closed`及额外worker收尾条件/OS采样，不能把这个字段当全进程线程数。

## 正式结果（非HTTP吞吐）

Xavier NX / L4T32.7.1 / CUDA10.2 / TRT8.2.1，原模型精度/shape不变，同一Cpp二进制、软件解码/det N1。每组26页预热后正/反/正三轮78页聚合；各104页候选结果一致、资源门槛全过。每组此前另26页参考＋26页串行校验，不混入这78页。

| 路径 | 页/秒 | 相对同步 |
|---|---:|---:|
| 同步0 | 0.297352 | 基线 |
| 异步rec1 | 0.285395 | −4.02% |
| 异步rec2 | 0.340100 | +14.38% |
| 异步rec4 | 0.350007 | +17.71% |

预定规则是“最快有效候选2%以内的最少槽”，rec2低约2.83%，因此选择4。不是事后选择最好单轮；0→1还包含fan-out/传输/预分配/信用重分配等变化，不能全归因多Stream。没有锁功耗/时钟，这只是冻结13图两种type的有限顺序实验，不是GT、稳定p95/p99、24h或当前默认硬解N2 HTTP的加速结论。

## 启动与回退（当前默认运行E162，不要重复启动）

必须先核对并正常停止当前5052的实际master，不照抄历史PID，不同时启动两套模型。

候选独立目录为`/home/xchq/work/autoocr/overlays/batch-ocr-plan2-e164-r1`，交付后读取自己的选择配置并在前台启动：

```bash
root=/home/xchq/work/autoocr
cd "$root/overlays/batch-ocr-plan2-e164-r1"
slots=$("$root/envs/business-py310/bin/python" -c 'import json; print(json.load(open("installation.json"))["selected_rec_slots"])')
AUTOOCR_REC_SLOTS="$slots" bash app/start_batch.sh
```

正常Ctrl+C/退出并确认清理后恢复E162：

```bash
env -u AUTOOCR_REC_SLOTS -u AUTOOCR_BATCH_DET -u AUTOOCR_BATCH_JPEG \
    -u AUTOOCR_BATCH_RUNTIME_BASE \
    bash /home/xchq/work/autoocr/overlays/batch-ocr-e161-r1/app/start_batch.sh
```

E162仍默认硬解＋det N2。健康检查用既有Python标准库（板端无curl）；批客户端仍为`app/smoke_batch.py`，接口/限制见[原批API](../app/BATCH_API.md)。

启用候选后，健康诊断的`runtime.session.rec_slots`及批响应`execution.rec_slots`应为4；`rec_submitted`体现实际提交，`rec_high_slots`只是逻辑占用高水位，不等于同时执行的kernel数。每批仍1～4图，每crop仍batch1。
