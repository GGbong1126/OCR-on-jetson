# 批量OCR交付与复盘索引

状态与最终实测数字以[计划](../../plan.md)最新验收及[技术日志E161](../../../TECHNICAL_LOG.md)为准；本文件是实现索引，不以文件存在代替上线成功。

E164增量：[Plan2实现/启动/限制](../multistream/README.md)与[完整计划](../../plan2.md)。独立`batch-ocr-plan2-e164-r1`候选支持rec1/2/4异步槽、共享有界crop队列、CTC/范围外CPU独立worker，显式stage/2；最终r19只补辅助线程启动失败清理。核心113/候选35/应用29/UBSan候选35板端无跳过，内部/公开/硬解N2及最终HTTP三周期102页通过。性能明确来自r18相同计时段：软件N1、IPC整链路三轮，rec4 0.350007页秒，比同步0.297352约+17.71%；不是当前默认HTTP加速。一次画像未观测目标跨流kernel重叠，不能写成已实证GPU kernel并行；Event区间也不是纯kernel时间。早前Event异常判断已纠正为终端误读，原始报告不变。原E162已经恢复，候选不默认推广。下表先描述默认E162，Plan2文件分工见新索引。

E162采用策略：用户接受小幅性能损失，官方启动入口改为默认硬解＋det N2，显式两个开关0可退回软件N1；不是把E161负收益改写成加速。面试可讲“集成Jetson NVJPG硬解与兼容shape动态合批、C++有界双池及跨进程buffer生命周期，并通过实际调用计数和完整OCR回归验证”；不要写成所有格式均硬解、所有模型／页面均batch=2、线程绑定GPU核心或已提升吞吐。

## 先读这张映射

| 文件 | 负责什么／重点 |
|---|---|
| `batch_app.cpp`、`batch_client.hpp` | C++输入2／调度1／结果2／事件1；四条常驻后端连接、批输入顺序恢复、SIGTERM与join |
| `../scheduler/bounded_queue.hpp` | 条件变量、数量＋字节有界队列、RAII信用；出队不等于GPU已经安全用完buffer |
| `../ipc/channel_io.py`、`worker_bridge.py` | 单I/O所有者、增量帧、2CPU算法／单模型执行线程、完成事件及finalizer；不是各线程绑一个GPU核心 |
| `../ipc/ocr_worker_adapter.py`、`ocr_response.py` | 复用原六模型OCR及原图msg组装；隔离每页外壳、保护YOLO可变包装器；不接Extract |
| `../jpeg_pixels.hpp`、`../patches/nvjpeg-partial-mcu.patch` | 支持范围内真实NVJPG解码、fancy色度恢复、BGR交接；项目内修复SDK末尾补齐行问题，不改驱动 |
| `serve_http.py`、`serve_backend.py`、`start_batch.sh` | 一个Gunicorn worker＋4HTTP线程、独立模型后端与C++子进程、健康、准入、私有运行目录及安全退出 |
| `smoke_batch.py`、`verify_http.py`、`../ipc/verify_ocr_workers.py` | 用户批测试、三周期HTTP验收、固定13图的串行／线程池／N2／硬解对照；不存正文、不调用Extract |

完整请求／限额／启动／可选开关／Windows转发／回退说明见[BATCH_API.md](BATCH_API.md)。所有源文件在Windows的`edge_deploy/jetson/batch_ocr`，部署路径在Jetson项目的`overlays/batch-ocr-e161-r1`。

## 能讲清的设计边界

- C++负责应用调度和硬解，原Python/CV算法与TensorRT适配保留，不是重写全套OCR为C++。两个Python算法线程可重叠库调用与GPU等待，不宣称突破纯Python GIL或线性CPU加速。
- 一个GPU宿主执行入口仍可提交N=2；仅同type、同实际shape的det合批，其它模型N=1。先推进另一页的layout/cell/YOLO/增强依赖，避免抱着det等待一个无法产生的det；收集空等100ms受页deadline限制。
- IPC是私有AF_UNIX＋24B大端帧头＋严格JSON元信息／二进制；模型张量留后端注册表，用handle控制。首版有内存拷贝，非共享内存／零拷贝，也不承诺跨模型CUDA kernel同时运行。
- 页面信用镜像而非重复相加，CPU finalizer／GPU安全完成后才释放；新前端另256MiB，加核心768MiB合计1GiB静态显式账本，不是立即分配，也不是模型／库／JSON DOM的RSS硬上限。
- 原图坐标／Markdown严格比较，不只比较文本框数量。GT准确率与回归一致性是两件事；坏图逐项失败，模型故障锁存、不自动重试或错误回退。
- JPEG不是只看文件后缀，硬解包含完整颜色转换成本；范围外合法格式走既有软件路径，硬解执行错误不是静默重试软件。NVJPG实际提交、像素一致与完整业务一致分别验证。
- 原5051保持停止，原发布／Extract／许可计数未改；新5052只loopback，开机策略／BSP／桌面／挂载／功耗时钟均不改。重启板子后的自启行为不属于本次新入口交付。
