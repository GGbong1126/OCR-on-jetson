# 板端配置与适配器快照

这些小型文件于 2026-11-16 从原板端只读取回，102 个取回成员已逐一核对 SHA256。没有取回或发布模型、引擎、插件二进制和运行环境。

| 本仓库位置 | 原板端相对项目根的位置 |
| --- | --- |
| `../server.py`、`../layout/` 等业务源码 | `releases/jetson-r2/` |
| `base_factory/jetson/`、`base_factory/business/` | `releases/jetson-r2/edge_deploy/jetson/`、`business/` |
| `rec-wide-e124-r1/` | `overlays/rec-wide-e124-r1/`（不含引擎） |
| `batch-ocr-threaded-e167-r1/start_threaded.sh`、`selection.json` | `overlays/batch-ocr-threaded-e167-r1/` |
| `../edge_deploy/jetson/batch_ocr/` | E167 overlay 内对应的 C++、Python、CMake、limits 和第三方头文件 |

`base_factory` 保留原基础发布适配器版本，用于理解固定 SHA 依赖；开发目录中的同名文件可能是后续版本，二者不能直接互换。E167 批组件与本次板端源码对照，在统一换行后没有差异。

配置中的路径和 SHA 用于描述原部署依赖，部分指向本仓库未包含的引擎、报告或插件。模型相关词典、tokenizer 和依赖版本同样需要外部准备。公开业务源码中的四个私有许可常量已脱敏，因此它也不能冒充原发布的逐字节完整镜像。

这不是自动安装目录；不要直接覆盖已运行的原发布。移植时先准备独立目录，核对模型、适配器、环境与完整性检查，再做输出一致性和资源验收。历史实验控制器中包含停服/启动/采样流程，应在明确的隔离实验任务中使用。
