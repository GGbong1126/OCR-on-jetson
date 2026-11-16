# 完整业务的TensorRT OCR部署

2026-09-25 E93：WSL服务、发布包、venv、实验目录和TensorRT SDK已按用户要求删除；本文为历史记录，不再是可直接执行的当前部署步骤。Jetson部署及Windows源码保留，详见[技术日志](../TECHNICAL_LOG.md)E93。

## E76当前：其余模型CUDA部署

2026-09-18已切换到`autoocr-business-gpu-20260918-r1`和独立`.venv-gpu-r1`。增加`AUTOOCR_OTHER_BACKEND=cuda`（默认仍cpu），必须与`AUTOOCR_OCR_BACKEND=trt_fp32`配合。三图实际业务验证及正常重启通过，细节见TECHNICAL_LOG E76；以下E75数据是历史已验证版本。

- OCR det/rec仍使用原TensorRT FP32引擎及显式超profile CPU回退。
- Layout/Cell使用onnxruntime-gpu 1.19.2的CUDAExecutionProvider；显式关闭TF32，每session arena上限768MiB、按需扩展、卷积工作区限制，禁止运行错误自动切换Provider。arena上限不是进程/设备总显存上限。
- YOLO/NER/增强使用torch 2.8.0+cu126和torchvision 0.23.0+cu126。FP32、关闭TF32；增强batch=1，保留512切片与原预后处理，不启用原GPU默认FP16分支。原server默认行为不变，部署通过显式参数注入。
- `gpu_models.py`实际执行5个模块的合成预热，layout/cell通过一次ORT trace统计CUDA/CPU节点执行，不把GPU Provider注册当作执行证据。私有临时trace只含合成输入执行信息，汇总后移除。CUDA不支持的形状/控制等节点仍可在CPU上，health显示节点计数，不声称所有算子GPU。
- health的`models_execution`区分预热/真实调用/错误，失败锁定不健康，拒绝原可选模块捕获异常后返回部分业务成功。真实NER仍仅按原业务需要调用。
- 新建`.venv-gpu-r1`，不替换原`.venv`；官方cu126 PyTorch安装后，install_gpu_dependencies.py从原已验证环境清单固定其余依赖，使用ORT GPU替换CPU包（不能同环境共存两者）。保留本机TRT10.7.0官方wheel和SDK，不安装WSL显示驱动。
- 打包使用`release_business.py build --gpu-models --trt-engine-dir ...`创建新`gpu_business_source_models`包；旧CPU/TRT包内容不改。新unit同时检查OCR与其他模块CUDA预热，不接受旧CPU健康响应。

版本依据：[PyTorch官方历史版本](https://pytorch.org/get-started/previous-versions/)提供2.8.0/0.23.0的cu126配对；[ORT CUDA文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)列明1.19.x对应CUDA12/cuDNN9、import torch的依赖加载方式与use_tf32/显存选项。安装CUDA toolkit本身不会把CPU PyTorch变成GPU版。

E76新包48文件、1121415318字节，manifest SHA256 `1a7c6e7b142dc34049c4a365de2df5fac6761dd64e560a67ab110af9c8843b0d`。三图142框及指定字段严格一致；det3/rec198、layout3/cell2/YOLO3/NER3/增强24批实际GPU调用，无错误/超profile回退。设备显存快照3465MiB（不是进程峰值）；4GB余量有限，禁止随意增加worker/增强batch，不同时跑另一套模型服务或GPU构建任务。未做长文本NER/所有尺寸的显存上界证明。

当前只读检查：

```bash
release="$HOME/projects/autoocr-business/releases/autoocr-business-gpu-20260918-r1"
"$HOME/projects/autoocr-business/.venv-gpu-r1/bin/python" -B \
  "$release/edge_deploy/business/service_health.py" --wait-ready --timeout 15 \
  --backend tensorrt_fp32_with_ort_fallback --other-backend cuda
curl --noproxy 127.0.0.1 --fail --max-time 15 http://127.0.0.1:5051/healthz
```

按需维护窗口`sudo systemctl restart autoocr-business.service`，本轮正常重启已通过。不要重跑绑定旧PID的一次性cutover_gpu_r1.sh/restart_gpu_r1.sh。回退对象优先E75 TRT＋CPU模块：旧unit在`~/projects/autoocr-business/deployment_backups/20260918-gpu-r1/autoocr-business-trt-r1.service`，SHA256 `ffcee7333ab01abdd18e9db8df9ee31d3510f6257b6625d641e5b3a6bdf409f3`；旧release和`.venv`均保留。回退必须先停止当前服务、校验备份/旧包，再恢复该unit、daemon-reload/start并检查旧GPU OCR readiness；不回滚许可/计数/HOME。新unit SHA256 `acba3cf36569bd7f595d4afd47d1cf2b2169301dc46d5936b3a261738b83e5fe`。未修改冻结包文档、未生成本轮新tar归档或外发。

## E75历史版本说明

此版本只将原server.py的OCR检测/识别张量执行接入TensorRT FP32。版式、YOLO、单元格、NER和增强仍CPU；不是整条业务全GPU。原API、图像处理、字典优先级、CTC、字段归属和授权计数不改。不要启用未通过多图回归的补白分桶或失败FP16引擎。

## 后端与输入覆盖

`AUTOOCR_OCR_BACKEND=cpu`是默认值；`trt_fp32`启用本版本。start.sh中的AUTOOCR_USE_GPU=false控制其他原业务模型，不表示OCR的TensorRT被禁用。

trt_models.json绑定已构建引擎和来源ONNX的SHA256。det为1×3×960×[672..736]，opt宽704；rec为1×3×48×[32..1536]，opt宽160。启动逐项核对引擎、源ONNX、实际profile、CPU备用session及TRT10.7.0，并实际执行两次合成张量预热。

- 输入在profile范围内：原张量交给GPU，不变更输入/精度/后处理。
- 超范围：将同一张量交给原ORT CPU session；计入ort_fallback_out_of_profile_calls，不偷偷拉伸、截断或加桶。
- CUDA执行、形状或非有限输出错误：不自动回退重试，标记不健康；包装器拒绝原逐框异常捕获后留下的部分结果。后续由运维重启/回退，不伪报GPU成功。

health的ocr_backend为tensorrt_fp32_with_ort_fallback，ocr_execution分别包含det/rec真实调用、超范围CPU调用、启动预热、错误计数和最近输入形状。启动预热不计入真实业务调用；成功接口响应与调用计数增长共同验证GPU确实进入业务链路。计数仅供运行诊断，不是授权余额。

## 环境与打包

沿用本机business venv，只安装本机官方TensorRT10.7.0 cp310 wheel（--no-index --no-deps），不安装onnxruntime-gpu或替换PyTorch。运行时需LD_LIBRARY_PATH包含TensorRT10.7.0/lib和CUDA12.6/lib64。

release_business.py build增加`--trt-engine-dir`，只复制trt_models.json指定的两个已审核FP32引擎、GPU适配代码及原完整业务资产；生成trt_business_source_models类型的新目录。默认CPU包行为保留。启动前仍验证完整清单，既有CPU r2绝不原地修改。

本包使用设备上已构建的可信engine，不自动下载或编译新engine。engine不是跨平台通用ONNX，新设备/不同TRT版本需要重新构建和回归；参见[NVIDIA运行时说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/architecture/capabilities.html)和[动态形状说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-dynamic-shapes.html)。

## 切换与验证

1. 候选包构建后运行verify_trt_business.py：原server OCR类、真实图片CPU/GPU文字与坐标对照及超范围张量回退验证。不导入完整server、不激活、不扣次。
2. 维护窗口确认无业务请求，备份实际CPU unit。仅停这一个服务，不并行运行共享Linux计数状态的两套服务。
3. 完整unit设置新release路径、原User/HOME/runtime/venv、AUTOOCR_OCR_BACKEND=trt_fp32和上述LD_LIBRARY_PATH；仍单sync worker、loopback5051。ExecStartPost明确带`--backend tensorrt_fp32_with_ort_fallback`，不能接受旧CPU服务的health冒充GPU。
4. 新服务ready后，Windows原生基准对照同图增强OCR→Extract，并检查GPU计数增长、是否发生CPU回退。不以HTTP200或health标签单独宣称GPU业务验证成功。
5. 启动/回归失败时停止候选，恢复已备份CPU unit并验证CPU ready；只恢复代码/模型入口，不回滚许可证、余额或已用码。有效Extract超时不自动重试。

正常启停继续使用autoocr-business.service；不要手动运行第二个start.sh。健康查询会在单worker长请求期间排队，不应据此频繁重启。首次调用、热运行、HTTP耗时及纯推理耗时分开记录；单次业务对照不是benchmark。

这是操作说明，实际执行结果以工作区TECHNICAL_LOG最新条目和技术报告为准。冷启动、故障注入、长期稳定性及人工准确率未因集成而自动通过。

## E75本机已完成的交付

2026-09-18实际部署：`~/projects/autoocr-business/releases/autoocr-business-trt-20260918-r1`，47个payload文件、1121396793字节。manifest SHA256为f3d99a9f0279d9f87748d3c4c5818f8b384d4c3e91da9a04fceb8e0de4d1fcd9。同父目录.tar.gz归档992691142字节，SHA256为0471cffa0f3cfc075507318e96602f5c66be54165bb7de96fea8fd659aaac60e。这是内部交付，不包含venv、SDK动态库、机器许可/计数或私人输入。

预检006在原生OCR类上55条非空文字/坐标一致，GPU det1/rec56；超范围det/rec合成张量分别回退CPU且输出相等。完整服务三图0075/0093/0096的13/56/73框、Markdown及6项字段与Windows严格一致；业务实际GPU det3/rec198、回退0、错误0。报告在temp/business_trt_acceptance_r1/，没有保存正文/字段值。

22:29:34～22:29:50切换成功，22:34:49～22:35:07正常重启成功（Asia/Shanghai）；当前MainPID11301、NRestarts0、enabled/active/running。重启后业务统计清零，det/rec预热各1次且ready。未重发有效Extract来填充计数。

日常只读检查：

```bash
release="$HOME/projects/autoocr-business/releases/autoocr-business-trt-20260918-r1"
"$HOME/projects/autoocr-business/.venv/bin/python" -B \
  "$release/edge_deploy/business/service_health.py" \
  --wait-ready --timeout 15 --backend tensorrt_fp32_with_ort_fallback
curl --noproxy 127.0.0.1 --fail --max-time 15 http://127.0.0.1:5051/healthz
```

按需在无请求的维护窗口使用`sudo systemctl restart autoocr-business.service`；不要重跑cutover_trt_r1.sh，它绑定切换前PID和哈希。长OCR时不要把单worker健康查询排队误判成故障。

CPU r2配置备份：`~/projects/autoocr-business/deployment_backups/20260918-trt-r1/autoocr-business-cpu-r2.service`，SHA256=60428f2d8da6dc893fbb1eb95dd5f3ce42c08464fcfd70ee2fb0f02955363e6b；当前GPU unit SHA256=ffcee7333ab01abdd18e9db8df9ee31d3510f6257b6625d641e5b3a6bdf409f3。

确需回退时，先核对上述两哈希、DropInPaths无额外修改、CPU r2包完整及无在途业务；正常stop后只用该备份以root:root/0644恢复/etc/systemd/system/autoocr-business.service，再verify、daemon-reload、start，使用CPU r2原service_health.py默认CPU检查确认ready。不要恢复更早的/mnt/d unit，不恢复余额/已用码、不删runtime/HOME缓存。失败回退分支本轮未触发，不宣称已做故障恢复演练。

冻结包内说明是构建时版本，当前实测结果记录在工作区文档，不修改已发布目录来补日志。当前虚拟环境只新增本地官方tensorrt10.7.0包，SDK仍位于~/sdk；新机不能只复制engine就视为部署完成。完整服务时延仍包含其他CPU模型，单次HTTP观察不作为正式GPU加速比。
