# 原完整OCR与worker的最小适配（E160）

[ocr_worker_adapter.py](../ipc/ocr_worker_adapter.py)在独立进程内复用原load_business／process，六个OCR模型layout、cell、yolo、enhance、det、rec经E159桥接委派。原文件／权重／engine／阈值／CTC／融合规则不改，Extract不动，run_ner固定False；原factory的NER启动预热单独记录，不把它计成业务NER或声称未加载NER。

## 真实执行边界

1. 独立执行线程调用原factory，模型只初始化一套，保留原rec宽GPU overlay及CPU范围外路由；reference和candidate复用这一套模型。CPU工作线程不重新load_models。
2. 捕获各模型原run，安装进程内proxy。CPU线程借threading.local取得自己的api、调用计数和错误标记，准备Tensor并等待；执行线程只调用捕获的原run，不递归进proxy。get_inputs／get_outputs读取已有Python元信息，不从CPU线程改context。
3. 内部输出保留float32／int32，不给检测数量强转FP32；IPC仍只传handle／输出字节数，不把模型大张量来回发送。CPU消费前复制为独立可写NumPy数组，释放Call后也不会读到下一任务覆盖的数据，代价是一份明确的输出拷贝，不宣称零拷贝。
4. 每页浅拷贝无模型加载的pipeline及OCR engine外壳，深拷贝LayoutFusion配置；字符表及只读模型参数复用。原源码process没有运行中修改pipeline属性；该事实不是任意未来包装器都线程安全的承诺。
5. Ultralytics predictor有可变状态，整个predict用一个锁保护，返回前深拷贝Results，禁止stream生成器逃出锁。锁可能覆盖等待模型的时间，但GPU执行不依赖CPU池或此锁；其它页仍能在不同模型阶段推进。增强包装器保持eval／CPU宿主输入、FP32、batch1，不去掉原锁、不在worker调用half或CUDA。

只有一个执行线程提交模型，不代表GPU只有一个计算核心；当前仍各模型N1，不把“多线程整页在途”称为GPU多模型kernel同时运行或batch吞吐提升。2305～3200的rec按原规则在该受控执行入口运行ORT CPU，统计必须标出，不称全部GPU。

## 输入、错误与清理

- 此适配只接受encoded。先按原Pillow格式识别／verify、检查实际宽高≤8192及≤1600万像素，再进原process；坏输入／raw BGR拒绝，不把BGR字节误当JPEG。保留format是内部页属性，协议没有任意路径字段。
- 每页在项目暂存目录创建0700临时子目录，以受校验image_id命名.img输入，process保留原缩图／中间JPEG90／可选增强语义。页结束移除自己的临时输入及中间图，不保存原文；硬解候选未接，不额外转JPEG作为硬解交接。
- 原业务部分try/except会吞掉模型错误。proxy额外锁存当前页failed标记，即使process返回success也拒绝部分结果；GPU层原错误锁存仍保留，不能重试推理或因执行错误转CPU。
- GPU同步／隔离规则不变。close不恢复proxy，以免取消中的CPU finalizer重新取得直连GPU访问；只由执行线程安全关闭原模型，之后等待CPU finalizer／清额度，不能由超时强行free。
- 输出拷贝、原算法／库内部和CPU fallback内存尚非统一逐笔预算；768MiB是显式预约策略，不是跨进程RSS硬上限。

## 固定验证

[verify_ocr_workers.py](../ipc/verify_ocr_workers.py)固定E153原13图的前两图×imageType3／1，共4页。先同一执行线程串行原process作为参考，再两CPU算法worker＋一个执行线程经私有四通道IPC处理，验证客户端仅板端线程，不是正式C++应用；C++消费者／schema用独立回归验证。

逐页要求完整data、全文／Text／Markdown／RawLines／special／NER空值／框及坐标严格一致、模型调用数一致，按image_id反序collect。模型数值误差不是本轮门槛；有共同空文本仍非GT。独立合成W2305检查原CPU范围外路由恰增1调用、GPU调用不增加且输出finite；不据此更改业务张量、引擎或宣称长行精度。

启动前MemAvailable≥8GiB，模型加载后≥2GiB；原OCR服务必须inactive。使用原服务CUDA10.2库路径，OpenBLAS／OMP固定1，原loader仍设torch2／cv2 1；不改功耗、时钟、系统或桌面。报告只保存哈希／布尔／shape／调用统计，不落盘全文或图片。参考／候选模型相同、无新引擎构建，不计时宣传吞吐。

使用新报告名复跑，禁止覆盖失败记录：

```bash
LD_LIBRARY_PATH=/home/xchq/work/autoocr/lib/cuda10.2:/usr/local/cuda-10.2/targets/aarch64-linux/lib \
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /home/xchq/work/autoocr/envs/business-py310/bin/python -B ipc/verify_ocr_workers.py \
 --output /home/xchq/work/autoocr/staging/NEW_STAGE/business.json
```

本轮不包含正式C++双池／批入口／跨进程统一额度、JPEG硬解质量及HTTP／长稳／性能资源验收；这些仍按plan后续步骤推进，不直接替换原服务。
