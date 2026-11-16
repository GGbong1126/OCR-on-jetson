# Python worker桥接（E159）

[worker_bridge.py](../ipc/worker_bridge.py)把E158 I/O循环和实际算法线程分开：I/O所有者1线程、CPU算法2线程、GPU执行1线程。不新增依赖，不给每个CPU线程复制模型。仍在独立staging验证，不恢复原HTTP服务、Extract不动。

## 调用与所有权

I/O所有者构造`WorkerBridge(io, algorithm, runtime_factory)`，循环调用`io.poll()`和`bridge.step()`。runtime_factory在GPU线程执行；runtime的run／synchronize／close必须由同一GPU线程调用。模型仍N1；桥接明确拒绝allow_det_batch，E153 N2负收益不推广。

CPU函数签名`algorithm(payload_bytes, image_type, api) -> JSON bytes`。每页最多一个未消费模型请求：

```python
with api.infer('det', (Tensor('x', shape_tuple, immutable_fp32_bytes),)) as outputs:
    # 在此完成本次模型输出的CPU消费／后处理
    ...
```

CPU发布请求后在条件变量等待，等待时不持锁，也不占GPU线程。I/O所有者注册Tensor、发model.ready；C++发model.run后，独立GPU线程执行并同步；I/O发model.done，收到model.resume后才唤醒CPU。context退出发送consumed并等待I/O释放Call，之后才允许下一次模型请求。CPU不得直接使用原TensorRT context。

E159当时模型输出仅独立、不可变的FP32 bytes＋shape。E160扩展内部Tensor输出为float32／int32，输入仍仅FP32，保持形状乘积与64MiB总量检查；完整OCR消费者转换成独立可写NumPy副本，避免torch.from_numpy／NMS写入只读bytes。范围见[OCR worker](OCR_WORKERS.md)。不能跨context保存无界数组／Tensor别名；Python局部及第三方库内存不是自动受全进程RSS硬限制。

## 通知、背压与结果

- 2个活动Job，每Job一个模型Request／一个CPU完成槽；GPU一个任务／完成槽。待执行CPU队列受活动2页约束，不用无界Future／callback队列。
- 内核AF_UNIX socketpair只传唤醒字节，数据仍在受锁mailbox；非阻塞发送满时合并通知。I/O唤醒回调仅排空字节，状态迁移在step完成。组连接已关闭时，selector仍能接收晚到的安全完成／finalizer。
- 事件队列满则保留请求或已完成GPU输出，不重复执行模型；最终image.done事件无空间时仍保留活动页和完成槽，不丢结果。
- 新事件`image.done`载荷0B，字段image_id、status及非成功code，与image.result状态规则相同。它只表示CPU阶段结束／结果可收集；不是发送完成或GT正确性。model.resume ACK不再被误认为整页已经结束。
- C++新增`--workers SOCKET`：同PID四连接、两页合成模型交接，允许image.done与其它页model事件交错，最终反序collect。旧单连接／--channels用例不要求新事件，保持不变。

## 取消、失败与退出

取消／原绝对deadline／断连由I/O转为Job取消并notify_all。CPU在checkpoint或infer等待处醒来，最终必须报告finished；任意CPU算法不能被安全强杀，长CPU步骤应插入checkpoint。CPU异常只返回固定状态，不把异常正文送入IPC。

运行中取消保留原Batch输入和活动页，直至GPU完成且CPU finalizer确认结束。GPU异常不自动CPU回退、不重试推理，锁存gpu_failed；只有runtime.synchronize确认安全才允许Session.complete和关闭runtime。

同步也失败时，任务／输入／runtime隔离保留，close有界等待后返回False，不声称资源已释放。显式request_quiescence只能在同GPU线程重试同步，不能重试推理、reset或清锁存；若后续确认安全才可收尾。未知硬件状态不是用超时强行free的理由。runtime.close异常仍锁存失败，重复close不能改报成功。

stop关闭连接组并取消任务；调用者继续poll／step到安全结束。close(timeout)负责有界收尾、join及注销唤醒FD；部分线程启动失败会先唤醒并join已启动线程。不会改原服务、系统或开机设置。

## 真实det窄验证

[det_worker_adapter.py](../ipc/det_worker_adapter.py)仅接原det FP32 N1 engine，复用原trt82_backend.Session。模型创建、所有GPU调用、额外安全同步、close均在同一GPU线程，输出独立bytes后才交CPU。

[verify_det_workers.py](../ipc/verify_det_workers.py)固定E153原13图中的前两个不同文件，校验图片／原预处理哈希。私有四连接由独立验证客户端线程驱动，CPU worker实际软件解码并通过AST复用原_det_run预后处理；用ExitStack让输出租约覆盖整个原方法后处理。参考2次和候选2次GPU调用共用一份engine／一个context，不同时加载两套模型；参考张量是额外诊断内存，不算生产页预算。

逐页检查输入／概率图逐字节、框完全相等、image_id、反序collect及关闭错误计数。只输出技术哈希与数量，不保存图片／正文。非硬解、业务缩图／增强、layout／rec／CTC、HTTP、Extract、GT或吞吐测试；不能据此称完整多线程OCR已部署。

复跑必须用新报告名及相同冻结输入，原服务保持停止，不重跑择优：

```bash
export LD_LIBRARY_PATH=/home/xchq/work/autoocr/lib/cuda10.2:/usr/local/cuda-10.2/targets/aarch64-linux/lib
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 /home/xchq/work/autoocr/envs/business-py310/bin/python -B ipc/verify_det_workers.py --output NEW_REPORT.json
```

M0／M2仍未全部完成：C++双池与GPU调度器正式应用组合、其它模型和完整OCR阶段安全适配、跨进程统一额度、硬解质量、批HTTP和端到端验收待后续。现有C++消费者是联调探针，不是完整应用线程池；Python两个worker也不是“每线程绑定GPU核心”。
