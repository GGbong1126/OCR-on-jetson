# 会话与资源生命周期（E156）

实现为[stage_session.py](../ipc/stage_session.py)，使用Python标准库，**不是OCR算法后端或上线服务**。它把E155的元信息校验接到有界内存注册表、控制应答与状态转换；C++业务JSON消费者、真实算法／NumPy／TensorRT适配、CPU任务池、I/O事件循环及HTTP仍待集成。原probe默认、原服务和Extract不变。

## 新增应答规则

沿用24B帧、ocr.stage/1及严格JSON。现有schema尚未部署为业务服务，本轮增加以下白名单，不自动放开probe连接：

| 消息 | 字段／载荷 | 规则 |
|---|---|---|
| session.ready | 公共schema/op/session/remaining_ms，0B | 后端启动生成新随机128bit会话标识；新连接组必须属于同一个活会话 |
| rpc.reply | for_op、status=ok/error、code，0B | 帧头request_id等于该请求；成功code=OK，失败仅固定枚举，不传异常／路径／正文 |
| model.failed | call_ids、code=GPU_FAILED，0B | 后端内部GPU失败锁存事件；不是来自客户端的合法控制请求，也不证明GPU已安全停止 |
| image.result | image_id、status=ok/error/cancelled | 成功为1～8MiB JSON对象字节；非成功0B并必须带code，cancelled仅配CANCELLED |

错误码：BUSY（额度／入口繁忙）、LIMIT（资源上限）、STATE（阶段／ID错误）、SESSION（旧会话）、TIMEOUT、CANCELLED、GPU_FAILED、DISCONNECTED。无任意错误字符串。原图像／模型规格仍按STAGES.md；结果目前只检验UTF-8／JSON对象、重复键及非有限数字，不校验完整原OCR响应字段，也不宣称识别正确。

控制通道由可信I/O层分配，不由帧中字符串决定：input只收submit／cancel，gpu只收run，result只收resume／collect／cancel。每通道独立递增request_id，语义失败也消耗已接受会话内的ID，不缓存重放或自动执行重试；旧session不推进水位。不能使用跨三个socket的单一水位，否则较晚到达的低ID会被误拒绝。未知schema／操作／方向／坏帧使连接作废；有合法schema但处于错误状态则回固定错误应答。实际连接组绑定／握手超时由未来I/O适配器完成，本轮不实现公网认证。

## 谁拥有状态

Session只允许创建它的I/O线程修改；GPU和CPU线程须把完成事件投递回I/O所有者，不能直接跨线程调用方法。当前没有创建这些工作池或事件循环，不把单所有者断言当作整条OCR已经线程安全。

模型调用ID与buffer handle在会话内单调分配、不复用；图片ID可在结果交付后重新使用，但内部Page／Batch以对象身份验归属。晚到的旧Page／Batch不能改写新同名图片或新会话。每图最多一个未释放调用，最多4个在途Page、2个活动页；不存在无限增长的请求回包缓存或ID墓碑表。

## 正常路径

```text
submit → queued → activate → active → ready → model.run → running
         ↑入口额度                 ↓模型实际安全完成
finish ← active ← release_call ← resumed ← model.resume ← done
  ↓
complete → image.collect → delivering → delivered → 删除Page
```

- submit持有原始输入bytes，排队图像计入入口64MiB；activate后输入转入每页256MiB预约，活动2页合512MiB。这里只是本注册表额度与计数，不把完整预约实际malloc。
- ready接收后端拥有的不可变bytes及float32规格；当前合成测试实际使用12KiB的det输入，没有NumPy／CUDA对象支持。深拷贝对外规格，避免返回字典被修改后破坏注册表。ready队列计权≤64MiB；出队到running仍持输入及页预约。
- run只选ready且未停止的调用；单次仅一个Batch，det N2需显式opt-in、不同图片且type／规格一致。模型deadline取本地阶段120秒、请求剩余时间、每图原始绝对deadline的最小值，不能靠新请求延长单图时限。
- complete只能由内部确认安全完成的执行事件调用。正常输出留到resume和CPU显式release_call；resume应答不是“CPU已不用张量”，不能立即释放。输出总字节超限时标记LIMIT并完成该调用的收尾，不把已完成任务永远留在running。
- finish是CPU最终完成事件，要求没有未归还调用，校验结果字节后退还活动预约。image.collect仅取结果、不重新计算；Reply携带Page交付令牌，结果额度一直保留到发送成功后的delivered。重复collect／resume／complete／交付确认拒绝。
- 安全释放调用时清空Call里的inputs／outputs，交付确认时清空Page.result，避免持有旧令牌仍间接保留已删除注册表数据。外部直接取得的bytes别名仍由调用者及时丢弃，不把删字典项等同于整个进程对象已销毁。
- I/O事件的remaining_ms是协议上限信息，不是新的任务起点；接收方必须沿用自己的绝对截止时间。Session.expire必须由未来I/O事件循环定期调用，本轮不会靠后台定时线程自动执行。

## 取消、超时、断连与GPU失败

- 排队任务停止可直接释放入口输入；活动CPU任务停止后仍持页预约，直到CPU finalizer调用retire，确认不再访问本页。
- 正在执行的模型停止，只标记stopping，保留输入和Batch，拒绝提前retire。GPU真正安全完成后调用complete，丢弃停止页的迟到输出，再由CPU retire。两页N2中取消一页时，另一页仍可正常完成。
- GPU错误调用gpu_fault锁存，拒绝新submit／run，不转CPU、不重试N1。错误不等于同步成功，运行中的资源保持隔离；若始终不能确认安全完成，就保留隔离状态，由未来后端停机策略处理，本轮不实现GPU reset或自动恢复。
- disconnect停止接收，清理尚未活动的排队项，但不释放活动输入；仍需安全complete／retire。正在发送的结果不自动删除，关闭socket并丢弃该待发送帧后，I/O调用abandon_delivery才归还交付额度。旧会话对象不能重新握手复用。
- 第一个终止原因保留，不因后续取消／断连覆盖；已开始发送的响应不改正文或状态。任何停止通知／future唤醒仍需实际CPU/I/O适配器接线，本轮只实现状态与归属规则。

## 额度范围与测试限制

本轮用真实合成bytes约束入口、注册表张量、ready引用字节及结果；活动页预约按E155 limits.json。测试部分用小额度注入来触发边界，不分配512／768MiB。模型输出每调用最多16个bytes对象、总≤64MiB；实际模型适配仍需验证输出布局及数值有效性。

这不是跨进程统一记账：外部调用者持有的别名、算法未登记临时内存、NumPy／Pillow／Torch／TRT内部、JSON DOM、transport转换与内核缓冲未受此核心约束，768MiB仍不是服务RSS上限。测试只有合成载荷／假时钟／模拟模型完成；真实AF_UNIX测试验证握手、阶段事件／应答、ID及反序收集，不冒充C++业务解析、GPU／OCR／硬解、吞吐或长稳验收。
