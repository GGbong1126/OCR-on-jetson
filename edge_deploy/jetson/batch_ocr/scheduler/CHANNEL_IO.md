# 多通道连接组与I/O循环（E158）

[channel_io.py](../ipc/channel_io.py)实现私有Unix socket的单所有者、增量非阻塞I/O；复用Session、stage schema和Python selectors，不新增依赖。**它仍未接原OCR／NumPy／GPU或HTTP，不自动恢复原服务，Extract不动。**

## 连接与绑定

三条控制连接input／gpu／result，加一条独立只收事件的events连接，共4条。这落实E155独立事件接收职责，不把“三条控制”误写成总共三个socket。

1. 后端以同一个Session在每条连接发送session.ready。客户端固定nonce；随后发送request_id=1的channel.bind，字段channel只能为四个角色之一，载荷0B。
2. 后端核对schema、session、SO_PEERCRED同UID且与首个有效绑定的客户端PID相同、角色唯一，回同ID的channel.bound。绑定不是公网认证，不把PID元信息当凭证；PID由内核取得。跨进程移交／重连恢复暂不支持。
3. 每条控制通道从应用request_id=2开始、各自递增；绑定ID已占水位。events的后端事件ID从2递增，不能由客户端发送业务帧。
4. 4个角色到齐前不处理应用请求。未绑定错误连接单独关闭、不替换合法角色；已绑定任一连接断开／超时／协议错误，整个会话连接组关闭，不偷偷接新组重试。

最多4条已接收连接，首条连接起5秒内完成整组绑定；未绑定单连接也有5秒时限，非法额外PID在握手前拒绝。只有正式绑定后才固定客户端PID。私有父目录0700、socket0600、同UID及inode清理由既有private_listener管理；listener由调用者持有，ChannelServer关闭selector时不提前关闭listener，保持原清理规则。

## 收发与背压

- selectors只在I/O所有者线程使用，Linux默认实现由实机报告确认；每次每条连接最多处理64KiB，短读写／EAGAIN／EINTR保存进度，不等整帧收完才处理其它通道。
- 先读24B头和≤4096B元信息，校验通道／op／session／shape等，再申请body与转换副本额度；超时也在校验后、申请前复查。解码器不多读下一帧，可处理拆包／粘包。
- 一条控制连接最多一个处理中的应答，发送完成前不读下一条请求，由socket提供背压；events队列最多4项。事件队列满时不增加event_id，model_ready先确认空间再改变Session状态。
- body按2×长度预留bytearray→bytes的瞬时转换空间；输出队列按整帧字节计权，合并受128MiB传输额度约束，另预留4条连接的小头／元信息空间。输出分段memoryview，不拼接大帧。它不是跨进程统一RSS约束，JSON DOM、库内部、内核缓冲、外部别名仍未计入。
- 绑定／接收一帧／发送一帧默认5秒，接收时限从第一个实际读到的字节起，不随部分进展刷新。已绑定空闲通道无5秒空闲踢出，避免将真实模型等待误判为坏连接；单图／模型仍由Session原始绝对deadline约束。
- poll最多休眠50ms，前后调用Session.expire；不创建额外定时线程或忙等。输出时限跨过截止后不调用delivered。

## 所有权与退出

仅当一帧所有字节均已被本地socket接受时回调Session.delivered；这不是客户端已解析确认或跨崩溃恰好一次承诺。失败时先关闭网络连接、丢弃待发视图，再disconnect并abandon_delivery，包括应答入队失败的结果，不使delivering页遗留。

group stop只关闭I/O并停止接收，不释放running模型输入或活动页预约。I/O所有者仍可确认安全model_complete及CPU retire；有活动页时close拒绝销毁selector／状态所有者，不能伪造GPU完成。最终close要求图片／running／传输额度清零。未来算法桥接必须提供真实完成／finalizer，未知GPU安全状态不能靠超时或本组件自动恢复。

poll返回已处理控制的少量元信息通知；当前可信I/O调用者可调用model_ready／model_complete，只有前者／后者的空间检查与事件交接，不运行CPU/GPU工作池。队列满时已完成的执行结果需要未来有界完成队列保留并择时再交接，不能再次执行GPU。测试Harness的Future／queue及自动合成driver仅为夹具，不是已实现的生产worker桥接。

## 检查范围

纯逻辑用例覆盖绑定schema／ID水位、增量收发／双帧、校验前不申请body、绝对时限及短写；Linux真实socket覆盖同PID四角色、重复／旧会话绑定、其它PID拒绝、绑定时限、慢半包与其它通道进展、满事件队列、阻塞结果发送时保持额度及其它控制进展、运行中断连保留输入直到安全收尾。

C++ stage_client_probe新增`--channels SOCKET`，实际打开4条同PID连接、各自绑定并核对nonce／角色／应答ID，在events接ready／done，在各控制socket推进两图9请求和反序结果收集。原单连接探针及全部旧测试仍保留。合成输入／输出规格沿用E157，无实际模型或OCR文字，不证明GPU并发、吞吐、硬解质量或完整M0／M2／M3。

新的隔离源码根（ipc／scheduler／third_party同级）复跑，不覆盖旧报告：

```bash
source /home/xchq/work/autoocr/envs/business-py310/bin/activate
g++ -std=c++14 -O2 -Wall -Wextra -Werror -pthread ipc/ipc_probe.cpp -o ipc_probe
g++ -std=c++14 -O2 -Wall -Wextra -Werror -pthread ipc/stage_client_probe.cpp -o stage_client_probe
AUTOOCR_IPC_PROBE="$PWD/ipc_probe" AUTOOCR_STAGE_PROBE="$PWD/stage_client_probe" \
  python -B -m unittest discover -s ipc -p 'test_*.py' -v
```
