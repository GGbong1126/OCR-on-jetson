# IPC v1：隔离传输小闭环（E154）

这是C++／Python的**传输层与合成数据探针**，不是OCR后端或新HTTP接口。原服务保持停止，原模型／Extract不变。先冻结以下字节契约再实现；完整阶段操作、线程池及其预算仍须后续验收。

## 报文

Linux本机`AF_UNIX/SOCK_STREAM`，路径socket置于当前用户拥有的0700私有目录，socket0600；两端连接后用`SO_PEERCRED`确认同UID，再收发。禁止自动删除已有路径；退出仅清理本实例拥有的socket inode。不传FD、路径、命令、pickle或Base64。

固定24字节头，所有整数为大端无符号整数；禁止直接发送C++原生struct内存：

| 偏移 | 长度 | 字段／约束 |
|---|---:|---|
| 0 | 4 | magic=`AOCR` |
| 4 | 2 | version=1 |
| 6 | 2 | flags=0；未知位拒绝 |
| 8 | 4 | JSON元信息字节数，1～4096 |
| 12 | 4 | 二进制载荷字节数，0～64MiB；具体操作可更严格 |
| 16 | 8 | request_id，1～2^63−1 |

随后是恰好指定长度的UTF-8 JSON及二进制，不带分隔符。长度必须先验证再分配；关闭／截断／超时后整条连接废弃，不尝试扫描magic恢复。完整发送／接收须循环处理短读写、EINTR、EOF和粘包；发送抑制SIGPIPE。一个连接由单个调用者拥有，不允许多线程交错操作；本轮探针可连续发送两帧，再按ID逐一核对回复，不实现业务乱序调度。

## 当前唯一操作：probe

元信息必须恰有以下字段，拒绝额外键、重复键、非法UTF-8、NaN／Infinity、非整数尺寸、bool冒充整数及未知操作：

```json
{"dtype":"uint8","image_id":"synthetic_1","layout":"HWC","op":"probe","shape":[2,4,3],"stride":12}
```

- image_id为1～64个ASCII字母／数字／`._-`；shape为`[H,W,3]`，H/W各1～8192、H×W≤1600万。
- 紧密HWC三通道字节，stride必须为W×3；payload_bytes必须为H×stride。没有补白、JPEG往返或张量数值转换。这里不声称BGR／RGB识别效果或NVJPG硬解。
- Python读取元信息并完成上述检查后才申请载荷缓冲；通过后原样回传同ID、元信息和载荷。错误关闭连接，不回传输入、堆栈或文档正文。
- C++当前只解析固定头和长度；JSON保持不透明字节，探针要求回包与已知请求逐字节相同，绝不依据未解析JSON执行操作。**完整C++业务元信息解析／OCR阶段操作尚未实现**，届时采用成熟解析器并单独冻结schema，不手写通用JSON解析器。Extract不在操作白名单。

## 时限与资源

- send／recv使用同一个`steady_clock/monotonic`绝对deadline，部分进展不重置时限；不是硬实时保证。探针每次连接整个往返默认2秒，故障用例用更短时限；未来模型推理不能套用此探针时限。
- 单帧头24B、元信息≤4KiB、载荷≤64MiB；probe像素规则进一步限制载荷≤48000000B。当前测试串行单连接，不按连接／线程复制模型。
- C++分段发送，避免构造头＋载荷的大副本；接收方独立持有缓冲。Python接收转换为bytes时有一次复制，不声称零拷贝。当前单探针每端至多保留两份请求／回复载荷；这是显式载荷预算，不是RSS／内核socket缓冲／库内存的总上限。
- 本轮仅使用极小合成字节，超长测试只送非法头，不分配64MiB以上缓冲。后续M2 CPU输入池／结果池数量、各队列数量／字节、全局在途配额、最大等待／推理超时必须另冻结；**不把传输上限当作已实现的队列背压**。

## 构建与验收

复用GCC7.5和标准库，不安装系统包；本板无CMake，直接编译命令先实测，CMake目标仅供已有CMake的环境：

```bash
g++ -std=c++14 -O2 -Wall -Wextra -Werror -pthread ipc_probe.cpp -o ipc_probe
./ipc_probe --self-test
AUTOOCR_IPC_PROBE="$PWD/ipc_probe" python -B -m unittest test_ipc_wire -v
```

覆盖正常双向／ID／二进制、分片与连续多帧、长度／版本／flags、JSON／shape／stride、对端退出／截断／超时、同UID和socket生命周期。窗口内FD／线程／内存只作诊断；没有真实OCR质量、GPU并行、吞吐提升或长稳结论。

实现依据：[Linux unix(7)](https://man7.org/linux/man-pages/man7/unix.7.html)、[poll(2)](https://man7.org/linux/man-pages/man2/poll.2.html)、[Python 3.10 socket](https://docs.python.org/3.10/library/socket.html)；实际兼容性以本板4.9内核／Python3.10／GCC7.5测试为准，不套用新版专有API。
