# C++阶段消费者（E157）

本轮补成熟JSON解析／校验与实际C++↔Python阶段交接，**仍为隔离合成验证，不是OCR服务或完整M2**。原probe默认保留，不加载模型或改动Extract／原发布／开机设置。

## 依赖与输入边界

- 项目内固定[nlohmann/json v3.12.0](../third_party/nlohmann_json/README.md)，原单头与MIT不修改，校验官方SHA。不依赖系统开发包或联网CMake。
- [stage_json.hpp](../ipc/stage_json.hpp)对照Python stage_contract实现E156全部阶段操作的字段白名单、严格整数／bool区分、session／ID、模型绑定、shape／字节乘积和限长。未知操作含Extract／NER拒绝；只校验规格，不证明张量数值／模型准确。
- 使用成熟解析器callback拒绝各层重复键，含转义后同名键；拒绝坏UTF-8、未配对代理项、BOM、原始NUL、NaN／Inf／溢出、尾随数据／注释／逗号。元信息仍1～4096B；深度上限16足以覆盖已有阶段schema，不接任意文档作为元信息。
- 成功结果最多8MiB、JSON对象，最大深度64；Python新共享validate_result也用同样结果规则，Session.finish复用。它不是原OCR响应字段校验。只约束独立新协议，不改变原HTTP返回或Windows业务。
- C++ transport增加显式validator重载，在元信息接收后、payload.resize之前校验，之后再读二进制；旧echo重载仍可用。解析后及传输结束重查同一绝对deadline，不靠零长度载荷跳过截止检查。

## 消费而非echo

[stage_client_probe.cpp](../ipc/stage_client_probe.cpp)实际解析握手并固定session，用收到的ready中的image_id／call_id发run，核对done、resume应答、最终image_id及JSON结果。两图ready反序，末尾反序collect；中间故意提前collect验证STATE应答。请求与应答同时核对帧ID、op／for_op、session和状态码，不因字节可解析就当成功。

Python测试端通过Session处理真实AF_UNIX帧，合成BGR每图6B、det输入每图12288B、模拟输出4B，最终仅synthetic标记和图片ID。没有真实像素解码、NumPy、GPU、OCR文字或HTTP；单条受控测试连接承担各逻辑通道，未来正式三通道绑定／I/O事件循环仍未实现，不能宣布M0全完成。

探针整体绝对2秒，只用于小合成流程；真实算法适配必须沿用每页300秒及模型120秒等阶段预算，不能照搬该探针等待。当前只验证N1模型调用，N2候选仍不默认启用，不把这条串行阶段示例写成C++线程池吞吐提升。

## 验收与复跑

Python fixture先独立核对期望，再让C++接受／拒绝同一批元信息与结果字节，覆盖全部模型／操作、类型／尺寸／重复键／解析及限长边界。另测错误session、帧ID、for_op、重复键、非法元信息声明64MiB却不发载荷时立即拒绝，以及无进展超时。只保存技术用例名／计数和源码指纹，不写原始请求／OCR正文。

在新的隔离源码根中（ipc／scheduler／third_party同级）复跑，勿覆盖旧二进制或报告：

```bash
source /home/xchq/work/autoocr/envs/business-py310/bin/activate
g++ -std=c++14 -O2 -Wall -Wextra -Werror -pthread ipc/ipc_probe.cpp -o ipc_probe
g++ -std=c++14 -O2 -Wall -Wextra -Werror -pthread ipc/stage_client_probe.cpp -o stage_client_probe
AUTOOCR_IPC_PROBE="$PWD/ipc_probe" AUTOOCR_STAGE_PROBE="$PWD/stage_client_probe" \
  python -B -m unittest discover -s ipc -p 'test_*.py' -v
```

CMake仅提供独立目标，板端未安装CMake、未实测配置。成功结果的DOM／转换复制仍要单列内存，不据此宣称768MiB已限制RSS或实现零拷贝。真实GPU故障／长稳／完整业务／并发和硬解质量不属于本探针已验收能力。
