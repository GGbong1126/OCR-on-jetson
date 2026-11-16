# Jetson 板端开发测试服务（E129）

E160最新（2026-10-30）：独立staging已接原六模型及完整Python OCR worker，4页经2CPU＋单执行线程／四通道IPC与串行参考严格一致，98项／26项×20轮／UBSan 14项通过。原服务仍inactive／MainPID0／enabled，原发布、宽rec overlay及GDM／SSH／FRP不变；未创建HTTP／调用Extract或改授权。正式C++双池应用／批入口及性能验收仍待做，不因该小闭环自动恢复服务，详见[范围](batch_ocr/scheduler/OCR_WORKERS.md)。

用户于2026-10-06明确确认板端不启用授权限制。本入口不要求`license.dat`，不执行硬件激活、授权余额初始化或`/Extract`扣次；不创建假许可、不清空或迁移原计数。原Windows正式版、`releases/jetson-r2`及E124/E127交付包保留，不以全局开关改变正式版默认行为。

E159最新：独立目录已验证2CPU＋1GPU worker桥接，以及两张真图的原det N1／NumPy／四通道IPC；概率图／框与一份原engine串行参考完全一致。88项／17项×20轮／UBSan 14项通过，无worker／socket残留；仍非完整OCR或正式C++双池应用／HTTP上线。未启动服务，原发布12项＋N2及OCR inactive／MainPID0／enabled、GDM6153／SSH5833／FRP5617不变，不自动恢复原服务，Extract未改未调，详见[worker范围](batch_ocr/scheduler/WORKERS.md)。

E158当时：只在独立目录验证三控制＋独立事件四连接组及非阻塞I/O；当时未加载模型或接生产worker。该阶段限制已由E159的det窄验证部分推进，不表示完整业务服务已接入。

E157最新：只在独立目录验证C++JSON阶段消费及合成Unix交接，57项、UBSan 12项和20轮通过，没有加载模型或启动HTTP。原12项发布＋N2及OCR inactive／MainPID0／enabled、GDM6153／SSH5833／FRP5617运行状态未变；项目内新增固定JSON单头而非系统安装。正式多通道I/O／真实OCR适配尚未完成，不自动恢复原服务，Extract未改未调，详见[消费者范围](batch_ocr/scheduler/CPP_STAGE.md)。

E156最新：只在独立暂存目录验证会话／字节注册表、错误应答及合成阶段IPC，45项无跳过、20项核心重复30轮通过。取消／超时／断连与GPU错误锁存是状态规则及模拟事件验收，不是实际CUDA故障恢复或OCR上线；原发布12项＋N2指纹及OCR inactive／MainPID0／enabled、GDM／SSH／FRP运行状态未变。没有加载模型或调用OCR／Extract，不自动恢复原服务，详见[生命周期范围](batch_ocr/scheduler/LIFECYCLE.md)。

E155最新：仅阶段校验、C++有界队列／额度及合成2-1-2协作测试，无模型／HTTP／Extract执行。50轮原生、25项IPC/schema及UBSan通过，最终原发布12项指纹＋N2候选未变；OCR保持inactive／MainPID0／enabled，GDM6153／SSH5833／FRP5617运行。尚未接入真实OCR线程池／批入口或跨进程内存约束，不自动恢复服务；复跑与范围见[独立实机记录](batch_ocr/README.md)。

E154只运行独立C++／Python合成IPC探针，无模型／HTTP／Extract执行。终态再次核实：开发服务仍inactive／MainPID0／enabled，GDM／SSH／FRP保持active及原PID，原发布12项指纹与N2候选SHA不变（E155按原报告更正计数），可用内存约13.07GiB；不是恢复OCR、替换N1或接入新批入口，现有恢复边界不变。

**E153最新核实（2026-10-29）：用户已重启，OCR与GDM曾按原设置自启；为续接既有隔离构建窗口，OCR已正常停止，MainPID=0／inactive，未disable。GDM保持active，未结束新桌面会话。** OCR／Extract仍随共用进程暂不可用，代码与原发布未改。新det N2已独立构建并通过GPU／26组离线质量对照，尚未部署到服务、没有恢复HTTP，不能将预计算结果回放当在线批调度；以下停止／失联描述为历史状态。

E153补测终态：不同图片同type／shape的10个N2组＋6尾项，26/26质量／坐标一致，NER业务0；原7模型和宽rec未改。raw-det候选均值慢3.37%、预热全机NvMap多约1.30GiB，不推广原服务默认。全部测试任务已退出，最终可用内存约13.07GiB；只是空载端点，不是服务显存或长稳验收。新C++线程池／IPC／批HTTP尚未实现，按plan继续，不自动启动原服务。

**断线前最后确认（2026-10-29，E150／E151）：用户要求先停止OCR，开发服务inactive／dead、5051无监听，OCR／Extract暂不可用；enabled开机设置未改、未主动恢复。E151构建时SSH意外断开，失联后当前状态未知，重连须先检查；若已重启，原enabled服务可能自启。** 为释放显示事件积压，另经用户确认工作已保存后停止GDM并清理残留greeter会话，回收约7.19GiB内核池内存；用户Xorg／nvargus仍保留，不承诺GUI可用。没有主动改FRP、重启或改开机策略。此为运行态规避，重新启桌面是否复发尚未测。恢复命令见下文，本轮未执行恢复；详细证据见[技术日志E150／E151](../TECHNICAL_LOG.md)。

历史E129验收：服务曾在板端运行并配置开机启动。13张现有图片两路径26组HTTP验收、正常重启后4组复测通过，共30次OCR与30次Extract成功；4组跨重启输出一致。未进行整机重启、并发、GT或速度基准测试。

## 隔离范围

- 服务名：`autoocr-jetson-dev.service`。原`autoocr-jetson.service`继续保留，不能与开发版同时占用5051。
- 启动包：`/home/xchq/work/autoocr/overlays/jetson-dev-e129-r1`。
- 独立运行目录：`/home/xchq/work/autoocr/runtime/jetson-dev-e129-r1`。
- 监听地址：仅`127.0.0.1:5051`；路由另拒绝非loopback来源，不信任`X-Forwarded-For`。没有打开公网端口或更改FRP配置。
- 单同步worker、单线程、不预加载CUDA模型。七模块沿用既有FP32 TensorRT，rec启用已验证的1537～2304宽profile；原范围和超范围CPU策略不变。CPU前后处理仍保留。
- `/healthz`明确返回`service_mode=jetson_development_no_license`与`authorization`中的硬件许可/用量限制`disabled`。这是开发模式，不代表已获得正式许可。

## 在板子上查看与测试

```bash
systemctl status autoocr-jetson-dev.service --no-pager
curl --noproxy '*' --fail http://127.0.0.1:5051/healthz
```

业务接口仍为`POST /OCR`（`img`纯Base64、`imageType`字符串1/2/3）和`POST /Extract`（`ocrResult`、`topFields`、`nerFields`）。请求和返回格式沿用原协议；`/OCR`不调用NER，`/Extract`按原条件执行NER，但本测试入口不扣次。

如需从Windows测试，先在一个终端保持SSH隧道：

```powershell
ssh -p SSH_PORT -o HostKeyAlias=jetson.local -o StrictHostKeyChecking=yes -o ExitOnForwardFailure=yes -N -L 127.0.0.1:15051:127.0.0.1:5051 USER@JETSON_HOST
```

在另一终端检查：

```powershell
curl.exe --noproxy "*" --fail http://127.0.0.1:15051/healthz
```

随后API客户端访问`http://127.0.0.1:15051/OCR`或`/Extract`。密码由SSH交互输入，不写到脚本或文档里。隧道只开放电脑本机，关闭该SSH终端即可关闭隧道。

## 停止、重启和回退

```bash
sudo systemctl stop autoocr-jetson-dev.service
sudo systemctl start autoocr-jetson-dev.service
sudo systemctl restart autoocr-jetson-dev.service
```

需要取消开机启动并停止：

```bash
sudo systemctl disable --now autoocr-jetson-dev.service
```

原正式服务仍需满足它自己的许可要求；停止开发版不等于正式版自动可用，不自动启动旧服务。手工执行启动包还需要显式`AUTOOCR_ENABLE_DEV_NO_AUTH=1`，且会拒绝其它继承的`AUTOOCR_*`覆盖变量。

## 实现与验收边界

独立Flask应用只注册三个业务/健康路由，复用原模型加载和原OCR处理器。`/Extract`使用相同业务函数字节码的独立绑定，仅将`deal_counter(1)`依赖替换为无存储的开发用量策略；不修改原函数、原许可校验函数或源文件。识别正文不写到测试日志/归档，临时图片沿用原接口的请求结束清理。

启动健康检查与正常重启并不证明所有CUDA故障均能自动恢复：当前模型错误会锁存并返回503，503本身不保证worker退出。仅单worker顺序HTTP验收，不是公网安全评估、并发/小时级压测、GT准确率或性能基准。实际测试数量、结果及证据哈希以`TECHNICAL_LOG.md` E129为准。
