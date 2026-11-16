# 宽输入业务服务交付包（E127，未启用）

E124～E126验证的是离线业务及可选宽FP32适配。E127补齐独立启动包和健康检查兼容性，**没有办理激活、安装/启用新systemd服务或启动HTTP**。板端正式运行目录`/home/xchq/work/autoocr/runtime/license.dat`当前不存在，真实HTTP联调受既有许可条件限制。

## 文件与行为

独立目录：`/home/xchq/work/autoocr/overlays/rec-wide-service-e127-r1`。

- `start_rec_wide.sh`：默认只做`--check`，显式`--serve`才可能启动服务。缺少许可先退出78，不导入业务、不加载模型、不初始化计数；有许可时仍调用原`serve_jetson.py --check-license`，不接受借用/覆盖许可位置，不提供激活选项。
- `health_rec_wide.py`：区分`--mode on/off`；检查七模型、原FP32精度、错误/关闭状态、预热记录、计数未倒退，宽模式额外核对两个profile与至少三次宽引擎预热。正常范围外CPU调用不等于错误；CPU执行错误仍不就绪。仅查询固定loopback地址，不读业务正文、不扣次、不跟随重定向。
- `autoocr-jetson-wide.service`：**未安装的候选unit**。沿用单同步worker、单线程、禁止模型预加载的进程边界；仅绑定127.0.0.1:5051。缺少许可/预检失败的78退出码不自动重启，避免无意义重试。原`autoocr-jetson.service`完整保留，不自动停止它来抢端口。

旧`health_jetson.py`只接受`rec.backend=tensorrt82_cuda`，会误拒绝宽适配的`tensorrt82_fp32_wide_overlay`；新检查器是独立文件，**未就地修改r2或E124包**。E124的原工厂仍负责许可、计数、健康拒绝与业务接口，没有旁路WSGI实现。

## 当前可做的安全检查

以下在板端终端运行；不需要输入或展示申请码/激活码：

```bash
bash /home/xchq/work/autoocr/overlays/rec-wide-service-e127-r1/start_rec_wide.sh --check
```

目前预期返回“Existing Jetson license is absent…”并退出78，这是明确的未满足条件，**不是服务已就绪**。不要伪造许可文件或修改校验函数让它通过。若继承了其它`AUTOOCR_*`位置/模型覆盖变量，会更早拒绝；使用干净终端，不把秘密值打印到日志。

## 真正启用前仍需完成

1. 用户明确决定恢复原板端激活流程，由原授权机制获得该板有效许可；不复制Windows/WSL许可，不重置计数。
2. 用`--check`验证原许可与包完整性，确认5051空闲；再在明确的维护窗口启用单worker loopback服务。
3. 验证真实`/healthz`、同图`/OCR`与必要`/Extract`、错误拒绝及正常停止/重启。此前允许的正常扣次不等于允许跳过许可。
4. 确认后才安装候选unit；回滚停止新服务并恢复原unit，二者不能同时占用5051。候选unit固定宽模式on；若手工运行off模式，健康检查也必须用off，不能只改unit环境变量却保留on检查。

`Restart=on-failure`只处理进程失败；当前工厂遇到模型错误会锁存并拒绝请求，但**单纯返回503不保证进程退出或systemd自动重启**。运行中GPU错误的监督/重启策略、超时中止、并发和真实HTTP故障恢复仍待联调，本轮不能宣称已完成自动恢复。

本轮测试包括无网络的健康协议回归、历史真实模型统计的离线重放、缺许可的启动拒绝及unit语法检查。它们不是新一次GPU推理或真实HTTP验收。技术证据见`TECHNICAL_LOG.md` E127。
