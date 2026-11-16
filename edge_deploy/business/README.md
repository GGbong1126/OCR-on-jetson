# WSL 完整业务部署（TensorRT＋ORT CUDA＋PyTorch CUDA）

2026-09-25 E93：本文描述的WSL部署已按用户要求卸载，服务及autoocr-business/autoocr-edge目录、TensorRT SDK均已删除。下文保留历史实验/复现意义，不代表当前仍运行，也不能直接执行启停/回退命令。Windows源码、Jetson部署不变；少量本机许可/扣次状态保留，见[技术日志E93](../TECHNICAL_LOG.md)。

范围：直接加载项目server.py，保留POST /OCR、POST /Extract和授权/计数。E76在E75的TensorRT OCR上增加ORT CUDA版式/单元格、PyTorch CUDA的YOLO/NER/增强，全部模型有实际GPU调用证据；不是仅独立det+rec演示，也不是所有算子全GPU。

2026-09-18 当前实测（E76）：三图增强OCR→Extract共142框及指定字段与Windows严格一致；GPU调用det3/rec198、layout3/cell2/YOLO3/NER3/增强24批，无错误、OCR回退0。正常重启后MainPID11852、NRestarts0、enabled/active/running、5051 ready。**用systemctl管理，不要重复安装、激活或手动start.sh抢占端口。** 见[GPU部署说明](GPU_DEPLOYMENT.md)和[API](API_USAGE.md)。字段GT质量、故障注入、WSL冷启动及长期稳定性仍未验收；以下旧CPU步骤为历史说明。

## 目录与约束

E68～E72已完成WMIC兼容和CPU r2独立发布。E75继续保留同一用户/HOME/runtime/venv及正常授权路径，在独立GPU发布中仅增加原OCR session适配。两版包都保持不可变，旧CPU r2及服务配置可作限定回退；详见[GPU部署说明](GPU_DEPLOYMENT.md)。

- 当前源码/模型/两TRT引擎使用 `~/projects/autoocr-business/releases/autoocr-business-gpu-20260918-r1`。旧TRT r1和CPU r2保留；修改Windows工作区不会自动更新已部署包，变更需新release及回归，不原地修改发布目录。
- 当前环境 `~/projects/autoocr-business/.venv-gpu-r1`；原`.venv`不变，用于旧版本回退。不修改 autoocr-edge/.venv。
- 运行文件 `~/projects/autoocr-business/runtime`：license.dat、server.log、temp、requirements-resolved.txt。业务日志/临时结果可能含正文，不直接分享或入 Git。
- 原计数/已用充值码还在HOME/.cache/sys_config_d及HOME/.local/share/.python_sys_cache，不能当普通缓存清理，不随通用包分发或随版本回滚；详见[发布说明](RELEASE.md)。
- 默认仅 `127.0.0.1:5051`，一个 Gunicorn sync worker、一个线程，不 preload、不 reload。此服务没有新增网络认证/TLS，不向公网或局域网开放。
- 当前OCR的TensorRT profile：det H960/W672～736、rec H48/W32～1536、batch1；超范围明确走原ORT CPU，health记录次数。不改输入来套profile；不使用分桶补白/混合精度候选。其他模型CUDA、FP32/noTF32，增强batch1适配4GB显存。
- `/healthz` 仅在授权及所有模型初始化成功后提供，不能代替真实 OCR/Extract 功能验收。单 worker 正在执行长请求时健康查询也可能排队。

## 1. 安装（仅首次）

```bash
bash /mnt/d/docuyyy/autoocr/edge_deploy/business/install.sh
```

安装脚本拒绝覆盖已存在的 `.venv`。中途失败先查看错误，不删除环境；可使用其 Python 分别重试失败的 pip 命令。不要运行根目录原 requirements.txt，其中有拼接错误和 Windows/GPU 环境遗留。

依赖采用原环境直接依赖版本和 PyTorch 2.8.0 / torchvision 0.23.0 CPU 配对。安装后执行 pip check，并保存实际 freeze；requirements.txt 不是完全锁定的传递依赖清单。参考 [PyTorch 官方历史版本](https://docs.pytorch.org/get-started/previous-versions/)。

## 2. 本机合法授权（用户在 WSL 终端操作）

不能复制 Windows 许可证就假定 WSL 已授权；不生成管理员激活码、不绕过验证、不在聊天/日志粘贴授权码。

```bash
export AUTOOCR_SOURCE_ROOT=/mnt/d/docuyyy/autoocr
export AUTOOCR_RUNTIME_DIR="$HOME/projects/autoocr-business/runtime"
"$HOME/projects/autoocr-business/.venv/bin/python" \
  "$AUTOOCR_SOURCE_ROOT/edge_deploy/business/serve_business.py" --activate
```

终端显示设备请求码，交给有权限的授权管理员，再在隐藏输入提示处填写激活码。流程复用 server.py 原验证/许可证写入方法；不会调用 Tk 弹窗。非交互输入或验证失败时退出，不后台循环重试。

当前 server.py 仍有计数存储与 `/Extract` 成功后的扣减，本次保留。用户此前“只需硬件授权”的历史描述不等于当前文件已经删除计数；若需要变更，单独确认后处理。设备指纹目前仍沿用原实现，其中 Linux 的 WMIC 调用会失败并使用原有占位项；不能声称已完成可靠的 WSL 硬件绑定加固，也不在本次静默更换指纹算法使许可证失效。

## 3. 原始前台启动（新环境或回退用；当前已托管，勿直接执行）

```bash
bash /mnt/d/docuyyy/autoocr/edge_deploy/business/start.sh
```

采用正式 WSGI server 而非 Flask 开发服务器，参见 [Flask Gunicorn 部署说明](https://flask.palletsprojects.com/en/stable/deploying/gunicorn/)。不要直接运行 `gunicorn server:app_online`：该方式不会执行原 main() 的模型加载/授权，可能得到未初始化服务。必须使用本目录 create_app 工厂。

运行脚本会先核验授权，再在 worker 内加载完整模型；图像增强/YOLO/NER 等任一未加载时拒绝 readiness，避免“端口能访问但少了业务功能”。默认最大 HTTP body=24MiB。加载失败保留 runtime/server.log，不删除许可证或改验收开关。

## 4. 本地检查

另开 WSL 终端：

```bash
curl --noproxy 127.0.0.1 --fail --max-time 15 http://127.0.0.1:5051/healthz
```

应返回 status=ready，scope=full_business 和各模型就绪标记。然后按项目原 V1.1 协议提交 /OCR 的 `img`（纯 Base64）和 `imageType`，再以其结果调用 /Extract；不修改输入协议，不把仅健康检查成功称为完整功能通过。/Extract 可能扣减现有业务次数，正式验收前注意当前授权余额。

现阶段systemd托管已配置并验证启动/受控重启（E61），未引入Docker、Windows开机任务、反向代理或外网访问。WSL 当前约4GiB内存/1GiB交换空间，全业务推理内存峰值仍须测量；若 OOM，保留错误再调整资源，不能通过跳过模型冒充完整部署。

## 本轮迁移改动

server.py：Tk 可选导入（GUI 默认逻辑不改）；Paddle 包装器移到实际回退分支，四 ONNX 可用时不强装 Paddle；日志/许可证路径可配置。修复 ONNX layout 分支原先跳过 YOLO 加载的遗漏，保留原 YOLO 阈值及融合逻辑。该修复可能改变原 ONNX 版式输出，必须作为业务回归项，不声称与旧错误分支逐字等价。

server_config.py 新增 AUTOOCR_BASE_DIR，原 Windows/EXE 默认保持。业务 launcher 不改原服务器输入输出协议，也不禁用授权或计数。部署单元测试是模拟控制流验证，不等于 WSL 模型/HTTP 验收。

layout/__init__.py 的包装器导出也改为按需导入：首次 WSL 导入测试发现只移动 server.py 中的 import 不够，导入 layout.layout_fusion 仍会先执行包初始化并隐式加载 Paddle。修正后不安装 Paddle 也能导入业务 ONNX 路径；明确使用 Paddle 包装器仍需要其依赖。

## E57 自定义层与接口验证

best1.pt依赖ultralytics.nn.bifpn.Concat_BiFPN，它是原Windows环境本地新增层，不在标准ultralytics8.3.28包里。layout/yolo_layers/bifpn.py保存原实现，layout/yolo_compat.py只在原模块缺失时注册兼容路径；Windows已有该模块时保持原实现。原类三个权重归一化后对两个输入加权拼接的计算不改，不用通用BiFPN替代。业务入口关闭YOLO自动pip安装，保留模型加载错误可见。

一次真实CPU HTTP冒烟已通过：006.jpg/imageType3，55块、坐标范围正确、请求后health仍ready。wmic告警不阻止当前许可验证，本轮不更改指纹算法以免使刚激活的许可失效。原502来源未单独查明；当前采用明确本机地址无代理直连，避免依赖环境代理。

可重复运行（不会执行有效Extract扣次，默认只测health/参数校验，加--image测OCR）：

```bash
"$HOME/projects/autoocr-business/.venv/bin/python" \
  /mnt/d/docuyyy/autoocr/edge_deploy/business/smoke_business.py \
  --image /mnt/d/docuyyy/autoocr/test/006.jpg
```

有效Extract在上面命令增加 `--extract`，可能消费一次授权计数。E61用户已持续允许本项目必要业务验证扣次，不需逐次确认，仍避免无意义重复请求。客户端将OCR结果在内存交给Extract，仅输出顶层键数量，不打印字段值。增强另用 `--image-type 1` 或 `2`；imageType1已在E60完成接口冒烟，不同时启动多个请求。不要将块数作为准确率或业务全部通过的证明。

E57时服务由会话进程承载；E61已改为systemd托管，当前应使用文末命令启停。若端口已占用，先检查现有unit/health，不重复启动或直接杀未知进程。

## E59 增强开启请求（E60已执行，无需重复）

用户已完成有效Extract冒烟：OCR单次82.1008秒、Extract单次2.6064秒，业务状态检查通过。returned_field_count=2表示msg的title和NER两个顶层键，不能据此判断NER非空或字段内容正确；无需再重复扣次测试。

先使用同一图片测试imageType1（源码中1/2均进入增强分支），不添加--extract：

```bash
"$HOME/projects/autoocr-business/.venv/bin/python" \
  /mnt/d/docuyyy/autoocr/edge_deploy/business/smoke_business.py \
  --image /mnt/d/docuyyy/autoocr/test/006.jpg \
  --image-type 1
```

只输出技术统计；不发送有效Extract，不进入其扣次路径。等待该请求完成再做其他检查，避免单worker排队。若超时/错误，先保留输出分析：业务OCR阈值180秒、客户端240秒，当前不自动放宽；增强效果和是否回退仍需结合证据核验。增强测试本轮尚未执行，之后再安排进程托管/重启恢复。

## E60 托管前检查（已于E61完成切换）

用户已完成增强开启请求，image_type="1"、57块、单次85.7434秒、坐标范围通过；末行imageType3仅为固定提示，不是此次参数。当前OCR（增强开/关）及有效Extract的单图接口冒烟已通过，输出数量不是质量结论。

已只读确认systemd249正在作为PID1运行，autoocr-business.service不存在，127.0.0.1:5051仍由此前Gunicorn进程监听。准备沿用本目录start.sh、q1220身份及现有runtime，配置单worker服务托管和失败重启；先确认允许短暂停机，再停止已确认的旧实例，避免端口冲突。本轮没有安装unit/变更启停，不需要修改wsl.conf或shutdown WSL。

后续先以health检查新服务，再做受控重启/健康恢复；不会为验证托管反复发送扣次Extract。参考[Gunicorn systemd部署](https://gunicorn.org/deploy/)。[Microsoft WSL说明](https://learn.microsoft.com/en-us/windows/wsl/systemd)指出systemd服务本身不会保持WSL实例存活，因此Linux托管不等于Windows开机无人干预常驻，后者尚未配置。

## E61 当前操作：systemd托管已启用

安装位置：`/etc/systemd/system/autoocr-business.service`（root:root、0644），源码副本在本目录。运行身份仍为q1220，HOME=/home/q1220，复用原runtime和venv；不以root运行模型，不重新激活，不重置计数。首次启动和受控restart已实测恢复ready，仍单worker/CPU/本机5051。

查看技术状态及健康（无扣次）：

```bash
systemctl show autoocr-business.service -p ActiveState -p SubState -p UnitFileState -p MainPID -p NRestarts
curl --noproxy 127.0.0.1 --fail --max-time 15 http://127.0.0.1:5051/healthz
```

按需启停（先确保没有业务请求；sudo在自己的终端输入密码，不发送密码到聊天）：

```bash
sudo systemctl restart autoocr-business.service
sudo systemctl stop autoocr-business.service
sudo systemctl start autoocr-business.service
```

以上三条是独立操作示例，不必整段连续执行。无需手动activate虚拟环境。start/restart会等待ExecStartPost健康检查通过；最多等待180秒，unit启动超时240秒。端口占用由ExecStartPre拒绝；遇到失败先看本地日志，不抢杀未知进程。

日志只在本机查看，可能包含正文，**不要原样分享**：

```bash
journalctl -u autoocr-business.service -n 60 --no-pager
```

业务日志仍在runtime/server.log。UMask0077影响新建文件，不代表旧日志已改权限。当前没有新增日志轮转/保留期限方案，磁盘与日志治理后续处理。

异常退出配置10秒后重启、30分钟最多3次启动（包括手动重启）；超过限制时先修复原因，再执行：

```bash
sudo systemctl reset-failed autoocr-business.service
sudo systemctl start autoocr-business.service
```

配置策略已经安装；本轮只验证受控restart，没有SIGKILL等故障注入。启动后health监测不是持续watchdog，进程活着但无响应不会仅凭此探针触发systemd重启。

回退到旧前台方式（仅托管有问题且决定回退时）：

```bash
sudo systemctl disable --now autoocr-business.service
"$HOME/projects/autoocr-business/.venv/bin/python" \
  /mnt/d/docuyyy/autoocr/edge_deploy/business/service_health.py --check-port
# 只有上一条确认端口空闲才执行
bash /mnt/d/docuyyy/autoocr/edge_deploy/business/start.sh
```

保留unit和许可文件，不删除环境。再次托管前先正常退出自己的前台服务，确认5051空闲，再`sudo systemctl enable --now autoocr-business.service`。

新环境安装单元需先审阅本目录service文件中的用户/绝对路径，完成安装和合法激活，并确认同名unit不存在、旧实例已安全退出。然后才复制：`sudo install -o root -g root -m 0644 autoocr-business.service /etc/systemd/system/autoocr-business.service`（在本目录运行），再`sudo systemd-analyze verify /etc/systemd/system/autoocr-business.service`、`sudo systemctl daemon-reload`及`sudo systemctl enable --now autoocr-business.service`。当前环境已安装，不要重复此流程覆盖配置。

已enable意味着配置为WSL发行版启动时启动，不等于Windows启动WSL/持续保活；本轮未重启WSL验证冷启动，也未配置Windows计划任务。下一阶段若需要Windows开机常驻，单独安排该层，并做字段质量/长期运行验收。

## E62 当前下一步：业务调用方与字段质量联调

本机Windows访问127.0.0.1:5051及一条增强OCR→Extract实际链路已通过：57块、OCR89.53秒、Extract2.50秒，1个非空NER字段，title为空。服务仍active/ready；客户端没有打印正文。此次为systemd环境与Windows调用方的集成验证，不是又一轮性能benchmark。

新版smoke客户端输出returned_top_level_key_count、nonempty_top_field_count、ner_field_count、nonempty_ner_field_count；旧returned_field_count保留且明确只数顶层键。OCR输出正文字符数，最后说明跟随实际imageType，避免固定提示造成误解。客户端改动无需服务重启。

已交付[API_USAGE.md](API_USAGE.md)：包含Windows/WSL命令、接口入参、Markdown前三行约定、非空字段含义、超时/幂等限制及隐私注意事项。现在用有权处理的代表性图片核对业务必需字段，暂不改阈值或以更多框冒充准确性提升。没有真实标注，title为空尚不能判为错误；NER字段也可能由规则兜底，需人工核对。

本轮有效Extract已使用持续扣次授权执行一次，未读取余额，也未自动重试。后续必要验证同样按需调用；本机部署闭环已建立，不必再重复006示例。

## E63 多图功能完成，优先处理质量疑点

0075/0093/0096的增强OCR→Extract均完成，各一次有效Extract（可能正常扣次），无自动重试；批次状态completed_pending_manual_review。服务仍同一主进程、active/ready。新增批量工具及[验收说明](ACCEPTANCE.md)，184项本机测试通过；无需重启服务。

项目根目录temp/business_acceptance_r1内的summary.json和manual_review.md均被Git忽略，无正文/字段值。原图检查提示0093分类号/页码/性别可能漏提取；0096打码字段非空、多编号归属需核验。没有完整人工标注，不将非空数量当正确率，不推测打码内容。

下一步检查字段从版式/识别到Markdown/Extract的归属链路，并确认业务期望；独立Linux发布包仍待准备，当前不搬运模型或扩大测试次数。

## E64历史：Windows原生服务对照等待Windows授权

用户选择以Windows结果为回归基准。已增加[两端对照工具及说明](WINDOWS_COMPARE.md)，默认Windows5052、WSL5051，同图比较文本/坐标/字段，只保存技术统计，最多每图两次有效Extract，无自动重试。195项本机测试通过；WSL服务未改动。

当前Windows原项目license校验未通过且wmic不可用，未启动Windows服务或执行真实两端比较。请先确定基准是当前server.py还是历史EXE，并按原正常流程恢复有效Windows授权；不能将Windows客户端访问WSL5051误当作两端对照，也不能复制WSL许可证或改校验以求通过。

## E70 当前结果：本组三图与Windows基准一致

上述E64阻塞已在E68修复；E69/E70实际运行当前Windows server.main，两轮共对照0075/0093/0096。各13/56/73框逐字、坐标、顺序、Markdown以及指定字段存在性/值均一致，未观察到本组三图的迁移回归。结果与边界见[WINDOWS_COMPARE.md](WINDOWS_COMPARE.md)。保留原人工验收pending，不为追求“全通过”忽略共同错误或推测遮挡内容。

下一阶段整理独立发布与运维交付：源码/模型资产清单、与许可证/计数/日志分离的运行目录、启动/停止/恢复说明和发布回归。当前尚未建立或切换新发布目录，仍由现有WSL systemd托管，勿重复start.sh抢占5051，也不要复制Windows许可证充当WSL授权。故障注入、冷启动和长期稳定性仍未验收。

## E71 当前交付：独立候选包已建立，运行入口尚未切换

上述E70的“未建立发布目录”已更新：当前候选为autoocr-business-cpu-20260918-r2，Windows产物在temp/releases，WSL暂存在~/projects/autoocr-business/releases/同名目录。40个payload文件约889 MiB，gzip归档约817 MiB，两端逐项哈希/文件集合校验通过。原服务仍依赖/mnt/d，没有新包模型加载或业务推理结果。

使用[RELEASE.md](RELEASE.md)安排维护窗口：先停止旧单worker，审阅并验证专用systemd路径覆盖，沿用现有q1220/HOME/runtime/venv启动新候选，检查实际模型路径和必要业务回归，再验证恢复。不能并行启动两套进程共享计数，也不能回滚机器状态。发布工具不会自动安装unit、激活或重试扣次；本轮没有这些动作。

## E72 当前运行：r2独立目录已上线

E71的待办已执行。首次切换后完成0096增强OCR→Extract与Windows基准对照：73框逐字/顺序/坐标、Markdown以及6项指定字段严格相同，2次有效Extract尝试全成功、无重试；这是回归一致性，不是GT准确率。

systemd 249实测空RequiresMountsFor没有清除/mnt/d，已备份实际unit并安装[完整r2配置](systemd/autoocr-business-r2.service)，不再使用中间drop-in。21:48:15～21:48:27受控停启后MainPID10886、NRestarts0、enabled/active/running，health全部ready；清单40文件哈希不变。技术摘要见../../temp/business_release_r2_acceptance/deployment_review.md。临时Windows5052已关闭，只保留WSL5051。

原user/HOME/runtime/venv及许可证、正常计数路径保留，不自动重新激活。当前只做CPU内部本机部署，未开放网络或接入TensorRT。冻结r2包内旧版RELEASE.md的drop-in方法不要再执行；按[当前运维说明](RELEASE.md)启停和限定回退。故障注入/失败回退、WSL冷启动、新机环境重建、长期稳定性及人工字段质量仍待完成。
