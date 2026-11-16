# 完整业务发布包与恢复流程（CPU历史基线）

2026-09-25 E93：WSL部署及旧回退包现已按用户要求清理，下面路径/回退步骤仅保留历史意义，不可直接恢复已删除版本。Windows原项目及既有归档未删，授权/扣次状态没有重置；见[技术日志](../TECHNICAL_LOG.md)E93。

E76更新：当前运行`autoocr-business-gpu-20260918-r1`及独立`.venv-gpu-r1`，OCR/版式/单元格/YOLO/NER/增强均有GPU执行。E75 TRT＋CPU模块版与原venv、CPU r2都保留。当前GPU路径、计数观测及限定回退见[GPU_DEPLOYMENT.md](GPU_DEPLOYMENT.md)。下文E71/E72属于CPU历史，不要用其更早的/mnt/d回退步骤覆盖当前部署。

本包是当前server.py的内部源码/模型快照，不是EXE、Docker镜像或离线Python环境。原业务接口、默认模型、授权生成/验证和计数算法不改。没有绕过激活，也不包含管理员生成器。源码仍含原有授权实现，应按受控内部交付保管，不能把“排除运行文件”理解为“可以公开整个包”。第三方模型/代码再分发权限需另行审查。

2026-09-18 E72：r2已在WSL独立发布目录运行，通过0096增强OCR→Extract与Windows基准对照及一次受控停启。最终使用完整unit，不使用E71提议的路径drop-in。当前文档已修正该流程；冻结r2包内的RELEASE.md仍是E71版本，不要照其旧drop-in步骤操作，也不要原地修改冻结包。

## 包含与排除

- 显式清单收集server及依赖、四个业务ONNX、YOLO权重与自定义层、NER推理文件、增强唯一.pth、OCR JSON字符表与现有rec/inference.yml。保留原代码YAML优先、JSON回退的选择，不静默换字典。
- 不递归复制项目。排除license.dat、计数/充值记录、日志/temp、原图/识别正文、训练checkpoint/指标、.git、IDE配置、Windows激活工具、venv和TensorRT实验引擎。
- release_manifest.json逐项记录路径/大小/SHA256，最后写入，作为完整构建标志。缺文件、内容改变、多余文件/目录、软链接/junction均拒绝验证。SHA256是完整性校验，不是签名或可信来源证明；只接收可信内部包。
- deployment_environment.json只记录已验证WSL Python和已安装包的名称/版本，不记录索引凭据/直连URL。它是观测快照，不是带wheel哈希的离线锁文件；requirements.txt仍是直接依赖。新机安装仍需受控包源，不能复制Windows虚拟环境。

## 构建与检查（不会导入业务或扣次）

在项目根目录使用release_business.py。所有输出必须是新名称；失败目录/压缩包保留，不应发布，也不要原地补成“成功”。

1. 在现有WSL business venv记录环境：

```bash
"$HOME/projects/autoocr-business/.venv/bin/python" \
  /mnt/d/docuyyy/autoocr/edge_deploy/business/release_business.py environment \
  --output /mnt/d/docuyyy/autoocr/temp/business_environment_r1.json
```

2. 在Windows原项目根目录构建（下列r2若已存在不要重复；有变更必须换release-id及输出名）：

```powershell
& 'D:\Anaproject\envs\env1\python.exe' edge_deploy/business/release_business.py build `
  --source . --output temp/releases/autoocr-business-cpu-20260918-r2 `
  --release-id autoocr-business-cpu-20260918-r2 `
  --environment temp/business_environment_r1.json
& 'D:\Anaproject\envs\env1\python.exe' edge_deploy/business/release_business.py archive `
  --root temp/releases/autoocr-business-cpu-20260918-r2 `
  --output temp/releases/autoocr-business-cpu-20260918-r2.tar.gz
```

3. 仅将可信包展开到一个**全新、空的暂存目录**，不要以root向服务目录解包，不覆盖current或现有发布目录。然后使用包中验证器检查：

```bash
release="$HOME/projects/autoocr-business/releases/autoocr-business-cpu-20260918-r2"
"$HOME/projects/autoocr-business/.venv/bin/python" \
  "$release/edge_deploy/business/release_business.py" verify --root "$release"
```

验证要求精确文件集合：不要在发布目录添加pycache、日志或README笔记。启动入口设置PYTHONDONTWRITEBYTECODE=1；验证成功不代表模型已加载或业务接口已验收。原清单/摘要另存在报告目录；需要真实性保障时另做发布签名，不以自带清单替代签名。

## 三类目录不能混用

| 类型 | 当前实例 | 处理方式 |
|---|---|---|
| 发布源码和模型 | `~/projects/autoocr-business/releases/<release-id>` | 每版新目录，不原地修改 |
| Python依赖 | `~/projects/autoocr-business/.venv` | 当前先复用；新机单独安装、pip check，记录版本 |
| 机器运行状态 | `~/projects/autoocr-business/runtime` 及下列HOME状态 | 原机同用户保留，不能作为通用包复制或回滚 |

特别注意：server现有Linux计数/已用充值码状态在`~/.cache/sys_config_d/`，另有`~/.local/share/.python_sys_cache/`备份路径。它们不是普通可清理缓存。许可证在runtime/license.dat；日志/temp也在runtime。不要删HOME缓存来“重装”，不要换HOME或用root启动，不要复制Windows许可证到Linux。不打印、手工编辑、回滚余额/已用码记录；发布回退只换代码/模型，继续使用最新机器状态。保护/备份这些状态时须停业务并按授权管理流程处理，本工具不自动备份或恢复它们。

新机器需要正常合法激活；原WSL同机器同用户只更换发布代码路径时仍需校验既有许可，不应重新生成或覆盖许可。原始Linux硬件指纹的历史限制未在此包中修复。

## 发布切换：维护窗口、完整unit与合并配置核验

当前服务已由autoocr-business.service运行r2；不要重复首次切换。本节用于审阅下一版发布流程，切换前必须更新release路径、记录当前配置并安排维护窗口。避免同时加载两套完整模型或多个进程并发修改同一计数状态。

E72修正：在本机systemd 249上，空的RequiresMountsFor=没有清除基础unit中的旧路径。实测WorkingDirectory/ExecStart虽已是r2，但RequiresMountsFor仍含/mnt/d/docuyyy/autoocr。核对[v249解析实现](https://raw.githubusercontent.com/systemd/systemd/v249/src/core/load-fragment.c)的config_parse_unit_requires_mounts_for，空值直接结束解析、不执行清空。不能把其他列表指令的reset规则套用到这里，也不能只检查服务ready就认定旧挂载依赖已经移除。

1. 确认无请求处理，检查FragmentPath、DropInPaths、所有Exec和环境。备份实际安装的unit及本次已知drop-in到新备份目录并记SHA256，不以工作区副本冒充实际配置，不覆盖已有备份。
2. 正常stop、确认5051空闲，保留User/Group=q1220及原HOME/runtime/venv。旧/mnt/d工作区不是不可变回退版本，回退前须确认其源码仍符合预期。
3. 审阅[本机r2完整unit](systemd/autoocr-business-r2.service)，安装到/etc/systemd/system/autoocr-business.service。仅移除已经核对哈希、属于本次的20-release-r2.conf，不删除其他管理员配置。完整unit的RequiresMountsFor只写WSL目录，保留原单worker、CPU、loopback、超时、权限与重启限制。
4. systemd-analyze verify后daemon-reload，再检查有效RequiresMountsFor不含/mnt/，WorkingDirectory及全部Exec入口均来自对应release，User/HOME/runtime未变，然后start。验证器通过不等于业务验收。
5. 检查enabled/active/running、health及新模型路径，做必要单图增强OCR→Extract回归，再安排一次受控停启并确认ready。长请求时health也会排队；健康探测不能代替请求排空的维护约定。
6. 启动失败先保留技术诊断，只恢复已备份的代码入口；不自动删许可、激活或反复重试业务调用。有效Extract超时可能已扣次。无启动失败时不要reset-failed。

本轮一次性脚本cutover_release_r2.sh（中间尝试）和finalize_release_r2.sh（修正）记录已执行步骤，绑定旧PID及配置哈希，不是可重复执行的通用安装器。20-release-r2.conf仅为历史证据，不作为推荐配置。两阶段最终均正常启动，未触发脚本的失败回退分支。

## 回退、日常启停及限制

- 本机最终配置是完整unit，回退不能只删除drop-in。原unit备份为`/home/q1220/projects/autoocr-business/deployment_backups/20260918-r2/autoocr-business.service.before`，SHA256=`fc7b5e3eb12332fa67b902fc72b5bcec9e32bce82d5423e66fec7692a5de87f9`。中间drop-in另存同目录20-release-r2.conf.intermediate，仅供审计，不要恢复它。
- 需要回退时，先确认旧/mnt/d代码/模型仍完整、记录当前有效配置及备份哈希、排空业务并stop。仅将上述原unit以root:root/0644恢复到/etc/systemd/system/autoocr-business.service，保留未知管理员drop-in并评估其影响；verify、daemon-reload、核对旧入口后start及health。不要广泛删除或用systemctl revert清空未知配置。此失败回退尚未实测，不能写成“故障恢复验收完成”。
- 不删除当前runtime/HOME状态，不从旧备份恢复计数。若许可失败，走原授权排查，不用自动重新激活制造通过。
- 日常按需单独执行`sudo systemctl stop/start/restart autoocr-business.service`，不要连跑三条；启动限流后先查原因，必要时再reset-failed和start。启动探针不等于持续watchdog。
- 长请求期间单worker的health也会排队；这不自动证明服务故障。当前没有增加反向代理/TLS、Windows开机任务、冷启动验收或长期压力测试。
- r2现已完成发布目录启动、真实增强OCR→Extract对照及受控停启后ready，可以标记“本机CPU业务版本已部署”。不是故障注入、冷启动、新机器安装或长期稳定性验收。

日常查看（不加载第二套模型、不扣次）：

```bash
systemctl show autoocr-business.service -p ActiveState -p SubState -p MainPID -p NRestarts -p WorkingDirectory -p RequiresMountsFor
curl --noproxy 127.0.0.1 --fail --max-time 15 http://127.0.0.1:5051/healthz
```

维护窗口内按需执行一次`sudo systemctl restart autoocr-business.service`；不要额外手动运行start.sh/start_release.sh。本轮最终MainPID10886、NRestarts0，RequiresMountsFor仅WSL home发布/业务目录及PrivateTmp引入的/var/tmp；没有实际卸载Windows盘验证，不能外推为整台WSL不依赖Windows宿主。
