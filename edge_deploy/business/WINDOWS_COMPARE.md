# Windows 基准与 WSL 完整业务对照

目标：同一张原图分别经过两端完整 `/OCR` → `/Extract`，以 Windows 返回为回归基准。不是以 Windows 结果充当人工真值，也不据单次时延评判部署性能。

## 当前状态（E70）：三张图的完整业务回归一致

2026-09-18已扩展0075/0096，沿用当前Windows server.main作基准、现有WSL部署作候选，均imageType=1，OCR→各端自身结果送Extract，严格比较不忽略空格/数字/符号。

| 图片 | Windows / WSL块数 | 文字、顺序、坐标 | 完整Markdown | 6项字段存在性/值 |
|---|---|---|---|---|
| 0075 | 13 / 13 | 完全一致 | 完全一致 | 全部一致 |
| 0093（E69） | 56 / 56 | 完全一致 | 完全一致 | 全部一致 |
| 0096 | 73 / 73 | 完全一致 | 完全一致 | 全部一致 |

三张共142框，全匹配、每对IoU=1。0075的三个NER字段双方缺失；0093的空字段见下文；0096的6项字段双方非空且相同。Windows同样输出不代表人工真值正确，原有遮挡字段/编号归属疑点保留。只是本组既有样本的跨平台兼容证据，不是全量无回归或准确率100%。

本轮temp/business_windows_wsl_r2/summary.json（2张）与r1（1张）共同构成证据；三图便读汇总在r2/comparison_review.md。Windows r2记录的4个源文件、4个ONNX/YOLO模型哈希及依赖均与r1相同，实际CPU、torch8/OpenCV1；WSL仍CPU、torch2/OpenCV1、单worker，不作为受控性能比较。详细单次时延、图片哈希见技术日志E70。

本轮4次有效OCR和4次有效Extract尝试全成功，无重试；未修改业务代码、授权/计数或生成器。临时Windows PID53408已核对身份后停止，5052关闭；WSL仍enabled/active/running，MainPID9924、NRestarts0、health ready。compare_business的11项模拟测试本轮再次通过。下一步整理独立部署包与启动/恢复流程，不必继续重跑同组图片；人工质量验收仍另行进行。

## E69历史：0093真实同图完整业务结果一致

2026-09-18实际运行Windows当前server.main（原默认设备选择，临时127.0.0.1:5052）与WSL已部署服务（127.0.0.1:5051）。两端输入同一份0093.jpg字节、imageType="1"，各自OCR结果送各自Extract，字段定义相同。

| 比较内容 | 结果 |
|---|---|
| 56个文字块 | 逐字、顺序、坐标全部一致，56对IoU=1、无未匹配项 |
| Markdown | 完整字符串一致，共278字符；块文本227字符 |
| title / 出生年月 | 非空，分别13/9字符，字段值一致 |
| class / pagenumber | 两端均存在但为空串 |
| 姓名 / 性别 | 两端均缺失 |

结论：本图没有观察到WSL引入的结果差异。E63的空字段现象在Windows基准也存在；这不是字段准确率通过，不把Windows当人工真值，也不外推历史EXE或其他图片。

技术结果在temp/business_windows_wsl_r1/summary.json，便读摘要在同目录comparison_review.md；未保存私人OCR正文或字段值。Windows运行元数据在temp/windows_reference_r1/runtime_info.json，记录实际四个CPUExecutionProvider、CPU增强/NER、torch8/OpenCV1、依赖版本、部分源文件/模型哈希。WSL为CPU、torch2/OpenCV1、单Gunicorn worker。Windows/WSL单次OCR约17.069/87.247秒；未经预热重复、线程/环境不同，不作为受控性能或WSL开销结论。

本轮2次有效OCR、2次有效Extract（各端一次），全成功、无重试；沿用原正常扣次路径，未审计余额。临时Windows PID50688完成后已关闭；WSL仍enabled/active/running、MainPID9924、NRestarts0、health ready。下一步扩展0075/0096，再分别核验人工质量；本轮未追加这两张的有效请求。

## E68历史：兼容读取已修复，本机原Windows许可证通过

用户确认只需保持原生成/校验逻辑，随机填充仍按原规则。已新增根目录windows_hardware.py，并将server.py四处WMIC读取接入适配层。实际导入Windows server.py后validate_license为true、license.dat字节未变，无需重新激活；没有生成真实激活码、修改计数或加载模型。Windows全套215项测试与WSL新增20项测试通过；两端真实OCR对照仍未执行。下文E64～E67为历史诊断与方案，不再表示本机仍被许可校验阻塞。

E64历史检查（2026-09-18）：WSL `127.0.0.1:5051/healthz` ready；未发现 Windows 原生 OCR 监听5051/5052。原 Windows Python 环境导入当时server.py后，原项目license.dat的validate_license返回false，同时wmic不存在。E68已修复并验证本机旧许可通过，未覆盖许可内容。另一个generate_12.py进程未被停止或操作。

不要在这个状态下将“没跑 Windows”记录为对照通过。E65用户已确认使用当前项目server.py，不是EXE。先处理该源码在当前Windows上的授权依赖兼容问题，再按正常流程校验/激活。不要复制 WSL 许可证到 Windows，不跳过校验，不自动生成或重置许可/次数。

E65只读检查：Windows版本10.0.26200；Get-Command找不到wmic，标准路径C:/Windows/System32/wbem/WMIC.exe也不存在。源码生成请求码经build_plain_bytes调用_get_disk_serial，Windows分支直接运行wmic；这意味着重新激活也依赖缺失命令，不能只让用户反复启动。Get-WindowsCapability在线只读查询仍要求管理员提升，未取得组件状态、未安装任何组件。

E66用户澄清：以前修复方式是给Windows系统补装WMIC，并非修改源码。因此不能用代码仍调用wmic推断“以前没有修过”。[微软移除时间线](https://support.microsoft.com/en-us/servicing/os/windows/docs/2025/09/windows-management-instrumentation-command-line-wmic-removal-from-windows)说明升级25H2会移除已装WMIC，2026年又进一步移除FoD支持；系统更新是当前缺失的候选原因，但未核对本机补装日期/具体更新，不作确定归因。先核对系统变化；本轮不改授权代码，不重装组件、不重新激活。

[Microsoft Learn的2026年8月更新](https://learn.microsoft.com/en-us/windows/whats-new/deprecated-features)说明Windows 11 24H2及以上已移除WMIC，且不再提供该FoD；WMI本身未移除。不照搬旧版补装教程。建议经用户确认后，仅为Windows硬件读取增加PowerShell/CIM兼容实现，保留真实硬件校验、授权验证和计数，不改OCR流程及WSL分支。旧指纹包含OS版本和格式等因素，无法保证兼容读取后旧许可一定通过；如不通过仍需正常重新激活，不能写入新指纹强行放行。当前未实施修复。

## E67 建议的最小兼容补丁（尚未实施）

用户希望无需重装WMIC，并保持授权码及激活工具兼容。建议只为Windows硬件读取增加PowerShell/CIM路径：查询同一Win32_DiskDrive.SerialNumber属性，输出JSON，不把表格显示当数据；不改为卷序列号/UUID/MAC，不擅自大写、删除内部字符、换磁盘或改截断规则。WMIC仍存在的环境可保留旧路径；多盘排序、主板列内空白和CPU回退值需专门核对，不能假设两种命令输出格式天然一致。Linux/WSL分支、授权编码/验证、计数协议和业务推理不在此补丁范围。

2026-09-18只读实测：CIM查询成功，本机1块盘且序列号非空；现有license.dat内请求码可解析，CIM值按原“超过19字符取末16字符”规则处理后，与旧请求中身份相符。只输出数量/匹配布尔值，不显示标识、请求码或许可值，没有写文件、激活或扣次。此结果只证明旧请求中的硬盘身份能通过CIM读到，不代表旧WMIC所有字段的原始输出已重建，也不等于license指纹校验通过。

兼容要求必须分开：

1. **编码/激活协议保持**：保留server.py生成和验证函数及generate_12.py，固定虚构序列号与相同随机状态时生成请求相同；使用现有generate_12生成的激活码应由现有server验证通过。本轮提取纯函数进行12/19/29字符虚构序列号检查，均匹配原截断和激活协议，不导入业务模型、不生成真实设备激活码。
2. **请求字符串不一定恒定**：原build_plain_bytes使用随机位置/填充，一般同硬件也可能生成不同请求；恰好19字符占满负载的边界没有填充空间，本轮换随机状态仍得到相同请求。不能宣称“所有序列号每次请求都不同”，也不能用固定随机种子把请求变固定。激活码绑定完整请求字符串：请求变了，旧激活码不能拿来配新请求；新请求仍可由原工具生成对应激活码。
3. **旧license.dat免重激活是另一项**：指纹包含CPU、Windows版本/架构、主板、硬盘及旧输出格式。CIM单项序列号匹配不足以保证完整哈希相同；不能修改已存指纹或跳过校验。若用户额外要求旧请求逐字复用，应单独设计真实设备核对与请求持久化，不靠存储值冒充实时硬件；现有补丁方案不默认增加此行为。

另外，generate_12.py对应硬件激活；32Activation.py对应次数充值。源码显示后者使用24字节/Base85/17字符硬盘片段，而当前server计数码使用21字节/Base58/14字符片段，是另一个已有协议差异，不能因都是32字符就承诺互通。此次未修改或实际兑换次数码，后续如需要须单独核对实际在用充值工具。

参考：[Microsoft Win32_DiskDrive定义](https://learn.microsoft.com/en-us/windows/win32/cimwin32prov/win32-diskdrive)、[Get-CimInstance说明](https://learn.microsoft.com/en-us/powershell/module/cimcmdlets/get-ciminstance?view=powershell-7.5)。源码兼容补丁与完整激活测试尚未执行，本轮为方案和非写入诊断，不宣称Windows服务已恢复。

## E68 补丁范围与验证结果

- 仅server.py四个WMIC调用改为windows_hardware.check_output，新增显式import；其余指纹拼接/哈希/校验不变。工具只支持原有三种查询，拒绝任意命令；存在WMIC时保留查询路径，不存在时通过系统PowerShell的CIM查询相同类/属性。非Windows继续原subprocess行为。
- CIM使用固定白名单脚本、NoProfile/NonInteractive、JSON及UTF-8，15秒超时，不用shell拼接、不安装组件或改执行策略，不记录真实硬件数据。已有WMIC执行失败不会静默切换身份；CIM失败不伪造序列号。空盘标识仍无法生成请求。
- 保留提供者枚举顺序、不改大小写/内部字符/截断规则。主板兼容文本采用列宽max(列名、各行值长度)+2空格，再交原解析器；本机旧指纹实测相同，但不能据此保证所有地区/多盘/历史WMIC输出都一致。没有靠旧license值冒充实时读取，也没有添加请求持久化/复用或固定随机种子。
- 回归测试以改动前AST哈希锁定46个授权/计数相关函数及常量；指纹函数除了读取函数名外AST不变，非Windows磁盘分支AST完全不变。generate_12.py和32Activation.py整文件哈希未变，未修复/迁移后者的独立充值协议差异。
- 5种虚构序列号×3种随机状态，旧WMIC路径和CIM路径在同数据/同随机状态下请求逐字相同；原generate_12签发的对应激活码通过，错误码被拒绝。另有格式、空值、顺序、UTF-8、白名单、异常脱敏、超时和Linux转发测试。Windows全套215项通过，WSL新增20项通过；这是模拟协议/兼容测试，不是给真实机器自动激活。
- 本机实际无WMIC时，读取的硬盘身份仍与旧请求一致。随后真实导入server模块，只调用validate_license：true；license.dat前后字节完全相同、pipeline未初始化。没有修改余额、写授权、执行激活器或发送有效业务请求。现有WSL health仍ready，未重启。

源码分发必须带上根目录windows_hardware.py，preflight资产清单已加入它；不是给旧EXE热补丁，未做PyInstaller构建验证。下一步可按下面命令正常启动Windows5052并与WSL5051对照0093，不需要再重新激活本机。

## 对照条件

- 建议先比较 `数字化演示图片/0093.jpg`，imageType="1"；它在 E63 有待核验的空字段，适合区分 Windows 既有行为与迁移差异。
- Windows 用独立 loopback5052，WSL 保持现有5051。不能在 Windows 上请求5051就称为“Windows 推理”：该端口目前通往 WSL。
- 使用当前源码时先保留原 Windows 默认设备选择，记录其实际 ONNX/Paddle分支、ORT providers、GPU选择和依赖；WSL当前是CPU。不要悄悄把 Windows 改成CPU后称为原默认基准。若有差异，再做控制变量实验。
- WSL读取 `/mnt/d/docuyyy/autoocr` 下源码和模型，Windows对应同一磁盘目录；但运行进程可能加载了更早源码/环境覆盖配置，仍须确认加载时机和实际配置。相同目录不等于已经证明运行时完全相同。
- 当前工作区 server.py 已包含迁移时的 YOLO加载分支修复，不能把它等同于历史EXE。若用户指定EXE，应另记版本与模型，不替用户选择另一个基准。
- 两端请求的原图字节、imageType及字段定义一致。每端的 Extract 使用**该端自己的OCR结果**，这是端到端对照；它本身不能隔离NER差异。若需定位，可后续给两端相同OCR结果做单独Extract实验，当前工具不会自动追加扣次。

## 后续复现（使用新运行目录）

E68本机读取已修复、原许可正常；E69/E70已完成r1/r2。以下仅供后续有代码/环境变更时复现，当前无需重跑。临时入口调用真实server.main，不改默认设备、阈值或授权，新增只读/__reference_info并将日志/temp放独立runtime；不是生产部署方案。先确认5052空闲，runtime/output目录必须新建，不要覆盖已有报告：

```powershell
cd /path/to/OCR-on-jetson
& 'D:\Anaproject\envs\env1\python.exe' edge_deploy/business/windows_reference.py --runtime temp/windows_reference_r3
```

不要为了启动而绕过授权；保持现有WSL服务不变。确认5052由上述Windows进程持有、模型全部加载并记录实际后端，再在另一个PowerShell执行：

```powershell
cd /path/to/OCR-on-jetson
& 'D:\Anaproject\envs\env1\python.exe' `
  edge_deploy/business/compare_business.py `
  --reference-url http://127.0.0.1:5052 `
  --candidate-url http://127.0.0.1:5051 `
  --images '数字化演示图片\0093.jpg' `
  --image-type 1 --extract `
  --output-dir temp/business_windows_wsl_r3
```

上例r3仅为复现示例，尚未执行。E70已将--images设为'数字化演示图片\0075.jpg' '数字化演示图片\0096.jpg'并保留--extract，完成r2四次有效提取。工具不依赖Windows原生服务不存在的healthz；只预检空参数拒绝协议，这本身不证明模型已加载或进程身份。还需核对/__reference_info与Windows端口进程；对照结束只停止本次临时Windows入口，WSL继续由systemd管理。无需为复现重新激活或修改计数。

## 报告含义与安全限制

`summary.json`只保存技术信息：原图SHA256、端口标签、字符数、框坐标、相等标志、匹配索引、请求次数/状态/单次时延。不保存OCR正文、Markdown内容、姓名等字段值、Base64、任意服务端错误正文或授权信息。对比用完整响应只留在进程内存；不改变现有服务自己的日志/临时文件规则。

- 文字：有序文本列表是否一致、文本多重集合是否一致、Markdown原文和头三行/正文行是否一致。多重集合一致不证明顺序/字段归属一致。
- 框：按原图坐标互为唯一最佳IoU>=0.5匹配，分别报告匹配文本相等与坐标相等；有歧义、不足阈值或原图尺寸不同均不强行配对。不是拆分/合并匹配器，也不是检测精度验收标准。
- 字段：title/class/pagenumber及姓名/性别/出生年月，逐项比较存在性和完整字符串。缺失与空串不同；双方都空/缺失不代表识别正确。当前严格保留空格、数字和符号，不套用前期单独页码样本的空格豁免。
- `completed_compared_not_accuracy_validated`表示请求和比较完成，不表示没有差异或GT准确。要查看具体比较布尔值、未匹配项；不要只看顶层status。
- 最多5张显式图片；默认不发有效Extract，`--extract`每图最多两次，分别在Windows与WSL正常计数。串行、无重试、任一异常停止；新输出目录，失败不覆盖，不自动续跑。超时的Extract可能已扣次。
- 不同端口是必要检查，不是服务身份证明；相同端口的localhost/IPv4/IPv6别名均被拒绝，避免明显的自我对照。

E64历史：本机195项模拟/单元测试通过，其中11项新增覆盖文字/几何拆分、重排、匹配歧义、字段缺失/空串、同端口别名、无重试和不落盘私人正文，当时尚无真实Windows推理对照。E69再次运行这11项通过，临时入口语法编译通过，真实启动/元数据端点及同图OCR/Extract成功；仍未进行两端GT验证。
