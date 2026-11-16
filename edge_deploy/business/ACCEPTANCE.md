# 少量图片的业务验收

目的：检查现有CPU业务服务对多张真实图片的接口兼容性、字段返回和坐标范围，并准备人工质量核对。不是重复benchmark，也不把接口成功或非空字段称为正确率。

## 首轮样本

使用已有`数字化演示图片`中的0075.jpg、0093.jpg、0096.jpg，按该顺序串行请求。选择依据是前期独立模型实验的尺寸覆盖（det宽736/672/704）和0096曾出现较长rec输入；这是便利抽样，未证明完整业务代表性，不是新增独立测试集，也不把历史shape限制套到当前CPU服务。

统一imageType="1"。每张先OCR，再将真实OCR msg提交Extract：topFields请求title/class/pagenumber，NER请求姓名/性别/出生年月（与现有标签名称一致）。三个固定头字段可以为空；缺少字段可能是图片不含该内容，也可能是识别/归属问题，不能只按非空与否评定质量。

## 工具行为和隐私

`acceptance_business.py`：标准库客户端，复用现有接口校验，只访问本机。最多5张显式输入，默认**不**发有效Extract，须加--extract；用户已允许本项目必要验证正常扣次。

- 运行前读取所有输入并绑定SHA256；重复图片、缺失/过大输入、已存在输出目录在业务调用前拒绝。
- 串行执行，无并发、无自动重试。单张失败立即停止，剩余样本标记not_attempted，不跳过失败样本把批次伪装为通过。
- 每个请求之前与完成后更新技术checkpoint。Extract发送前即计attempt；若超时或中断，可能已扣次，不能因没有成功响应就盲目重跑。
- `summary.json`只保存状态、图片文件名/哈希、调用次数/单次时延、数量、字段存在/非空/长度；不保存Base64、OCR正文、字段值、服务器错误正文、许可证或余额。
- `manual_review.md`是空白判定清单，**不包含可直接评估准确率的文本结果**。以原图和本地调用方结果核对，填写“正确/错误/图中不存在/待确认”；不要把私人正文粘到聊天。
- 输出放项目`temp/`（已在.gitignore中），不覆盖旧批次、不自动断点续跑。失败时保留目录，先检查原因再决定后续范围。

## 首轮命令（已有报告时不要重复执行）

Windows PowerShell，项目根目录：

```powershell
& 'D:\Anaproject\envs\env1\python.exe' `
  edge_deploy/business/acceptance_business.py `
  --images '数字化演示图片\0075.jpg' '数字化演示图片\0093.jpg' '数字化演示图片\0096.jpg' `
  --image-type 1 --extract `
  --output-dir temp/business_acceptance_r1
```

该命令最多发3次有效Extract，每次可能扣一次；不代表实际余额变化已核实。客户端请求超时240秒，服务既有OCR业务阈值180秒没有为验收调整。终端只打印技术统计和进度。

## 结果如何判断

- `completed_pending_manual_review`：所有计划请求完成且前后健康检查通过，**质量尚未通过**。
- `stopped_on_error_no_retry`：至少一阶段失败或最终健康失败；查看completed_count、每张status/failed_stage、HTTP/业务status；error_type仅异常类型，避免泄露正文。
- 意外强杀时可能保留`running`/`*_in_flight`：执行是否完成/是否扣次未知，不自动重放。

人工验收还要明确哪些字段是该图片必需、哪些不适用；核对文字、页码/符号、框位置和字段归属。没有人工确认就保持pending。后续独立Linux发布包可以准备资产清单，但不能把打包当成质量问题已经解决。

## E63首轮已完成，无需重跑

三个OCR和三个Extract均成功，最终health通过；0075/0093/0096返回13/56/73块。报告仍为completed_pending_manual_review，未转换为准确率通过。

原图仅做存在性检查：0093分类相关标记、页脚数字、性别栏内容可见而对应字段空；0096遮挡字段非空，多个编号归属也待核验。打码区域不得推测/恢复原值，不作为正常可识别正样本。统计和问题提示见本地manual_review.md。

## E69/E70跨平台回归已补齐，人工判定仍保留

已用当前Windows server.py作基准，对照同一0075/0093/0096图片字节、imageType=1与相同6项字段定义。两端共142框文字/顺序/坐标、完整Markdown及每图字段存在性和值均一致；0075三个NER缺失、0093空字段、0096遮挡字段非空/多编号归属现象在Windows也有对应输出。

因此本组样本未观察到WSL迁移引入的结果差异，但不将completed_pending_manual_review改成GT通过、不覆盖原人工表。报告在temp/business_windows_wsl_r1和r2，三图汇总见r2/comparison_review.md；不是新增独立精度集。后续可以推进部署交付，同时保留质量待办，无需为相同结论重复OCR/扣次。
