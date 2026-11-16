# 可回滚的宽输入识别适配（E124）

本适配复用E123已验收的FP32宽识别引擎，不是新增量化。不覆盖`jetson-r2`，不改原图片预处理、裁剪、CTC、版式、字段、NER或授权规则。默认关闭；板端验收结果和限制以`TECHNICAL_LOG.md`的E124～E126条目为准。

E126持续运行已完成（2026-10-06）：45张既有图片×普通/增强，90次预热后连续完成三轮270次业务，预热后观察1023.474秒（约17.06分钟）；360/360结果和模型路由与既有基准一致。19564次页内GPU识别含120次宽引擎调用，无CPU回退/错误。逐页全局NvMap最大增长0.3125MiB、进程RSS39.19MiB、最低可用内存5986.11MiB、SwapFree无下降，冻结门槛通过；另有200个正式期5秒采样。进程退出后NvMap相对空闲只多0.3125MiB，不能把关闭会话前后和退出进程后混作同一个释放时点。原包不变、HTTP关闭；不是并发/小时级、GT或真实>2304输入驻留验收，没有新增性能声明。证据与归档打印修补详情见技术日志E126。

2026-10-06验收已完成：182/182离线页面回归、开/关新进程回滚、四个边界路由全部通过；预热后104页约6.01分钟业务快照NvMap增长0、RSS最大增长21.625MiB。68项本机/37项板端相关测试及技术归档复核通过。状态为`offline_adapter_validated_not_http_enabled`，**没有启用HTTP，也不是小时级长稳或代表性超长文字精度验证**。

## 包与选择规则

E125扩大回归（2026-10-06）：从既有testpic的673图中排除旧13图，预先固定另外32图，普通/增强64组原版与候选配对、128次完整业务全部一致。3745个输入形状/dtype/哈希及CTC序列逐次一致；3719次原GPU输出字节不变，26次原CPU识别由宽GPU接管，候选页内CPU为0。两侧这26组原始输出存在字节差异，未做新数值容差验收，不能将CTC相同解释为浮点输出相等。

累计覆盖45个不同图片文件，不是673图全部处理，也不保证文档独立。新样本识别宽32～1970，没有2200～2304或>2304真实输入；原13图最大宽2081的证据仍保留。该轮为单次固定顺序、含逐调用哈希探针的覆盖验证，不是新性能/GT/长稳结论。77项本机/33项板端相关测试、45成员归档复核通过，包内容未变，HTTP仍关闭。后续有限持续运行结果见上方E126；真实极宽文字与并发资源仍未验收。

板端包位于`/home/xchq/work/autoocr/overlays/rec-wide-e124-r1`，包含：

- `serve_rec_wide.py`：独立业务工厂，复用原r2的业务/模型加载。
- `rec_wide_adapter.py`：双引擎路由、校验、预热、关闭处理。
- `config.json`：固定原配置/模型/适配源码的哈希及profile。
- `rec_wide_fp32.engine`：E123引擎的独立完整副本，不依赖实验目录软链接。

| 输入（原始NCHW张量） | 开启适配 | 关闭适配 |
|---|---|---|
| N=1,C=3,H=48，W=32～1536 | 原FP32 GPU引擎 | 原FP32 GPU引擎 |
| N=1,C=3,H=48，W=1537～2304 | 新FP32 GPU引擎，opt=1696 | 原ORT CPU回退 |
| 超出两引擎范围 | 原ORT CPU策略，不填充/缩放/截断 | 原ORT CPU策略 |
| 推理异常、非有限结果、后端已关闭 | 拒绝继续，不借CPU掩盖错误 | 保留原错误策略 |

“范围外回退”不保证任意错误形状都可被原ONNX接受；CPU执行失败仍会拒绝结果。当前预设是batch1，不代表并发或batch部署验收。

使用新工厂时，设置`AUTOOCR_REC_WIDE_CONFIG`为该包`config.json`才开启适配；**未设置变量时走原配置，空字符串或不存在/校验失败的配置直接报错**，不会静默降级。启动前校验原r2三份关键适配代码、原部署及新包；新引擎实际接口/profile必须匹配，min/opt/max三个零输入全部预热后才接入。

哈希用于防止意外漂移，不是签名分发或对具有目录写权限的攻击者的安全边界。

## 不启动HTTP的加载检查

以下只加载并执行合成模型预热，不处理私人图片，不进入激活、授权检查或扣次。报告已存在时拒绝覆盖；每次使用新目录。

```bash
BASE=/home/xchq/work/autoocr
OVERLAY="$BASE/overlays/rec-wide-e124-r1"
export PYTHONDONTWRITEBYTECODE=1
export AUTOOCR_SOURCE_ROOT="$BASE/releases/jetson-r2"
export AUTOOCR_RUNTIME_DIR="$BASE/runtime/rec-wide-offline"
export AUTOOCR_LOG_FILE=/dev/null
export LD_LIBRARY_PATH="$BASE/lib/cuda10.2:/usr/local/cuda-10.2/targets/aarch64-linux/lib"
export AUTOOCR_REC_WIDE_CONFIG="$OVERLAY/config.json"
mkdir -p "$BASE/reports"
OUT=$(mktemp -d "$BASE/reports/rec-wide-probe.XXXXXX")
"$BASE/envs/business-py310/bin/python" "$OVERLAY/serve_rec_wide.py" \
  --probe-models "$OUT/enabled.json"

# 新进程回到原配置；不是运行中热切换。
unset AUTOOCR_REC_WIDE_CONFIG
"$BASE/envs/business-py310/bin/python" "$OVERLAY/serve_rec_wide.py" \
  --probe-models "$OUT/disabled.json"
```

不能在一个已加载模型的进程里修改环境变量就声称完成回滚。应正常停止该进程，去掉变量，然后启动新进程；原`serve_jetson.py`入口也完全不受此可选包影响。不删除旧引擎/新包以实现回滚。

## 业务入口边界

`load_business()`供离线业务调用使用，与原`--probe-models`一样不创建HTTP。正常返回`server,pipeline,models,config`，需要在`finally`中调用`rec_wide_adapter.close_models(models)`释放所有GPU会话。

`create_app()`提供未来服务入口：关闭适配时直接委托原工厂；开启时仍依次执行原硬件许可校验、模型加载、原计数初始化，并保留请求前/后后端健康拒绝规则。**这不是免激活HTTP入口，也不在本轮启动HTTP、修改systemd或完成真实授权联调。**入口的调用顺序及故障拒绝先用替身单元测试验证，不能写成已完成真实HTTP验收。

E124离线计划为三个全新进程：关闭26页→开启26页预热＋4×26页连续处理→关闭26页，合计182页。使用既有13图的普通/增强路径，逐页核对原业务签名和各模型调用数；开启进程另验证1536/1537/2304/2305四个零输入，每个两次，CPU原ONNX同输入对照。

资源指标是预热后逐页/周期快照，冻结门槛为NvMap增长≤64MiB、RSS增长≤128MiB、可用内存≥1024MiB、SwapFree下降≤16MiB。不是瞬时峰值，也不是小时级压测；RSS和NvMap有重叠，不能相加称总显存。边界零输入并非2304附近代表性文字，多图片GT和真实服务并发验收仍需另做。
