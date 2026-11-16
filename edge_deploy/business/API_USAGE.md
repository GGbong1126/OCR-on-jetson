# 本机业务接口联调

适用systemd托管的完整业务服务。E76为TensorRT OCR＋ORT CUDA版式/单元格＋PyTorch CUDA YOLO/NER/增强；请求/响应协议不变。当前启停及回退见[GPU部署说明](GPU_DEPLOYMENT.md)。

`/healthz`新增实际GPU调用/超范围CPU回退/错误计数，ocr_backend应为tensorrt_fp32_with_ort_fallback。计数是进程内诊断，重启归零，与授权余额无关。旧acceptance_business.py默认按CPU后端验收，不直接用它判断新GPU服务失败；当前可使用smoke_business.py、compare_business.py，以及带显式GPU backend参数的service_health.py。

E76还要求other_models_backend=cuda，models_execution列出5模块的实际/预热调用、错误及ORT预热节点Provider计数。启动检查同时传`--backend tensorrt_fp32_with_ort_fallback --other-backend cuda`；GPU模型故障返回503，不把被原代码捕获的异常伪装成部分成功。部分ORT节点及超profile OCR仍可CPU，详见日志。

## 地址与调用顺序

当前WSL与同一台Windows电脑均可访问 `http://127.0.0.1:5051`（Windows健康访问已实测）。仅限本机，不给其他电脑使用此地址；未配置局域网/公网认证、TLS或浏览器跨域策略。Python/curl调用成功不代表跨域网页已经联调通过。

1. `GET /healthz`：模型初始化状态，不扣次，不证明识别准确率。
2. `POST /OCR`：单张图片Base64 → 原图尺寸、文本块/坐标、Markdown。
3. `POST /Extract`：上一步返回的`msg`对象＋字段定义 → 结构化字段。正常完成NER时可能扣一次；不自动重试这个请求。

Content-Type为`application/json`。当前接口的成功判断需要同时满足HTTP200、`isSuccess == true`、业务`status == "200"`；业务参数错误可能仍是HTTP200，不可只看HTTP状态。服务层超限/网络错误也可能返回其他HTTP状态或非JSON，调用方需处理。

## OCR请求

```json
{
  "img": "图片文件的纯Base64字符串",
  "imageType": "1"
}
```

上面img是示意，不是可执行测试图片。实际用`base64.b64encode(image_bytes).decode("ascii")`生成；禁止data:image前缀、换行或空白。imageType必须传字符串："1"/"2"进入增强分支，"3"跳过增强；当前1和2共用实现。

成功响应的`msg`含`total_pages`和`pages`，每页含`width`/`height`/`markdown`/`blocks`；每块含`text`和`x1,y1,x2,y2`，坐标为原图像素。Markdown固定前三行用于标题、分类号、页码，可以为空；后续正文用于Extract，**不要删除空行或自行重排前三行**。

## Extract请求

使用OCR响应的`msg`对象，不是整个响应外层，也不是只传blocks。例如客户端在内存中构造：

```python
payload = {
    "ocrResult": ocr_response["msg"],
    "topFields": [
        {"name": "title", "type": "String", "description": "title"}
    ],
    "nerFields": [
        {"name": "姓名", "description": "姓名"}
    ],
}
```

两个字段数组均须提供，可其中一个为空，不能同时为空。topFields中name/description非空，type为String/Number/Boolean；不要假定服务会无条件返回所有自定义字段。

此示例成功响应的`msg`顶层含title和NER；title可能为空、NER可能是空字典。两个顶层键不等于两个已识别实体。NER字段还可能使用现有规则兜底；有值也不等于与人工标注相符。

## 现成的隐私友好客户端

客户端仅依赖Python标准库，仅打印数量/时延，不打印正文或字段值。不传--image时，只检查健康及空参数拒绝，不发有效OCR/Extract。

Windows PowerShell（本机已有Python路径）：

```powershell
& 'D:\Anaproject\envs\env1\python.exe' `
  '/path/to/OCR-on-jetson\edge_deploy\business\smoke_business.py' `
  --image '/path/to/OCR-on-jetson\test\006.jpg' `
  --image-type 1 --extract
```

WSL：

```bash
"$HOME/projects/autoocr-business/.venv/bin/python" \
  /mnt/d/docuyyy/autoocr/edge_deploy/business/smoke_business.py \
  --image /mnt/d/docuyyy/autoocr/test/006.jpg \
  --image-type 1 --extract
```

这是两种环境的**任选其一**示例，不需要重复执行。每次加--extract均可能扣次；用户已持续允许本项目必要验证扣次，但不应因超时自动重放。请求可能已执行、客户端却没收到结果，当前没有幂等键/去重协议。

## 当前限制与验收方式

- 单worker串行执行，CPU OCR样例此前约82～86秒；先按顺序请求，不并发压测。健康查询在忙时也会排队。
- 客户端默认请求超时240秒，服务OCR业务阈值180秒（完成推理后检查，非硬中断）；这不是性能SLA。
- HTTP请求体上限24MiB，包含Base64膨胀后的JSON；现有smoke客户端限制原图16MiB。尚未验证所有格式、大尺寸图片和资源边界。
- 用有权处理的代表性业务图片，人工核对标题/分类号/页码、关键字段、漏字及原图坐标。接口成功和字段非空仅是功能证据，不是质量验收。
- 原服务日志可能含正文，在本机核对，不原样上传或入Git。Windows开机自动启动WSL、长期运行、独立Linux发布包和GPU后端接入均属后续工作。

少量多图验收入口见[ACCEPTANCE.md](ACCEPTANCE.md)：串行请求、失败停止、不自动重试扣次，仅保存技术统计与空白人工清单。
