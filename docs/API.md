# API 适配文档（推理服务调用指南）

> 本文档面向两类读者：① 自部署本推理服务后需要对接调用的开发者；② 参考作者演示实例接口格式的研究者。
> 自部署时，密钥对由你自己通过环境变量 `API_KEYS_JSON` 配置（见仓库 README 与 deploy 脚本），下文示例中的 `webapp-001` / `你的密钥` 均为占位符。

## 1. 接口概览

- 服务基址: `https://<你的应用>.azurewebsites.net`（作者演示实例见 README，不保证长期可用）
- 数据格式: JSON（预测接口支持 multipart 与 base64 两种上传）
- 鉴权: API Key + HMAC-SHA256 签名 + 时间戳防重放 + 每 Key 限流
- 最大上传体积: 6MB（由服务端 MAX_CONTENT_MB 控制）

可用接口:
- GET /health（公开）
- GET /labels（公开）
- POST /predict（需鉴权）
- POST /predict_base64（需鉴权）

---

## 1.1 鉴权说明

为防止接口被第三方盗刷，预测接口（/predict、/predict_base64）启用了签名校验。

### 请求头

| 请求头 | 说明 |
|--------|------|
| X-Api-Key | API 密钥 ID（服务端配置） |
| X-Timestamp | Unix 秒级时间戳 |
| X-Nonce | 随机字符串（防重放，每次请求必须不同） |
| X-Signature | HMAC-SHA256 签名（hex） |

### 签名算法

```text
string_to_sign = METHOD + "\n" + PATH + "\n" + TIMESTAMP + "\n" + NONCE + "\n" + SHA256_HEX(body)
signature      = HMAC_SHA256(secret, string_to_sign) 的 hex 字符串
```

- METHOD：大写，如 `POST`
- PATH：如 `/predict_base64`
- body：原始请求体字节（base64 JSON 或 multipart 整体）

### 服务端校验规则

1. API Key 必须存在于服务端 `API_KEYS_JSON` 配置中
2. 时间戳与服务器时间偏差不能超过 300 秒（防旧请求重放）
3. 同一 Nonce 在 300 秒内只能使用一次（防重放）
4. 每个 API Key 默认每分钟最多 60 次调用（`RATE_LIMIT_PER_MINUTE` 可调）

### 错误码

- 401 unauthorized: invalid api key / timestamp expired / signature mismatch / replay detected
- 429 rate limit exceeded

> 注意：浏览器端的 secret 理论上可被查看源码提取，签名主要防止"只知道网址"的盗刷。
> 若需更高安全级别，建议在自己的后端做一次转发（浏览器 → 你的后端 → Azure），secret 只放在服务端。

---

## 2. 健康检查

### GET /health

用途:
- 检查服务是否可用
- 检查模型是否成功加载
- 查看运行时信息（版本、依赖、颜色校正开关）

示例返回:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "models/best.onnx",
  "model_error": null,
  "rev": "2026-07-22-hotfix-4",
  "runtime": {
    "python_version": "3.12.13",
    "onnxruntime_ok": true,
    "numpy_ok": true,
    "color_preprocess_enabled": true,
    "color_preprocess_ready": true,
    "model_exists": true,
    "model_size_bytes": 12368980
  }
}
```

---

## 3. 标签元数据

### GET /labels

用途:
- 获取舌象类别列表（21类）
- 获取体质类别列表（9类）

示例返回:

```json
{
  "labels": ["健康舌", "薄白苔", "红舌"],
  "constitutions": ["平和质", "气虚质", "阳虚质"]
}
```

---

## 4. 预测接口（文件上传）

### POST /predict

请求:
- Content-Type: multipart/form-data
- 鉴权请求头: X-Api-Key / X-Timestamp / X-Nonce / X-Signature
- 字段:
  - file: 图片文件（jpg/jpeg/png）

Python 示例（推荐，自动处理签名）:

```python
import requests
from auth_utils import make_auth_headers

API_KEY = "webapp-001"
SECRET = "你的密钥"
URL = "https://<你的应用>.azurewebsites.net/predict"

with open("tongue.jpg", "rb") as f:
    files = {"file": f}
    # multipart 的 body 是 requests 自动生成的，签名前需要构造一次
    req = requests.Request("POST", URL, files=files).prepare()
    headers = make_auth_headers(API_KEY, SECRET, "POST", "/predict", req.body)
    resp = requests.post(URL, data=req.body, headers={**headers, "Content-Type": req.headers["Content-Type"]}, timeout=30)
print(resp.json())
```

成功返回:

```json
{
  "detected_count": 1,
  "detections": [
    {
      "class": 9,
      "name": "白苔舌",
      "confidence": 0.9345,
      "box": [102, 75, 445, 420]
    }
  ],
  "constitution": {
    "primary": "阳虚质",
    "scores": {
      "平和质": 20.0,
      "气虚质": 25.0,
      "阳虚质": 30.0
    }
  },
  "inference_ms": 378.09
}
```

错误码:
- 400: No file uploaded / Empty file name / Empty file content
- 500: 服务内部错误（模型未就绪、推理异常等）

---

## 5. 预测接口（Base64，推荐给低代码/插件）

### POST /predict_base64

请求:
- Content-Type: application/json
- Body:

```json
{
  "image_base64": "..."
}
```

说明:
- 支持纯 base64 字符串
- 支持 Data URL 格式（例如 `data:image/jpeg;base64,xxxx`）
- 需携带鉴权请求头（见 1.1 节）

Python 示例:

```python
import base64, json, requests
from auth_utils import make_auth_headers

API_KEY = "webapp-001"
SECRET = "你的密钥"
URL = "https://<你的应用>.azurewebsites.net/predict_base64"

with open("tongue.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

body = json.dumps({"image_base64": b64}).encode()
headers = make_auth_headers(API_KEY, SECRET, "POST", "/predict_base64", body)
resp = requests.post(URL, data=body, headers={**headers, "Content-Type": "application/json"}, timeout=30)
print(resp.json())
```

错误码:
- 400: Missing image_base64 / Invalid base64 payload / Empty image bytes
- 401: 鉴权失败（见 1.1 节）
- 429: 触发限流
- 500: 推理异常

---

## 6. 结果字段说明

- detected_count: 检测目标数
- detections[].class: 类别ID（0~20）
- detections[].name: 类别中文名
- detections[].confidence: 置信度
- detections[].box: 边框像素坐标 [x1, y1, x2, y2]
- constitution.primary: 主体质
- constitution.scores: 各体质百分比
- inference_ms: 推理耗时（毫秒）

---

## 7. 秒哒接入方式（已检索官方文档）

检索依据:
- 百度智能云秒哒文档《自定义API或插件接入应用》
- 链接: https://cloud.baidu.com/doc/MIAODA/s/3mj9fhz2o

官方文档核心点:
- 一次性使用: 直接“API 对接外部系统”
- 可复用能力: 创建“自定义插件”，后续在多个应用中通过 @插件名 调用
- 提示词建议格式: 功能 + API地址 + 输入参数 + 输出字段

### 7.1 方案A: 秒哒中直接对接 API（最快）

在秒哒中给出如下需求（可直接复制）:

```text
创建一个舌诊分析页面。
调用接口: POST https://<你的应用>.azurewebsites.net/predict_base64
请求参数: image_base64(string)
返回字段: detected_count, detections[].name, detections[].confidence, detections[].box, constitution.primary, constitution.scores, inference_ms
页面需求:
1) 支持上传图片并转base64
2) 点击“开始分析”后请求接口
3) 展示主诊断体质、舌象特征列表、推理耗时
4) 错误时展示error字段
```

为什么推荐 /predict_base64:
- 低代码平台对 multipart 兼容性不一
- JSON + base64 更容易在可视化流程中编排

### 7.2 方案B: 秒哒自定义插件（可复用）

适用于你后续多个应用都要调用舌诊能力。

插件定义建议:
- 插件名: tongue_diagnose_api
- 方法: POST
- URL: https://<你的应用>.azurewebsites.net/predict_base64
- 请求体: { image_base64: string }
- 响应体: 与第5节一致

在秒哒应用里调用示例提示词:

```text
请使用 @tongue_diagnose_api 生成一个小程序页面：
上传舌象图片 -> 调用插件 -> 展示体质结论与舌象特征明细。
```

---

## 8. 小程序/前端调用示例

### 8.1 JavaScript fetch（Web/H5，含签名）

浏览器端用 Web Crypto API 计算 HMAC-SHA256：

```javascript
const API_KEY = "webapp-001";
const SECRET = "你的密钥";
const BASE = "https://<你的应用>.azurewebsites.net";

async function sha256Hex(text) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2, "0")).join("");
}

async function hmacSha256Hex(secret, text) {
  const key = await crypto.subtle.importKey(
    "raw", new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" }, false, ["sign"]
  );
  const sig = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(text));
  return [...new Uint8Array(sig)].map(b => b.toString(16).padStart(2, "0")).join("");
}

function randomNonce() {
  return [...crypto.getRandomValues(new Uint8Array(8))].map(b => b.toString(16).padStart(2, "0")).join("");
}

async function predictBase64(imageBase64) {
  const body = JSON.stringify({ image_base64: imageBase64 });
  const timestamp = Math.floor(Date.now() / 1000).toString();
  const nonce = randomNonce();
  const bodyHash = await sha256Hex(body);
  const stringToSign = ["POST", "/predict_base64", timestamp, nonce, bodyHash].join("\n");
  const signature = await hmacSha256Hex(SECRET, stringToSign);

  const resp = await fetch(BASE + "/predict_base64", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Api-Key": API_KEY,
      "X-Timestamp": timestamp,
      "X-Nonce": nonce,
      "X-Signature": signature,
    },
    body,
  });
  return await resp.json();
}
```

### 8.2 Python requests（含签名）

```python
import base64, json, requests
from auth_utils import make_auth_headers

API_KEY = "webapp-001"
SECRET = "你的密钥"
url = "https://<你的应用>.azurewebsites.net/predict_base64"

with open("tongue.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

body = json.dumps({"image_base64": b64}).encode()
headers = make_auth_headers(API_KEY, SECRET, "POST", "/predict_base64", body)
resp = requests.post(url, data=body, headers={**headers, "Content-Type": "application/json"}, timeout=30)
print(resp.json())
```

---

## 9. 对下一个程序员的实现建议

- 鉴权已实现（API Key + HMAC-SHA256 签名）:
  - 服务端通过环境变量 `API_KEYS_JSON` 配置密钥对，例如 `{"webapp-001": "sk-live-xxxx"}`
  - `AUTH_ENABLED=false` 可临时关闭鉴权（调试用）
  - 更高安全级别建议 Azure APIM / 自建后端转发
- 保持接口兼容:
  - /predict 给传统上传场景
  - /predict_base64 给秒哒/低代码/插件场景
- 秒哒/低代码平台接入注意:
  - 签名需要在请求时计算 HMAC-SHA256，低代码平台如果无法执行自定义代码，
    可让平台调用你自己写的一个轻量转发接口（转发层负责签名）
- 版本治理:
  - 每次发版更新 APP_REV
  - 用 /health 验证 rev 与 model_loaded
- 回归建议:
  - 使用 regress20.py 做发版后固定回归（需同步加上签名头）

---

## 10. 发版后验收清单

1. GET /health: status=ok, model_loaded=true
2. GET /health: rev 为本次发布号
3. GET /health: color_preprocess_enabled=true
4. POST /predict_base64: 带签名返回 detected_count 与 constitution.primary
5. 批量回归（需先配置密钥）:

```bash
# Windows PowerShell
$env:TONGUE_API_KEY="webapp-001"
$env:TONGUE_API_SECRET="你的密钥"
python azure_appservice/regress20.py

# Linux/macOS
export TONGUE_API_KEY="webapp-001"
export TONGUE_API_SECRET="你的密钥"
python azure_appservice/regress20.py
```

---

## 11. 直接给 AI 的前端生成提示词

下面这段可以直接复制给秒哒、Copilot、Cursor 或其他前端 AI 生成器，用来生成一个同时适配电脑和手机的舌诊前端。

```text
请帮我生成一个“中医舌诊前端系统”，要求是响应式 Web 应用，电脑和手机都能正常访问与使用。

核心目标：
1. 用户可以登录系统。
2. 用户可以上传舌象图片，调用已存在的舌诊插件/API，得到分析结果。
3. 用户可以查看历史记录、保存每次分析结果。
4. 页面在桌面端和手机端都要有良好的体验，手机优先但不牺牲桌面布局。

技术要求：
1. 使用现代前端技术栈，优先 React + TypeScript + Vite。
2. 界面必须响应式，支持桌面、平板、手机三种尺寸。
3. 需要清晰的导航、卡片式布局、表单和结果展示区。
4. 需要登录页、首页、分析页、历史记录页、个人中心页。
5. 需要考虑移动端触摸操作、单手操作、图片上传、结果卡片阅读体验。

接口对接：
1. 舌诊分析接口：POST https://<你的应用>.azurewebsites.net/predict_base64
2. 请求格式：JSON
3. 请求字段：image_base64
4. 返回字段至少展示：detected_count、detections[].name、detections[].confidence、detections[].box、constitution.primary、constitution.scores、inference_ms

登录与数据记录：
1. 设计一个简单的登录/注册流程。
2. 登录后要能保存用户的分析记录。
3. 每条记录至少包含：用户ID、时间、图片、分析结果、主体质、检测到的舌象特征、耗时。
4. 历史记录页支持分页、搜索、按时间筛选、按主体质筛选。
5. 如果后端接口未提供，先在前端代码中预留 API 接口层和 mock 数据层，方便后续接真实后端。

UI/UX 要求：
1. 首页要能快速开始分析。
2. 分析页要有上传区、拍照/相册入口、分析按钮、结果展示区。
3. 历史页要能清楚看到每次体质结论和图片缩略图。
4. 结果页要突出显示主诊断体质，并用列表展示舌象特征。
5. 整体风格要专业、清爽、适合医疗健康场景，不要花哨。

交付要求：
1. 直接输出可运行的前端项目代码结构。
2. 提供必要的组件、页面、路由、API 封装、状态管理、类型定义。
3. 代码要保证手机和电脑都能使用。
4. 如果需要，你可以同时生成一个后端接口契约说明，方便我后续对接登录和数据保存。

请直接开始生成项目，不要只写思路。
```

### 可选增强版提示词

如果你想让 AI 写得更像成品，可以再补一句：

```text
请优先做成一个 PWA 风格的 Web 前端，支持浏览器直接访问、移动端适配、桌面端大屏展示，并预留后续接入微信小程序或低代码平台的可能性。
```
