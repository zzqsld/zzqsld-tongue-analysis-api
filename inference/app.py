#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Azure App Service 版舌诊 API（ONNX Runtime + Flask）。"""

from __future__ import annotations

import io
import os
import json
import platform
import sys
import time
import base64
import binascii
import logging
from functools import wraps
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

from auth_utils import NonceCache, RateLimiter, verify_signature

if TYPE_CHECKING:
    import numpy as np

LOGGER = logging.getLogger("tongue-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ==================== 基础配置 ====================
MODEL_PATH = os.getenv("MODEL_PATH", "models/best.onnx")
INPUT_SIZE = int(os.getenv("INPUT_SIZE", "640"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.20"))
NMS_THRESHOLD = float(os.getenv("NMS_THRESHOLD", "0.50"))
# 结果过滤阈值：低于该置信度的检测框（多为文字/背景误检）不进入体质评分与响应
REPORT_CONF_THRESHOLD = float(os.getenv("REPORT_CONF_THRESHOLD", "0.30"))
MAX_CONTENT_MB = int(os.getenv("MAX_CONTENT_MB", "6"))
APP_REV = os.getenv("APP_REV", "local-dev")
ENABLE_COLOR_CORRECTION = os.getenv("ENABLE_COLOR_CORRECTION", "true").strip().lower() in {"1", "true", "yes", "on"}

# ==================== 鉴权配置 ====================
# API_KEYS_JSON: JSON 字符串，格式 {"<api_key>": "<secret>", ...}
# 例如：API_KEYS_JSON='{"webapp-001": "sk-live-xxxxxxxx"}'
API_KEYS: Dict[str, str] = {}
try:
    API_KEYS = json.loads(os.getenv("API_KEYS_JSON", "{}"))
except json.JSONDecodeError:
    LOGGER.error("API_KEYS_JSON 格式错误，鉴权将拒绝所有请求")
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
TIMESTAMP_WINDOW = int(os.getenv("TIMESTAMP_WINDOW", "300"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
NONCE_CACHE = NonceCache(ttl=TIMESTAMP_WINDOW)
RATE_LIMITER = RateLimiter(per_minute=RATE_LIMIT_PER_MINUTE)

# CORS 允许的源，逗号分隔；为空则允许全部
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

try:
    from tongue_preprocess import normalize_tongue_frame
    COLOR_PREPROCESS_READY = True
except Exception:
    normalize_tongue_frame = None
    COLOR_PREPROCESS_READY = False


# 19 类（修复版数据集 /mnt/workspace/v8data/shezhenv3-coco，benchmark yolo12s 模型）
# 相比旧 21 类：移除了空类 脾胃凸 / 心肺凸；修正命名 botaishe=剥苔（非薄白苔）、huataishe=滑苔（非花苔舌）
FEATURE_NAMES: List[str] = [
    "健康舌", "剥苔", "红舌", "紫舌", "胖大舌", "瘦舌",
    "红点舌", "裂纹舌", "齿痕舌", "白苔舌", "黄苔舌", "黑苔舌",
    "滑苔", "肾区凹", "肾区凸", "肝胆凹", "肝胆凸",
    "脾胃凹", "心肺凹",
]

CONSTITUTIONS: List[str] = [
    "平和质", "气虚质", "阳虚质", "阴虚质",
    "痰湿质", "湿热质", "血瘀质", "气郁质", "特禀质",
]

TONGUE_FEATURE_MAP: Dict[int, Dict[str, Dict[str, float]]] = {
    0:  {"weights": {"平和质": 1.00}},
    1:  {"weights": {"平和质": 0.50, "气虚质": 0.20, "阳虚质": 0.15, "痰湿质": 0.15}},
    2:  {"weights": {"阴虚质": 0.40, "湿热质": 0.35, "气郁质": 0.25}},
    3:  {"weights": {"血瘀质": 0.60, "阳虚质": 0.20, "湿热质": 0.10, "气郁质": 0.10}},
    4:  {"weights": {"阳虚质": 0.40, "气虚质": 0.30, "痰湿质": 0.30}},
    5:  {"weights": {"阴虚质": 0.60, "血瘀质": 0.20, "气虚质": 0.20}},
    6:  {"weights": {"湿热质": 0.45, "阴虚质": 0.30, "血瘀质": 0.25}},
    7:  {"weights": {"阴虚质": 0.50, "气虚质": 0.20, "阳虚质": 0.15, "平和质": 0.15}},
    8:  {"weights": {"气虚质": 0.40, "阳虚质": 0.25, "痰湿质": 0.25, "湿热质": 0.10}},
    9:  {"weights": {"阳虚质": 0.30, "气虚质": 0.25, "痰湿质": 0.25, "平和质": 0.20}},
    10: {"weights": {"湿热质": 0.50, "阴虚质": 0.25, "气郁质": 0.25}},
    11: {"weights": {"阳虚质": 0.50, "湿热质": 0.30, "阴虚质": 0.20}},
    12: {"weights": {"阴虚质": 0.45, "气虚质": 0.30, "平和质": 0.25}},
    13: {"weights": {"阳虚质": 0.50, "阴虚质": 0.30, "气虚质": 0.20}},
    14: {"weights": {"痰湿质": 0.40, "湿热质": 0.35, "阳虚质": 0.25}},
    15: {"weights": {"气虚质": 0.40, "阴虚质": 0.30, "阳虚质": 0.30}},
    16: {"weights": {"气郁质": 0.45, "湿热质": 0.30, "血瘀质": 0.25}},
    17: {"weights": {"气虚质": 0.50, "阳虚质": 0.30, "阴虚质": 0.20}},
    18: {"weights": {"气虚质": 0.45, "阳虚质": 0.35, "阴虚质": 0.20}},
}


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024
if ALLOWED_ORIGINS:
    CORS(app, origins=ALLOWED_ORIGINS)
else:
    CORS(app)


def require_auth(view_func):
    """API Key + HMAC-SHA256 签名 + 时间戳防重放 + 限流 装饰器。"""

    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not AUTH_ENABLED:
            return view_func(*args, **kwargs)

        api_key = request.headers.get("X-Api-Key", "")
        timestamp = request.headers.get("X-Timestamp", "")
        nonce = request.headers.get("X-Nonce", "")
        signature = request.headers.get("X-Signature", "")

        if not api_key or api_key not in API_KEYS:
            return jsonify({"error": "unauthorized: invalid api key"}), 401
        if not RATE_LIMITER.allow(api_key):
            return jsonify({"error": "rate limit exceeded"}), 429

        body = request.get_data() or b""
        ok, reason = verify_signature(
            api_key=api_key,
            secret=API_KEYS[api_key],
            method=request.method,
            path=request.path,
            timestamp=timestamp,
            nonce=nonce,
            provided_signature=signature,
            body=body,
            window=TIMESTAMP_WINDOW,
        )
        if not ok:
            return jsonify({"error": f"unauthorized: {reason}"}), 401
        if NONCE_CACHE.seen(nonce):
            return jsonify({"error": "unauthorized: replay detected"}), 401

        return view_func(*args, **kwargs)

    return wrapper


NP = None


def get_np():
    global NP
    if NP is None:
        try:
            import numpy as np
            NP = np
        except Exception as exc:
            raise RuntimeError(f"numpy 导入失败: {exc}") from exc
    return NP


def build_session(model_path: str) -> Any:
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError(f"onnxruntime 导入失败: {exc}") from exc

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型不存在: {model_path}")
    providers = ["CPUExecutionProvider"]
    # 推理性能优化：开启全部图优化；线程数默认用满可用核（B1 单核也无副作用）
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.enable_mem_pattern = True
    sess_opts.enable_cpu_mem_arena = True
    return ort.InferenceSession(model_path, providers=providers, sess_options=sess_opts)


MODEL_LOAD_ERROR = None
SESSION = None
INPUT_NAME = ""


def ensure_session() -> None:
    """Try to build ONNX session when missing; keep the last error for diagnostics."""
    global SESSION, INPUT_NAME, MODEL_LOAD_ERROR
    if SESSION is not None:
        return
    try:
        SESSION = build_session(MODEL_PATH)
        INPUT_NAME = SESSION.get_inputs()[0].name
        MODEL_LOAD_ERROR = None
    except Exception as exc:
        SESSION = None
        INPUT_NAME = ""
        MODEL_LOAD_ERROR = str(exc)


def probe_runtime() -> Dict[str, Any]:
    """Collect lightweight runtime diagnostics for quick cloud troubleshooting."""
    diag: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "model_path": MODEL_PATH,
        "model_abs_path": os.path.abspath(MODEL_PATH),
    }

    model_abs = diag["model_abs_path"]
    exists = os.path.exists(model_abs)
    diag["model_exists"] = exists
    diag["model_size_bytes"] = os.path.getsize(model_abs) if exists else 0

    try:
        import numpy as np
        diag["numpy_version"] = np.__version__
        diag["numpy_ok"] = True
    except Exception as exc:
        diag["numpy_version"] = None
        diag["numpy_ok"] = False
        diag["numpy_error"] = str(exc)

    try:
        import onnxruntime as ort
        diag["onnxruntime_version"] = ort.__version__
        diag["onnxruntime_ok"] = True
        diag["onnx_providers"] = ort.get_available_providers()
    except Exception as exc:
        diag["onnxruntime_version"] = None
        diag["onnxruntime_ok"] = False
        diag["onnxruntime_error"] = str(exc)

    diag["color_preprocess_enabled"] = ENABLE_COLOR_CORRECTION
    diag["color_preprocess_ready"] = COLOR_PREPROCESS_READY

    return diag


ensure_session()


def letterbox_resize(img: Image.Image, size: int = 640, color_level: int = 2) -> Tuple[np.ndarray, float, int, int, int, int, np.ndarray]:
    np = get_np()
    rgb = img.convert("RGB")
    src = np.array(rgb)

    if ENABLE_COLOR_CORRECTION and COLOR_PREPROCESS_READY and normalize_tongue_frame is not None:
        # 预处理模块使用 BGR，模型输入前再转回 RGB。
        bgr = src[:, :, ::-1].copy()
        bgr = normalize_tongue_frame(bgr, color_level=color_level)
        src = bgr[:, :, ::-1]

    h, w = src.shape[:2]
    scale = min(size / max(h, 1), size / max(w, 1))
    nh, nw = int(round(h * scale)), int(round(w * scale))

    resized = np.array(Image.fromarray(src).resize((nw, nh), Image.BILINEAR))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized

    blob = canvas.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None, ...]  # NCHW
    return blob, scale, left, top, w, h, src


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
    np = get_np()
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter + 1e-9
        iou = inter / union
        remain = np.where(iou <= threshold)[0]
        order = order[remain + 1]
    return keep


def parse_output(raw: np.ndarray, scale: float, left: int, top: int, src_w: int, src_h: int) -> List[dict]:
    """向量化解析 YOLO 输出：8400 行无需逐行 Python 循环。"""
    np = get_np()
    out = raw
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]
    if out.ndim == 2 and out.shape[0] == 4 + len(FEATURE_NAMES):
        out = out.T
    if out.ndim != 2 or out.shape[1] < 4 + len(FEATURE_NAMES):
        return []

    cls_scores = out[:, 4:4 + len(FEATURE_NAMES)]
    class_ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), class_ids]
    mask = scores >= CONF_THRESHOLD
    if not np.any(mask):
        return []

    boxes_xywh = out[mask, :4].astype(np.float32)
    scores = scores[mask]
    class_ids = class_ids[mask]

    # 归一化坐标映射回输入尺寸
    if float(np.max(boxes_xywh)) <= 2.0:
        boxes_xywh *= INPUT_SIZE

    cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = (cx - bw / 2.0 - left) / max(scale, 1e-9)
    y1 = (cy - bh / 2.0 - top) / max(scale, 1e-9)
    x2 = (cx + bw / 2.0 - left) / max(scale, 1e-9)
    y2 = (cy + bh / 2.0 - top) / max(scale, 1e-9)
    x1 = np.clip(x1, 0.0, src_w - 1.0)
    y1 = np.clip(y1, 0.0, src_h - 1.0)
    x2 = np.clip(x2, 0.0, src_w - 1.0)
    y2 = np.clip(y2, 0.0, src_h - 1.0)
    valid = (x2 > x1) & (y2 > y1)
    if not np.any(valid):
        return []

    boxes_np = np.stack([x1[valid], y1[valid], x2[valid], y2[valid]], axis=1)
    scores = scores[valid]
    class_ids = class_ids[valid]

    keep = nms(boxes_np, scores, NMS_THRESHOLD)
    return [
        {
            "class": int(class_ids[i]),
            "name": FEATURE_NAMES[int(class_ids[i])],
            "confidence": round(float(scores[i]), 4),
            "box": [int(round(v)) for v in boxes_np[i]],
        }
        for i in keep
    ]


# 苔类类别（用于苔覆盖率统计）
COATING_CLASS_IDS = {1, 9, 10, 11, 12}  # 薄白苔 / 白苔舌 / 黄苔舌 / 黑苔舌 / 花苔舌


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0~255) -> CIE L*a*b*（D65），纯 numpy 实现。"""
    np = get_np()
    arr = np.asarray(rgb, dtype=np.float64) / 255.0
    # 逆伽马校正：sRGB -> 线性 RGB
    linear = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    # 线性 RGB -> XYZ（D65）
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = linear @ matrix.T
    # XYZ -> Lab，D65 白点
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    delta = 6.0 / 29.0
    f = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3.0 * delta ** 2) + 4.0 / 29.0)
    lightness = 116.0 * f[..., 1] - 16.0
    a_chan = 500.0 * (f[..., 0] - f[..., 1])
    b_chan = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([lightness, a_chan, b_chan], axis=-1)


def _boxes_union_mask(shape: Tuple[int, int], detections: List[dict], class_filter=None) -> "np.ndarray":
    """生成检测框并集区域的布尔掩码（坐标为原图像素，自动裁剪到图内）。"""
    np = get_np()
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for det in detections:
        if class_filter is not None and int(det["class"]) not in class_filter:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in det["box"]]
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return mask


def compute_color_stats(enhanced_rgb: "np.ndarray", detections: List[dict]):
    """舌色统计：检测框并集（无检出时回退原图中心 72% 区域）的 RGB/Lab 均值。

    在增强后的图像（与送入模型推理的同一张图）上计算。任何异常返回 None。
    """
    try:
        np = get_np()
        h, w = enhanced_rgb.shape[:2]
        if detections:
            mask = _boxes_union_mask((h, w), detections)
            region = "detections"
        else:
            # 回退：原图中心 72% 区域
            cw = max(1, int(round(w * 0.72)))
            ch = max(1, int(round(h * 0.72)))
            x1 = (w - cw) // 2
            y1 = (h - ch) // 2
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y1 + ch, x1:x1 + cw] = True
            region = "center"

        pixels = enhanced_rgb[mask].astype(np.float64)
        sampled = int(pixels.shape[0])
        if sampled == 0:
            return None
        mean_rgb = pixels.mean(axis=0)
        mean_lab = _srgb_to_lab(pixels).mean(axis=0)
        return {
            "mean_rgb": [round(float(v), 1) for v in mean_rgb],
            "mean_lab": [round(float(v), 1) for v in mean_lab],
            "sampled_pixels": sampled,
            "region": region,
        }
    except Exception as exc:
        print(f"[warn] color_stats 计算失败: {exc}", flush=True)
        return None


def compute_coating_coverage(shape: Tuple[int, int], detections: List[dict]):
    """苔覆盖率：苔类检测框并集面积 / 全部检测框并集面积；无检出返回 None。"""
    try:
        if not detections:
            return None
        all_mask = _boxes_union_mask(shape, detections)
        total_area = int(all_mask.sum())
        if total_area <= 0:
            return None
        coating_mask = _boxes_union_mask(shape, detections, class_filter=COATING_CLASS_IDS)
        coating_area = int(coating_mask.sum())
        coverage = min(1.0, coating_area / float(total_area))
        return round(coverage, 3)
    except Exception as exc:
        print(f"[warn] coating_coverage 计算失败: {exc}", flush=True)
        return None


def score_constitution(detections: List[dict]) -> Tuple[str, Dict[str, float]]:
    scores = {c: 0.0 for c in CONSTITUTIONS}
    for det in detections:
        cid = int(det["class"])
        conf = float(det["confidence"])
        if cid not in TONGUE_FEATURE_MAP:
            continue
        for const, weight in TONGUE_FEATURE_MAP[cid]["weights"].items():
            scores[const] += conf * weight

    total = sum(scores.values())
    if total <= 0:
        return "平和质", scores
    for k in scores:
        scores[k] = round(scores[k] / total * 100.0, 1)
    best = max(scores.items(), key=lambda item: item[1])[0]
    return best, scores


def run_infer(image_bytes: bytes, color_level: int = 2) -> Tuple[List[dict], float, "np.ndarray"]:
    ensure_session()
    if SESSION is None:
        raise RuntimeError(f"模型未就绪: {MODEL_LOAD_ERROR or 'unknown error'}")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except Exception as exc:
        raise ValueError("上传内容不是有效图片") from exc

    blob, scale, left, top, src_w, src_h, enhanced_src = letterbox_resize(image, INPUT_SIZE, color_level=color_level)

    t0 = time.perf_counter()
    outputs = SESSION.run(None, {INPUT_NAME: blob})
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if not outputs:
        return [], elapsed_ms, enhanced_src
    dets = parse_output(outputs[0], scale, left, top, src_w, src_h)
    return dets, elapsed_ms, enhanced_src


def _parse_color_level(raw: Any) -> int:
    try:
        level = int(raw)
    except (TypeError, ValueError):
        return 2
    return level if level in (1, 2, 3) else 2


def _do_predict(image_bytes: bytes, color_level: int):
    """两个预测端点共用的执行与响应逻辑。"""
    try:
        detections, infer_ms, enhanced_src = run_infer(image_bytes, color_level=color_level)
        # 过滤低置信度框：避免把图片中的文字/背景误检为舌象特征
        detections = [d for d in detections if d["confidence"] >= REPORT_CONF_THRESHOLD]
        best, scores = score_constitution(detections)
        color_stats = compute_color_stats(enhanced_src, detections)
        coating_coverage = compute_coating_coverage(enhanced_src.shape[:2], detections)
        return jsonify(
            {
                "detected_count": len(detections),
                "detections": detections,
                "constitution": {"primary": best, "scores": scores},
                "inference_ms": round(infer_ms, 2),
                "color_level": color_level,
                "color_stats": color_stats,
                "coating_coverage": coating_coverage,
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        LOGGER.exception("推理失败")
        return jsonify({"error": "inference failed"}), 500


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "service": "tongue-api",
            "status": "running",
            "model_path": MODEL_PATH,
            "input_size": INPUT_SIZE,
            "auth_enabled": AUTH_ENABLED,
            "endpoints": ["/health", "/labels", "/predict", "/predict_base64"],
        }
    )


@app.route("/health", methods=["GET"])
def health():
    ensure_session()
    runtime = probe_runtime()
    return jsonify(
        {
            "status": "ok" if SESSION is not None else "degraded",
            "model_loaded": SESSION is not None,
            "model_path": MODEL_PATH,
            "model_error": MODEL_LOAD_ERROR,
            "rev": APP_REV,
            "runtime": runtime,
        }
    )


@app.route("/labels", methods=["GET"])
def labels():
    return jsonify({"labels": FEATURE_NAMES, "constitutions": CONSTITUTIONS})


@app.route("/predict", methods=["POST"])
@require_auth
def predict():
    color_level = _parse_color_level(request.form.get("color_level", "2"))

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty file name"}), 400

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Empty file content"}), 400

    return _do_predict(image_bytes, color_level)


@app.route("/predict_base64", methods=["POST"])
@require_auth
def predict_base64():
    payload = request.get_json(silent=True) or {}
    image_base64 = payload.get("image_base64", "")
    color_level = _parse_color_level(payload.get("color_level", 2))

    if not image_base64:
        return jsonify({"error": "Missing image_base64"}), 400

    # 兼容 data URL：data:image/jpeg;base64,xxxx
    if isinstance(image_base64, str) and "," in image_base64 and image_base64.strip().startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError):
        return jsonify({"error": "Invalid base64 payload"}), 400

    if not image_bytes:
        return jsonify({"error": "Empty image bytes"}), 400

    return _do_predict(image_bytes, color_level)


# 启动预载模型：gunicorn worker 启动时就完成加载与图优化，
# 避免第一个请求承担数十 MB 模型加载的冷启动开销（可用 PRELOAD_MODEL=false 关闭）
if os.getenv("PRELOAD_MODEL", "true").strip().lower() in {"1", "true", "yes", "on"}:
    ensure_session()


if __name__ == "__main__":
    # 本地调试入口；Azure 上由 gunicorn 启动。
    app.run(host="0.0.0.0", port=8000, debug=False)
