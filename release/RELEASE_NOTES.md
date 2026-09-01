## v2.0 复现权重包（12 模型 · 论文基准同口径复现 · 2026-09-01）

TCM-Tongue 论文（Gao & Jin, 2026）Table 2 全部 12 个基准模型的同口径复现权重与成绩表。
每个模型一个独立 zip（内含 `<模型名>/best.pt`），可按需单独下载；另附全量合集包与成绩 CSV。

**YOLO 系（Ultralytics 格式，Ultralytics ≥ 8.4 直接加载）**

| 附件 | 大小 | test mAP@0.5（论文原值） | SHA256 |
|---|---|---|---|
| `yolov7-tiny_best_v2.zip` | 10 MB | 41.30%（31.93%） | `4d955003…e707c` |
| `yolov7_best_v2.zip` | 66 MB | 41.00%（34.82%） | `2a2d2047…f65a` |
| `yolo12s_best_v2.zip` | 16 MB | 40.88%（35.23%） | `b13526d8…931f` |
| `yolov5l_best_v2.zip` | 94 MB | 40.00%（34.57%） | `c44c8669…7e47` |
| `yolov5m_best_v2.zip` | 44 MB | 39.60%（32.48%） | `9e632f33…1513` |
| `yolov8l_best_v2.zip` | 77 MB | 39.23%（34.95%） | `9db43673…a3e9` |
| `yolov8m_best_v2.zip` | 45 MB | 38.74%（34.77%） | `65f8333b…f745` |
| `yolov8s_best_v2.zip` | 19 MB | 36.75%（33.54%） | `33ed5b2c…4a85` |
| `yolo11s_best_v2.zip` | 16 MB | 36.54%（35.32%） | `e812429e…ce31` |
| `yolov5s_best_v2.zip` | 16 MB | 34.50%（31.83%） | `2b3abaff…4464` |

> 注：yolov7 / yolov7-tiny 为 WongKinYiu 官方仓库格式（非 Ultralytics），
> 加载需该仓库代码，且 torch≥2.6 需 `torch.load(..., weights_only=False)`。

**SSD 系（torchvision 格式，state_dict 含完整模型）**

| 附件 | 大小 | test mAP@0.5（论文原值） | SHA256 |
|---|---|---|---|
| `ssd300_vgg16_best_v2.zip` | 185 MB | 24.70%（28.86%） | `5f9b2ce5…0557` |
| `ssdlite320_mobilenet_v3_large_best_v2.zip` | 28 MB | 23.35%（23.20%） | `58d29394…fd6f` |

**成绩表与合集**

| 附件 | 说明 | SHA256 |
|---|---|---|
| `benchmark_results.csv` | YOLO 8 模型成绩（P/R/mAP/耗时） | `9085ba56…35e8` |
| `benchmark_results_v7.csv` | YOLOv7 系成绩 | `bfc71fe6…f947` |
| `benchmark_results_ssd.csv` | SSD 系成绩 | `ddaad3cd…034e` |
| `reproduction_weights_12models_v2.zip` | 全量合集（622 MB，12 权重 + 3 CSV） | `0b52f6f1…26c92` |

要点（详见仓库 `training/REPRODUCTION.md`）：

- 10 个 YOLO 复现模型 mAP@0.5 全部超过论文原值，平均 +4.91pp，最大 +9.37pp（YOLOv7-tiny）
- 全部模型：19 类修复版标签，输入 640×640，test 集 553 张 COCO 协议评估
- 训练平台：阿里云 PAI-DSW，NVIDIA A10，PyTorch 2.10.0+cu128，Ultralytics 8.4.51

许可：CC BY-NC 4.0（仅科研与非商业用途）

---

## v1.0 模型权重（YOLOv8s · 21 类 · test mAP@0.5 40.74%）

**附件 `tcm-tongue-detection-weights-v7.zip`（56 MB）包含：**

| 文件 | 大小 | 用途 | SHA256 |
|---|---|---|---|
| `best_v7.pt` | 21.5 MB | PyTorch 权重（训练/再训练/格式转换） | `200e9fa2ee09e973cc4f1645a6834a8e1d40f4eba927c7dcf6bf8501dad09bd4` |
| `best_v7.onnx` | 42.7 MB | ONNX 推理（本仓库推理服务直接使用） | `de73f9c7923e8d35a35de46f4e0b7f989c7a07096c4b5616dad740582eaff2dd` |
| `WEIGHTS.md` | — | 权重说明文档 | — |

### 加载配置（不一致会加载失败或结果错误）

- 模型架构：YOLOv8s（Ultralytics）
- 类别数 nc：**21**（类名表见仓库 `training/dataset.yaml`；其中 piweitu/xinfeitu 为数据集发布版空类）
- 输入分辨率：640×640
- 训练框架：Ultralytics 8.4.51 + PyTorch 2.10.0 (cu128)，阿里云 PAI-DSW NVIDIA A10
- 推理框架：Ultralytics ≥ 8.4（.pt）/ onnxruntime ≥ 1.18（.onnx）
- ONNX 参数：opset 19；输入 `images` [1,3,640,640]；输出 `output0` [1,25,8400]（25 = 4 + 21 类，xywh + 各类别置信度，需自行 NMS）

### 加载示例

```python
# PyTorch
from ultralytics import YOLO
model = YOLO("best_v7.pt")
model.predict("tongue.jpg", imgsz=640)

# ONNX Runtime
import onnxruntime as ort
sess = ort.InferenceSession("best_v7.onnx", providers=["CPUExecutionProvider"])
# 输入需 letterbox 到 640×640 并归一化，完整预处理见仓库 inference/app.py
```

### 指标（test 集 553 张，Ultralytics 官方 val，COCO 协议）

P 48.80% / R 41.61% / mAP@0.5 **40.74%** / mAP@0.5:0.95 29.93%
（超原论文同型号 YOLOv8s 基准 33.54% 共 7.2 个百分点；逐类别指标与 8 版迭代过程见仓库 `training/TRAINING.md`）

### 许可

权重以 **CC BY-NC 4.0** 发布：仅限科研与非商业用途，使用时请注明出处。
模型基于 Ultralytics YOLOv8 训练（AGPL-3.0），训练数据来自 TCM-Tongue 数据集（Gao & Jin, 2026，CC BY 4.0，DOI: 10.62762/BISH.2026.303296）。

> ⚠️ 模型输出仅供健康参考与科研使用，不构成医疗建议。
