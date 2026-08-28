# 模型权重说明（Release v1.0 · v7 模型）

## 文件清单

| 文件 | 大小 | 用途 | SHA256 |
|---|---|---|---|
| `best_v7.pt` | 21.5 MB | PyTorch 权重（训练/再训练/转换用） | `200e9fa2ee09e973cc4f1645a6834a8e1d40f4eba927c7dcf6bf8501dad09bd4` |
| `best_v7.onnx` | 42.7 MB | ONNX 推理（本仓库推理服务直接使用） | `de73f9c7923e8d35a35de46f4e0b7f989c7a07096c4b5616dad740582eaff2dd` |

## 加载配置（重要，不一致会加载失败或结果错误）

| 配置项 | 值 |
|---|---|
| 模型架构 | YOLOv8s（Ultralytics） |
| 类别数 nc | **21**（类名表见 `training/dataset.yaml`；其中 piweitu/xinfeitu 为数据集发布版空类） |
| 输入分辨率 | 640×640 |
| 训练框架 | Ultralytics 8.4.51 + PyTorch 2.10.0 (cu128) |
| 推理框架 | Ultralytics ≥ 8.4（.pt）/ onnxruntime ≥ 1.18（.onnx） |
| ONNX 参数 | opset 19；输入 `images` [1,3,640,640]；输出 `output0` [1,25,8400]（25 = 4 + 21 类，xywh+各类别置信度，需自行 NMS） |

## 加载示例

```python
# PyTorch
from ultralytics import YOLO
model = YOLO("best_v7.pt")
model.predict("tongue.jpg", imgsz=640)

# ONNX Runtime
import onnxruntime as ort
sess = ort.InferenceSession("best_v7.onnx", providers=["CPUExecutionProvider"])
# 输入需 letterbox 到 640×640 并归一化，完整预处理可参考 inference/app.py
```

## 指标（test 集 553 张，Ultralytics 官方 val，COCO 协议）

P 48.80% / R 41.61% / mAP@0.5 **40.74%** / mAP@0.5:0.95 29.93%
（超原论文同型号 YOLOv8s 基准 33.54% 共 7.2 个百分点；逐类别指标见 training/TRAINING.md）

## 许可

权重以 **CC BY-NC 4.0** 发布：仅限科研与非商业用途，使用时请注明出处。
