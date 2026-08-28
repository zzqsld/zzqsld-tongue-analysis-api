# 训练方法说明（v7 复现指南）

## 环境

| 项目 | 版本/配置 |
|---|---|
| 训练平台 | 阿里云 PAI-DSW |
| GPU | NVIDIA A10（24GB） |
| 框架 | PyTorch 2.10.0 + CUDA 12.8 |
| 训练库 | Ultralytics 8.4.51 |
| 基础模型 | YOLOv8s（COCO 预训练权重初始化） |

## 数据

- 数据集：TCM-Tongue（Gao & Jin, 2026，CC BY 4.0），6,719 张，21 个类名（其中 piweitu/xinfeitu 为发布版空类）
- 划分：train 5,594 / val 572 / test 553（官方分层划分原样使用）
- 配置：`dataset.yaml`（本目录，21 类）
- 已知数据瑕疵：test 集 `A (195).txt` 含 7 行越界标签（类别 21），评估时该图自动跳过；修复工具见 `../tools/fix_dataset_v8.py`

## v7 关键超参数（完整版见 args_v7.yaml）

| 配置项 | 值 |
|---|---|
| 输入分辨率 | 640×640 |
| epochs / patience | 240 / 45（实际早停于 epoch 99，best @54） |
| batch | 64 |
| 优化器 | SGD + 余弦退火（cos_lr），lr0=0.01 |
| 数据增强 | HSV(0.015/0.7/0.4)、translate 0.1、scale 0.5、fliplr 0.5、mosaic 1.0（close_mosaic=20） |
| AMP | 开启 |

## 版本演进（test 集官方 val，553 张，COCO 协议）

| 版本 | 模型 | 策略 | P/% | R/% | mAP@0.5/% | mAP@0.5:0.95/% |
|---|---|---|---|---|---|---|
| v1 | YOLOv8s | 基线训练（early stop @104） | 39.40 | 37.33 | 34.61 | 25.40 |
| v2 | YOLOv8s | v1 权重续训，lr0 0.005 + mixup 0.1 | 37.90 | 45.65 | 36.68 | 27.79 |
| v3 | YOLOv8s | v2 权重续训 100 epoch | 35.90 | 45.25 | 36.92 | 27.65 |
| v4 | YOLOv8s | 重训（early stop @117） | 40.34 | 44.73 | 36.84 | 27.12 |
| v5 | YOLOv8m | 大模型对照，batch 32（early stop @122） | 45.13 | 42.46 | 39.40 | **30.93** |
| v6 | YOLOv8s | 收回 s 级（early stop @68） | 46.51 | 41.57 | 39.04 | 27.94 |
| **v7** | YOLOv8s | SGD + 余弦退火（early stop @99，best @54） | **48.80** | 41.61 | **40.74** | 29.93 |
| v8 | YOLOv8s | 19 类标签修复版对照实验（best @16） | 38.66 | 41.01 | 35.94 | 26.18 |

发布权重为 **v7**。从零复现：直接跑 `train.py`；更接近原结果的路径是 v1→v2→v3 续训后再按 v7 配置收尾。

## 评估复现

```bash
yolo val model=best.pt data=dataset.yaml split=test
```

v7 参考结果（test 集）：P 48.80% / R 41.61% / mAP@0.5 40.74% / mAP@0.5:0.95 29.93%。
逐类别 AP 与论文基准的完整对比见根目录 README.md。
