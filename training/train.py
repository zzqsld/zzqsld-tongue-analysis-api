#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复现 v7 模型的训练脚本（YOLOv8s / 21 类 / TCM-Tongue 数据集）。

用法：
    python train.py --data /path/to/shezhenv3-coco/dataset.yaml

与 v7 完全一致的配置见 args_v7.yaml（Ultralytics 训练后自动生成）。
注意：v7 是从更早版本迭代而来（见 TRAINING.md 的版本演进），
从零复现可直接运行本脚本，指标可能略有差异（未做续训链）。
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='dataset.yaml', help='数据集 yaml 路径')
    ap.add_argument('--weights', default='yolov8s.pt', help='初始化权重（COCO 预训练）')
    ap.add_argument('--project', default='runs', help='输出目录')
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        # ---- v7 关键超参数（详见 args_v7.yaml / TRAINING.md）----
        epochs=240,          # 实际早停于 99（patience=45）
        patience=45,
        batch=64,
        imgsz=640,
        optimizer='SGD',
        lr0=0.01,
        cos_lr=True,         # 余弦退火
        close_mosaic=20,     # 最后 20 轮关闭 mosaic
        seed=42,
        amp=True,
        project=args.project,
        name='train',
    )


if __name__ == '__main__':
    main()
