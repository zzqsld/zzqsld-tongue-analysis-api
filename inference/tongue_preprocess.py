#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""舌象图像的轻量颜色归一化（无 OpenCV 依赖版）。"""

from __future__ import annotations

from PIL import Image, ImageEnhance
import numpy as np


def _center_crop(image: np.ndarray, ratio: float = 0.70) -> np.ndarray:
    height, width = image.shape[:2]
    crop_h = max(1, int(height * ratio))
    crop_w = max(1, int(width * ratio))
    top = max(0, (height - crop_h) // 2)
    left = max(0, (width - crop_w) // 2)
    return image[top:top + crop_h, left:left + crop_w]


def _robust_gray_world(image: np.ndarray) -> np.ndarray:
    roi = _center_crop(image, 0.72)
    if roi.size == 0:
        return image

    roi_f = roi.astype(np.float32)
    channel_means = roi_f.reshape(-1, 3).mean(axis=0)
    if np.any(channel_means <= 1e-6):
        return image

    target = float(channel_means.mean())
    gains = target / channel_means
    gains = np.clip(gains, 0.78, 1.28)

    balanced = image.astype(np.float32)
    balanced[..., 0] *= gains[0]
    balanced[..., 1] *= gains[1]
    balanced[..., 2] *= gains[2]
    return np.clip(balanced, 0, 255).astype(np.uint8)


def _light_normalization(image: np.ndarray) -> np.ndarray:
    """用亮度归一化近似 CLAHE 的核心目标：缓解过暗/过亮。"""
    work = image.astype(np.float32)
    luma = 0.114 * work[..., 0] + 0.587 * work[..., 1] + 0.299 * work[..., 2]
    mean_luma = float(luma.mean()) + 1e-6
    target = 128.0
    gain = np.clip(target / mean_luma, 0.82, 1.18)
    work *= gain
    return np.clip(work, 0, 255).astype(np.uint8)


def _gamma_adjust(image: np.ndarray) -> np.ndarray:
    gray = 0.114 * image[..., 0] + 0.587 * image[..., 1] + 0.299 * image[..., 2]
    mean_luma = float(gray.mean())
    if mean_luma < 96:
        gamma = 0.90
    elif mean_luma > 168:
        gamma = 1.05
    else:
        gamma = 1.00

    if abs(gamma - 1.0) < 1e-3:
        return image

    inv_gamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    return lut[image]


def _light_sharpen(image: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(image[:, :, ::-1])
    sharp = ImageEnhance.Sharpness(pil).enhance(1.12)
    arr = np.array(sharp)
    return arr[:, :, ::-1]


def normalize_tongue_frame(image: np.ndarray, color_level: int = 2) -> np.ndarray:
    """对舌象帧做轻量颜色归一化，输入输出均为 BGR 图像。"""
    if image is None or image.size == 0:
        return image

    work = image.copy()

    # 颜色增强级别：1=轻、2=中(默认)、3=强
    level = 2 if color_level not in (1, 2, 3) else int(color_level)

    work = _robust_gray_world(work)

    if level >= 2:
        work = _light_normalization(work)

    if level == 1:
        # 轻量级：只做白平衡和轻微锐化，最大程度保持原色。
        work = _light_sharpen(work)
        return work

    work = _gamma_adjust(work)

    if level >= 3:
        # 强化级：叠加一次亮度归一化，增强弱光下的可分辨性。
        work = _light_normalization(work)

    work = _light_sharpen(work)

    return work
