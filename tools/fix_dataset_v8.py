#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 shezhenv3-coco 数据集类别错位，生成 19 类连续标签版本（v8 训练用）。

背景：
  发布版数据集有 21 个类名（0-20），但：
  - piweitu(18)、xinfeitu(20) 在 train/val/test 全部为 0 实例（空类）
  - xinfeiao 实际标注用的是 19，脾胃区凹 piweiao 是 17
  - test/A (195).txt 含 7 行越界标签（类别 21），导致评估时整图被跳过
  旧模型因此只按 17 类（0-16）训练，piweiao/xinfeiao 共 494 个训练实例被丢弃。

修复内容（只改文本标签和 yaml，不动图片）：
  - 标签重映射：0-17 不变；19(xinfeiao) -> 18；18/20 不存在；>=21 的越界行删除
  - 新类名表 19 类：原 0-17 + xinfeiao(18)
  - 更新 dataset.yaml（nc: 19）和各划分 classes.txt
  - 修改前自动备份原 labels + yaml 到 <数据集>/labels_backup_before_v8fix.zip

用法：
  本地:  python fix_dataset_v8.py --root D:/TCM-Tongue/shezhenv3-coco
  DSW:   python fix_dataset_v8.py --root /mnt/workspace/shezhenv3-coco
"""
import argparse
import sys
import zipfile
from collections import Counter
from pathlib import Path

NAMES_21 = [
    'jiankangshe', 'botaishe', 'hongshe', 'zishe', 'pangdashe', 'shoushe', 'hongdianshe',
    'liewenshe', 'chihenshe', 'baitaishe', 'huangtaishe', 'heitaishe', 'huataishe',
    'shenquao', 'shenqutu', 'gandanao', 'gandantu', 'piweiao', 'piweitu', 'xinfeiao', 'xinfeitu',
]
# 新 19 类：去掉 piweitu(18) 和 xinfeitu(20)
NAMES_19 = [n for i, n in enumerate(NAMES_21) if i not in (18, 20)]
# 旧索引 -> 新索引（None = 删除该行）
REMAP = {i: i for i in range(18)}
REMAP[19] = 18
REMAP[20] = None   # 空类，不存在实例；若出现则删除
REMAP[18] = None   # 空类 piweitu


def remap_file(path: Path, stats: Counter):
    lines_out = []
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        if not line.strip():
            continue
        parts = line.split()
        try:
            cid = int(parts[0])
        except ValueError:
            stats['unparseable'] += 1
            continue
        if cid in REMAP and REMAP[cid] is not None:
            new = REMAP[cid]
            stats[f'cls_{new}'] += 1
            if new != cid:
                stats['remapped'] += 1
            lines_out.append(' '.join([str(new)] + parts[1:]))
        else:
            stats['dropped'] += 1   # 越界/空类标签
    path.write_text('\n'.join(lines_out) + ('\n' if lines_out else ''), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='shezhenv3-coco 数据集目录（含 train/val/test 的那一层）')
    args = ap.parse_args()
    root = Path(args.root)
    missing = [s for s in ['train', 'val', 'test'] if not (root / s / 'labels').is_dir()]
    if missing:
        print(f'[错误] {root} 下未找到这些划分的 labels 目录: {missing}')
        print('       --root 应指向包含 train/val/test 的那一层目录')
        return 1

    # 1) 备份原标签与配置
    backup = root / 'labels_backup_before_v8fix.zip'
    if backup.exists():
        print(f'[错误] 备份已存在（{backup}），疑似已修复过，退出以免重复映射')
        return 1
    with zipfile.ZipFile(backup, 'w', zipfile.ZIP_DEFLATED) as zf:
        for split in ['train', 'val', 'test']:
            for f in (root / split / 'labels').glob('*.txt'):
                zf.write(f, f'{split}/labels/{f.name}')
            ct = root / split / 'classes.txt'
            if ct.is_file():
                zf.write(ct, f'{split}/classes.txt')
        dy = root / 'dataset.yaml'
        if dy.is_file():
            zf.write(dy, 'dataset.yaml')
    print(f'[备份] {backup}')

    # 2) 重映射标签
    total = Counter()
    for split in ['train', 'val', 'test']:
        stats = Counter()
        for f in sorted((root / split / 'labels').glob('*.txt')):
            remap_file(f, stats)
        kept = sum(v for k, v in stats.items() if k.startswith('cls_'))
        print(f'[{split}] 保留 {kept} 实例，重映射 {stats["remapped"]} 条（19->18），'
              f'删除越界/空类 {stats["dropped"]} 条')
        total.update(stats)

    # 3) 删除 Ultralytics 标签缓存（不删会沿用旧的 21 类扫描结果）
    for cache in root.rglob('labels.cache'):
        cache.unlink()
        print(f'[清理] 删除标签缓存 {cache}')

    # 4) 写新 classes.txt 与 dataset.yaml
    for split in ['train', 'val', 'test']:
        ct = root / split / 'classes.txt'
        if ct.is_file():
            ct.write_text('\n'.join(NAMES_19) + '\n', encoding='utf-8')

    yaml_text = (
        '# TMC-Tongue 数据集配置（v8 修复版：19 类连续标签，已去除空类 piweitu/xinfeitu，\n'
        '# xinfeiao 由 19 重映射为 18；原标签备份在 labels_backup_before_v8fix.zip）\n'
        f'path: {root.as_posix()}\n'
        'train: train/images\n'
        'val: val/images\n'
        'test: test/images\n\n'
        f'nc: {len(NAMES_19)}\n'
        'names:\n' + ''.join(f'  {i}: {n}\n' for i, n in enumerate(NAMES_19))
    )
    (root / 'dataset.yaml').write_text(yaml_text, encoding='utf-8')
    # 同步一份 dataset_v8.yaml，避免覆盖原引用时丢失
    (root / 'dataset_v8.yaml').write_text(yaml_text, encoding='utf-8')

    # 4) 验证输出
    print('\n[验证] 修复后各类别实例数（全部划分合计）:')
    for i, n in enumerate(NAMES_19):
        print(f'  {i:2d} {n:14s} {total.get(f"cls_{i}", 0)}')
    print(f'\n[完成] nc=19，总计 {sum(v for k, v in total.items() if k.startswith("cls_"))} 实例，'
          f'删除 {total["dropped"]} 条越界/空类标签')
    print('[提示] 训练 v8 时请指向该目录的 dataset.yaml（或 dataset_v8.yaml）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
