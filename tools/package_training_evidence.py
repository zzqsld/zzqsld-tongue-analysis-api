#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
打包 PAI-DSW 上多版本舌诊模型训练数据，用于论文基准对比（配合《论文基准对比模板.md》）。

用法（在 DSW 的 /mnt/workspace 下执行）：
    python package_training_evidence.py                 # 只打包已有训练证据 + 汇总指标表
    python package_training_evidence.py --val-test      # 额外对每版 best.pt 在 test 集上跑一次官方 val（含逐类别 AP，耗时较长）
    python package_training_evidence.py --out /mnt/workspace/pack.zip

产物：
    training_evidence_pack_YYYYMMDD_HHMM.zip
      ├── summary.md            # 各版本总体指标汇总表（可直接填入论文模板第二/三节）
      ├── summary.csv           # 同上，CSV 格式
      ├── <版本>/runs/.../results.csv、args.yaml、曲线图、混淆矩阵等
      ├── <版本>/training_evidence*.zip（如存在，原样收录）
      ├── <版本>/val_test/...   # 仅 --val-test 时生成：test 集官方 val 结果（含逐类别 AP）
      └── logs/ 各版本训练日志与训练脚本副本
"""
import argparse
import csv
import json
import re
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

BASE = Path("/mnt/workspace")
VERSION_GLOB = "tcm_tongue_training*"

# Ultralytics results.csv 的指标列名
COL_P = "metrics/precision(B)"
COL_R = "metrics/recall(B)"
COL_M50 = "metrics/mAP50(B)"
COL_M5095 = "metrics/mAP50-95(B)"

# 常见 YOLO 变体参数量（M），用于填写对比表；找不到则留空
PARAM_TABLE = {
    "yolov8n": 3.2, "yolov8s": 11.1, "yolov8m": 25.9, "yolov8l": 43.6, "yolov8x": 68.2,
    "yolo11n": 2.6, "yolo11s": 9.4, "yolo11m": 20.1,
    "yolo26n": None, "yolo26s": None,
    "yolov5s": 7.0, "yolov5m": 20.9, "yolov5l": 46.2,
}

# 每个训练 run 目录下要收集的证据文件
RUN_FILE_PATTERNS = [
    "results.csv", "args.yaml", "results.png",
    "confusion_matrix.png", "confusion_matrix_normalized.png",
    "PR_curve.png", "P_curve.png", "R_curve.png", "F1_curve.png",
    "BoxPR_curve.png", "BoxP_curve.png", "BoxR_curve.png", "BoxF1_curve.png",
    "val_batch0_pred.jpg", "val_batch1_pred.jpg", "val_batch0_labels.jpg",
]


def log(msg):
    print(msg, flush=True)


def find_versions():
    versions = sorted(p for p in BASE.glob(VERSION_GLOB) if p.is_dir())
    return versions


def version_tag(version_dir: Path) -> str:
    """tcm_tongue_training_v6 -> v6；tcm_tongue_training -> v1"""
    m = re.search(r"_v(\d+)$", version_dir.name)
    return f"v{m.group(1)}" if m else "v1"


def parse_args_yaml(path: Path) -> dict:
    """尽量解析 Ultralytics args.yaml；无 yaml 库时退化为正则抓关键字段。"""
    keys = ["model", "imgsz", "epochs", "batch", "optimizer", "lr0", "lrf",
            "pretrained", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate",
            "scale", "fliplr", "flipud", "mosaic", "mixup", "data"]
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        import yaml
        data = yaml.safe_load(text) or {}
        return {k: data.get(k) for k in keys if k in data}
    except Exception:
        out = {}
        for k in keys:
            m = re.search(rf"^{k}:\s*(.+)$", text, re.M)
            if m:
                out[k] = m.group(1).strip()
        return out


def parse_results_csv(path: Path):
    """返回 (最佳行dict, 总行数)。以 mAP50-95(B) 最高的 epoch 为最佳。"""
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if (r.get(COL_M5095) or "").strip()]
    if not rows:
        return None, 0
    def fnum(r, k):
        try:
            return float(r[k])
        except Exception:
            return -1.0
    best = max(rows, key=lambda r: fnum(r, COL_M5095))
    return best, len(rows)


def fmt(v, nd=2):
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return ""


def model_params_m(args: dict):
    name = str(args.get("model", "")).lower().replace(".pt", "").replace(".yaml", "")
    name = Path(name).name
    return PARAM_TABLE.get(name)


def collect_run(run_dir: Path, stage: Path, rel: Path):
    """把一个 run 目录的证据文件复制到 staging，返回收集到的文件数。"""
    n = 0
    for pat in RUN_FILE_PATTERNS:
        src = run_dir / pat
        if src.is_file():
            dst = stage / rel / pat
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n += 1
    return n


def run_val_test(version_dir: Path, run_dir: Path, stage: Path, rel: Path):
    """可选：对 best.pt 在 test split 上跑官方 val，产出逐类别指标。"""
    weights = run_dir / "weights" / "best.pt"
    data_yaml = version_dir / "dataset_pai.yaml"
    if not weights.is_file():
        log(f"    [跳过 val] 未找到 {weights}")
        return None
    if not data_yaml.is_file():
        log(f"    [跳过 val] 未找到 {data_yaml}")
        return None
    try:
        from ultralytics import YOLO
    except ImportError:
        log("    [跳过 val] 未安装 ultralytics")
        return None

    out_dir = stage / rel / "val_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"    [val] {weights} 在 test 集评估中……")
    model = YOLO(str(weights))
    # project 指向 staging，val 产生的 PR 曲线等图也一并进压缩包
    metrics = model.val(data=str(data_yaml), split="test", verbose=False,
                        project=str(out_dir), name="ultralytics")
    per_class = {}
    names = getattr(metrics, "names", {}) or {}
    ap50 = getattr(metrics.box, "ap50", None)
    ap = getattr(metrics.box, "ap", None)
    # 关键：all_ap 的行号不是类别 id！ap_class_index[i] 才是第 i 行对应的真实类别。
    # test 集中 0 实例的类别不会出现在 all_ap 里，直接按下标映射会错位。
    aci_raw = getattr(metrics.box, "ap_class_index", None)
    aci = [int(x) for x in aci_raw] if aci_raw is not None else []
    if ap50 is not None and len(ap50):
        if len(aci) == len(ap50):
            for row, cid in enumerate(aci):
                cname = names.get(cid, str(cid))
                per_class[cname] = {
                    "AP50": round(float(ap50[row]) * 100, 2),
                    "AP50_95": round(float(ap[row]) * 100, 2) if ap is not None and row < len(ap) else None,
                }
        else:
            log(f"    [注意] ap_class_index({len(aci)}) 与 AP 数组({len(ap50)}) 长度不一致，"
                f"退化为按 names 顺序映射")
            for i, cname in names.items():
                if i >= len(ap50):
                    break
                per_class[cname] = {
                    "AP50": round(float(ap50[i]) * 100, 2),
                    "AP50_95": round(float(ap[i]) * 100, 2) if ap is not None and i < len(ap) else None,
                }
    # 记录 0 实例（未参与评估）的类别，避免误读
    nt = getattr(metrics.box, "nt_per_class", None)
    zero_classes = []
    if nt is not None and len(nt) == len(names):
        zero_classes = [names[i] for i in range(len(names)) if int(nt[i]) == 0]
    summary = {
        "weights": str(weights),
        "data": str(data_yaml),
        "nt_per_class": {names[i]: int(nt[i]) for i in range(len(names))} if nt is not None and len(nt) == len(names) else None,
        "zero_instance_classes": zero_classes,
        "precision": round(float(metrics.box.mp) * 100, 2),
        "recall": round(float(metrics.box.mr) * 100, 2),
        "mAP50": round(float(metrics.box.map50) * 100, 2),
        "mAP50_95": round(float(metrics.box.map) * 100, 2),
        "per_class": per_class,
    }
    (out_dir / "val_test_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser(description="打包多版本训练证据用于论文基准对比")
    ap.add_argument("--val-test", action="store_true", help="对每版 best.pt 在 test 集跑官方 val（含逐类别 AP）")
    ap.add_argument("--out", default="", help="输出 zip 路径（默认 /mnt/workspace/training_evidence_pack_时间戳.zip）")
    args = ap.parse_args()

    versions = find_versions()
    if not versions:
        log(f"[错误] 在 {BASE} 下未找到 {VERSION_GLOB} 目录")
        return 1
    log(f"[信息] 找到 {len(versions)} 个版本目录: {[v.name for v in versions]}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_zip = Path(args.out) if args.out else BASE / f"training_evidence_pack_{stamp}.zip"

    table_rows = []   # 汇总表
    val_rows = []     # test 集 val 结果

    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "pack"
        stage.mkdir(parents=True)

        for vdir in versions:
            tag = version_tag(vdir)
            log(f"\n[版本] {vdir.name} ({tag})")

            # 1) 找训练 run（含 results.csv 的目录）
            run_dirs = sorted({p.parent for p in vdir.rglob("results.csv")})
            if not run_dirs:
                log("    [警告] 未找到 results.csv，跳过指标汇总")
            for run_dir in run_dirs:
                rel = Path(vdir.name) / run_dir.relative_to(vdir)
                n = collect_run(run_dir, stage, rel)
                log(f"    run: {run_dir.relative_to(vdir)}（收集 {n} 个证据文件）")

                args_y = run_dir / "args.yaml"
                cfg = parse_args_yaml(args_y) if args_y.is_file() else {}
                best_row, epochs_done = (None, 0)
                if (run_dir / "results.csv").is_file():
                    best_row, epochs_done = parse_results_csv(run_dir / "results.csv")

                best_pt = run_dir / "weights" / "best.pt"
                weight_mb = round(best_pt.stat().st_size / 1024 / 1024, 1) if best_pt.is_file() else ""

                if best_row:
                    table_rows.append({
                        "版本": tag,
                        "run": str(run_dir.relative_to(vdir)),
                        "模型": cfg.get("model", ""),
                        "参数量/M": model_params_m(cfg) or "",
                        "权重/MB": weight_mb,
                        "P/%": fmt(best_row.get(COL_P)),
                        "R/%": fmt(best_row.get(COL_R)),
                        "mAP@0.5/%": fmt(best_row.get(COL_M50)),
                        "mAP@0.5:0.95/%": fmt(best_row.get(COL_M5095)),
                        "最佳epoch": best_row.get("epoch", ""),
                        "实际训练epoch": epochs_done,
                        "imgsz": cfg.get("imgsz", ""),
                        "batch": cfg.get("batch", ""),
                        "optimizer": cfg.get("optimizer", ""),
                        "lr0": cfg.get("lr0", ""),
                        "pretrained": cfg.get("pretrained", ""),
                    })

                if args.val_test:
                    vr = run_val_test(vdir, run_dir, stage, rel)
                    if vr:
                        vr["版本"] = tag
                        val_rows.append(vr)

            # 2) 版本级证据：training_evidence 目录/zip、dataset_pai.yaml
            for z in vdir.glob("training_evidence*.zip"):
                shutil.copy2(z, stage / vdir.name / z.name)
                log(f"    收录 {z.name} ({z.stat().st_size/1024/1024:.1f} MB)")
            yml = vdir / "dataset_pai.yaml"
            if yml.is_file():
                shutil.copy2(yml, stage / vdir.name / yml.name)

            # 3) 根目录日志与训练脚本
            n = tag[1:]
            log_dir = stage / "logs"
            log_dir.mkdir(exist_ok=True)
            candidates = [
                BASE / f"train_v{n}.log", BASE / f"continue_v{n}.log",
                BASE / f"continue_v{n}_export.py", BASE / f"train_modelscope_v{n}.py",
            ]
            if tag == "v1":
                candidates += [BASE / "train.log", BASE / "train_modelscope.py"]
            for c in candidates:
                if c.is_file():
                    shutil.copy2(c, log_dir / c.name)
                    log(f"    收录日志/脚本 {c.name}")

        # 4) 汇总表 summary.md / summary.csv
        if table_rows:
            headers = list(table_rows[0].keys())
            with open(stage / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=headers)
                w.writeheader()
                w.writerows(table_rows)

        md = ["# 各版本训练指标汇总（results.csv 最佳 epoch，按 mAP@0.5:0.95 取最优）", ""]
        md.append("| 版本 | 模型 | 参数量/M | 权重/MB | P/% | R/% | mAP@0.5/% | mAP@0.5:0.95/% | 最佳epoch | imgsz | batch | optimizer |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in table_rows:
            md.append(f"| {r['版本']} | {r['模型']} | {r['参数量/M']} | {r['权重/MB']} | {r['P/%']} | {r['R/%']} "
                      f"| {r['mAP@0.5/%']} | {r['mAP@0.5:0.95/%']} | {r['最佳epoch']} | {r['imgsz']} | {r['batch']} | {r['optimizer']} |")
        if val_rows:
            md += ["", "## test 集官方 val（含逐类别 AP，见各版本 val_test/val_test_metrics.json）", ""]
            md.append("| 版本 | P/% | R/% | mAP@0.5/% | mAP@0.5:0.95/% |")
            md.append("|---|---|---|---|---|")
            for r in val_rows:
                md.append(f"| {r['版本']} | {r['precision']} | {r['recall']} | {r['mAP50']} | {r['mAP50_95']} |")
        md += ["", f"打包时间：{datetime.now().isoformat(timespec='seconds')}"]
        (stage / "summary.md").write_text("\n".join(md), encoding="utf-8")

        # 5) 打 zip
        log(f"\n[打包] -> {out_zip}")
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for p in sorted(stage.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(stage).as_posix())

    log(f"[完成] {out_zip} ({out_zip.stat().st_size/1024/1024:.1f} MB)")
    log("把该 zip 下载到本地后，用 summary.md 的表格填写《论文基准对比模板.md》第二/三节；")
    log("逐类别 AP（模板第四节）需要 --val-test 重新打包或在各 val_test_metrics.json 中查看。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
