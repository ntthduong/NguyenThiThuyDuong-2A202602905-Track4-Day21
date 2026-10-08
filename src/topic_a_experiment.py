"""Chạy toàn bộ benchmark Topic A và tạo bằng chứng trong ``results/``.

Ví dụ:
    python -m src.topic_a_experiment --data-root data/kitti_mini --out-dir results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from starter.datasets import list_frames, load_frame
from starter.projection import perturb_extrinsic, velo_to_cam
from src.topic_a_metrics import object_record, pixel_shifts, points_in_object, project_full
from src.topic_a_visualize import (
    plot_sweep,
    render_comparison,
    render_failure,
    render_metric_failure,
    render_overlay,
)


def parse_levels(text: str) -> list[float]:
    try:
        return [float(value) for value in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Danh sách mức phải là các số ngăn cách bằng dấu phẩy") from exc


def evaluate_level(frames: list[dict], perturb_type: str, level: float, min_object_points: int):
    shifts: list[np.ndarray] = []
    object_rows: list[dict] = []
    n_finite = 0
    n_fov = 0

    for fr in frames:
        reference = project_full(fr["points"], fr["calib"], fr["image"].shape)
        if perturb_type == "yaw":
            candidate_calib = perturb_extrinsic(fr["calib"], yaw_deg=level)
        elif perturb_type == "translation":
            candidate_calib = perturb_extrinsic(fr["calib"], t_xyz_m=(0.0, level / 100.0, 0.0))
        else:
            raise ValueError(perturb_type)
        candidate = project_full(fr["points"], candidate_calib, fr["image"].shape)

        finite = np.isfinite(fr["points"][:, :3]).all(axis=1)
        n_finite += int(finite.sum())
        n_fov += int(candidate.mask.sum())
        shifts.append(pixel_shifts(reference, candidate))

        points_cam = velo_to_cam(fr["points"][:, :3], fr["calib"])
        for object_index, obj in enumerate(fr["labels"]):
            object_mask = points_in_object(points_cam, obj)
            if int(object_mask.sum()) < min_object_points:
                continue
            row = object_record(fr["frame_id"], object_index, obj, object_mask, candidate)
            row.update({"perturb_type": perturb_type, "level": level})
            object_rows.append(row)

    shift_values = np.concatenate(shifts) if shifts else np.empty(0)
    hit_rates = np.array([row["hit_rate"] for row in object_rows], dtype=float)
    total_object_points = sum(int(row["n_object_points"]) for row in object_rows)
    total_hits = sum(int(row["n_hits"]) for row in object_rows)
    summary = {
        "perturb_type": perturb_type,
        "level": level,
        "n_frames": len(frames),
        "n_objects": len(object_rows),
        "n_common_projected_points": len(shift_values),
        "fov_rate": n_fov / n_finite if n_finite else np.nan,
        "macro_hit_rate": float(np.mean(hit_rates)) if len(hit_rates) else np.nan,
        "micro_hit_rate": total_hits / total_object_points if total_object_points else np.nan,
        "median_shift_px": float(np.median(shift_values)) if len(shift_values) else np.nan,
        "p95_shift_px": float(np.percentile(shift_values, 95)) if len(shift_values) else np.nan,
    }
    return summary, object_rows


def add_alignment_metrics(summary_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Chuẩn hóa macro hit-rate theo baseline 0 của từng loại perturbation."""
    out = summary_df.copy()
    out["alignment_score"] = np.nan
    for perturb_type in out["perturb_type"].unique():
        group_mask = out["perturb_type"] == perturb_type
        baseline = out[group_mask & np.isclose(out["level"], 0)]
        if len(baseline) != 1:
            raise ValueError(
                f"{perturb_type} cần đúng một level 0 để tính alignment score; nhận được {len(baseline)}"
            )
        baseline_rate = float(baseline.iloc[0]["macro_hit_rate"])
        if not np.isfinite(baseline_rate) or baseline_rate <= 0:
            raise ValueError(f"Baseline macro hit-rate không hợp lệ: {baseline_rate}")
        out.loc[group_mask, "alignment_score"] = out.loc[group_mask, "macro_hit_rate"] / baseline_rate
    out["alignment_threshold"] = threshold
    out["drift_detected"] = out["alignment_score"] < threshold
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Topic A: sweep calibration drift và đo FOV, pixel shift, object hit-rate"
    )
    parser.add_argument("--data-root", default="data/kitti_mini", help="Dataset KITTI dùng cho benchmark")
    parser.add_argument("--out-dir", default="results", help="Thư mục lưu CSV và figures")
    parser.add_argument(
        "--yaw-degs", type=parse_levels, default=parse_levels("-3,-2,-1,-0.5,0,0.5,1,2,3"),
        help="Các mức yaw, ngăn cách bằng dấu phẩy",
    )
    parser.add_argument(
        "--translation-cm", type=parse_levels, default=parse_levels("0,2,5,10"),
        help="Các mức dịch ngang theo trục y LiDAR, đơn vị cm",
    )
    parser.add_argument("--min-object-points", type=int, default=5, help="Bỏ object có quá ít điểm LiDAR")
    parser.add_argument(
        "--alignment-threshold", type=float, default=0.85,
        help="Cảnh báo drift khi normalized alignment score thấp hơn ngưỡng này",
    )
    parser.add_argument("--frames", nargs="*", help="Frame ID; mặc định chạy tất cả frame")
    args = parser.parse_args()
    if not 0 < args.alignment_threshold <= 1:
        parser.error("--alignment-threshold phải nằm trong khoảng (0, 1]")

    frame_ids = args.frames or list_frames(args.data_root)
    if not frame_ids:
        raise SystemExit(f"Không tìm thấy frame trong {args.data_root}")
    print(f"Loading {len(frame_ids)} frames from {args.data_root} ...")
    frames = [load_frame(args.data_root, frame_id) for frame_id in frame_ids]

    summary_rows: list[dict] = []
    object_rows: list[dict] = []
    for perturb_type, levels in (("yaw", args.yaw_degs), ("translation", args.translation_cm)):
        for level in levels:
            summary, objects = evaluate_level(frames, perturb_type, level, args.min_object_points)
            summary_rows.append(summary)
            object_rows.extend(objects)
            print(
                f"{perturb_type:11s} {level:6g}: median_shift={summary['median_shift_px']:.2f}px "
                f"macro_hit={summary['macro_hit_rate']:.1%} fov={summary['fov_rate']:.1%}"
            )

    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df = add_alignment_metrics(pd.DataFrame(summary_rows), args.alignment_threshold)
    object_df = pd.DataFrame(object_rows)
    summary_df[summary_df["perturb_type"] == "yaw"].to_csv(out_dir / "yaw_perturb_sweep.csv", index=False)
    summary_df[summary_df["perturb_type"] == "translation"].to_csv(
        out_dir / "translation_sweep.csv", index=False
    )
    object_df.to_csv(out_dir / "object_metrics.csv", index=False)

    plot_sweep(summary_df, "yaw", figures_dir / "yaw_sweep_metrics.png")
    plot_sweep(summary_df, "translation", figures_dir / "translation_sweep_metrics.png")
    demo_frames = [("000019", "near"), ("000011", "medium"), ("000004", "far")]
    available = set(frame_ids)
    for frame_id, distance_tag in demo_frames:
        if frame_id in available:
            render_overlay(
                args.data_root, frame_id, figures_dir / f"demo_{distance_tag}_{frame_id}.png"
            )
    comparison_frame = "000011" if "000011" in available else frame_ids[0]
    render_comparison(
        args.data_root, comparison_frame, [0.0, 1.0, 3.0], figures_dir / f"yaw_comparison_{comparison_frame}.png"
    )
    failure = render_failure(args.data_root, object_df, figures_dir / "fail_01_yaw_drift.png")
    metric_failure = render_metric_failure(
        args.data_root,
        object_df,
        summary_df,
        figures_dir / "fail_02_translation_threshold.png",
    )
    (out_dir / "failure_case.json").write_text(
        json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "metric_failure_case.json").write_text(
        json.dumps(metric_failure, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Failure case: {failure}")
    print(f"Metric failure: {metric_failure}")
    print(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
