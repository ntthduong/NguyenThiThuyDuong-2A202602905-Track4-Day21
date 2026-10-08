"""Tạo biểu đồ, ảnh demo và failure case cho Topic A."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from starter.datasets import load_frame
from starter.projection import draw_box2d, overlay_points, perturb_extrinsic, velo_to_cam
from src.topic_a_metrics import points_in_object, project_full


def _annotate(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 560), 36), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return out


def render_overlay(data_root: str, frame_id: str, out_path: Path, yaw_deg: float = 0.0) -> None:
    fr = load_frame(data_root, frame_id)
    calib = perturb_extrinsic(fr["calib"], yaw_deg=yaw_deg)
    projection = project_full(fr["points"], calib, fr["image"].shape)
    vis = overlay_points(fr["image"], projection.uv[projection.mask], projection.depth[projection.mask])
    for obj in fr["labels"]:
        vis = draw_box2d(vis, obj.bbox, label=obj.type)
    vis = _annotate(vis, f"frame={frame_id}  yaw={yaw_deg:g} deg  points={projection.mask.sum()}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def render_comparison(data_root: str, frame_id: str, yaw_values: list[float], out_path: Path) -> None:
    fr = load_frame(data_root, frame_id)
    panels = []
    for yaw in yaw_values:
        calib = perturb_extrinsic(fr["calib"], yaw_deg=yaw)
        projection = project_full(fr["points"], calib, fr["image"].shape)
        vis = overlay_points(fr["image"], projection.uv[projection.mask], projection.depth[projection.mask])
        for obj in fr["labels"]:
            vis = draw_box2d(vis, obj.bbox, label=obj.type)
        panels.append(_annotate(vis, f"Yaw {yaw:g} deg"))
    comparison = cv2.hconcat(panels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), comparison)


def plot_sweep(summary: pd.DataFrame, perturb_type: str, out_path: Path) -> None:
    df = summary[summary["perturb_type"] == perturb_type].sort_values("level")
    x_label = "Yaw perturbation (degree)" if perturb_type == "yaw" else "Lateral translation (cm)"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(df["level"], df["median_shift_px"], "o-", label="Median")
    axes[0].plot(df["level"], df["p95_shift_px"], "s--", label="P95")
    axes[0].set(xlabel=x_label, ylabel="Pixel displacement", title="Reprojection displacement")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(df["level"], 100 * df["macro_hit_rate"], "o-", label="Macro object")
    axes[1].plot(df["level"], 100 * df["micro_hit_rate"], "s--", label="Micro point")
    axes[1].plot(df["level"], 100 * df["fov_rate"], "^:", label="All-point FOV")
    axes[1].plot(df["level"], 100 * df["alignment_score"], "D-.", label="Normalized alignment")
    threshold = float(df["alignment_threshold"].iloc[0])
    axes[1].axhline(100 * threshold, color="red", linestyle="--", alpha=0.7,
                    label=f"Drift threshold ({threshold:.2f})")
    axes[1].set(xlabel=x_label, ylabel="Rate (%)", title="Alignment metrics")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.suptitle(f"LiDAR-camera calibration sensitivity: {perturb_type}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_failure(
    data_root: str,
    object_df: pd.DataFrame,
    out_path: Path,
    failure_yaw: float = 3.0,
) -> dict[str, float | int | str]:
    """Chọn object giảm hit-rate nhiều nhất và vẽ calibration đúng/sai."""
    yaw_df = object_df[object_df["perturb_type"] == "yaw"]
    baseline = yaw_df[np.isclose(yaw_df["level"], 0)][
        ["frame_id", "object_index", "hit_rate"]
    ].rename(columns={"hit_rate": "baseline_hit_rate"})
    drift = yaw_df[np.isclose(yaw_df["level"], failure_yaw)].copy()
    candidates = drift.merge(baseline, on=["frame_id", "object_index"])
    candidates["hit_drop"] = candidates["baseline_hit_rate"] - candidates["hit_rate"]
    candidates = candidates[candidates["n_object_points"] >= 5]
    if candidates.empty:
        raise RuntimeError("Không tìm được object có >= 5 điểm để tạo failure case")
    worst = candidates.sort_values(["hit_drop", "n_object_points"], ascending=False).iloc[0]

    frame_id = str(worst["frame_id"])
    object_index = int(worst["object_index"])
    fr = load_frame(data_root, frame_id)
    obj = fr["labels"][object_index]
    points_cam = velo_to_cam(fr["points"][:, :3], fr["calib"])
    obj_mask = points_in_object(points_cam, obj)

    panels = []
    for yaw, title in [(0.0, "Correct calibration"), (failure_yaw, f"Yaw drift {failure_yaw:g} deg")]:
        projection = project_full(
            fr["points"], perturb_extrinsic(fr["calib"], yaw_deg=yaw), fr["image"].shape
        )
        vis = draw_box2d(fr["image"], obj.bbox, color=(0, 255, 0), label=obj.type)
        for u, v in projection.uv[obj_mask & projection.mask].astype(int):
            cv2.circle(vis, (int(u), int(v)), 4, (0, 0, 255), -1)
        rate = worst["baseline_hit_rate"] if yaw == 0 else worst["hit_rate"]
        panels.append(_annotate(vis, f"{title} | object hit-rate={100 * rate:.1f}%"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.hconcat(panels))
    return {
        "frame_id": frame_id,
        "object_index": object_index,
        "class": str(worst["class"]),
        "distance_m": float(worst["distance_m"]),
        "n_object_points": int(worst["n_object_points"]),
        "baseline_hit_rate": float(worst["baseline_hit_rate"]),
        "drift_hit_rate": float(worst["hit_rate"]),
        "hit_drop": float(worst["hit_drop"]),
        "failure_yaw": failure_yaw,
    }


def render_metric_failure(
    data_root: str,
    object_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_path: Path,
    translation_cm: float = 10.0,
) -> dict[str, float | int | str | bool]:
    """Minh họa drift dịch ngang mà normalized score không cảnh báo."""
    trans_objects = object_df[object_df["perturb_type"] == "translation"]
    baseline = trans_objects[np.isclose(trans_objects["level"], 0)][
        ["frame_id", "object_index", "hit_rate"]
    ].rename(columns={"hit_rate": "baseline_hit_rate"})
    drift = trans_objects[np.isclose(trans_objects["level"], translation_cm)].copy()
    candidates = drift.merge(baseline, on=["frame_id", "object_index"])
    # Chọn object nhiều điểm, baseline tốt và vẫn gần như nằm trong box sau drift.
    candidates = candidates[
        (candidates["baseline_hit_rate"] >= 0.95)
        & (candidates["hit_rate"] >= 0.90)
        & (candidates["n_object_points"] >= 5)
    ]
    if candidates.empty:
        raise RuntimeError("Không tìm được object phù hợp để minh họa metric failure")
    chosen = candidates.sort_values("n_object_points", ascending=False).iloc[0]

    summary_row = summary_df[
        (summary_df["perturb_type"] == "translation")
        & np.isclose(summary_df["level"], translation_cm)
    ].iloc[0]
    frame_id = str(chosen["frame_id"])
    object_index = int(chosen["object_index"])
    fr = load_frame(data_root, frame_id)
    obj = fr["labels"][object_index]
    points_cam = velo_to_cam(fr["points"][:, :3], fr["calib"])
    obj_mask = points_in_object(points_cam, obj)

    panels = []
    settings = [
        (fr["calib"], "Translation 0 cm", float(chosen["baseline_hit_rate"]), 1.0, False),
        (
            perturb_extrinsic(fr["calib"], t_xyz_m=(0.0, translation_cm / 100.0, 0.0)),
            f"Translation {translation_cm:g} cm",
            float(chosen["hit_rate"]),
            float(summary_row["alignment_score"]),
            bool(summary_row["drift_detected"]),
        ),
    ]
    for calib, title, hit_rate, score, detected in settings:
        projection = project_full(fr["points"], calib, fr["image"].shape)
        vis = draw_box2d(fr["image"], obj.bbox, color=(0, 255, 0), label=obj.type)
        for u, v in projection.uv[obj_mask & projection.mask].astype(int):
            cv2.circle(vis, (int(u), int(v)), 4, (0, 0, 255), -1)
        panels.append(_annotate(
            vis, f"{title} | hit={100 * hit_rate:.1f}% score={score:.3f} alert={detected}"
        ))

    cv2.imwrite(str(out_path), cv2.hconcat(panels))
    return {
        "frame_id": frame_id,
        "object_index": object_index,
        "class": str(chosen["class"]),
        "translation_cm": translation_cm,
        "n_object_points": int(chosen["n_object_points"]),
        "baseline_hit_rate": float(chosen["baseline_hit_rate"]),
        "drift_hit_rate": float(chosen["hit_rate"]),
        "median_shift_px": float(summary_row["median_shift_px"]),
        "p95_shift_px": float(summary_row["p95_shift_px"]),
        "alignment_score": float(summary_row["alignment_score"]),
        "alignment_threshold": float(summary_row["alignment_threshold"]),
        "drift_detected": bool(summary_row["drift_detected"]),
    }
