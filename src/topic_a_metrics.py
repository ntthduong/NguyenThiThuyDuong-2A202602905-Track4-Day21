"""Metric cho Topic A: đánh giá độ khớp LiDAR-camera khi extrinsic bị lệch."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from starter.kitti_io import KittiCalib, KittiObject
from starter.projection import cam_to_image, velo_to_cam


@dataclass
class FullProjection:
    """Kết quả projection giữ nguyên index của point cloud đầu vào."""

    uv: np.ndarray       # (N, 2), NaN nếu điểm không nằm trong ảnh
    depth: np.ndarray    # (N,), z trong camera frame
    mask: np.ndarray     # (N,), True nếu projection hợp lệ và trong ảnh


def project_full(points: np.ndarray, calib: KittiCalib, image_shape: tuple[int, ...]) -> FullProjection:
    """Project point cloud và khôi phục UV về đúng index của từng điểm gốc."""
    points_cam = velo_to_cam(points[:, :3], calib)
    uv_valid, _, mask = cam_to_image(points_cam, calib.P2, image_shape)
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    uv[mask] = uv_valid
    return FullProjection(uv=uv, depth=points_cam[:, 2], mask=mask)


def points_in_object(points_cam: np.ndarray, obj: KittiObject, margin_m: float = 0.0) -> np.ndarray:
    """Mask điểm nằm trong KITTI 3D box.

    KITTI lưu ``location`` tại tâm đáy box. Ta trừ tâm rồi quay ngược
    ``rotation_y`` để kiểm tra điểm trong hệ local của object.
    """
    relative = points_cam - obj.location
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    # Inverse của rotation quanh trục y camera.
    x_local = c * relative[:, 0] - s * relative[:, 2]
    z_local = s * relative[:, 0] + c * relative[:, 2]
    y_local = relative[:, 1]
    h, w, length = obj.dimensions
    finite = np.isfinite(points_cam).all(axis=1)
    return (
        finite
        & (np.abs(x_local) <= length / 2 + margin_m)
        & (np.abs(z_local) <= w / 2 + margin_m)
        & (y_local >= -h - margin_m)
        & (y_local <= margin_m)
    )


def points_in_bbox(uv: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Mask UV nằm trong 2D box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return (
        np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= x1)
        & (uv[:, 0] <= x2)
        & (uv[:, 1] >= y1)
        & (uv[:, 1] <= y2)
    )


def pixel_shifts(reference: FullProjection, candidate: FullProjection) -> np.ndarray:
    """Khoảng cách pixel của các điểm nhìn thấy trong cả hai projection."""
    common = reference.mask & candidate.mask
    return np.linalg.norm(candidate.uv[common] - reference.uv[common], axis=1)


def object_record(
    frame_id: str,
    object_index: int,
    obj: KittiObject,
    object_mask: np.ndarray,
    projection: FullProjection,
) -> dict[str, float | int | str]:
    """Tính tỷ lệ điểm của một 3D object chiếu vào đúng 2D label box."""
    point_ids = np.flatnonzero(object_mask)
    hits = points_in_bbox(projection.uv[point_ids], obj.bbox)
    n_points = len(point_ids)
    distance_m = float(np.linalg.norm(obj.location[[0, 2]]))
    return {
        "frame_id": frame_id,
        "object_index": object_index,
        "class": obj.type,
        "distance_m": distance_m,
        "n_object_points": n_points,
        "n_hits": int(hits.sum()),
        "hit_rate": float(hits.mean()) if n_points else np.nan,
        "truncated": float(obj.truncated),
        "occluded": int(obj.occluded),
    }
