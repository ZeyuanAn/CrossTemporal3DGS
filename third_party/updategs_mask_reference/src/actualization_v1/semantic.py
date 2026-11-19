"""Conservative region appearance features and geometry-only pair candidates.

This module deliberately does not assign instance identity or changed labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np


@dataclass(frozen=True)
class RegionFeature:
    frame_id: str
    region_id: int
    label: int | str | None
    embedding: np.ndarray
    area: int
    centroid_xy: tuple[float, float]
    mask: np.ndarray
    partial: bool = False


def normalize_embedding(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 0 else np.zeros_like(value)


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    yy = np.minimum((np.arange(shape[0]) * mask.shape[0] / shape[0]).astype(int), mask.shape[0] - 1)
    xx = np.minimum((np.arange(shape[1]) * mask.shape[1] / shape[1]).astype(int), mask.shape[1] - 1)
    return mask[yy[:, None], xx[None, :]].astype(bool)


def region_features(masks: np.ndarray, dense_features: np.ndarray, frame_id: str,
                    labels: Iterable[int | str | None] | None = None,
                    partial: Iterable[bool] | None = None) -> list[RegionFeature]:
    """Mean-pool each mask from CxHfxW or HfxWxC dense visual features."""
    masks = np.asarray(masks).astype(bool)
    if masks.ndim == 4:
        masks = masks.reshape((-1,) + masks.shape[-2:])
    if masks.ndim != 3:
        raise ValueError("masks must have shape (R,H,W)")
    features = np.asarray(dense_features, dtype=np.float32)
    if features.ndim != 3:
        raise ValueError("dense_features must be CxHfxW or HfxWxC")
    if features.shape[0] == features.shape[1]:
        # HxWxC convention; CxHfxW has unequal spatial dimensions here.
        features = np.moveaxis(features, -1, 0)
    labels = list(labels) if labels is not None else [None] * len(masks)
    partial = list(partial) if partial is not None else [False] * len(masks)
    output = []
    for rid, mask in enumerate(masks):
        small = _resize_mask(mask, features.shape[-2:])
        pixels = features[:, small]
        embedding = normalize_embedding(pixels.mean(axis=1) if pixels.size else np.zeros(features.shape[0]))
        ys, xs = np.where(mask)
        centroid = (float(xs.mean()), float(ys.mean())) if len(xs) else (float("nan"), float("nan"))
        output.append(RegionFeature(frame_id, rid, labels[rid], embedding, int(mask.sum()), centroid,
                                    mask.copy(), bool(partial[rid])))
    return output


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    a, b = normalize_embedding(left), normalize_embedding(right)
    return float(np.dot(a, b))


def cosine_difference(before: np.ndarray, after: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Per-vector visual embedding difference; never interpreted as semantic truth."""
    a = np.asarray(before, np.float32); b = np.asarray(after, np.float32)
    if a.shape != b.shape: raise ValueError("embedding shapes must match")
    an = np.linalg.norm(a, axis=axis, keepdims=True); bn = np.linalg.norm(b, axis=axis, keepdims=True)
    dot = np.sum(a * b, axis=axis, keepdims=True) / np.maximum(an * bn, 1e-8)
    return (1.0 - np.clip(dot, -1.0, 1.0)).squeeze(axis)


def contour(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    inner = np.zeros_like(mask)
    if min(mask.shape) > 2:
        inner[1:-1, 1:-1] = mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
    return mask & ~inner


def contour_similarity(left: np.ndarray, right: np.ndarray, aligned: bool = False) -> float | None:
    """Compare boundaries only for masks in the same pixel alignment."""
    if not aligned or np.shape(left) != np.shape(right):
        return None
    a, b = contour(left), contour(right)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def region_pair_candidates(before: Iterable[RegionFeature], after: Iterable[RegionFeature],
                           geometry_distance: dict[tuple[int, int], float] | None = None,
                           max_geometry_distance: float | None = None) -> list[dict[str, object]]:
    """Return geometry/appearance evidence; never emit same-instance or changed decisions."""
    candidates = []
    for old in before:
        for new in after:
            distance = None if geometry_distance is None else geometry_distance.get((old.region_id, new.region_id))
            if max_geometry_distance is not None and distance is not None and distance > max_geometry_distance:
                continue
            candidates.append({
                "before_frame_id": old.frame_id, "after_frame_id": new.frame_id,
                "before_region_id": old.region_id, "after_region_id": new.region_id,
                "label_match": bool(old.label is not None and old.label == new.label),
                "cosine": cosine_similarity(old.embedding, new.embedding),
                "geometry_distance": distance,
                "partial_object": bool(old.partial or new.partial),
                "contour_similarity": contour_similarity(old.mask, new.mask, aligned=False),
                "role": "geometry_appearance_candidate",
            })
    return candidates


def _feature_chw(value: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 3:
        raise ValueError("feature embedding must be a 3-D dense feature map")
    if value.shape[1:] == hw:
        return value
    if value.shape[:2] == hw:
        return np.moveaxis(value, -1, 0)
    raise ValueError(f"feature map shape {value.shape} does not match image HW={hw}")


def feature_difference(existing_after_embedding: np.ndarray, historical_render_embedding: np.ndarray,
                       H: int, W: int, masks: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Compute aligned, normalized dense feature differences for the same rendered image HW.

    The caller must provide embeddings extracted from the after image and the calibrated-pose
    historical render at the same image coordinates. This function never aligns raw source
    indices or declares a semantic/class/instance change.
    """
    def as_dense(value):
        value = np.asarray(value, dtype=np.float32)
        if value.ndim != 3:
            raise ValueError("feature embedding must be a 3-D dense feature map")
        if value.shape[1:] == (H, W):
            return value
        if value.shape[:2] == (H, W):
            return np.moveaxis(value, -1, 0)
        # EfficientSAM's Cx64x64 embedding is a valid image-aligned feature grid;
        # H,W describe the source image, not the encoder grid resolution.
        if value.shape[0] <= 1024:
            return value
        raise ValueError(f"feature map shape {value.shape} does not match image HW={(H, W)}")
    a, b = as_dense(existing_after_embedding), as_dense(historical_render_embedding)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"feature channel mismatch: {a.shape} vs {b.shape}")
    if a.shape[1:] != b.shape[1:]:
        raise ValueError(f"feature grid mismatch: {a.shape} vs {b.shape}")
    if a.shape != b.shape:
        raise ValueError(f"feature channel/grid mismatch: {a.shape} vs {b.shape}")
    eps = np.float32(1e-8)
    an = a / np.maximum(np.linalg.norm(a, axis=0, keepdims=True), eps)
    bn = b / np.maximum(np.linalg.norm(b, axis=0, keepdims=True), eps)
    cosine = np.sum(an * bn, axis=0).clip(-1.0, 1.0)
    l2 = np.linalg.norm(an - bn, axis=0)
    result = {"cosine": cosine.astype(np.float32), "l2": l2.astype(np.float32),
              "difference": (1.0 - cosine).astype(np.float32), "valid": np.isfinite(cosine)}
    if masks is not None:
        masks = np.asarray(masks).astype(bool)
        if masks.ndim == 2:
            masks = masks[None]
        if masks.ndim != 3 or masks.shape[1:] != (H, W):
            raise ValueError("masks must have shape (R,H,W) matching the aligned image")
        masks = np.asarray([_resize_mask(mask, a.shape[1:]) for mask in masks])
        result["region_mean_difference"] = np.asarray([
            result["difference"][mask].mean() if mask.any() else np.nan for mask in masks
        ], dtype=np.float32)
    return result

