"""Conservative local visible-change evidence for one RGB image pair.

This module is deliberately independent from the existing routing algorithms.  It
reports 2-D visible evidence only; it never proves physical removal.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from scipy.ndimage import binary_dilation, binary_erosion, label, uniform_filter

from .ssim_evidence import _fit_affine, _luma, ssim_evidence


def _validate_rgb(x, name, shape=None):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"{name} must have shape HxWx3")
    if shape is not None and x.shape != shape:
        raise ValueError("before and after shapes must match")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} must be finite")
    return np.clip(x, 0.0, 1.0)


def _mask(x, shape, name):
    x = np.asarray(x, dtype=bool)
    if x.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return x

def save_local_result_npz(path, result):
    p=Path(path)
    np.savez_compressed(p, support_valid=np.asarray(result['support_valid_mask'],bool), support_unknown=np.asarray(result['support_unknown_mask'],bool), change=np.asarray(result['change_mask'],bool), unknown=np.asarray(result['unknown_mask'],bool), pre_anchor_change=np.asarray(result['pre_anchor_change_mask'],bool), interior_anchor=np.asarray(result['interior_anchor_mask'],bool), unanchored_change_unknown=np.asarray(result['unanchored_change_unknown_mask'],bool), counts=np.asarray([int(np.asarray(result['support_valid_mask']).sum()),int(np.asarray(result['support_unknown_mask']).sum()),int(np.asarray(result['change_mask']).sum()),int(np.asarray(result['unknown_mask']).sum())],dtype=np.int64))
    return p

def component_interior_anchor(change_mask, valid_mask, alignment_radius):
    radius=int(alignment_radius)
    if radius<0: raise ValueError('alignment_radius must be non-negative')
    change=np.asarray(change_mask,bool)&np.asarray(valid_mask,bool); labels,count=label(change,structure=np.ones((3,3),dtype=np.uint8)); anchors=np.zeros_like(change)
    if radius==0: anchors=change.copy()
    elif count:
        kernel=np.ones((2*radius+1,2*radius+1),dtype=bool); anchors=binary_erosion(change,structure=kernel,iterations=1,border_value=0)
    ids=np.unique(labels[anchors]); lookup=np.zeros(count+1,dtype=bool); lookup[ids]=True; unanchored=change&~lookup[labels]
    return change&~unanchored, anchors&valid_mask, unanchored


def _local_distance(before, after, valid, radius):
    """Return same-anchor bidirectional 3x3 neighborhood tolerance.

    The returned best_offset is a forward-vote diagnostic only; it is not a
    cycle-consistent correspondence or camera registration estimate.
    """
    h, w, _ = before.shape
    best = np.full((h, w), np.inf, np.float32)
    best_center = np.full((h, w), np.inf, np.float32)
    best_offset = np.zeros((h, w, 2), np.int16)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y0, y1 = max(0, dy), min(h, h + dy)
            x0, x1 = max(0, dx), min(w, w + dx)
            sy0, sy1 = max(0, -dy), min(h, h - dy)
            sx0, sx1 = max(0, -dx), min(w, w - dx)
            pair = valid[y0:y1, x0:x1] & valid[sy0:sy1, sx0:sx1]
            delta = np.mean((before[y0:y1, x0:x1] - after[sy0:sy1, sx0:sx1]) ** 2, axis=2)
            patch_valid = uniform_filter(pair.astype(np.float32), size=3, mode="constant", cval=0.0)
            d = np.full((y1 - y0, x1 - x0), np.inf, np.float32)
            patch_error = uniform_filter(delta, size=3, mode="constant", cval=0.0)
            complete = patch_valid >= (1.0 - 1e-6)
            d[complete] = np.sqrt(np.maximum(patch_error[complete], 0.0))
            center = np.sqrt(np.mean((before[y0:y1, x0:x1] - after[sy0:sy1, sx0:sx1]) ** 2, axis=2))
            take = d < best[y0:y1, x0:x1]
            best[y0:y1, x0:x1][take] = d[take]
            best_center[y0:y1, x0:x1][take] = center[take]
            best_offset[y0:y1, x0:x1][take] = (dy, dx)
    # Symmetric max is conservative: either direction must be explainable.
    reverse = np.full((h, w), np.inf, np.float32)
    reverse_center = np.full((h, w), np.inf, np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y0, y1 = max(0, dy), min(h, h + dy)
            x0, x1 = max(0, dx), min(w, w + dx)
            sy0, sy1 = max(0, -dy), min(h, h - dy)
            sx0, sx1 = max(0, -dx), min(w, w - dx)
            pair = valid[y0:y1, x0:x1] & valid[sy0:sy1, sx0:sx1]
            delta = np.mean((after[y0:y1, x0:x1] - before[sy0:sy1, sx0:sx1]) ** 2, axis=2)
            patch_valid = uniform_filter(pair.astype(np.float32), size=3, mode="constant", cval=0.0)
            d = np.full((y1 - y0, x1 - x0), np.inf, np.float32)
            patch_error = uniform_filter(delta, size=3, mode="constant", cval=0.0)
            complete = patch_valid >= (1.0 - 1e-6)
            d[complete] = np.sqrt(np.maximum(patch_error[complete], 0.0))
            center = np.sqrt(np.mean((after[y0:y1, x0:x1] - before[sy0:sy1, sx0:sx1]) ** 2, axis=2))
            take = d < reverse[y0:y1, x0:x1]
            reverse[y0:y1, x0:x1][take] = d[take]
            reverse_center[y0:y1, x0:x1][take] = center[take]
    return np.maximum(best, reverse), np.maximum(best_center, reverse_center), best_offset


def _components(mask, valid_count):
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, count = label(mask, structure=structure)
    # Scale with the observed valid image area; never use a fixed 100-pixel gate.
    minimum = max(1, int(round(np.sqrt(max(valid_count, 1)) * 0.03)))
    keep = np.zeros_like(mask)
    for idx in range(1, count + 1):
        component = labels == idx
        if int(component.sum()) >= minimum:
            keep |= component
    return keep, minimum, int(count)


def _static_orb_registration(before, after, static_valid, grid=(4, 4)):
    """Audit-only 2-D static registration; never a 3-D/camera calibration."""
    report = {"status": "unavailable", "reliable": False, "is_camera_registration": False}
    try:
        import cv2
    except Exception as exc:
        report["reason"] = "cv2_unavailable:" + repr(exc)
        return report
    mask = (np.asarray(static_valid, bool).astype(np.uint8) * 255)
    b = np.uint8(np.clip(_luma(before) * 255.0, 0, 255))
    a = np.uint8(np.clip(_luma(after) * 255.0, 0, 255))
    orb = cv2.ORB_create(nfeatures=3000)
    kp1, de1 = orb.detectAndCompute(b, mask)
    kp2, de2 = orb.detectAndCompute(a, mask)
    if de1 is None or de2 is None:
        report.update({"status": "insufficient_features", "keypoints_before": len(kp1), "keypoints_after": len(kp2)})
        return report
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    fwd = bf.knnMatch(de1, de2, k=2); rev = bf.knnMatch(de2, de1, k=2)
    fwd_ids = {(m.queryIdx, m.trainIdx): m for pair in fwd if len(pair) == 2 and pair[0].distance < .75 * pair[1].distance for m in [pair[0]]}
    rev_ids = {(m.trainIdx, m.queryIdx): m for pair in rev if len(pair) == 2 and pair[0].distance < .75 * pair[1].distance for m in [pair[0]]}
    mutual = [m for key, m in fwd_ids.items() if key in rev_ids]
    if len(mutual) < 6:
        report.update({"status": "insufficient_mutual_matches", "mutual_matches": len(mutual)})
        return report
    src = np.float32([kp1[m.queryIdx].pt for m in mutual]); dst = np.float32([kp2[m.trainIdx].pt for m in mutual])
    matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=.8, maxIters=3000, confidence=.99)
    if matrix is None or inliers is None:
        report.update({"status": "ransac_failed", "mutual_matches": len(mutual)})
        return report
    keep = inliers.ravel().astype(bool); pred = cv2.transform(src[None], matrix)[0]; err = np.linalg.norm(pred - dst, axis=1)
    h, w = mask.shape; cells = {(min(3, max(0, int(p[1] / h * 4))), min(3, max(0, int(p[0] / w * 4)))) for p in src[keep]}
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0])); angle = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    dx, dy = float(matrix[0, 2]), float(matrix[1, 2]); fraction = max(abs(dx - round(dx)), abs(dy - round(dy)))
    inlier_fraction = float(keep.mean()); median_error = float(np.median(err[keep])) if keep.any() else float("inf")
    correspondence_reliable = bool(keep.sum() >= 10 and len(cells) >= 3 and inlier_fraction >= .6 and median_error <= .5)
    integer_compatible = bool(abs(scale - 1.) <= .01 and abs(angle) <= 1.0 and
                              abs(dx) <= 2.5 and abs(dy) <= 2.5 and fraction < .2)
    reliable = bool(correspondence_reliable and integer_compatible)
    report.update({"status": "reliable" if reliable else ("unsupported_transform" if correspondence_reliable else "unreliable_correspondence"), "reliable": reliable,
                   "correspondence_reliable": correspondence_reliable,
                   "integer_compatible": integer_compatible,
                   "geometric_veto": bool(correspondence_reliable and not integer_compatible),
                   "mutual_matches": len(mutual), "inliers": int(keep.sum()), "inlier_fraction": inlier_fraction,
                   "grid_cells": len(cells), "median_reprojection_error": median_error,
                   "dx": dx, "dy": dy, "scale": scale, "rotation_deg": angle,
                   "fractional_translation_error": fraction, "is_camera_registration": False})
    return report


def _raw_ecc_registration(before, after, static_valid, radius):
    """Continuous translation audit on raw luminance; never a 3-D calibration."""
    report = {"status": "unavailable", "reliable": False, "is_camera_registration": False}
    try:
        import cv2
    except Exception as exc:
        report["reason"] = "cv2_unavailable:" + repr(exc)
        return report
    mask = np.asarray(static_valid, bool)
    count = int(mask.sum()); report["static_count"] = count
    if count < 256:
        report["status"] = "insufficient_static"; return report
    ys, xs = np.where(mask); y0, y1 = int(ys.min()), int(ys.max()) + 1; x0, x1 = int(xs.min()), int(xs.max()) + 1
    template = np.asarray(_luma(before)[y0:y1, x0:x1], np.float32)
    observed = np.asarray(_luma(after)[y0:y1, x0:x1], np.float32)
    input_mask = mask[y0:y1, x0:x1].astype(np.uint8)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
    def solve(src, dst, m):
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cc, result = cv2.findTransformECC(src, dst, warp, cv2.MOTION_TRANSLATION, criteria, m, 5)
            return float(cc), float(result[0, 2]), float(result[1, 2]), None
        except Exception as exc:
            return None, None, None, repr(exc)
    cc, dx, dy, error = solve(template, observed, input_mask)
    back_cc, back_dx, back_dy, back_error = solve(observed, template, input_mask)
    if error is not None or cc is None:
        report.update({"status": "ecc_failed", "error": error, "backward_error": back_error,
                       "forward_cc": cc, "backward_cc": back_cc})
        return report
    cycle = float(np.hypot(dx + back_dx, dy + back_dy)) if back_dx is not None else float("inf")
    fractional = max(abs(dx - round(dx)), abs(dy - round(dy)))
    bounded = abs(dx) <= float(radius) + 0.5 and abs(dy) <= float(radius) + 0.5
    reliable = bool(cc >= .95 and back_cc is not None and back_cc >= .95 and cycle <= .2 and bounded)
    report.update({"status": "reliable" if reliable else "unreliable_continuous",
                   "reliable": reliable, "forward_cc": cc, "backward_cc": back_cc,
                   "dx": dx, "dy": dy, "backward_dx": back_dx, "backward_dy": back_dy,
                   "cycle_error": cycle, "fractional_translation_error": fractional,
                   "bounded_translation": bounded, "bbox_xyxy": [x0, y0, x1, y1],
                   "static_count": count, "is_camera_registration": False,
                   "noninteger": bool(fractional > .2)})
    return report


def build_local_change_evidence(before, after, feature_candidate, ssim_candidate,
                                valid, pre_masks=None, post_masks=None,
                                alignment_radius=2, *, external_pose_verified=False):
    """Build conservative local visible-change evidence.

    All image arrays are RGB HWC float01; candidates/valid are HW bool; SAM
    masks, when supplied, are NHW bool.  Candidate pixels explained by a local
    displacement are classified as unknown, never as change.
    """
    before = _validate_rgb(before, "before")
    after = _validate_rgb(after, "after", before.shape)
    h, w, _ = before.shape
    feature = _mask(feature_candidate, (h, w), "feature_candidate")
    ssim = _mask(ssim_candidate, (h, w), "ssim_candidate")
    valid = _mask(valid, (h, w), "valid")
    if not isinstance(alignment_radius, (int, np.integer)) or alignment_radius < 0:
        raise ValueError("alignment_radius must be a non-negative integer")
    if pre_masks is not None:
        pre_masks = _mask(np.asarray(pre_masks, dtype=bool), (len(pre_masks), h, w), "pre_masks")
    if post_masks is not None:
        post_masks = _mask(np.asarray(post_masks, dtype=bool), (len(post_masks), h, w), "post_masks")

    candidate = feature | ssim
    candidate_valid = candidate & valid
    empty = not bool(valid.any())
    pre_luma, post_luma = _luma(before), _luma(after)
    # If all valid pixels are candidates, there is no protected static subset;
    # use the valid image for exposure estimation, as the existing SSIM route does.
    has_static = bool(np.any(valid & ~candidate))
    fit_mask = valid & ~candidate if has_static else valid.copy()
    gain, offset, fit_used, clipped, fit_report = _fit_affine(pre_luma, post_luma, valid, fit_mask)
    initial_fit_report = dict(fit_report)
    initial_fit_status = {"fit_used": bool(fit_used), "fit_reliable": bool(fit_used and not clipped),
                          "gain_clipped": bool(clipped), "domain_source": fit_report.get("fit_domain_source"),
                          "fit_count": int(fit_mask.sum())}
    # A clipped affine fit on an all-candidate image is commonly caused by
    # viewpoint displacement, not exposure.  Do not let it erase shift evidence.
    global_fit_fallback = not has_static
    if clipped and global_fit_fallback:
        gain, offset = 1.0, 0.0
    fit_report["fit_count"] = int(fit_mask.sum())
    fit_report["excluded_count"] = int(valid.sum() - fit_mask.sum())
    fit_report["threshold_status"] = "trusted_static" if has_static else "global_fallback_untrusted"
    ssim_result = ssim_evidence([before], [after], valid_masks=[valid], fit_masks=[fit_mask])
    corrected_after = np.clip((after - offset) / max(gain, 1e-8), 0.0, 1.0)
    registration = _static_orb_registration(before, corrected_after, valid & ~candidate)
    ecc_registration = _raw_ecc_registration(before, after, valid & ~candidate, int(alignment_radius))
    aligned_before = before
    aligned_valid = valid
    distance, center_distance, offsets = _local_distance(before, corrected_after, valid, int(alignment_radius))
    # Establish a deterministic integer-image alignment from the modal offset
    # of low-residual complete patches.  In the sampler above (dy, dx) means
    # ``before[y, x]`` was compared with ``after[y-dy, x-dx]``.  Therefore the
    # corresponding before->after warp is ``(-dy, -dx)``.
    low = valid & np.isfinite(distance) & (distance <= 0.03)
    modal_offset = None
    support_count = 0
    if int(low.sum()) >= max(9, int(0.1 * valid.sum())):
        pairs, counts = np.unique(offsets[low].reshape(-1, 2), axis=0, return_counts=True)
        index = int(np.argmax(counts)); modal_offset = tuple(int(x) for x in pairs[index])
        support_count = int(counts[index])
    modal_support_ratio = support_count / max(int(low.sum()), 1)
    if registration.get("reliable"):
        modal_offset = (int(-round(registration["dy"])), int(-round(registration["dx"])))
        alignment_applied = True
    else:
        alignment_applied = modal_offset is not None and modal_support_ratio >= 0.5
    warp_shift = None
    if alignment_applied and modal_offset != (0, 0):
        warp_shift = (-modal_offset[0], -modal_offset[1])
        aligned_before = np.zeros_like(before)
        aligned_valid = np.zeros_like(valid)
        dy, dx = warp_shift
        y0, y1 = max(0, dy), min(h, h + dy)
        x0, x1 = max(0, dx), min(w, w + dx)
        sy0, sy1 = max(0, -dy), min(h, h - dy)
        sx0, sx1 = max(0, -dx), min(w, w - dx)
        aligned_before[y0:y1, x0:x1] = before[sy0:sy1, sx0:sx1]
        aligned_valid[y0:y1, x0:x1] = valid[sy0:sy1, sx0:sx1] & valid[y0:y1, x0:x1]
        valid = aligned_valid
        candidate_valid = candidate & valid
        fit_mask = valid & ~candidate if np.any(valid & ~candidate) else valid.copy()
        gain, offset, fit_used, clipped, fit_report = _fit_affine(
            _luma(aligned_before), _luma(corrected_after), valid, fit_mask)
        final_fit_report = dict(fit_report)
        corrected_after = np.clip((corrected_after - offset) / max(gain, 1e-8), 0.0, 1.0)
        before = aligned_before
        distance, center_distance, offsets = _local_distance(before, corrected_after, valid, 0)
    else:
        final_fit_report = dict(fit_report)
    static = valid & ~candidate & np.isfinite(distance)
    values = distance[static]
    median = float(np.median(values)) if values.size else 0.0
    mad = float(np.median(np.abs(values - median))) if values.size else 0.0
    threshold = max(0.03, median + 3.0 * mad)
    # Recompute SSIM on the final aligned/photometrically corrected pair.
    aligned_ssim = ssim_evidence([aligned_before], [corrected_after],
                                  valid_masks=[valid], fit_masks=[valid & ~candidate])
    aligned_ssim_map = np.asarray(aligned_ssim["corrected_maps"][0], np.float32)
    static_ssim = static & np.isfinite(aligned_ssim_map)
    ssim_values = aligned_ssim_map[static_ssim]
    ssim_median = float(np.median(ssim_values)) if ssim_values.size else 0.0
    ssim_mad = float(np.median(np.abs(ssim_values - ssim_median))) if ssim_values.size else 0.0
    structure_threshold = max(0.10, ssim_median + 3.0 * ssim_mad)
    aligned_rgb = np.sqrt(np.mean((aligned_before - corrected_after) ** 2, axis=2))
    rgb_values = aligned_rgb[static]
    rgb_median = float(np.median(rgb_values)) if rgb_values.size else 0.0
    rgb_mad = float(np.median(np.abs(rgb_values - rgb_median))) if rgb_values.size else 0.0
    rgb_structure_threshold = max(0.003, rgb_median + 3.0 * rgb_mad)
    explained = candidate_valid & (distance <= threshold)
    comparable = np.isfinite(distance)
    # Incomplete patches are unclassified, not evidence of change.
    identical_after_correction = bool(np.allclose(before[valid], corrected_after[valid], atol=1e-6)) if valid.any() else False
    global_shift_supported = global_fit_fallback and int(explained.sum()) > int(0.5 * valid.sum())
    ecc_uncertainty_veto = bool(ecc_registration.get("noninteger", False) and
                                ecc_registration.get("status") in {"unreliable_continuous", "reliable"})
    raw_geometric_veto = bool(registration.get("geometric_veto", False) or
                          (ecc_registration.get("reliable", False) and ecc_registration.get("noninteger", False)) or
                          ecc_uncertainty_veto)
    effective_geometric_veto = bool(raw_geometric_veto and not external_pose_verified)
    center_support = np.isfinite(center_distance) & (center_distance > threshold)
    center_core = uniform_filter(center_support.astype(np.float32), size=3,
                                 mode="constant", cval=0.0) >= 0.55
    raw_change = candidate_valid & ~explained & comparable & center_support & center_core
    # Newly exposed/unsupported bands can only be explained by the image
    # boundary; never promote their nearby tolerance fringe to change.
    invalid_fringe = binary_dilation(~valid, iterations=int(alignment_radius) + 1)
    raw_change &= ~invalid_fringe
    if global_shift_supported:
        raw_change &= uniform_filter(raw_change.astype(np.float32), size=5,
                                     mode="constant", cval=0.0) >= 0.20
    if global_fit_fallback and not global_shift_supported:
        raw_change[:] = False
    filtered_change, minimum_component_area, component_count = _components(raw_change, int(valid.sum()))
    change = filtered_change & valid
    rgb_change_mask = change.copy()
    saturated_mask = np.any(after >= 1.0, axis=2) | np.any(after <= 0.0, axis=2)
    fit_reliable_for_rescue = bool(fit_used and not clipped and fit_report.get("fit_count", 0) >= 16)
    rescue_domain = (feature & ssim & valid & ~saturated_mask &
                     (aligned_ssim_map >= structure_threshold) &
                     (aligned_rgb >= rgb_structure_threshold))
    structure_rescued = np.zeros_like(valid)
    if fit_reliable_for_rescue and alignment_applied and not effective_geometric_veto:
        rescued_components, rescue_minimum, _ = _components(rescue_domain, int(valid.sum()))
        labels, count = label(rescue_domain, structure=np.ones((3, 3), np.uint8))
        for idx in range(1, count + 1):
            component = labels == idx
            area = int(component.sum())
            anchors = int((component & (aligned_rgb >= rgb_structure_threshold)).sum())
            if area >= rescue_minimum and anchors >= max(1, int(np.ceil(0.20 * area))):
                structure_rescued |= component
    change |= structure_rescued
    camera_alignment_required = effective_geometric_veto
    if camera_alignment_required:
        change[:] = False
    raw_support_unknown=np.asarray(ssim_result['support_unknown'],bool); ssim_support_unknown=raw_support_unknown[0] if raw_support_unknown.ndim==3 else raw_support_unknown; change &= ~ssim_support_unknown
    # Invalid pixels remain unsupported/unknown; valid pixels with no complete
    # patch are unclassified, not change evidence.
    unknown = (candidate & valid & ~change) | ssim_support_unknown
    if identical_after_correction:
        unknown = ssim_support_unknown.copy()
    unknown &= ~change
    pre_anchor_change=change.copy(); change,interior_anchor,unanchored_change=component_interior_anchor(change,valid,int(alignment_radius)); unknown=(unknown|unanchored_change)&~change
    feature_supported = feature & valid
    ssim_only = ssim & ~feature & valid
    if empty:
        change[:] = False
        explained[:] = False
        unknown[:] = False
    diagnostics = {
        "source": "rgb_complete_3x3_patch_symmetric_distance+ssim_evidence+feature_candidate",
        "alignment_radius": int(alignment_radius),
        "integer_image_alignment": {"sampling_offset": None if modal_offset is None else list(modal_offset),
                                     "warp_shift": None if warp_shift is None else list(warp_shift),
                                     "support_count": support_count,
                                     "support_ratio": float(modal_support_ratio),
                                     "applied": bool(alignment_applied),
                                     "is_camera_registration": False},
        "static_registration": registration,
        "continuous_ecc_registration": ecc_registration,
        "ecc_uncertainty_veto": ecc_uncertainty_veto,
        "raw_geometric_veto": raw_geometric_veto,
        "effective_geometric_veto": effective_geometric_veto,
        "external_pose_verified": bool(external_pose_verified),
        "pose_veto_bypassed": bool(raw_geometric_veto and external_pose_verified),
        "camera_alignment_required": camera_alignment_required,
        "alignment_explained_rule": "same-anchor bidirectional complete-3x3 neighborhood tolerance <= threshold",
        "best_offset_role": "forward vote only; not cycle-consistent correspondence",
        "threshold": float(threshold), "threshold_floor": 0.03,
        "structure_rescue": {"source": "aligned_ssim_and_center_rgb",
                              "structure_threshold": structure_threshold,
                              "static_ssim_median": ssim_median, "static_ssim_mad": ssim_mad,
                              "center_rgb_threshold": rgb_structure_threshold,
                              "static_rgb_median": rgb_median, "static_rgb_mad": rgb_mad,
                              "fit_reliable": fit_reliable_for_rescue,
                              "alignment_required": True,
                              "region_seed_fraction": float((feature & ssim & valid).sum() / max(int(valid.sum()), 1)),
                              "rescued_count": int(structure_rescued.sum())},
        "static_median": median, "static_mad": mad,
        "brightness_gain": float(gain), "brightness_offset": float(offset),
        "brightness_fit_used": bool(fit_used), "brightness_gain_clipped": bool(clipped),
        "brightness_fit_report": fit_report,
        "photometric_fit": {"initial": initial_fit_status,
                             "initial_report": initial_fit_report,
                             "final": {"fit_used": bool(fit_used),
                                       "fit_reliable": bool(fit_used and not clipped),
                                       "gain_clipped": bool(clipped),
                                       "domain_source": final_fit_report.get("fit_domain_source"),
                                       "fit_count": int(fit_mask.sum())},
                             "final_report": final_fit_report},
        "ssim_threshold": float(ssim_result["threshold"]),
        "ssim_method": ssim_result["method"],
        "valid_count": int(valid.sum()), "candidate_count": int(candidate.sum()),
        "comparable_count": int(np.isfinite(distance).sum()),
        "invalid_count": int((~valid).sum()),
        "feature_supported_count": int(feature_supported.sum()),
        "ssim_only_count": int(ssim_only.sum()), "alignment_explained_count": int(explained.sum()),
        "change_count": int(change.sum()), "unknown_count": int(unknown.sum()), "pre_anchor_change_count": int(pre_anchor_change.sum()), "interior_anchor_count": int(interior_anchor.sum()), "unanchored_change_unknown_count": int(unanchored_change.sum()),
        "rgb_change_count": int(rgb_change_mask.sum()),
        "structure_rescued_count": int(structure_rescued.sum()),
        "saturated_count": int(saturated_mask.sum()),
        "component_count": component_count, "minimum_component_area": minimum_component_area,
        "pre_mask_count": 0 if pre_masks is None else int(pre_masks.shape[0]),
        "post_mask_count": 0 if post_masks is None else int(post_masks.shape[0]),
        "sam_role": "region/boundary audit only; no whole-object gate",
        "camera_registration": False,
        "view_change_handled": False,
        "low_residual_candidate_policy": "unknown; rendering error cannot establish real edit",
        "empty_valid": empty, "physical_removal_proof": False,
    }
    residual_output = np.where(np.isfinite(distance), distance, 0.0).astype(np.float32)
    return {"change_mask": change, "unknown_mask": unknown, "pre_anchor_change_mask":pre_anchor_change, "interior_anchor_mask":interior_anchor, "unanchored_change_unknown_mask":unanchored_change, "support_valid_mask":valid&~ssim_support_unknown, "support_unknown_mask":ssim_support_unknown,
            "effective_valid_mask": valid.copy(), "saturated_mask": saturated_mask,
            "rgb_change_mask": rgb_change_mask, "structure_rescued_mask": structure_rescued,
            "feature_supported_mask": feature_supported, "ssim_only_mask": ssim_only,
            "alignment_explained_mask": explained, "residual_map": residual_output,
            "diagnostics": diagnostics}
