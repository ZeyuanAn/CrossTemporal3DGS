"""Independent 2D evidence adapter for the six-method validation runner.

This module deliberately has no RGB-difference fallback.  EfficientSAM features
must come from ``model_cache`` (or the local checkpoint loader); otherwise the
result is explicitly unavailable rather than a fabricated evidence map.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .rgb_pair_mask_features import multichannel_evidence
from .ssim_evidence import ssim_evidence
from .local_change_evidence import build_local_change_evidence


def _rgb(x, name):
    raw = np.asarray(x)
    if not np.issubdtype(raw.dtype, np.floating):
        raise ValueError(f"{name} must be float01")
    a = np.asarray(x, dtype=np.float32)
    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError(f"{name} must be HWC RGB")
    if not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 1):
        raise ValueError(f"{name} must be finite float01")
    return a


def _valid(x, shape):
    a = np.asarray(x, dtype=bool)
    if a.shape != shape: raise ValueError("valid must match image H,W")
    return a


def _embeddings(renderer_rgb, target_rgb, model_cache):
    cache = model_cache or {}
    if "pre_embeddings" in cache and "post_embeddings" in cache:
        layout = cache.get("embedding_layout")
        if layout not in {"CHW", "1CHW"}:
            raise ValueError("cached embeddings require explicit embedding_layout='CHW' or '1CHW'; HWC is rejected")
        pre = np.asarray(cache["pre_embeddings"], np.float32); post = np.asarray(cache["post_embeddings"], np.float32)
        if layout == "CHW": pre = pre[None, ...]; post = post[None, ...]
        if pre.ndim != 4 or post.ndim != 4 or pre.shape != post.shape or pre.shape[0] != 1:
            raise ValueError("pre/post embeddings must match exact [1,C,Hf,Wf] layout")
        return pre, post, {"source": "cached_efficientsam_embeddings", "embedding_layout": "1CHW", "embedding_shape": list(pre.shape)}
    model = cache.get("model"); torch = cache.get("torch")
    if model is None or torch is None:
        checkpoint = Path(cache.get("checkpoint", ".cache/efficientsam/efficient_sam_vitt.pt"))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"EfficientSAM checkpoint unavailable: {checkpoint}")
        from ...scripts.actualization_v1.extract_render_features import load_fresh_model
        model, torch = load_fresh_model(checkpoint, cache.get("device", "cpu"))
    def tensor(a): return torch.from_numpy(a).permute(2, 0, 1)[None]
    device = next(model.parameters()).device
    with torch.inference_mode():
        pre = model.get_image_embeddings(tensor(renderer_rgb).to(device))[0].detach().cpu().numpy()
        post = model.get_image_embeddings(tensor(target_rgb).to(device))[0].detach().cpu().numpy()
    return pre[None, ...], post[None, ...], {"source": "fresh_local_efficientsam", "embedding_layout": "1CHW", "embedding_shape": [1, *pre.shape], "checkpoint": str(cache.get("checkpoint", ".cache/efficientsam/efficient_sam_vitt.pt"))}


def build_fusion_evidence(renderer_rgb, target_rgb, valid, *, device="cuda:0", model_cache=None, registered_pose_verified=False):
    """Build independent 2D evidence for one render/target pair.

    All image arrays are HWC float01. ``model_cache`` may contain paired
    EfficientSAM embeddings or a loaded ``model``/``torch`` pair. No holdout
    image enters fitting: the caller supplies only this pair and valid mask.
    """
    renderer_rgb = _rgb(renderer_rgb, "renderer_rgb"); target_rgb = _rgb(target_rgb, "target_rgb")
    if renderer_rgb.shape != target_rgb.shape: raise ValueError("renderer_rgb/target_rgb shape mismatch")
    valid = _valid(valid, renderer_rgb.shape[:2])
    pre, post, provenance = _embeddings(renderer_rgb, target_rgb, model_cache)
    if pre.ndim != 4 or post.ndim != 4 or pre.shape != post.shape or pre.shape[0] != 1:
        raise ValueError("pre/post embeddings must be exact [1,C,Hf,Wf]")
    from scipy.ndimage import zoom
    feature_valid = zoom(valid.astype(np.float32), (pre.shape[-2] / valid.shape[0], pre.shape[-1] / valid.shape[1]), order=0) >= .5
    feat = multichannel_evidence(pre, post, [renderer_rgb.shape[:2]], valid_masks=np.asarray([feature_valid]))
    feature_map = np.asarray(feat["residual_maps"][0], np.float32) / 2.0
    feature_mask = np.asarray(feat["multidim_candidate_mask"][0], bool)
    ssim = ssim_evidence([renderer_rgb], [target_rgb], valid_masks=[valid], fit_masks=[valid & ~feature_mask])
    ssim_map = np.asarray(ssim["corrected_maps"][0], np.float32)
    ssim_mask = np.asarray(ssim["candidate_masks"][0], bool)
    local = build_local_change_evidence(renderer_rgb, target_rgb, feature_mask, ssim_mask, valid, external_pose_verified=registered_pose_verified)
    score = np.maximum(feature_map, np.clip(ssim_map, 0.0, 1.0)).astype(np.float32); score[~valid] = 0
    change_mask = np.asarray(local["change_mask"], bool) & valid
    unknown_mask = (np.asarray(local["unknown_mask"], bool) | ~valid) & ~change_mask
    stable_mask = valid & ~unknown_mask & ~change_mask
    region = {"status": "unavailable", "semantic_labels": False, "provenance": "No EfficientSAM proposal model supplied; no region labels fabricated."}
    cache = model_cache or {}
    if cache.get("proposal_masks") is not None:
        masks = np.asarray(cache["proposal_masks"], bool)
        if masks.ndim == 3 and masks.shape[1:] == valid.shape:
            region = {"status": "available", "semantic_labels": False, "source": "cached_efficientsam_proposals", "masks": masks & valid}
    diagnostics = {"provenance": provenance, "device": device, "feature": feat.get("threshold_report"), "ssim": ssim.get("reports", []), "local": local["diagnostics"], "pair_local_fit": True, "detection_target_observed": True, "registered_pose_verified": bool(registered_pose_verified), "oracle_used": False, "external_gt_used": False, "output_partition": {"valid_count": int(valid.sum()), "change_count": int(change_mask.sum()), "unknown_count": int(unknown_mask.sum()), "stable_count": int(stable_mask.sum()), "mutually_exclusive": bool(not np.any(change_mask & unknown_mask) and not np.any(change_mask & stable_mask) and not np.any(unknown_mask & stable_mask)), "outside_valid_is_unknown": bool(np.all(unknown_mask[~valid]))}}
    return {"score": score, "change_mask": change_mask, "unknown_mask": unknown_mask, "stable_mask": stable_mask, "feature_candidate": feature_mask, "ssim_candidate": ssim_mask, "region_labels": region, "diagnostics": diagnostics}
