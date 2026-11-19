"""Initial, view-scoped BU mask from 2-D reference evidence.

This adapter reports visible image agreement only. It does not infer historical
3-D uncertainty, object identity, or semantic/physical edit labels.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import numpy as np


class InitialEvidenceMask:
    """Generate a conservative trusted-pixel mask for one same-camera pair.

    Args:
        checkpoint_path: Explicit EfficientSAM checkpoint path.
        device: Torch device used by EfficientSAM.
        analysis_width: Width to which both RGB images are resized for analysis.
        grid_sizes: Point-grid sizes passed to the reference multiscale proposer.
        reference_root: Optional directory containing ``efficient_sam`` and
            ``src``. Defaults to this candidate's
            ``third_party/updategs_mask_reference``; can be overridden with
            ``UPDATEGS_MASK_REFERENCE_ROOT``.
        proposal_batch_size: Number of point prompts per model batch.
    """

    def __init__(self, checkpoint_path, device="cuda", analysis_width=640,
                 grid_sizes=(8,), *, reference_root=None, proposal_batch_size=8,
                 alignment_radius=2, refine_unknown=True,
                 fuse_h0_proposals=True):
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self.device = str(device)
        self.analysis_width = int(analysis_width)
        self.grid_sizes = tuple(int(size) for size in grid_sizes)
        self.proposal_batch_size = int(proposal_batch_size)
        self.alignment_radius = int(alignment_radius)
        self.refine_unknown = bool(refine_unknown)
        self.fuse_h0_proposals = bool(fuse_h0_proposals)
        if self.analysis_width < 1 or not self.grid_sizes or any(x < 1 for x in self.grid_sizes):
            raise ValueError("analysis_width and grid_sizes must be positive")
        if self.proposal_batch_size < 1 or self.alignment_radius < 0:
            raise ValueError("proposal_batch_size must be positive and alignment_radius non-negative")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError("EfficientSAM checkpoint does not exist: {}".format(self.checkpoint_path))

        root = (reference_root or os.environ.get("UPDATEGS_MASK_REFERENCE_ROOT") or
                (Path(__file__).resolve().parents[1] / "third_party" / "updategs_mask_reference"))
        self.reference_root = Path(root).expanduser().resolve()
        if not (self.reference_root / "efficient_sam").is_dir() or not (self.reference_root / "src").is_dir():
            raise FileNotFoundError("reference_root must contain efficient_sam/ and src/: {}".format(self.reference_root))

        self._prepend_path(self.reference_root)
        try:
            import torch
            from efficient_sam.efficient_sam import build_efficient_sam
            self._torch = torch
            self._proposals = importlib.import_module("src.actualization_v1.scar_mask_proposals")
            self._local = importlib.import_module("src.actualization_v1.local_change_evidence")
            self._ssim = importlib.import_module("src.actualization_v1.ssim_evidence")
            from scipy.ndimage import distance_transform_edt
            from utils.pixel_masks import pixel_confidence_mask
            self._distance_transform_edt = distance_transform_edt
            self._pixel_confidence_mask = pixel_confidence_mask
        except ImportError as exc:
            message = "InitialEvidenceMask requires PyTorch and the reference modules from {!s}.".format(
                self.reference_root)
            if "scipy" in str(exc).lower():
                message += (" SciPy is required by local_change_evidence; install a build compatible "
                            "with the active Python/NumPy environment or add its package directory to PYTHONPATH.")
            else:
                message += " Verify torch is installed and reference_root contains the expected source modules."
            raise ImportError(message) from exc
        try:
            self.model = build_efficient_sam(192, 3, checkpoint=str(self.checkpoint_path))
            self.model.to(self.device).eval()
        except Exception as exc:
            raise RuntimeError("failed to load EfficientSAM checkpoint {!s} on {!r}: {}".format(
                self.checkpoint_path, self.device, exc)) from exc

    @staticmethod
    def _prepend_path(path):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)

    def _rgb(self, image):
        torch = self._torch
        if not torch.is_tensor(image) or image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("inputs must be CxHxW torch tensors with C=3")
        image = image.detach().to(device="cpu", dtype=torch.float32).numpy()
        if not np.isfinite(image).all():
            raise ValueError("inputs must contain only finite values")
        return np.clip(np.moveaxis(image, 0, -1), 0., 1.)

    @staticmethod
    def _merge_supported_pre_proposals(change, raw_unknown, refined_unknown, pre_masks):
        """Conservatively extend change from H0 proposals with strong raw support."""
        change = np.asarray(change, dtype=bool)
        raw_unknown = np.asarray(raw_unknown, dtype=bool)
        refined_unknown = np.asarray(refined_unknown, dtype=bool)
        masks = np.asarray(pre_masks, dtype=bool)
        if masks.ndim != 3 or masks.shape[1:] != change.shape:
            raise ValueError("pre_masks must have shape (N,H,W) matching the evidence grid")
        area_total = float(change.size)
        added = np.zeros_like(change)
        accepted = 0
        audit = []
        for index, proposal in enumerate(masks):
            area = int(proposal.sum())
            fraction = area / area_total
            if area == 0:
                audit.append({"index": index, "accepted": False, "reason": "empty"})
                continue
            change_fraction = float(np.count_nonzero(proposal & change) / float(area))
            unknown_fraction = float(np.count_nonzero(proposal & raw_unknown) / float(area))
            reasons = []
            if fraction < 0.003 or fraction > 0.10:
                reasons.append("area")
            if change_fraction < 0.25:
                reasons.append("change_support")
            if change_fraction - unknown_fraction < 0.10:
                reasons.append("change_unknown_margin")
            keep = not reasons
            audit.append({"index": index, "accepted": keep, "area_fraction": fraction,
                          "raw_change_fraction": change_fraction,
                          "raw_unknown_fraction": unknown_fraction,
                          "rejection_reasons": reasons})
            if keep:
                accepted += 1
                # Existing raw change and all (refined) unknown remain rejected.
                # Only currently trusted pixels can be newly marked as change.
                added |= proposal & ~change & ~refined_unknown
        return change | added, added, accepted, audit

    def __call__(self, rendered, target, return_debug=False):
        """Return HxW uint8, where one means trusted and zero means change/unknown.

        Public argument order stays ``(rendered, target)`` for BU integration.
        Internally, target is historical H0 (before) and rendered is the
        adapted BU comparison image (after). With no historical alpha, every
        source pixel is provisionally valid.
        """
        torch = self._torch
        before_rgb, after_rgb = self._rgb(target), self._rgb(rendered)
        if before_rgb.shape != after_rgb.shape:
            raise ValueError("rendered and target must have matching HxW")
        h, w = before_rgb.shape[:2]
        analysis_width = min(self.analysis_width, w)
        ah = max(1, int(round(h * analysis_width / float(w))))
        before_t = torch.from_numpy(np.ascontiguousarray(np.moveaxis(before_rgb, -1, 0)))[None]
        after_t = torch.from_numpy(np.ascontiguousarray(np.moveaxis(after_rgb, -1, 0)))[None]
        before_t = torch.nn.functional.interpolate(before_t, size=(ah, analysis_width), mode="bilinear", align_corners=False)
        after_t = torch.nn.functional.interpolate(after_t, size=(ah, analysis_width), mode="bilinear", align_corners=False)
        before = np.moveaxis(before_t[0].numpy(), 0, -1).astype(np.float32)
        after = np.moveaxis(after_t[0].numpy(), 0, -1).astype(np.float32)

        input_image = torch.from_numpy(np.ascontiguousarray(np.moveaxis(before, -1, 0)))[None].to(self.device)
        target_image = torch.from_numpy(np.ascontiguousarray(np.moveaxis(after, -1, 0)))[None].to(self.device)
        valid = np.ones((ah, analysis_width), dtype=bool)
        with torch.inference_mode():
            before_embedding = self.model.get_image_embeddings(input_image)[0].detach().cpu().numpy()
            after_embedding = self.model.get_image_embeddings(target_image)[0].detach().cpu().numpy()
        feature_map = 1. - np.clip(np.sum(
            before_embedding / np.maximum(np.linalg.norm(before_embedding, axis=0, keepdims=True), 1e-8) *
            (after_embedding / np.maximum(np.linalg.norm(after_embedding, axis=0, keepdims=True), 1e-8)), axis=0), -1., 1.)
        feature_map = torch.nn.functional.interpolate(
            torch.from_numpy(feature_map)[None, None], size=(ah, analysis_width),
            mode="bilinear", align_corners=False)[0, 0].numpy()
        feature_threshold = float(np.median(feature_map) + 3. * np.median(np.abs(feature_map - np.median(feature_map))))
        feature_candidate = valid & (feature_map >= max(1e-6, feature_threshold))
        ssim_result = self._ssim.ssim_evidence([before], [after], valid_masks=[valid])
        ssim_candidate = np.asarray(ssim_result["candidate_masks"], dtype=bool)
        if ssim_candidate.ndim == 3:
            ssim_candidate = ssim_candidate[0]

        # H0 is the historical GT target; BU render is the comparison image.
        proposal_domain = valid.copy()
        if self.fuse_h0_proposals:
            pre_masks = self._proposals.predict_multiscale(
                self.model, torch, input_image, proposal_domain, grid_sizes=self.grid_sizes,
                batch_size=self.proposal_batch_size)[0]
        else:
            pre_masks = np.empty((0, ah, analysis_width), dtype=bool)
        # H0 pre proposals are fused below. The reference classifier does not
        # use post proposals, so skip their otherwise redundant inference.
        post_masks = np.empty((0, ah, analysis_width), dtype=bool)
        pre_masks = np.asarray(pre_masks, dtype=bool)
        post_masks = np.asarray(post_masks, dtype=bool)
        # The reference implementation does not use proposal pixels; H0
        # proposals are fused explicitly below under a conservative rule.
        local = self._local.build_local_change_evidence(
            before, after, feature_candidate, ssim_candidate, valid,
            pre_masks=pre_masks, post_masks=post_masks,
            alignment_radius=self.alignment_radius, external_pose_verified=False)
        unknown = np.asarray(local["unknown_mask"], dtype=bool)
        change = np.asarray(local["change_mask"], dtype=bool)
        refined_unknown = unknown.copy()
        confidence_score = None
        if self.refine_unknown:
            # Project-specific recovery heuristic: retain raw unknown only when
            # photometric confidence is low or it lies close to raw change.
            # Similar-looking distant real changes can be missed by this rule.
            confidence_render = rendered.detach().to(device=self.device, dtype=torch.float32)
            confidence_target = target.detach().to(device=self.device, dtype=torch.float32)
            _, confidence_full = self._pixel_confidence_mask(
                confidence_target, confidence_render, ssim_threshold=0.85,
                analysis_width=analysis_width, return_score=True)
            confidence_score = torch.nn.functional.interpolate(
                confidence_full[None, None], size=(ah, analysis_width),
                mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
            near_change = np.zeros_like(change)
            if change.any():
                near_change = self._distance_transform_edt(~change) <= 2.0
            refined_unknown = unknown & ((confidence_score < 0.85) | near_change)
        fused_change = change.copy()
        proposal_added = np.zeros_like(change)
        accepted_proposal_count = 0
        proposal_audit = []
        if self.fuse_h0_proposals:
            fused_change, proposal_added, accepted_proposal_count, proposal_audit = (
                self._merge_supported_pre_proposals(change, unknown, refined_unknown, pre_masks))
        analysis_mask = (valid & ~fused_change & ~refined_unknown).astype(np.uint8)
        mask_t = torch.from_numpy(analysis_mask)[None, None].float()
        mask = torch.nn.functional.interpolate(mask_t, size=(h, w), mode="nearest")[0, 0].to(
            device=self.device, dtype=torch.uint8)
        if not return_debug:
            return mask
        labels = np.zeros((ah, analysis_width), dtype=np.uint8)
        labels[refined_unknown] = 1
        labels[fused_change] = 2
        labels_t = torch.from_numpy(labels)[None, None].float()
        labels_full = torch.nn.functional.interpolate(labels_t, size=(h, w), mode="nearest")[0, 0].to(
            device=self.device, dtype=torch.uint8)
        raw_unknown_t = torch.from_numpy(unknown.astype(np.uint8))[None, None].float()
        raw_unknown_full = torch.nn.functional.interpolate(
            raw_unknown_t, size=(h, w), mode="nearest")[0, 0].to(device=self.device, dtype=torch.uint8)
        refined_unknown_t = torch.from_numpy(refined_unknown.astype(np.uint8))[None, None].float()
        refined_unknown_full = torch.nn.functional.interpolate(
            refined_unknown_t, size=(h, w), mode="nearest")[0, 0].to(device=self.device, dtype=torch.uint8)
        proposal_added_t = torch.from_numpy(proposal_added.astype(np.uint8))[None, None].float()
        proposal_added_full = torch.nn.functional.interpolate(
            proposal_added_t, size=(h, w), mode="nearest")[0, 0].to(device=self.device, dtype=torch.uint8)
        residual = torch.nn.functional.interpolate(
            torch.from_numpy(np.asarray(local["residual_map"], np.float32))[None, None],
            size=(h, w), mode="bilinear", align_corners=False)[0, 0].to(self.device)
        debug = {
            "class_labels": labels_full,
            "raw_unknown": raw_unknown_full,
            "refined_unknown": refined_unknown_full,
            "score": residual,
            "analysis_hw": (ah, analysis_width),
            "valid_policy": "all pixels valid: historical alpha unavailable",
            "valid_mask": torch.ones((h, w), dtype=torch.uint8, device=self.device),
            "external_pose_verified": False,
            "sam_pre_masks": pre_masks,
            "sam_post_masks": post_masks,
            "sam_pre_count": int(len(pre_masks)),
            "sam_post_count": int(len(post_masks)),
            "sam_mask_influence": ("explicit_conservative_h0_pre_proposal_fusion" if self.fuse_h0_proposals
                                    else "disabled"),
            "h0_proposal_fusion_enabled": self.fuse_h0_proposals,
            "h0_proposals_accepted": accepted_proposal_count,
            "h0_proposal_added_pixel_count_analysis": int(proposal_added.sum()),
            "h0_proposal_added_pixels": proposal_added_full,
            "h0_proposal_audit": proposal_audit,
            "post_proposals_generated": False,
            "h0_proposal_fusion_policy": ("project-specific: proposal area 0.3%-10%, raw change support >=25%, "
                                           "raw change fraction exceeds raw unknown by >=10 percentage points; "
                                           "raw change and refined unknown retain priority; not semantic ground truth"),
            "unknown_refinement_enabled": self.refine_unknown,
            "unknown_refinement": ("project_specific: raw_unknown AND (combined_pixel_confidence < 0.85 "
                                    "OR distance_to_raw_change <= 2 analysis pixels); may miss "
                                    "high_similarity changes far from raw change"),
            "evidence": local["diagnostics"],
            "semantic_labels": False,
            "physical_removal_proof": False,
        }
        return mask, debug
