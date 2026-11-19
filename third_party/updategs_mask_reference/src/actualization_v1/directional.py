"""CPU directional evidence helpers; outputs are candidates, never delete decisions."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, zoom


def _finite_array(x, name, ndim=None):
    a = np.asarray(x, dtype=float)
    if ndim is not None and a.ndim != ndim: raise ValueError(f"{name} must have ndim={ndim}")
    if not np.isfinite(a).all(): raise ValueError(f"{name} must be finite")
    return a


def _resize(field, hw):
    if field.shape[-2:] == tuple(hw): return field
    factors = [1.] * (field.ndim - 2) + [hw[0] / field.shape[-2], hw[1] / field.shape[-1]]
    return zoom(field, factors, order=1, mode="nearest", prefilter=False)


def _thresholds(values, dynamic, left_default=-.17, right_default=.28):
    if not dynamic: return left_default, right_default, {"mode": "fixed", "fallback": False, "peaks": [], "histogram_source": "signed_map", "sigma": 1., "actual_used": [left_default, right_default]}
    x = np.clip(values[np.isfinite(values)], -1., 1.)
    fallback={"mode": "fallback", "fallback": True, "reason": "insufficient_or_constant_signed_values", "peaks": [], "histogram_source": "signed_map", "sigma": 1., "actual_used": [left_default, right_default]}
    if x.size < 8 or np.ptp(x) < 1e-8: return left_default, right_default, fallback
    hist, edges = np.histogram(x, bins=50, range=(-1., 1.)); smooth = gaussian_filter1d(hist.astype(float), 1.)
    peaks = np.flatnonzero((smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] >= smooth[2:])) + 1
    neg = peaks[edges[peaks] < 0]; pos = peaks[edges[peaks] > 0]; zero=int(np.searchsorted(edges,0.))
    # SCAR side selection is by the highest smoothed peak amplitude on each
    # side, never by nearest-bin selection.  A strong zero-mode is accepted as
    # the opposing mode only when no positive signed peak exists.
    def strongest(indices):
        return int(indices[np.argmax(smooth[indices])]) if len(indices) else None
    if not len(neg):
        neg=np.asarray([int(np.argmax(np.where(edges[:-1] < 0, smooth, -np.inf)))])
    # A zero/unchanged mode is the second peak for the common one-sided
    # SCAR case (for example values near -0.4 and 0.0).
    if not len(pos) and smooth[zero] > 0:
        pos=np.asarray([zero], dtype=int)
    if not len(neg) or not len(pos):
        fallback["reason"]="missing_signed_peak"; fallback["peaks"]=[float((edges[i]+edges[i+1])/2.) for i in peaks]
        return left_default, right_default, fallback
    npk=strongest(neg); ppk=strongest(pos)
    if ppk is None: ppk=zero
    lo,hi=min(npk,zero),max(npk,zero); left=float((edges[int(np.argmin(smooth[lo:hi+1])+lo)]+edges[int(np.argmin(smooth[lo:hi+1])+lo)+1])/2.)
    lo,hi=min(zero,ppk),max(zero,ppk); right=float((edges[int(np.argmin(smooth[lo:hi+1])+lo)]+edges[int(np.argmin(smooth[lo:hi+1])+lo)+1])/2.)
    if not (np.isfinite(left) and np.isfinite(right) and left<=0<=right):
        fallback["reason"]="invalid_signed_valley"; return left_default,right_default,fallback
    report={"mode":"dynamic_valley","fallback":False,"histogram_bins":50,"sigma":1.,"histogram_source":"signed_map","peaks":[float((edges[i]+edges[i+1])/2.) for i in peaks],"actual_used":[left,right]}
    return left,right,report


def signed_feature_masks(pre_embeddings, post_embeddings, image_sizes, valid_masks=None,
                         left_threshold=-.17, right_threshold=.28, dynamic=True,
                         pca_samples_per_view=4096):
    """Fit one joint pre/post PCA and return anonymous signed directional masks."""
    pre = _finite_array(pre_embeddings, "pre_embeddings", 4); post = _finite_array(post_embeddings, "post_embeddings", 4)
    if pre.shape != post.shape: raise ValueError("pre_embeddings and post_embeddings must match")
    v, c, h, w = pre.shape
    if len(image_sizes) != v: raise ValueError("image_sizes must contain one (H,W) per view")
    sizes = [tuple(map(int, s)) for s in image_sizes]
    if any(len(s) != 2 or min(s) <= 0 for s in sizes): raise ValueError("invalid image_sizes")
    if pca_samples_per_view < 1: raise ValueError("pca_samples_per_view must be positive")
    valid = np.ones((v, h, w), bool) if valid_masks is None else np.asarray(valid_masks, bool)
    if valid.shape != (v, h, w): raise ValueError("valid_masks must have shape (V,H,W)")
    samples=[]
    for i in range(v):
        flat = np.concatenate([pre[i].transpose(1,2,0).reshape(-1,c), post[i].transpose(1,2,0).reshape(-1,c)], axis=0)
        ids = np.linspace(0, len(flat)-1, min(pca_samples_per_view, len(flat)), dtype=int)
        samples.append(flat[ids])
    all_samples=np.concatenate(samples, axis=0); mean=all_samples.mean(0); centered=all_samples-mean
    cov=centered.T @ centered / max(len(centered)-1, 1); eigval,eigvec=np.linalg.eigh(cov); pc=eigvec[:,-1]
    pc[np.argmax(np.abs(pc))] *= 1 if pc[np.argmax(np.abs(pc))] >= 0 else -1
    variance=float(max(eigval[-1],0)); explained=float(variance / max(float(eigval.sum()), 1e-12))
    signed=[]; neg=[]; pos=[]; resize_reports=[]
    for i,(H,W) in enumerate(sizes):
        s=(pre[i].transpose(1,2,0)-post[i].transpose(1,2,0)-mean*0) @ pc
        s=_resize(s[None,...], (H,W))[0]; vm=_resize(valid[i][None,None].astype(float), (H,W))[0,0] >= .5; s=np.where(vm,s,0.)
        signed.append(s); resize_reports.append([H,W]); neg.append(vm & (s < left_threshold)); pos.append(vm & (s > right_threshold))
    sm=np.asarray(signed); threshold_reports=[]; lows=[]; highs=[]
    for s in sm:
        lo,hi,rep=_thresholds(s,dynamic,left_threshold,right_threshold); lows.append(lo); highs.append(hi); threshold_reports.append(rep)
    neg=np.asarray([(_resize(valid[i][None,None].astype(float),sizes[i])[0,0]>=.5)&(sm[i] < lows[i]) for i in range(v)])
    pos=np.asarray([(_resize(valid[i][None,None].astype(float),sizes[i])[0,0]>=.5)&(sm[i] > highs[i]) for i in range(v)])
    return {"signed_maps": sm, "negative_masks": np.asarray(neg,bool), "positive_masks": np.asarray(pos,bool),
            "pca_component": pc, "explained_variance": explained, "thresholds": {"negative": np.asarray(lows), "positive": np.asarray(highs)},
            "report": {"joint_pre_post": True, "sample_count": int(len(all_samples)), "resize_image_sizes": resize_reports, "thresholds_per_view": threshold_reports, "threshold_mode": "per_view"}}


def trace_direction(contribution, visibility, center_inside, center_valid, occluded=None,
                    min_views=2, min_weight=1e-4, vote_threshold=.1, prune_tau=.4):
    """AlphaT/visibility-aware auxiliary trace; center votes are never synthesized."""
    a=_finite_array(contribution,"contribution",2); vis=_finite_array(visibility,"visibility",2)
    if a.shape != vis.shape: raise ValueError("contribution and visibility shapes must match")
    if (a<0).any() or (vis<0).any() or np.any(a>vis+1e-6): raise ValueError("contribution must be <= visibility")
    v,n=a.shape; inside=np.asarray(center_inside,bool); valid=np.asarray(center_valid,bool)
    if inside.shape != (v,n) or valid.shape != (v,n): raise ValueError("center masks must be (V,N)")
    occ=np.zeros((v,n),bool) if occluded is None else np.asarray(occluded,bool)
    if occ.shape != (v,n): raise ValueError("occluded must be (V,N)")
    seen=(vis>=min_weight)&~occ; valid_seen=seen&valid; seen_count=valid_seen.sum(0)
    center_seen=valid_seen; center_count=center_seen.sum(0); outside_count=(center_seen & ~inside).sum(0)
    ratios=np.divide(a, np.maximum(vis,min_weight), out=np.zeros_like(a), where=vis>=min_weight)
    vote=np.divide((ratios*valid_seen).sum(0), np.maximum(valid_seen.sum(0),1), where=valid_seen.sum(0)>0, out=np.zeros(n))
    raw_candidate=(seen_count>=min_views)&(vote>=vote_threshold)
    outside_ratio=outside_count/np.maximum(center_count,1); candidate=raw_candidate&(center_count>=min_views)
    kept=candidate&(outside_ratio<=prune_tau); retention=float(kept.sum()/max(candidate.sum(),1))
    observed_fraction=valid_seen.mean(axis=0); occlusion_fraction=occ.mean(axis=0)
    return {"candidate":candidate,"candidate_mask":candidate,"kept":kept,"kept_mask":kept,
            "retention_rate":retention,"vote":vote,"seen_count":seen_count,"valid_seen_count":seen_count,
            "center_count":center_count,"outside_count":outside_count,"outside_ratio":outside_ratio,
            "raw_candidate":raw_candidate,"raw_candidate_count":int(raw_candidate.sum()),
            "classifiable_count":int(candidate.sum()),"observed_fraction":observed_fraction,
            "occlusion_fraction":occlusion_fraction,"unknown_center":raw_candidate & (center_count<min_views),"center_unknown":~valid|occ,
            "report":{"alphaT_auxiliary":True,"center_vote_used":False,"oob_not_counterevidence":True,
                      "visibility_fraction_is_contribution_over_visibility":True,"pixel_occlusion_veto":True}}


def classify_directions(negative_trace, positive_trace, min_retention=0., min_margin=.1,
                        min_candidates=20, criterion="relative_retention"):
    """Classify anonymous directional candidates; never emits an automatic delete."""
    if criterion not in ("relative_retention", "absolute_retention"):
        raise ValueError("criterion must be relative_retention or absolute_retention")
    def read(x):
        candidate=np.asarray(x.get("candidate",x.get("candidate_mask")),bool)
        kept_value=x.get("kept",x.get("kept_mask",None))
        kept=np.asarray(candidate if kept_value is None and float(x.get("retention_rate",0.))>0 else (np.zeros_like(candidate) if kept_value is None else kept_value),bool)
        return int(candidate.sum()), int(kept.sum()), float(x.get("retention_rate",0.))
    nc,nk,nr=read(negative_trace); pc,pk,pr=read(positive_trace); gap=nr-pr
    neg_eligible=nc>=min_candidates and nk>0 and (criterion=="relative_retention" or nr>=min_retention)
    pos_eligible=pc>=min_candidates and pk>0 and (criterion=="relative_retention" or pr>=min_retention)
    available = (["negative"] if nc >= min_candidates else []) + (["positive"] if pc >= min_candidates else [])
    high_gap = nr - pr
    if nc < min_candidates and pc < min_candidates: status,reason="ambiguous","no_candidates"
    elif neg_eligible and not pos_eligible:
        status,reason="pre_candidate_requires_segmentation","one_direction_requires_segmentation_validation"
    elif pos_eligible and not neg_eligible:
        status,reason="pre_candidate_requires_segmentation","one_direction_requires_segmentation_validation"
    elif neg_eligible and pos_eligible and abs(high_gap)<min_margin: status,reason="ambiguous_retention_requires_segmentation","both_high_close"
    elif neg_eligible and high_gap>=min_margin: status,reason="candidate","negative_relative_retention_margin"
    elif pos_eligible and -high_gap>=min_margin: status,reason="candidate","positive_relative_retention_margin"
    else: status,reason="ambiguous","retention_or_margin_insufficient"
    primary = available[0] if len(available)==1 else ("negative" if high_gap > 0 else "positive" if high_gap < 0 else None)
    return {"old_direction": None, "new_direction": None,
            "old_direction_candidate": None, "new_direction_candidate": None,
            "old_candidates": ["negative", "positive"] if status == "ambiguous_retention_requires_segmentation" else available,
            "candidate_directions": available, "primary_candidate": primary,
            "high_gap": float(high_gap),
            "negative": {"candidate_count":nc,"kept_count":nk,"retention_rate":nr,"eligible":neg_eligible},
            "positive": {"candidate_count":pc,"kept_count":pk,"retention_rate":pr,"eligible":pos_eligible},
            "status":status,"reason":reason,"automatic_delete":False,
            "requires_validated_segmentation": True,
            "criteria":{"criterion":criterion,"min_retention":float(min_retention),"min_margin":float(min_margin),"min_candidates":int(min_candidates),"relative_retention":criterion=="relative_retention","absolute_optional":criterion=="absolute_retention"},
            "raw_n_kept":{"negative":nk,"positive":pk},"raw_n_voted":{"negative":nc,"positive":pc}}
