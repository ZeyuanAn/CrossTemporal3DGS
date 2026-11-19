"""Fresh EfficientSAM RGB-pair features plus anonymous signed PCA proxies.

This is 2D diagnostic input only. The negative/positive maps are PCA-signed
proxies, not calibrated 3D traces and not edit-label directions.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

from .directional import signed_feature_masks
from .semantic import feature_difference


def channel_cosine_residual(pre_embedding: np.ndarray, post_embedding: np.ndarray) -> np.ndarray:
    """Unsigned per-pixel channel residual; zero vectors compare as zero residual."""
    pre=np.asarray(pre_embedding,np.float32); post=np.asarray(post_embedding,np.float32)
    if pre.shape!=post.shape or pre.ndim!=3: raise ValueError("embeddings must match CxHfxW")
    pn=np.linalg.norm(pre,axis=0); qn=np.linalg.norm(post,axis=0); dot=np.sum(pre*post,axis=0)
    residual=np.zeros(pn.shape,np.float32); both=(pn>0)&(qn>0)
    residual[both]=np.clip(1.-dot[both]/(pn[both]*qn[both]),0.,2.)
    return residual


def joint_residual_threshold(residual_maps: np.ndarray, valid_masks: np.ndarray | None = None) -> tuple[float,dict[str,object]]:
    values=np.asarray(residual_maps,np.float32); valid=np.ones(values.shape,dtype=bool) if valid_masks is None else np.asarray(valid_masks,bool)
    x=values[np.isfinite(values)&valid]; x=np.clip(x,0.,2.)
    if not len(x): return 1e-6,{"count":0,"median":0.,"mad":0.,"otsu":0.,"threshold":1e-6,"floor":1e-6}
    hist,edges=np.histogram(x,bins=256,range=(0.,2.)); centers=(edges[:-1]+edges[1:])/2.; weight=np.cumsum(hist); total=weight[-1]; sums=np.cumsum(hist*centers); mean=sums[-1]/max(total,1); denom=weight*(total-weight); between=(sums-mean*weight)**2/np.maximum(denom,1e-12); otsu=float(centers[int(np.argmax(between))]); med=float(np.median(x)); mad=float(np.median(np.abs(x-med))); threshold=max(1e-6,otsu,med+3.*mad)
    return threshold,{"count":int(len(x)),"median":med,"mad":mad,"otsu":otsu,"threshold":threshold,"floor":1e-6}


def multichannel_evidence(pre_embeddings: np.ndarray, post_embeddings: np.ndarray,
                          image_sizes, valid_masks=None) -> dict[str,object]:
    """Joint-view unsigned 2D diagnostic evidence; does not alter PCA/sign baseline."""
    from scipy.ndimage import zoom
    pre=np.asarray(pre_embeddings,np.float32); post=np.asarray(post_embeddings,np.float32)
    if pre.shape!=post.shape or pre.ndim!=4: raise ValueError("embeddings must be VxCxHfxW and match")
    v,_,hf,wf=pre.shape; valid=np.ones((v,hf,wf),bool) if valid_masks is None else np.asarray(valid_masks,bool)
    if valid.shape!=(v,hf,wf): raise ValueError("valid_masks must match feature grid")
    base=np.asarray([channel_cosine_residual(pre[i],post[i]) for i in range(v)])
    threshold,stats=joint_residual_threshold(base,valid); maps=[]; masks=[]
    for i,(h,w) in enumerate(image_sizes):
        r=zoom(base[i],(h/hf,w/wf),order=1,mode='nearest',prefilter=False); vm=zoom(valid[i].astype(float),(h/hf,w/wf),order=0,mode='nearest')>=.5; maps.append(r.astype(np.float32)); masks.append(vm&(r>=threshold))
    return {"residual_feature_grid":base,"residual_maps":maps,"multidim_candidate_mask":np.asarray(masks,bool),"threshold":threshold,"threshold_report":stats,"physical_direction_claim":False,"edit_label_inference":False,"method":"joint_multiview_unsigned_channel_cosine_residual"}


def signed_pair_diagnostics(pre_embeddings: np.ndarray, post_embeddings: np.ndarray,
                            image_sizes, valid_masks=None) -> dict[str, object]:
    """Run shared pre/post PCA and return both anonymous directional proxies."""
    signed = signed_feature_masks(pre_embeddings, post_embeddings, image_sizes, valid_masks=valid_masks)
    return {"signed_maps": signed["signed_maps"], "negative_masks": signed["negative_masks"],
            "positive_masks": signed["positive_masks"], "pca_component": signed["pca_component"],
            "report": signed["report"], "thresholds": signed["thresholds"],
            "diagnostic_trace_proxies": ("negative", "positive"),
            "physical_direction_claim": False, "edit_label_inference": False}


def extract_rgb_pair_features(pairs, checkpoint: str | Path = ".cache/efficientsam/efficient_sam_vitt.pt",
                              device: str = "cpu", valid_masks=None) -> dict[str, object]:
    """Fresh-extract paired RGB embeddings and shared-PCA masks for multiple views.

    ``pairs`` is an iterable of ``(view_id, pre_rgb_path, post_rgb_path)`` tuples.
    The local EfficientSAM model is loaded once; each pair uses the same model and
    preprocessing. No P06 cached embedding is read.
    """
    from ...scripts.actualization_v1.extract_render_features import fresh_pair_embeddings, load_fresh_model
    checkpoint = Path(checkpoint).resolve(); model, torch = load_fresh_model(checkpoint, device)
    pre_embeddings=[]; post_embeddings=[]; sizes=[]; records=[]; feature_maps=[]
    for view_id, pre_path, post_path in pairs:
        pre, post, self_embedding, hw = fresh_pair_embeddings(model, torch, Path(pre_path), Path(post_path))
        pre_embeddings.append(pre); post_embeddings.append(post); sizes.append(hw)
        diff = feature_difference(pre, post, hw[0], hw[1])["difference"]
        feature_maps.append(diff)
        records.append({"view_id": view_id, "pre_rgb": str(Path(pre_path).resolve()), "post_rgb": str(Path(post_path).resolve()), "pre_sha256": hashlib.sha256(Path(pre_path).read_bytes()).hexdigest(), "post_sha256": hashlib.sha256(Path(post_path).read_bytes()).hexdigest(), "image_hw": list(hw), "embedding_shape": list(pre.shape), "self_compare_max_abs": float(np.max(np.abs(pre-self_embedding)))})
    pre_array=np.asarray(pre_embeddings); post_array=np.asarray(post_embeddings)
    valid_masks_supplied = valid_masks is not None
    if valid_masks is None: valid_masks=np.ones((len(sizes), pre_array.shape[-2], pre_array.shape[-1]), dtype=bool)
    signed=signed_pair_diagnostics(pre_array, post_array, sizes, valid_masks)
    return {"pre_embeddings":pre_array, "post_embeddings":post_array, "feature_difference":feature_maps, "signed_maps":signed["signed_maps"], "negative_masks":signed["negative_masks"], "positive_masks":signed["positive_masks"], "diagnostic_trace_proxies":signed["diagnostic_trace_proxies"], "physical_direction_claim":False, "edit_label_inference":False, "provenance":{"checkpoint":str(checkpoint),"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),"model_rerun":True,"records":records,"valid_masks_supplied":valid_masks_supplied,"no_gt_labels":True}}
