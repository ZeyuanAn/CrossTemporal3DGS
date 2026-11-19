"""Optional SCaR-style multiscale 2D EfficientSAM proposal generation."""
from __future__ import annotations
import numpy as np
from .exact_mask_overlap import ExactMaskOverlap


def grid_points(domain: np.ndarray, grid_sizes=(16, 32)) -> np.ndarray:
    domain = np.asarray(domain).astype(bool); ys, xs = np.where(domain)
    if not len(xs): return np.empty((0, 2), dtype=np.float32)
    points=[]
    for size in grid_sizes:
        for gy in np.linspace(0, domain.shape[0]-1, size).round().astype(int):
            for gx in np.linspace(0, domain.shape[1]-1, size).round().astype(int):
                if domain[gy, gx]: points.append((gx, gy))
    return np.unique(np.asarray(points, dtype=np.float32), axis=0)


def mask_stability(logits: np.ndarray, delta: float = 0.05) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    return (np.logical_and(logits > -delta, logits > delta).sum(axis=(-2, -1)) /
            np.maximum((logits > -delta).sum(axis=(-2, -1)), 1)).astype(np.float32)


def mask_nms(masks: np.ndarray, scores: np.ndarray, iou_threshold: float = .7) -> np.ndarray:
    masks=np.asarray(masks).astype(bool); scores=np.asarray(scores,float); order=np.argsort(-scores); keep=[]; overlap=ExactMaskOverlap(masks)
    for idx in order:
        if all(overlap.iou(int(idx), j) < iou_threshold for j in keep): keep.append(int(idx))
    return np.asarray(keep,dtype=np.int64)


def filter_proposals(masks, scores, stability, min_score=.7, min_stability=.9, min_area=100, nms_iou=.7):
    masks=np.asarray(masks).astype(bool); scores=np.asarray(scores,float); stability=np.asarray(stability,float)
    keep=(scores>=min_score)&(stability>=min_stability)&(masks.reshape(len(masks),-1).sum(1)>=min_area)
    indices=np.flatnonzero(keep); nms=mask_nms(masks[indices],scores[indices],nms_iou) if len(indices) else np.empty(0,dtype=np.int64)
    selected=indices[nms] if len(indices) else indices
    suppressed=[]
    # Record the actual highest-scoring selected overlap suppressor without changing NMS output.
    selected_set=set(selected.tolist())
    for idx in indices:
        if idx in selected_set: continue
        overlap=ExactMaskOverlap(masks[[idx]+selected.tolist()]); overlaps=[j for pos,j in enumerate(selected,1) if overlap.iou(0,pos) >= nms_iou]
        suppressed.append({"original_index":int(idx),"suppressed_by_original_index":int(overlaps[0]) if overlaps else None})
    reasons=[]
    areas=masks.reshape(len(masks),-1).sum(1)
    for i in range(len(masks)):
        reason=[]
        if scores[i]<min_score: reason.append('score')
        if stability[i]<min_stability: reason.append('stability')
        if areas[i]<min_area: reason.append('small_area')
        if i not in selected_set and not reason: reason.append('nms')
        reasons.append(reason)
    counts={"input":int(len(masks)),"score_rejected":int(np.sum(scores<min_score)),"stability_rejected":int(np.sum(stability<min_stability)),"small_rejected":int(np.sum(areas<min_area)),"after_nms":int(len(selected)),"selected_original_indices":selected.tolist(),"suppressed_original_indices":suppressed,"proposal_filter_records":[{"original_index":int(i),"score":float(scores[i]),"stability":float(stability[i]),"area":int(areas[i]),"score_pass":bool(scores[i]>=min_score),"stability_pass":bool(stability[i]>=min_stability),"area_pass":bool(areas[i]>=min_area),"reject_reasons":reasons[i]} for i in range(len(masks))]}
    return masks[selected], scores[selected], stability[selected], counts


def predict_multiscale(model, torch, image, domain, grid_sizes=(16,32), batch_size=8, min_score=.7, min_stability=.9, min_area=100, nms_iou=.7, return_diagnostics=False, return_audit_masks=False):
    points=grid_points(domain,grid_sizes); device=next(model.parameters()).device
    all_masks=[]; all_scores=[]; all_stability=[]; all_points=[]; embedding=model.get_image_embeddings(image.to(device))
    for start in range(0,len(points),batch_size):
        chunk=points[start:start+batch_size]; q=len(chunk); pt=torch.full((1,q,3,2),-1.,device=device); pt[0,:,0]=torch.from_numpy(chunk).to(device); labels=torch.full((1,q,3),-1,dtype=torch.int64,device=device); labels[0,:,0]=1
        with torch.inference_mode(): logits,scores=model.predict_masks(embedding,pt,labels,True,image.shape[-2],image.shape[-1],image.shape[-2],image.shape[-1])
        logits=logits.detach().cpu().numpy()[0]; scores=scores.detach().cpu().numpy()[0]; all_masks.append(logits>0); all_scores.append(scores); all_stability.append(mask_stability(logits)); all_points.extend([p for p in chunk for _ in range(3)])
    if not all_masks:
        empty=(np.empty((0,*domain.shape),bool),np.empty(0),np.empty(0),np.empty((0,2)),{"input":0,"after_nms":0})
        if not return_diagnostics: return empty
        result=empty + ({"grid_sizes":list(grid_sizes),"query_count":0,"raw_proposal_count":0},)
        return result + (np.empty((0,*domain.shape),bool),) if return_audit_masks else result
    masks=np.concatenate(all_masks,axis=0).reshape(-1,*domain.shape); scores=np.concatenate(all_scores,axis=0).reshape(-1); stability=np.concatenate(all_stability,axis=0).reshape(-1); points=np.asarray(all_points,np.float32)
    raw_scores=scores.copy(); raw_stability=stability.copy(); raw_areas=masks.reshape(len(masks),-1).sum(1).astype(np.int64); original_points=points.copy(); masks,scores,stability,counts=filter_proposals(masks,scores,stability,min_score,min_stability,min_area,nms_iou)
    selected=np.asarray(counts["selected_original_indices"],dtype=np.int64)
    filtered_points=original_points[selected]
    if not return_diagnostics: return masks,scores,stability,filtered_points,counts
    counts["grid_sizes"]=list(grid_sizes); counts["query_count"]=int(len(points)//3); counts["raw_proposal_count"]=int(len(points)); counts["raw_points_xy"]=[p.tolist() for p in points[::3]]
    records=[]; raw_masks=np.concatenate(all_masks,axis=0).reshape(-1,*domain.shape)
    for i in range(len(points)):
        ys,xs=np.where(raw_masks[i]); records.append({"original_index":int(i),"query_index":int(i//3),"proposal_index":int(i%3),"point_xy":points[i].tolist(),"score":float(raw_scores[i]),"stability":float(raw_stability[i]),"area":int(raw_areas[i]),"bbox_xyxy":None if not len(xs) else [int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())]})
    counts["raw_proposal_records"]=records
    return (masks,scores,stability,filtered_points,counts,raw_masks) if return_audit_masks else (masks,scores,stability,filtered_points,counts)


def match_whole_proposals(pre_masks, post_masks, valid_mask, minimum_iou=.5, minimum_area_ratio=.67):
    pre_masks=np.asarray(pre_masks).astype(bool); post_masks=np.asarray(post_masks).astype(bool); valid=np.asarray(valid_mask).astype(bool)
    records=[]; ambiguous=np.zeros_like(valid); pre_support=np.zeros_like(valid); post_support=np.zeros_like(valid); matched_post=set()
    pre_masks=pre_masks & valid; post_masks=post_masks & valid
    all_masks=np.concatenate([pre_masks,post_masks],axis=0); overlap=ExactMaskOverlap(all_masks)
    for pi, pre in enumerate(pre_masks):
        pa=int(pre.sum()); best=None
        for qi, post in enumerate(post_masks):
            qa=int(post.sum()); iou=float(overlap.iou(pi,len(pre_masks)+qi)); ratio=float(min(pa,qa)/max(pa,qa)) if max(pa,qa) else 0.
            row={"pre_index":pi,"post_index":qi,"direct_object_iou":iou,"area_ratio":ratio,"pre_area":pa,"post_area":qa}
            if best is None or (iou,ratio)>(best["direct_object_iou"],best["area_ratio"]): best=row
        if best is None: records.append({"pre_index":pi,"post_index":None,"direct_object_iou":None,"area_ratio":None,"pre_area":pa,"post_area":None,"role":"pre_object_difference_support"}); pre_support|=pre; continue
        same=best["direct_object_iou"]>=minimum_iou and best["area_ratio"]>=minimum_area_ratio; best["role"]="ambiguous" if same else "pre_object_difference_support"; records.append(best)
        if same: ambiguous|=pre; matched_post.add(best["post_index"])
        else: pre_support|=pre
    for qi, post in enumerate(post_masks):
        if qi not in matched_post:
            post_support |= post
            records.append({"pre_index":None,"post_index":int(qi),"direct_object_iou":None,"area_ratio":None,"pre_area":None,"post_area":int((post&valid).sum()),"role":"post_unmatched_object_support","physical_addition_proof":False})
    return {"records":records,"pre_object_support_mask":(np.any(pre_masks,axis=0)&valid),"pre_only_validated_mask":pre_support&~ambiguous,"ambiguous_mask":ambiguous,"post_support_mask":post_support,"unknown_mask":np.zeros_like(valid),"physical_addition_proof":False}


def filter_trace_supported(masks, trace_mask, minimum_support_fraction=.5, minimum_iou=.15, max_area_fraction=.5, proposal_validation='trace_iou'):
    masks=np.asarray(masks).astype(bool); trace=np.asarray(trace_mask).astype(bool); image_area=trace.size; selected=[]; records=[]
    for i, mask in enumerate(masks):
        area=int(mask.sum()); overlap=int((mask&trace).sum()); support=float(overlap/area) if area else 0.; union=int((mask|trace).sum()); iou=float(overlap/union) if union else 0.; area_fraction=float(area/image_area)
        reasons=[]
        if support < minimum_support_fraction: reasons.append('trace_support_fraction')
        if proposal_validation == 'trace_iou' and iou < minimum_iou: reasons.append('trace_iou')
        if proposal_validation not in ('trace_iou','support_precision'): raise ValueError('unknown proposal_validation mode')
        if proposal_validation == 'support_precision' and overlap < 100: reasons.append('support_pixels')
        if area_fraction > max_area_fraction: reasons.append('max_object_area_fraction')
        passed=not reasons
        if passed: selected.append(i)
        records.append({'proposal_index':int(i),'area':area,'area_fraction':area_fraction,'support_pixels':overlap,'support_fraction':support,'trace_iou':iou,'proposal_validation':proposal_validation,'pass':passed,'reject_reason':None if passed else reasons})
    return np.asarray(selected,dtype=np.int64), records


def filter_cached_proposals(masks, scores, stability, trace_mask, minimum_support_fraction=.5, minimum_iou=.15, max_area_fraction=.5, proposal_validation='trace_iou'):
    """Apply the exact live multiscale filter to cached proposals; never reruns a model."""
    selected, audit = filter_trace_supported(masks, trace_mask, minimum_support_fraction, minimum_iou, max_area_fraction, proposal_validation)
    return {"masks": np.asarray(masks)[selected], "scores": np.asarray(scores)[selected], "stability": np.asarray(stability)[selected], "selected_original_indices": selected, "audit": audit, "model_rerun": False}
