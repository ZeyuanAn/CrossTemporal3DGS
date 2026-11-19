"""Multiscale local SSIM evidence; diagnostic only, never a physical edit proof."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter
from .rgb_pair_mask_features import joint_residual_threshold

SIGMAS=(1.5,3.0,6.0); C1=.01**2; C2=.03**2

def _luma(rgb):
    x=np.asarray(rgb,np.float32)
    if x.ndim==2: return x
    if x.ndim!=3 or x.shape[-1]!=3: raise ValueError('RGB must be HxWx3 or luminance HxW')
    return .2126*x[...,0]+.7152*x[...,1]+.0722*x[...,2]

def _fit_affine(pre,post,valid,fit_mask=None):
    domain=valid if fit_mask is None else (valid & np.asarray(fit_mask,bool)); fit_source='valid_all' if fit_mask is None else 'valid_and_fit_mask'
    x=pre[domain].astype(np.float64); y=post[domain].astype(np.float64)
    fit_count=int(len(x)); excluded_count=int(valid.sum()-fit_count)
    if fit_count<2: return 1.,0.,False,False,{'fit_domain_source':fit_source,'fit_count':fit_count,'excluded_count':excluded_count,'fit_reason':'empty_fit_mask','residual_mad':None,'inlier_fraction':0.}
    if np.ptp(x)<1e-8: return 1.,0.,False,False,{'fit_domain_source':fit_source,'fit_count':fit_count,'excluded_count':excluded_count,'fit_reason':'low_texture','residual_mad':None,'inlier_fraction':0.}
    a,b=1.,0.; clipped=False
    for _ in range(5):
        r=y-(a*x+b); scale=np.median(np.abs(r))+1e-6; w=np.minimum(1.,1.345*scale/np.maximum(np.abs(r),1e-8))
        # The former least-squares system multiplied both A and y by w,
        # therefore its statistical weights are q=w**2. Centering the scalar
        # regression avoids an SVD/LAPACK call and its platform-sensitive path.
        q=np.square(w,dtype=np.float64); q_sum=float(np.sum(q))
        if not np.isfinite(q_sum) or q_sum<=0.:
            return 1.,0.,False,False,{'fit_domain_source':fit_source,'fit_count':fit_count,'excluded_count':excluded_count,'fit_reason':'degenerate_weight_sum','residual_mad':None,'inlier_fraction':0.}
        x_mean=float(np.sum(q*x)/q_sum); y_mean=float(np.sum(q*y)/q_sum)
        x_centered=x-x_mean; y_centered=y-y_mean
        weighted_variance=float(np.sum(q*x_centered*x_centered))
        variance_floor=np.finfo(np.float64).eps*max(q_sum,1.)*max(1.,abs(x_mean))**2
        if not np.isfinite(weighted_variance) or weighted_variance<=variance_floor:
            return 1.,0.,False,False,{'fit_domain_source':fit_source,'fit_count':fit_count,'excluded_count':excluded_count,'fit_reason':'degenerate_weighted_variance','residual_mad':None,'inlier_fraction':0.}
        weighted_covariance=float(np.sum(q*x_centered*y_centered))
        na=weighted_covariance/weighted_variance
        nb=y_mean-na*x_mean
        ca=float(np.clip(na,.25,4.)); clipped|=ca!=na; a,b=ca,nb
    if clipped: b=float(np.median(y-a*x))
    residual=y-(a*x+b); mad=float(np.median(np.abs(residual-np.median(residual)))); inlier=float(np.mean(np.abs(residual-np.median(residual))<=max(3.*mad,1e-6)))
    return a,b,True,clipped,{'fit_domain_source':fit_source,'fit_count':fit_count,'excluded_count':excluded_count,'fit_reason':'gain_clipped' if clipped else 'ok','residual_mad':mad,'inlier_fraction':inlier}

def _ssim(pre,post,sigma,valid):
    w=np.maximum(gaussian_filter(valid.astype(float),sigma,mode='reflect'),1e-8); mu1=gaussian_filter(pre*valid,sigma,mode='reflect')/w; mu2=gaussian_filter(post*valid,sigma,mode='reflect')/w; v1=np.maximum(gaussian_filter(pre*pre*valid,sigma,mode='reflect')/w-mu1*mu1,0.); v2=np.maximum(gaussian_filter(post*post*valid,sigma,mode='reflect')/w-mu2*mu2,0.); cov=gaussian_filter(pre*post*valid,sigma,mode='reflect')/w-mu1*mu2
    out=((2*mu1*mu2+C1)*(2*cov+C2))/((mu1*mu1+mu2*mu2+C1)*(v1+v2+C2)); out=np.clip(out,-1.,1.); support=(w>=.5)&valid; out[~support]=0.; return out,support

def ssim_evidence(pres, posts, valid_masks=None, fit_masks=None):
    """Return raw/corrected SSIM structural evidence for one or more RGB pairs."""
    pre_list=list(pres) if isinstance(pres,(list,tuple)) else [pres]; post_list=list(posts) if isinstance(posts,(list,tuple)) else [posts]
    if len(pre_list)!=len(post_list): raise ValueError('pre/post counts must match')
    maps=[]; corrected_maps=[]; reports=[]; valids=[]; support_unknowns=[]; support_scales=[]
    for i,(pr,po) in enumerate(zip(pre_list,post_list)):
        y1=_luma(pr); y2=_luma(po)
        if y1.shape!=y2.shape: raise ValueError('pre/post RGB shapes must match')
        valid=np.ones(y1.shape,bool) if valid_masks is None else np.asarray(valid_masks[i] if np.asarray(valid_masks).ndim==3 else valid_masks,bool)
        if valid.shape!=y1.shape: raise ValueError('valid mask shape mismatch')
        if (not np.isfinite(y1[valid]).all()) or (not np.isfinite(y2[valid]).all()): raise ValueError('RGB inputs must be finite inside valid region')
        fm=None if fit_masks is None else np.asarray(fit_masks[i] if np.asarray(fit_masks).ndim==3 else fit_masks,bool)
        a,b,fit,clipped,fit_report=_fit_affine(y1,y2,valid,fm); fit_report['fit_mask_count']=int((valid if fm is None else (valid&fm)).sum()); corrected=np.clip((np.where(valid,y2,0.)-b)/max(a,1e-8),0.,1.); y1_work=np.where(valid,y1,0.); y2_work=np.where(valid,y2,0.)
        raw_sc=[]; cor_sc=[]; support_sc=[]
        for sigma in SIGMAS:
            raw_map,raw_support=_ssim(y1_work,y2_work,sigma,valid); cor_map,cor_support=_ssim(y1_work,corrected,sigma,valid); raw_sc.append(.5*(1-raw_map)); cor_sc.append(.5*(1-cor_map)); support_sc.append(raw_support & cor_support)
        support_valid=valid & np.logical_and.reduce(support_sc); support_unknown=valid & ~support_valid
        raw=np.nanmean(raw_sc,axis=0); cor=np.nanmean(cor_sc,axis=0); raw[~support_valid]=0.; cor[~support_valid]=0.; maps.append(raw.astype(np.float32)); corrected_maps.append(cor.astype(np.float32)); valids.append(support_valid); support_unknowns.append(support_unknown); support_scales.append(support_sc)
        reports.append({'view_index':i,'gain':a,'offset':b,'fit_used':fit,'fit_reliable':fit and not clipped,'gain_clipped':clipped,'offset_recomputed_after_gain_clip':bool(clipped),'fit_reason':fit_report['fit_reason'],'fit_report':fit_report,'fit_mask_count':fit_report['fit_mask_count'],'fit_domain_source':fit_report['fit_domain_source'],'excluded_count':fit_report['excluded_count'],'residual_mad':fit_report['residual_mad'],'inlier_fraction':fit_report['inlier_fraction'],'weighted_gaussian_stats':True,'boundary_trust':False,'sigmas':list(SIGMAS),'raw_means':[float(np.nanmean(x)) for x in raw_sc],'corrected_means':[float(np.nanmean(x)) for x in cor_sc],'valid_count':int(valid.sum()),'translation_warning':'SSIM is not a registration solution; translations may produce evidence'})
    all_maps=np.asarray(corrected_maps); all_valid=np.asarray(valids); threshold,threshold_report=joint_residual_threshold(all_maps,all_valid)
    candidates=all_maps>=threshold; candidates[~all_valid]=False
    pack=lambda x: x[0] if len(x)==1 else np.asarray(x)
    return {'raw_maps':np.asarray(maps), 'corrected_maps':all_maps, 'candidate_masks':candidates, 'support_valid':pack(all_valid), 'support_unknown':pack(support_unknowns), 'support_sigma_1.5':pack([x[0] for x in support_scales]), 'support_sigma_3':pack([x[1] for x in support_scales]), 'support_sigma_6':pack([x[2] for x in support_scales]), 'threshold':threshold, 'threshold_report':threshold_report, 'reports':reports, 'method':'multiscale_local_ssim_luma_exposure_corrected', 'physical_removal_proof':False, 'diagnostic_only':True}
