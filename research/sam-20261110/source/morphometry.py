"""Fixed SAM mask-derived geometry proxies, not verified grain measurements.
Author: Francisco Angulo de Lafuente. Assistance: Codex. SPDX-License-Identifier: Apache-2.0
"""
import numpy as np

DIAMETERS=np.array([.002,.0063,.02,.063,.2,.63,2.,6.3,20.,63.,200.],dtype=np.float64)
DIMENSION=44


def describe_masks(masks,shape=(518,518),ppm=5.0):
    if tuple(shape)!=(518,518) or ppm!=5.0:raise ValueError('Physical crop contract changed')
    selected=[];union=np.zeros(shape,dtype=bool)
    for item in masks:
        mask=np.asarray(item['segmentation'])
        if mask.shape!=tuple(shape) or mask.dtype!=bool:raise ValueError('Invalid binary mask')
        area=int(mask.sum())
        if area<16 or area>.25*np.prod(shape):continue
        if mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any():continue
        yy,xx=np.nonzero(mask);w=int(xx.max()-xx.min()+1);h=int(yy.max()-yy.min()+1)
        score=float(item['predicted_iou'])
        if not np.isfinite(score):raise ValueError('Invalid mask quality')
        selected.append((area,2*np.sqrt(area/np.pi)/ppm,np.log(max(w,h)/min(w,h)),score))
        union|=mask
    if not selected:return np.zeros(DIMENSION,dtype=np.float64),0
    values=np.asarray(selected,dtype=np.float64);area,diameter,aspect,quality=values.T
    hit=diameter[:,None]<=DIAMETERS[None,:]
    cdfs=[(hit*weight[:,None]).sum(0)/weight.sum() for weight in (np.ones(len(area)),area,diameter**3)]
    quantiles=np.quantile(diameter,[.1,.25,.5,.75,.9])
    tail=[np.log1p(len(area)),area.sum()/np.prod(shape),union.sum()/np.prod(shape),aspect.mean(),np.log(diameter).std(),quality.mean()]
    result=np.concatenate([*cdfs,quantiles,tail])
    assert result.shape==(DIMENSION,) and np.isfinite(result).all()
    for cdf in cdfs:assert np.all(np.diff(cdf)>=-1e-12) and abs(cdf[-1]-1)<1e-12
    return result,len(area)


def aggregate(features,feature_ids,sample_ids):
    features=np.asarray(features,dtype=np.float64);feature_ids=np.asarray(feature_ids,dtype=str)
    if features.shape!=(127,DIMENSION) or not np.isfinite(features).all():raise ValueError('Incomplete photo descriptor table')
    if len(feature_ids)!=127 or len(sample_ids)!=24 or len(set(sample_ids))!=24 or set(feature_ids)!=set(sample_ids):raise ValueError('Physical sample mapping mismatch')
    rows=[]
    for sample in sample_ids:
        selected=features[feature_ids==sample]
        if len(selected)==0:raise ValueError('Missing physical sample')
        rows.append(np.r_[selected.mean(0),selected.std(0,ddof=0)])
    return np.stack(rows)
