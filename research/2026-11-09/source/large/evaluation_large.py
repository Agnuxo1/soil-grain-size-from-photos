"""The single registered large-CLS pooling rule."""
if not __debug__:
    raise RuntimeError('Optimised Python is forbidden')
import numpy as np


def aggregate_large(features,feature_ids,ids):
    features=np.asarray(features,dtype=np.float32);feature_ids=np.asarray(feature_ids,dtype=str)
    assert features.ndim==3 and features.shape[1:]==(5,1024) and len(features)==len(feature_ids)
    assert len(ids)==len(set(ids)) and set(feature_ids)==set(ids) and np.isfinite(features).all()
    norms=np.linalg.norm(features,axis=-1,keepdims=True);assert np.all(norms>1e-12)
    normalized=features/np.maximum(norms,1e-12)
    rows=[]
    for name in ids:
        selected=normalized[feature_ids==name,1:5,:].reshape(-1,1024)
        assert len(selected)>0
        rows.append(np.concatenate([selected.mean(axis=0),selected.std(axis=0,ddof=0)]))
    result=np.vstack(rows);assert result.shape==(len(ids),2048) and np.isfinite(result).all()
    return result
