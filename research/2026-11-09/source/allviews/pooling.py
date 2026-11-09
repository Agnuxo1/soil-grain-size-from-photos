if not __debug__:
    raise RuntimeError('Optimised Python is forbidden')
import numpy as np


def aggregate(raw,feature_ids,ids,views):
    raw=np.asarray(raw,dtype=np.float32);feature_ids=np.asarray(feature_ids,dtype=str)
    assert raw.shape==(127,5,384) and np.isfinite(raw).all()
    assert len(ids)==len(set(ids))==24 and len(feature_ids)==127 and set(feature_ids)==set(ids)
    assert views in [(1,2,3,4),(0,1,2,3,4)]
    norm=np.linalg.norm(raw,axis=-1,keepdims=True);assert np.all(norm>1e-12)
    raw=raw/np.maximum(norm,1e-12)
    result=[]
    for name in ids:
        selected=raw[feature_ids==name][:,list(views),:].reshape(-1,384)
        result.append(np.concatenate([selected.mean(axis=0),selected.std(axis=0,ddof=0)]))
    output=np.vstack(result);assert output.shape==(24,768) and np.isfinite(output).all()
    return output
