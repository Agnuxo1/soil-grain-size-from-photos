"""Pure calculations for the one preregistered Soil experiment."""
if not __debug__:
    raise RuntimeError('Optimised Python is forbidden: admission assertions must remain active')
import hashlib
import json
import numpy as np


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_identity(contract, sources, file_names, stage):
    count=contract['pilot_rows'] if stage=='pilot' else contract['rows']
    return {'experiment':contract['experiment'],'stage':stage,'rows':count,'sources':sources,
            'asset_manifest_sha256':contract['asset_manifest_sha256'],
            'weights_sha256':contract['weights_sha256'],'model_config_sha256':contract['model_config_sha256'],
            'runtime':contract['runtime'],'timm_version':contract['timm_version'],
            'row_order_sha256':hashlib.sha256('\n'.join(file_names[:count]).encode()).hexdigest(),
            'batch_images':4,'workers':0,'precision':contract['precision']}


def verify_summary(summary, contract, sources, file_names, stage):
    identity=expected_identity(contract,sources,file_names,stage)
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    assert summary['identity']==identity and summary['config_digest']==digest,'Inference identity mismatch'
    assert summary['experiment']==contract['experiment'] and summary['stage']==stage
    assert summary['images']==identity['rows']
    assert summary['descriptor_shape']==[identity['rows'],5,768]
    assert summary['checkpoint_complete'] is True and summary['exclusive_writer_lock'] is True
    assert summary['accuracy_evidence'] is False and summary['feasibility_gate_passed'] is True
    assert np.isfinite(summary['minimum_cls_cosine']) and summary['minimum_cls_cosine']>=contract['minimum_cls_cosine']
    return digest


def aggregate_patch_stats(stats, feature_ids, sample_ids):
    stats=np.asarray(stats)
    assert stats.ndim==3 and stats.shape[1:]==(5,768) and np.isfinite(stats).all()
    feature_ids=np.asarray(feature_ids,dtype=str)
    assert len(feature_ids)==len(stats) and len(sample_ids)==len(set(sample_ids))
    assert set(feature_ids)==set(sample_ids),'Missing or unexpected sample IDs'
    assert np.all(stats[:,:,384:]>=0),'Patch standard deviations cannot be negative'
    rows=[]
    for name in sample_ids:
        selected=stats[feature_ids==name,1:5,:].reshape(-1,768)
        assert len(selected)>0
        rows.append(np.concatenate([selected.mean(axis=0),selected.std(axis=0,ddof=0)]))
    result=np.vstack(rows)
    assert result.shape==(len(sample_ids),1536) and np.isfinite(result).all()
    return result


def sample_emd(target,prediction,weights):
    target=np.asarray(target,dtype=float);prediction=np.asarray(prediction,dtype=float)
    weights=np.asarray(weights,dtype=float)
    assert target.shape==prediction.shape and target.ndim==2 and target.shape[1]==11
    assert weights.shape==(10,) and np.all(weights>0) and abs(weights.sum()-5)<1e-10
    assert np.isfinite(target).all() and np.isfinite(prediction).all()
    for a in [target,prediction]:
        assert np.min(a)>=-1e-7 and np.max(a)<=100+1e-7 and np.all(np.diff(a,axis=1)>=-1e-7)
        assert np.allclose(a[:,-1],100,rtol=0,atol=1e-7)
    return (np.abs(target[:,:-1]-prediction[:,:-1])*weights).sum(axis=1)


def paired_cluster_bootstrap(baseline_loss,candidate_loss,reps=10000,seed=20261109):
    baseline_loss=np.asarray(baseline_loss,dtype=float);candidate_loss=np.asarray(candidate_loss,dtype=float)
    assert baseline_loss.shape==candidate_loss.shape==(20,24)
    assert np.isfinite(baseline_loss).all() and np.isfinite(candidate_loss).all()
    # A physical sample is the cluster. Twenty repeated partitions are not independent observations.
    improvement=(baseline_loss-candidate_loss).mean(axis=0)
    rng=np.random.default_rng(seed);indices=rng.integers(0,24,size=(reps,24))
    means=improvement[indices].mean(axis=1)
    interval=np.quantile(means,[.025,.975])
    delta=float(improvement.mean())
    return {'mean_emd_reduction':delta,'paired_cluster_95pct_ci':interval.tolist(),
            'preregistered_gate_passed':bool(delta>=1.0 and interval[0]>0),
            'clusters':24,'repeats_reuse_same_clusters':20,'bootstrap_replicates':reps,
            'bootstrap_seed':seed,'per_sample_mean_improvement':improvement.tolist()}
