"""Private training-only evaluation; never reads new test descriptors or submits."""
if not __debug__:
    raise RuntimeError('Optimised Python is forbidden: admission assertions must remain active')
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from evaluation_core import file_sha,verify_summary,aggregate_patch_stats,sample_emd,paired_cluster_bootstrap


def main():
    r=Path(__file__).resolve().parent;old=Path('D:/PROJECTS/183_Machine_Learning')
    seal=json.loads((r/'EVALUATION_SOURCE_MANIFEST.json').read_text())
    assert all(file_sha(r/name)==digest for name,digest in seal.items()),'Evaluator changed after freeze'
    gpu=r/'gpu-source-v2'
    inference=json.loads((r/'GPU_SOURCE_MANIFEST_V2.json').read_text())
    assert inference==json.loads((gpu/'SOURCE_MANIFEST.json').read_text())
    assert all(file_sha(gpu/name)==digest for name,digest in inference.items())
    baseline=json.loads((r/'BASELINE_REPRODUCTION.json').read_text())
    assert baseline['repeated_metrics_match'] is True and baseline['test_control_csv_bytes_exact'] is True
    assert {n:importlib.metadata.version(n) for n in baseline['runtime']}==baseline['runtime']
    for name,digest in baseline['source_and_artifact_sha256'].items():
        assert file_sha(Path(name))==digest,'Recovered baseline input changed'
    control=json.loads((r/'BASELINE_LOCAL_ARRAY_MANIFEST.json').read_text())
    assert all(file_sha(r/'baseline-private'/name)==digest for name,digest in control.items())
    labels=pd.read_csv(old/'data/Training_labels_updated.csv')
    ids=labels.sample_id.astype(str).tolist();target=labels.iloc[:,1:].to_numpy(dtype=float)
    assert len(ids)==len(set(ids))==24 and target.shape==(24,11)
    weights=np.diff(np.log10(np.asarray(labels.columns[1:],dtype=float)))
    assert baseline['row_order_sha256']==__import__('hashlib').sha256('\n'.join(ids).encode()).hexdigest()
    assert baseline['target_order_sha256']==__import__('hashlib').sha256(json.dumps(list(labels.columns[1:])).encode()).hexdigest()
    x=np.load(r/'baseline-private/train-features.npy',allow_pickle=False)
    bpred=np.load(r/'baseline-private/repeated-oof.npy',allow_pickle=False)
    bloss=np.load(r/'baseline-private/per-sample-losses.npy',allow_pickle=False)
    assert x.shape==(24,2562) and bpred.shape==(20,24,11) and bloss.shape==(20,24)
    np.testing.assert_allclose(bloss,np.asarray([sample_emd(target,p,weights) for p in bpred]),rtol=0,atol=1e-12)
    assert abs(float(bloss.mean())-baseline['mean_emd'])<1e-10
    contract=json.loads((gpu/'RUN_CONTRACT.json').read_text())
    assets=r/'server-assets';manifest=json.loads((assets/'ASSET_MANIFEST.json').read_text())
    assert file_sha(assets/'ASSET_MANIFEST.json')==contract['asset_manifest_sha256']
    assert all(file_sha(assets/name)==v['sha256'] for name,v in manifest.items())
    with np.load(assets/'train-reference.npz',allow_pickle=False) as ref:
        names=ref['file_names'].astype(str).tolist();feature_ids=ref['sample_ids'].astype(str)
        cls_reference=ref['features'].copy()
    bundle=r/'train-private'
    receipt=json.loads((bundle/'COLLECTION_RECEIPT.json').read_text())
    job=json.loads((r/'TRAIN_JOB_READBACK.json').read_text())
    assert job['id']==receipt['job_id'] and job['state']=='succeeded' and job.get('exit_code')==0
    assert job['project']=='soil-grain-size' and job['command']==['timeout','15m','/home/ai/venvs/training/bin/python','run_safe.py','--stage','train']
    assert receipt['stage']=='train' and receipt['source_manifest']==inference
    assert set(receipt['files_sha256'])=={'summary.json','patch-statistics.npy','cls-control.npy','checkpoint.npz'}
    assert all(file_sha(bundle/name)==digest for name,digest in receipt['files_sha256'].items())
    summary=json.loads((bundle/'summary.json').read_text())
    digest=verify_summary(summary,contract,inference,names,'train')
    stats=np.load(bundle/'patch-statistics.npy',allow_pickle=False)
    cls=np.load(bundle/'cls-control.npy',allow_pickle=False)
    assert stats.shape==(127,5,768) and cls.shape==cls_reference.shape==(127,5,384)
    assert np.isfinite(cls).all()
    den=np.linalg.norm(cls,axis=-1)*np.linalg.norm(cls_reference,axis=-1);assert np.all(den>1e-12)
    cos=(cls*cls_reference).sum(axis=-1)/den
    assert float(cos.min())>=contract['minimum_cls_cosine']
    assert abs(float(cos.min())-summary['minimum_cls_cosine'])<1e-7
    for name in ['patch-statistics.npy','cls-control.npy']:
        assert summary['outputs_sha256'][name]==file_sha(bundle/name)
    with np.load(bundle/'checkpoint.npz',allow_pickle=False) as checkpoint:
        assert str(checkpoint['digest'].item())==digest and int(checkpoint['processed'])==127
        assert np.array_equal(checkpoint['cls'],cls) and np.array_equal(checkpoint['stats'],stats)
    patch=aggregate_patch_stats(stats,feature_ids,ids);joined=np.column_stack([x,patch])
    assert joined.shape==(24,4098) and np.isfinite(joined).all()
    module_path=old/'src/soil_gsd_feature_fusion_audit.py'
    spec=importlib.util.spec_from_file_location('recovered_pure_math',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    predictions=[];losses=[];fold_details=[]
    for seed in range(20260925,20260945):
        pred=np.zeros_like(target);seen=np.zeros(24,dtype=int)
        for tr,va in KFold(5,shuffle=True,random_state=seed).split(target):
            assert set(tr).isdisjoint(va)
            pred[va]=module.fold_predict(joined[tr],target[tr],joined[va],alpha=1.)
            seen[va]+=1
            fold_details.append({'seed':seed,'train_ids':[ids[i] for i in tr],'valid_ids':[ids[i] for i in va]})
        assert np.all(seen==1)
        predictions.append(pred);losses.append(sample_emd(target,pred,weights))
    candidate=np.asarray(losses);effect=paired_cluster_bootstrap(bloss,candidate)
    out=r/'evaluation-private';assert not out.exists(),'Keep every original evaluation; do not overwrite'
    out.mkdir();np.save(out/'repeated-oof.npy',np.asarray(predictions));np.save(out/'per-sample-losses.npy',candidate)
    np.save(out/'train-features.npy',joined)
    report={'experiment':contract['experiment'],'scope':'Exploratory training-only validation on24physicalsamples',
            'baseline_mean_emd':float(bloss.mean()),'candidate_mean_emd':float(candidate.mean()),**effect,
            'repeat_emd':candidate.mean(axis=1).tolist(),'features':4098,'configuration_count':1,
            'folds':fold_details,'inference_job_id':job['id'],'inference_config_digest':digest,
            'evaluator_source_manifest':seal,'inference_source_manifest':inference,
            'baseline_array_manifest':control,'train_collection_manifest':receipt,
            'output_sha256':{name:file_sha(out/name) for name in ['repeated-oof.npy','per-sample-losses.npy','train-features.npy']},
            'test_permission_issued':False,'independent_review_complete':False,
            'limitations':'24historically selected samples; repeated partitions reuse them; Android/iPhone domain shift. Not an official score or causal claim.'}
    (r/'CANDIDATE_EVALUATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['baseline_mean_emd','candidate_mean_emd','mean_emd_reduction','paired_cluster_95pct_ci','preregistered_gate_passed','test_permission_issued']}))


if __name__=='__main__':main()
