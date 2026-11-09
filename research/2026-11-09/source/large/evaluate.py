"""One private exploratory comparison, training only, without test permissions."""
if not __debug__:
    raise RuntimeError('Optimised Python is forbidden')
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from evaluation_core import sample_emd,paired_cluster_bootstrap,file_sha
from evaluation_large import aggregate_large


def main():
    r=Path(__file__).resolve().parent;prior=r.parent/'soil-rescue-20261109';old=Path('D:/PROJECTS/183_Machine_Learning')
    seal=json.loads((r/'EVALUATION_SOURCE_MANIFEST.json').read_text())
    assert 'BASELINE_REPRODUCTION.json' in seal and 'BASELINE_LOCAL_ARRAY_MANIFEST.json' in seal
    assert all(file_sha(r/name)==digest for name,digest in seal.items())
    baseline=json.loads((r/'BASELINE_REPRODUCTION.json').read_text())
    assert baseline['repeated_metrics_match'] is True and baseline['test_control_csv_bytes_exact'] is True
    assert {n:importlib.metadata.version(n) for n in baseline['runtime']}==baseline['runtime']
    assert all(file_sha(Path(name))==digest for name,digest in baseline['source_and_artifact_sha256'].items())
    control=json.loads((r/'BASELINE_LOCAL_ARRAY_MANIFEST.json').read_text())
    assert all(file_sha(prior/'baseline-private'/name)==digest for name,digest in control.items())
    labels=pd.read_csv(old/'data/Training_labels_updated.csv');ids=labels.sample_id.astype(str).tolist()
    target=labels.iloc[:,1:].to_numpy(dtype=float);weights=np.diff(np.log10(np.asarray(labels.columns[1:],dtype=float)))
    assert len(ids)==len(set(ids))==24 and target.shape==(24,11)
    assert baseline['row_order_sha256']==hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    assert baseline['target_order_sha256']==hashlib.sha256(json.dumps(list(labels.columns[1:])).encode()).hexdigest()
    x=np.load(prior/'baseline-private/train-features.npy',allow_pickle=False)
    bpred=np.load(prior/'baseline-private/repeated-oof.npy',allow_pickle=False)
    bloss=np.load(prior/'baseline-private/per-sample-losses.npy',allow_pickle=False)
    assert x.shape==(24,2562) and bpred.shape==(20,24,11) and bloss.shape==(20,24)
    np.testing.assert_allclose(bloss,np.asarray([sample_emd(target,p,weights) for p in bpred]),rtol=0,atol=1e-12)
    gpu=r/'gpu-source';sources=json.loads((r/'GPU_SOURCE_MANIFEST.json').read_text())
    assert sources==json.loads((gpu/'SOURCE_MANIFEST.json').read_text())
    assert all(file_sha(gpu/name)==digest for name,digest in sources.items())
    contract=json.loads((gpu/'RUN_CONTRACT.json').read_text())
    assets=prior/'server-assets'
    assert file_sha(assets/'ASSET_MANIFEST.json')==contract['asset_manifest_sha256']
    asset_manifest=json.loads((assets/'ASSET_MANIFEST.json').read_text())
    assert file_sha(assets/'train-reference.npz')==asset_manifest['train-reference.npz']['sha256']
    with np.load(prior/'server-assets/train-reference.npz',allow_pickle=False) as ref:
        names=ref['file_names'].astype(str).tolist();feature_ids=ref['sample_ids'].astype(str)
    assert len(names)==127 and set(feature_ids)==set(ids)
    identity={'experiment':contract['experiment'],'stage':'train','rows':127,'sources':sources,
      'contract_sha256':file_sha(gpu/'RUN_CONTRACT.json'),'runtime':contract['runtime'],'timm_version':contract['timm_version'],
      'row_order_sha256':hashlib.sha256('\n'.join(names).encode()).hexdigest()}
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    bundle=r/'train-download-001/payload';receipt=json.loads((bundle/'COLLECTION_RECEIPT.json').read_text())
    job=json.loads((r/'TRAIN_JOB_READBACK.json').read_text())
    assert job['id']==receipt['job_id'] and job['state']=='succeeded' and job['exit_code']==0
    assert job['project']=='soil-grain-size' and job['command']==['timeout','15m','/home/ai/venvs/training/bin/python','run_safe.py','--stage','train']
    assert receipt['source_manifest']==sources and receipt['contract_sha256']==identity['contract_sha256']
    assert receipt['stage']=='train' and receipt['config_digest']==digest
    assert set(receipt['files_sha256'])=={'summary.json','cls.npy','checkpoint.npz'}
    assert all(file_sha(bundle/name)==v for name,v in receipt['files_sha256'].items())
    summary=json.loads((bundle/'summary.json').read_text())
    assert summary['identity']==identity and summary['config_digest']==digest
    assert summary['checkpoint_complete'] is True and summary['accuracy_evidence'] is False
    assert summary['exclusive_writer_lock'] is True and summary['precision_gate_passed'] is True
    smoke=json.loads((r/'SMOKE_VERIFICATION.json').read_text())
    assert smoke['complete_smoke_artifacts_and_checkpoint_verified'] is True and smoke['test_extraction_admitted'] is False
    assert summary['minimum_precision_cosine']==smoke['minimum_precision_cosine']>=.9999
    assert summary['images']==127 and summary['feature_shape']==[127,5,1024]
    features=np.load(bundle/'cls.npy',allow_pickle=False)
    assert features.shape==(127,5,1024) and np.isfinite(features).all()
    assert all(file_sha(bundle/name)==v for name,v in summary['outputs_sha256'].items())
    with np.load(bundle/'checkpoint.npz',allow_pickle=False) as checkpoint:
        assert int(checkpoint['processed'])==127 and str(checkpoint['digest'].item())==digest
        assert np.array_equal(checkpoint['features'],features)
        assert abs(float(checkpoint['minimum_precision_cosine'])-summary['minimum_precision_cosine'])<1e-8
    joined=np.column_stack([aggregate_large(features,feature_ids,ids),x[:,768:]])
    assert joined.shape==(24,3842)
    source=old/'src/soil_gsd_feature_fusion_audit.py'
    spec=importlib.util.spec_from_file_location('historical_pure_math',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    predictions=[];losses=[];repeat_rows=[]
    for number,seed in enumerate(range(20260925,20260945)):
        pred=np.zeros_like(target);base=np.zeros_like(target);seen=np.zeros(24,dtype=int)
        for tr,va in KFold(5,shuffle=True,random_state=seed).split(target):
            assert set(tr).isdisjoint(va)
            base[va]=module.fold_predict(x[tr],target[tr],x[va],alpha=1.)
            pred[va]=module.fold_predict(joined[tr],target[tr],joined[va],alpha=1.);seen[va]+=1
        assert np.all(seen==1)
        np.testing.assert_allclose(base,bpred[number],rtol=0,atol=1e-10)
        loss=sample_emd(target,pred,weights);predictions.append(pred);losses.append(loss)
        repeat_rows.append({'seed':seed,'baseline_emd':float(bloss[number].mean()),'candidate_emd':float(loss.mean())})
    candidate=np.asarray(losses);effect=paired_cluster_bootstrap(bloss,candidate)
    out=r/'evaluation-private';assert not out.exists();out.mkdir()
    np.save(out/'train-features.npy',joined);np.save(out/'repeated-oof.npy',np.asarray(predictions));np.save(out/'per-sample-losses.npy',candidate)
    report={'experiment':contract['experiment'],'scope':'Exploratory training-only evaluation,24physicalsamples',
      'baseline_mean_emd':float(bloss.mean()),'candidate_mean_emd':float(candidate.mean()),**effect,
      'features':3842,'configuration_count':1,'repeat_results':repeat_rows,
      'baseline_recomputed_all20partitions':True,'evaluator_source_manifest':seal,
      'inference_source_manifest':sources,'inference_job_id':job['id'],'inference_config_digest':digest,
      'train_collection_receipt_sha256':file_sha(bundle/'COLLECTION_RECEIPT.json'),
      'output_sha256':{n:file_sha(out/n) for n in ['train-features.npy','repeated-oof.npy','per-sample-losses.npy']},
      'test_permission_issued':False,'independent_result_review_complete':False,
      'limitations':'24historically selected samples;20partitions reuse them. Exploratory selection; CI does not correct prior selection. Android/iPhone shift remains. No officialscore/rank/safety claim.'}
    (r/'CANDIDATE_EVALUATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['baseline_mean_emd','candidate_mean_emd','mean_emd_reduction','paired_cluster_95pct_ci','preregistered_gate_passed','test_permission_issued']}))


if __name__=='__main__':main()
