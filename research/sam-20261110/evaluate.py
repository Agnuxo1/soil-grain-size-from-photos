"""One fixed geometry augmentation; cached exact E016-style controls are not retuned."""
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parent
prior=Path('D:/PROJECTS/.cognition/soil-rescue-20261109');old=Path('D:/PROJECTS/183_Machine_Learning')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
baseline=json.loads((prior/'BASELINE_REPRODUCTION.json').read_text())
assert baseline['repeated_metrics_match'] and baseline['test_control_csv_bytes_exact']
assert {n:importlib.metadata.version(n) for n in baseline['runtime']}==baseline['runtime']
assert all(sha(Path(n))==h for n,h in baseline['source_and_artifact_sha256'].items())
control=json.loads((prior/'BASELINE_LOCAL_ARRAY_MANIFEST.json').read_text())
assert all(sha(prior/'baseline-private'/n)==h for n,h in control.items())
labels=pd.read_csv(old/'data/Training_labels_updated.csv');ids=labels.sample_id.astype(str).tolist();target=labels.iloc[:,1:].to_numpy(dtype=float)
assert len(ids)==len(set(ids))==24 and target.shape==(24,11)
assert hashlib.sha256('\n'.join(ids).encode()).hexdigest()==baseline['row_order_sha256']
assert hashlib.sha256(json.dumps(list(labels.columns[1:])).encode()).hexdigest()==baseline['target_order_sha256']
weights=np.diff(np.log10(np.asarray(labels.columns[1:],dtype=float)))
spec=importlib.util.spec_from_file_location('frozen_evaluation_core',prior/'evaluation_core.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
x=np.load(prior/'baseline-private/train-features.npy',allow_pickle=False)
bpred=np.load(prior/'baseline-private/repeated-oof.npy',allow_pickle=False);bloss=np.load(prior/'baseline-private/per-sample-losses.npy',allow_pickle=False)
assert x.shape==(24,2562) and bpred.shape==(20,24,11) and bloss.shape==(20,24)
np.testing.assert_allclose(bloss,np.array([core.sample_emd(target,p,weights) for p in bpred]),atol=1e-12,rtol=0)
assert abs(bloss.mean()-34.45786163217882)<1e-10
bundle=root/'private-retrieved/result';receipt=json.loads((bundle/'COLLECTION_RECEIPT.json').read_text())
expected=json.loads((root/'EXPECTED_FULL.json').read_text());job=json.loads((root/'FULL_JOB_TERMINAL.json').read_text())
assert job['id']==expected['job_id']==receipt['job_id'] and job['state']=='succeeded' and job.get('exit_code')==0
assert job['command']==expected['command'] and job['project']=='soil-grain-size'
assert receipt['source_manifest_verified'] and receipt['rows']==127 and receipt['features']==44
assert all(sha(bundle/n)==h for n,h in receipt['files_sha256'].items())
summary=json.loads((bundle/'summary.json').read_text());identity=summary['identity']
assert hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()==receipt['config_digest']==expected['config_digest']
assert identity['sources']==expected['source_files']
assert all(sha(root/'source'/n)==h for n,h in expected['source_files'].items())
asset=prior/'server-assets';assert sha(asset/'ASSET_MANIFEST.json')==identity['asset_manifest_sha256']
asset_manifest=json.loads((asset/'ASSET_MANIFEST.json').read_text())
assert sha(asset/'train-reference.npz')==asset_manifest['train-reference.npz']['sha256']
with np.load(asset/'train-reference.npz',allow_pickle=False) as ref:
    names=ref['file_names'].astype(str).tolist();photo_ids=ref['sample_ids'].astype(str)
assert hashlib.sha256('\n'.join(names).encode()).hexdigest()==identity['row_order_sha256']
assert hashlib.sha256('\n'.join(photo_ids).encode()).hexdigest()==identity['sample_mapping_sha256']
sys.path.insert(0,str(root/'source'));from morphometry import aggregate
features=np.load(bundle/'features.npy',allow_pickle=False);extra=aggregate(features,photo_ids,ids)
assert extra.shape==(24,88)
joined=np.c_[x,extra];assert joined.shape==(24,2650)
module_path=old/'src/soil_gsd_feature_fusion_audit.py'
spec=importlib.util.spec_from_file_location('frozen_fold_predict',module_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
predictions=[];losses=[]
for seed in range(20260925,20260945):
    pred=np.zeros_like(target);seen=np.zeros(24,dtype=int)
    for tr,va in KFold(5,shuffle=True,random_state=seed).split(target):
        assert set(tr).isdisjoint(va)
        pred[va]=module.fold_predict(joined[tr],target[tr],joined[va],alpha=1.);seen[va]+=1
    assert np.all(seen==1)
    predictions.append(pred);losses.append(core.sample_emd(target,pred,weights))
candidate=np.array(losses);effect=core.paired_cluster_bootstrap(bloss,candidate,reps=10000,seed=20261110)
out=root/'evaluation-private';out.mkdir(exist_ok=False)
np.save(out/'repeated-oof.npy',np.array(predictions));np.save(out/'per-sample-losses.npy',candidate);np.save(out/'train-features.npy',joined)
report={'experiment':identity['experiment'],'scope':'Exploratory24physicaltraininggroups;20partitions reuse groups, not480independent observations',
        'baseline_mean_emd':float(bloss.mean()),'candidate_mean_emd':float(candidate.mean()),**effect,
        'feature_dimensions':2650,'added_dimensions':88,'configuration_count':1,'baseline_control_pins_verified':True,
        'inference_job_id':job['id'],'inference_config_digest':expected['config_digest'],
        'outputs_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()},
        'test_permission_issued':False,'independent_results_review_complete':False,
        'limitations':'SAM mask proxies are not validated grains/sieve mass; historical selection and phone domain shift persist.'}
(root/'CANDIDATE_EVALUATION.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('baseline_mean_emd','candidate_mean_emd','mean_emd_reduction','paired_cluster_95pct_ci','preregistered_gate_passed','test_permission_issued')}))
