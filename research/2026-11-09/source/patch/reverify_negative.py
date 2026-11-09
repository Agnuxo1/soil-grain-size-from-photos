"""Recompute the recorded negative under the completed baseline-proof binding."""
if not __debug__:
    raise RuntimeError('Optimised Python is forbidden')
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from evaluation_core import file_sha,sample_emd,paired_cluster_bootstrap
r=Path(__file__).resolve().parent
seal=json.loads((r/'EVALUATION_COMPLETE_MANIFEST.json').read_text())
assert all(file_sha(r/name)==digest for name,digest in seal.items())
baseline=json.loads((r/'BASELINE_REPRODUCTION.json').read_text())
assert {n:importlib.metadata.version(n) for n in baseline['runtime']}==baseline['runtime']
assert all(file_sha(Path(name))==v for name,v in baseline['source_and_artifact_sha256'].items())
report=json.loads((r/'CANDIDATE_EVALUATION.json').read_text());assert report['test_permission_issued'] is False
assert all(file_sha(r/'baseline-private'/name)==v for name,v in report['baseline_array_manifest'].items())
assert all(file_sha(r/'evaluation-private'/name)==v for name,v in report['output_sha256'].items())
old=Path('D:/PROJECTS/183_Machine_Learning');labels=pd.read_csv(old/'data/Training_labels_updated.csv')
ids=labels.sample_id.astype(str).tolist();target=labels.iloc[:,1:].to_numpy(dtype=float)
assert hashlib.sha256('\n'.join(ids).encode()).hexdigest()==baseline['row_order_sha256']
weights=np.diff(np.log10(np.asarray(labels.columns[1:],dtype=float)))
x=np.load(r/'baseline-private/train-features.npy',allow_pickle=False)
joined=np.load(r/'evaluation-private/train-features.npy',allow_pickle=False)
expected=np.load(r/'evaluation-private/repeated-oof.npy',allow_pickle=False)
base_expected=np.load(r/'baseline-private/repeated-oof.npy',allow_pickle=False)
source=old/'src/soil_gsd_feature_fusion_audit.py'
spec=importlib.util.spec_from_file_location('bound_historical_math',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
losses=[];base_losses=[]
for k,seed in enumerate(range(20260925,20260945)):
    pred=np.zeros_like(target);bpred=np.zeros_like(target)
    for tr,va in KFold(5,shuffle=True,random_state=seed).split(target):
        pred[va]=module.fold_predict(joined[tr],target[tr],joined[va],alpha=1.)
        bpred[va]=module.fold_predict(x[tr],target[tr],x[va],alpha=1.)
    np.testing.assert_allclose(pred,expected[k],rtol=0,atol=1e-10)
    np.testing.assert_allclose(bpred,base_expected[k],rtol=0,atol=1e-10)
    losses.append(sample_emd(target,pred,weights));base_losses.append(sample_emd(target,bpred,weights))
effect=paired_cluster_bootstrap(base_losses,losses)
assert effect['preregistered_gate_passed'] is False
assert abs(effect['mean_emd_reduction']-report['mean_emd_reduction'])<1e-10
np.testing.assert_allclose(effect['paired_cluster_95pct_ci'],report['paired_cluster_95pct_ci'],rtol=0,atol=1e-10)
proof={'complete_manifest_sha256':file_sha(r/'EVALUATION_COMPLETE_MANIFEST.json'),
 'baseline_proof_sha256':file_sha(r/'BASELINE_REPRODUCTION.json'),'historical_executed_module_sha256':file_sha(source),
 'verifier_source_sha256':file_sha(Path(__file__).resolve()),'original_result_sha256':file_sha(r/'CANDIDATE_EVALUATION.json'),
 'all20baseline_and_candidate_partitions_recomputed_match':True,'negative_gate_reconfirmed':True,
 'baseline_mean_emd':float(np.mean(base_losses)),'candidate_mean_emd':float(np.mean(losses)),**effect,
 'original_sources_results_unchanged':True,'test_permission_issued':False}
assert all(file_sha(r/name)==digest for name,digest in seal.items())
(r/'CANDIDATE_REVERIFICATION.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({k:proof[k] for k in ['all20baseline_and_candidate_partitions_recomputed_match','negative_gate_reconfirmed','baseline_mean_emd','candidate_mean_emd','test_permission_issued']}))
