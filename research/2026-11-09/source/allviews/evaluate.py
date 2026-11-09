"""One training-only CPU coverage ablation; fixed recipe and complete input binding."""
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
from pooling import aggregate


def main():
    r=Path(__file__).resolve().parent;prior=r.parent/'soil-rescue-20261109';old=Path('D:/PROJECTS/183_Machine_Learning')
    seal=json.loads((r/'SOURCE_MANIFEST.json').read_text())
    assert 'BASELINE_REPRODUCTION.json' in seal and all(file_sha(r/name)==digest for name,digest in seal.items())
    baseline=json.loads((r/'BASELINE_REPRODUCTION.json').read_text())
    assert baseline['repeated_metrics_match'] is True and baseline['test_control_csv_bytes_exact'] is True
    assert {n:importlib.metadata.version(n) for n in baseline['runtime']}==baseline['runtime']
    assert all(file_sha(Path(name))==digest for name,digest in baseline['source_and_artifact_sha256'].items())
    controls=json.loads((r/'BASELINE_LOCAL_ARRAY_MANIFEST.json').read_text())
    assert all(file_sha(prior/'baseline-private'/name)==v for name,v in controls.items())
    contract=json.loads((r/'INPUT_CONTRACT.json').read_text())
    assets=prior/'server-assets';assert file_sha(assets/'ASSET_MANIFEST.json')==contract['asset_manifest_sha256']
    manifest=json.loads((assets/'ASSET_MANIFEST.json').read_text())
    assert file_sha(assets/'train-reference.npz')==manifest['train-reference.npz']['sha256']
    with np.load(assets/'train-reference.npz',allow_pickle=False) as ref:
        raw=ref['features'].copy();feature_ids=ref['sample_ids'].astype(str)
    labels=pd.read_csv(old/'data/Training_labels_updated.csv');ids=labels.sample_id.astype(str).tolist()
    target=labels.iloc[:,1:].to_numpy(dtype=float);weights=np.diff(np.log10(np.asarray(labels.columns[1:],dtype=float)))
    assert hashlib.sha256('\n'.join(ids).encode()).hexdigest()==baseline['row_order_sha256']
    assert hashlib.sha256(json.dumps(list(labels.columns[1:])).encode()).hexdigest()==baseline['target_order_sha256']
    x=np.load(prior/'baseline-private/train-features.npy',allow_pickle=False)
    base_pred=np.load(prior/'baseline-private/repeated-oof.npy',allow_pickle=False)
    base_losses=np.load(prior/'baseline-private/per-sample-losses.npy',allow_pickle=False)
    assert x.shape==(24,2562) and base_pred.shape==(20,24,11) and base_losses.shape==(20,24)
    np.testing.assert_array_equal(aggregate(raw,feature_ids,ids,(1,2,3,4)).astype(np.float64),x[:,:768])
    np.testing.assert_allclose(base_losses,np.asarray([sample_emd(target,p,weights) for p in base_pred]),rtol=0,atol=1e-12)
    joined=np.column_stack([aggregate(raw,feature_ids,ids,(0,1,2,3,4)),x[:,768:]])
    assert joined.shape==(24,2562)
    source=old/'src/soil_gsd_feature_fusion_audit.py'
    spec=importlib.util.spec_from_file_location('bound_historical_math',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    preds=[];losses=[];repeat_rows=[]
    for k,seed in enumerate(range(20260925,20260945)):
        pred=np.zeros_like(target);bpred=np.zeros_like(target);seen=np.zeros(24,dtype=int)
        for tr,va in KFold(5,shuffle=True,random_state=seed).split(target):
            assert set(tr).isdisjoint(va)
            bpred[va]=module.fold_predict(x[tr],target[tr],x[va],alpha=1.)
            pred[va]=module.fold_predict(joined[tr],target[tr],joined[va],alpha=1.);seen[va]+=1
        assert np.all(seen==1)
        np.testing.assert_allclose(bpred,base_pred[k],rtol=0,atol=1e-10)
        loss=sample_emd(target,pred,weights);losses.append(loss);preds.append(pred)
        repeat_rows.append({'seed':seed,'candidate_emd':float(loss.mean()),'baseline_emd':float(base_losses[k].mean())})
    effect=paired_cluster_bootstrap(base_losses,losses)
    out=r/'evaluation-private';assert not out.exists();out.mkdir()
    np.save(out/'train-features.npy',joined);np.save(out/'repeated-oof.npy',np.asarray(preds));np.save(out/'per-sample-losses.npy',np.asarray(losses))
    report={'experiment':'SOIL-ALL5-VIEWS-20261109','configuration_count':1,'features':2562,
      'baseline_mean_emd':float(base_losses.mean()),'candidate_mean_emd':float(np.mean(losses)),**effect,
      'repeat_results':repeat_rows,'four_crop_control_features_byte_exact':True,'baseline20partitions_recomputed_match':True,
      'source_manifest':seal,'train_reference_sha256':file_sha(assets/'train-reference.npz'),
      'output_sha256':{n:file_sha(out/n) for n in ['train-features.npy','repeated-oof.npy','per-sample-losses.npy']},
      'test_permission_issued':False,'independent_result_review_complete':False,
      'limitations':'Exploratory24historically selected samples,20reusedpartitions,priorselectionnotcorrectedbyCI;Android/iPhoneshift. Not anofficialscore/rank.'}
    (r/'CANDIDATE_EVALUATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['baseline_mean_emd','candidate_mean_emd','mean_emd_reduction','paired_cluster_95pct_ci','preregistered_gate_passed','test_permission_issued']}))


if __name__=='__main__':main()
