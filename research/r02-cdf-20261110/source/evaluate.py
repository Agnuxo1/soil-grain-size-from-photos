"""Execute the single preregistered training-only R02 comparison."""
if not __debug__:
    raise RuntimeError('Optimized Python is forbidden')
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import control
from readout import fold_predict


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def execute():
    started=time.monotonic();s=Path(__file__).resolve().parent;r=s.parent
    seal=json.loads((s/'SOURCE_MANIFEST.json').read_text())
    def verify():
        actual={p.name for p in s.iterdir() if p.is_file()}
        assert actual==set(seal)|{'SOURCE_MANIFEST.json'},'Source membership drift'
        assert all(sha(s/n)==v for n,v in seal.items()),'Source digest drift'
        assert not any(p.is_dir() for p in s.iterdir()),'Unexpected source directory'
        contract=json.loads((s/'INPUTS.json').read_text())
        assert {p.name for p in (r/'inputs').iterdir()}==set(contract['files'])
        assert all(sha(r/'inputs'/n)==x['sha256'] for n,x in contract['files'].items()),'Input drift'
        assert all(sha(Path(x['original']))==x['sha256'] for x in contract['files'].values()),'Original input drift'
        assert {n:importlib.metadata.version(n) for n in contract['runtime']}==contract['runtime'],'Runtime drift'
        return contract
    contract=verify()
    assert not (r/'result').exists(),'Closed result must not be overwritten'
    lock=r/'evaluation.lock'
    with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    # Importing the control module only defines its functions; its main is not run.
    x=np.load(r/'inputs/features.npy',allow_pickle=False)
    baseline_oof=np.load(r/'inputs/control_oof.npy',allow_pickle=False)
    baseline_loss=np.load(r/'inputs/control_losses.npy',allow_pickle=False)
    labels=pd.read_csv(r/'inputs/labels.csv');ids=labels.sample_id.astype(str).tolist()
    columns=labels.columns[1:].tolist();y=labels.iloc[:,1:].to_numpy(dtype=float)
    weights=np.diff(np.log10(np.asarray(columns,dtype=float)))
    assert x.shape==(24,2562) and y.shape==(24,11)
    assert baseline_oof.shape==(20,24,11) and baseline_loss.shape==(20,24)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert hashlib.sha256('\n'.join(ids).encode()).hexdigest()==contract['row_order_sha256']
    assert hashlib.sha256(json.dumps(columns).encode()).hexdigest()==contract['target_order_sha256']
    assert weights.shape==(10,) and (weights>0).all() and abs(weights.sum()-5)<1e-12
    def loss(pred):
        assert pred.shape==(24,11) and np.isfinite(pred).all()
        assert (pred>=-1e-7).all() and (pred<=100+1e-7).all()
        assert (np.diff(pred,axis=1)>=-1e-7).all() and np.allclose(pred[:,-1],100,atol=1e-7,rtol=0)
        return (np.abs(y[:,:10]-pred[:,:10])*weights).sum(axis=1)
    np.testing.assert_allclose(baseline_loss,np.asarray([loss(p) for p in baseline_oof]),rtol=0,atol=1e-12)
    assert abs(float(baseline_loss.mean())-contract['baseline_mean_emd'])<1e-10
    predictions=[];losses=[];max_control_difference=0.;fold_count=0
    for k,seed in enumerate(range(20260925,20260945)):
        b=np.zeros_like(y);p=np.zeros_like(y);seen=np.zeros(24,dtype=int)
        for tr,va in KFold(5,shuffle=True,random_state=seed).split(x):
            assert set(tr).isdisjoint(va) and len(tr)+len(va)==24
            b[va]=control.fold_predict(x[tr],y[tr],x[va],alpha=1.)
            p[va]=fold_predict(x[tr],y[tr],x[va],weights)
            seen[va]+=1;fold_count+=1
            if time.monotonic()-started>540:raise TimeoutError('Preregistered CPU wall time exceeded')
        assert (seen==1).all()
        np.testing.assert_allclose(b,baseline_oof[k],atol=1e-10,rtol=0)
        max_control_difference=max(max_control_difference,float(np.max(np.abs(b-baseline_oof[k]))))
        predictions.append(p);losses.append(loss(p))
    assert fold_count==100
    predictions=np.asarray(predictions);losses=np.asarray(losses)
    paired=(baseline_loss-losses).mean(axis=0)
    rng=np.random.default_rng(20261110)
    samples=rng.integers(0,24,size=(10000,24))
    ci=np.quantile(paired[samples].mean(axis=1),[.025,.975])
    delta=float(paired.mean());gate=bool(delta>=1. and ci[0]>0.)
    verify()
    out=r/'result';out.mkdir(exist_ok=False)
    np.save(out/'repeated-oof.npy',predictions)
    np.save(out/'per-sample-losses.npy',losses)
    report={'experiment':'SOIL-R02-CDF-20261110','configuration_count':1,'fold_count':fold_count,
      'baseline_mean_emd':float(baseline_loss.mean()),'candidate_mean_emd':float(losses.mean()),
      'mean_emd_reduction':delta,'paired_cluster_95pct_ci':ci.tolist(),
      'preregistered_gate_passed':gate,'bootstrap_seed':20261110,'bootstrap_replicates':10000,
      'clusters':24,'repeats_reuse_same_clusters':20,
      'per_sample_mean_improvement':paired.tolist(),
      'repeat_candidate_means':losses.mean(axis=1).tolist(),
      'control_all100fits_reproduced':True,'control_max_abs_difference':max_control_difference,
      'no_test_access':True,'test_permission_issued':False,'independent_result_review_complete':False,
      'elapsed_seconds':time.monotonic()-started,'runtime':contract['runtime'],
      'source_manifest_sha256':sha(s/'SOURCE_MANIFEST.json'),
      'input_manifest_sha256':sha(s/'INPUTS.json'),
      'output_sha256':{n:sha(out/n) for n in ['repeated-oof.npy','per-sample-losses.npy']},
      'limitations':'Exploratory24reusedsamples,overlappingfolds,priorselectionnotcorrectedbyCI,Android/iPhonedomainshift;notindependentcohortorofficialresult.'}
    (out/'EVALUATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['baseline_mean_emd','candidate_mean_emd',
       'mean_emd_reduction','paired_cluster_95pct_ci','preregistered_gate_passed','elapsed_seconds']}))


if __name__=='__main__':execute()
