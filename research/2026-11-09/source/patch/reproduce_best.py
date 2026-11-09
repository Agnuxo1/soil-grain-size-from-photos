"""Reproduce the frozen best recipe without executing or overwriting the original main."""
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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

r=Path(__file__).resolve().parent;old=Path('D:/PROJECTS/183_Machine_Learning')
proof=json.loads((r/'CURRENT_DATA_VERIFICATION.json').read_text())
assert proof['pagination_complete'] and not proof['metadata_mismatches']
source=old/'src/soil_gsd_feature_fusion_audit.py'
spec=importlib.util.spec_from_file_location('historical_soil_fusion',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
paths=[source,old/'data/Training_labels_updated.csv',old/'data/sample_submission.csv',old/'data/ppm_updated.csv',
       old/'artifacts/public_outputs/dino/dinov2_physical_features.npz',old/'artifacts/pavlo_multitask_v1/features.npz',
       old/'artifacts/feature_fusion_audit_20260926/report.json',old/'artifacts/feature_fusion_audit_20260926/feature_concat_submission.csv']
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
for name,digest in proof['current_official_csv_bytes_exact'].items():assert hashes[str(old/'data'/name)]==digest
labels=pd.read_csv(old/'data/Training_labels_updated.csv');template=pd.read_csv(old/'data/sample_submission.csv')
ids=labels.sample_id.astype(str).tolist();test_ids=template.sample_id.astype(str).tolist()
assert len(ids)==len(set(ids))==24 and len(test_ids)==len(set(test_ids))==10 and set(ids).isdisjoint(test_ids)
columns=list(labels.columns[1:]);target=labels.iloc[:,1:].to_numpy(dtype=float)
assert target.shape==(24,11) and np.isfinite(target).all()
weights=np.diff(np.log10(np.asarray(columns,dtype=float)))
assert abs(weights.sum()-5)<1e-12 and np.all(weights>0)
dino=module.aggregate_dino(paths[4],ids);physical=module.aggregate_physical(paths[5],ids)
assert dino.shape==(24,768) and physical.shape==(24,1794)
x=np.column_stack([dino,physical]);historical=json.loads(paths[6].read_text())
reference={int(v['seed']):v for v in historical['repeat_results']}
losses=[];rows=[];predictions=[]
for seed in range(20260925,20260945):
    pred=np.zeros_like(target);counts=np.zeros(24,dtype=int)
    for tr,va in KFold(n_splits=5,shuffle=True,random_state=seed).split(target):
        assert set(tr).isdisjoint(va)
        pred[va]=module.fold_predict(x[tr],target[tr],x[va],alpha=1.)
        counts[va]+=1
    assert np.all(counts==1)
    per_sample=np.sum(np.abs(target[:,:-1]-pred[:,:-1])*weights,axis=1)
    score=float(per_sample.mean());expected=float(reference[seed]['feature_concat_emd'])
    assert abs(score-expected)<1e-8,'Historical repeated OOF drift'
    rows.append({'seed':seed,'emd':score,'historical_emd':expected,'absolute_difference':abs(score-expected)})
    losses.append(per_sample);predictions.append(pred)
assert abs(np.mean([q['emd'] for q in rows])-34.45786163217881)<1e-8
# Reproduce the already-submitted control only; these inputs never select a new configuration.
all_dino=module.aggregate_dino(paths[4],ids+test_ids);dtest=all_dino[len(ids):]
with np.load(paths[5],allow_pickle=True) as cache:
    feature_ids=cache['test_ids'].astype(str).tolist();assert len(set(feature_ids))==len(test_ids) and set(feature_ids)==set(test_ids)
    positions={name:i for i,name in enumerate(feature_ids)}
    ptest=np.vstack([cache['test'][positions[name]] for name in test_ids]).astype(np.float64)
xtest=np.column_stack([dtest,ptest]);scaler=StandardScaler().fit(x)
model=Ridge(alpha=1.,solver='lsqr').fit(scaler.transform(x),module.encode_sqrt_mass(target))
computed=module.decode_sqrt_mass(model.predict(scaler.transform(xtest)))
archived=pd.read_csv(paths[7]);assert archived.sample_id.astype(str).tolist()==test_ids
assert list(archived.columns)==list(template.columns)
difference=float(np.max(np.abs(computed-archived[columns].to_numpy(dtype=float))))
out=r/'baseline-private';out.mkdir(exist_ok=True)
reconstructed=template.copy();reconstructed.loc[:,columns]=computed
csv=out/'control-recomputed.csv';reconstructed.to_csv(csv,index=False,float_format='%.6f')
assert csv.read_bytes()==paths[7].read_bytes(),'Best official control CSV bytes differ under its original six-decimal writer'
assert difference<=.500001e-6,'Difference exceeds the recorded six-decimal quantisation'
np.save(out/'repeated-oof.npy',np.asarray(predictions));np.save(out/'per-sample-losses.npy',np.asarray(losses))
np.save(out/'train-features.npy',x)
summary={'scope':'Reproduction of historical exploratory repeated sample-level CV; all20partitions reuse24samples',
 'samples':24,'repeats':20,'features':2562,'mean_emd':float(np.mean(losses)),
 'historical_mean_emd':34.45786163217881,'repeated_metrics_match':True,'repeats_details':rows,
 'raw_prediction_vs_six_decimal_csv_maximum_percentage_difference':difference,
 'test_control_csv_bytes_exact':True,'test_control_values_reproduced':True,
 'reconstructed_csv_sha256':hashlib.sha256(csv.read_bytes()).hexdigest(),
 'source_and_artifact_sha256':hashes,
 'runtime':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn']},
 'row_order_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
 'target_order_sha256':hashlib.sha256(json.dumps(columns).encode()).hexdigest(),
 'new_candidate_selected_or_submitted':False}
(r/'BASELINE_REPRODUCTION.json').write_text(json.dumps(summary,indent=2)+'\n')
assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'Original inputs changed during recovery'
print(json.dumps({k:summary[k] for k in ['samples','repeats','mean_emd','repeated_metrics_match','raw_prediction_vs_six_decimal_csv_maximum_percentage_difference','test_control_csv_bytes_exact','new_candidate_selected_or_submitted']}))
