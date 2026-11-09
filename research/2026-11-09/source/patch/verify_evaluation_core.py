"""Independent fixtures for scientific units and admission identity; CPU only."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluation_core import aggregate_patch_stats,sample_emd,paired_cluster_bootstrap,verify_summary,file_sha

r=Path(__file__).resolve().parent
weights=np.diff(np.log10([.002,.0063,.02,.063,.2,.63,2,6.3,20,63,200]))
a=np.zeros((2,11));a[:,-1]=100
b=np.full((2,11),100.)
assert np.array_equal(sample_emd(a,a,weights),[0,0])
np.testing.assert_allclose(sample_emd(a,b,weights),[500,500],rtol=0,atol=1e-12)
c=a.copy();c[:,0]=20;c[:,1:]=100
expected=20*np.log10(.0063/.002)+100*np.log10(200/.0063)
np.testing.assert_allclose(sample_emd(a,c,weights),[expected,expected],rtol=0,atol=1e-10)
negative=a.copy();negative[:,1]=-1
try:sample_emd(a,negative,weights)
except AssertionError:pass
else:raise AssertionError('Invalid CDF accepted')

stats=np.zeros((3,5,768),dtype=np.float32)
stats[:,0,:]=9999 # Full-frame view is explicitly excluded.
stats[0,1:,:]=2;stats[1,1:,:]=6;stats[2,1:,:]=10
out=aggregate_patch_stats(stats,['A','A','B'],['B','A'])
np.testing.assert_array_equal(out[0,:768],np.full(768,10))
np.testing.assert_array_equal(out[0,768:],np.zeros(768))
np.testing.assert_array_equal(out[1,:768],np.full(768,4))
np.testing.assert_array_equal(out[1,768:],np.full(768,2))
np.testing.assert_array_equal(aggregate_patch_stats(stats[[2,0,1]],['B','A','A'],['B','A']),out)

base=np.full((20,24),10.)
strong=paired_cluster_bootstrap(base,base-2)
assert strong['preregistered_gate_passed'] and strong['paired_cluster_95pct_ci']==[2.,2.]
small=paired_cluster_bootstrap(base,base-.5);assert not small['preregistered_gate_passed']
one=np.zeros((20,24));one[:,0]=48
sparse=paired_cluster_bootstrap(base,base-one)
assert sparse['mean_emd_reduction']==2 and sparse['paired_cluster_95pct_ci'][0]==0
assert not sparse['preregistered_gate_passed'],'One outlier cluster cannot pass CI gate'
assert paired_cluster_bootstrap(base[::-1],(base-one)[::-1])==sparse

contract=json.loads((r/'gpu-source-v2/RUN_CONTRACT.json').read_text())
sources=json.loads((r/'GPU_SOURCE_MANIFEST_V2.json').read_text())
with np.load(r/'server-assets/train-reference.npz',allow_pickle=False) as ref:
    names=ref['file_names'].astype(str).tolist()
identity={'experiment':'SOIL-PATCH-STATS-20261109','stage':'train','rows':127,'sources':sources,
 'asset_manifest_sha256':contract['asset_manifest_sha256'],'weights_sha256':contract['weights_sha256'],
 'model_config_sha256':contract['model_config_sha256'],'runtime':contract['runtime'],
 'timm_version':'1.0.30','row_order_sha256':hashlib.sha256('\n'.join(names).encode()).hexdigest(),
 'batch_images':4,'workers':0,'precision':contract['precision']}
summary={'identity':identity,'config_digest':hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),
 'experiment':'SOIL-PATCH-STATS-20261109','stage':'train','images':127,'descriptor_shape':[127,5,768],
 'checkpoint_complete':True,'exclusive_writer_lock':True,'accuracy_evidence':False,
 'feasibility_gate_passed':True,'minimum_cls_cosine':.99995}
verify_summary(summary,contract,sources,names,'train')
rejected=[]
for name in ['stage','rows','sources','asset_manifest_sha256','weights_sha256','model_config_sha256',
             'runtime','timm_version','row_order_sha256','batch_images','workers','precision']:
    bad=copy.deepcopy(summary);bad['identity'][name]='mismatch'
    bad['config_digest']=hashlib.sha256(json.dumps(bad['identity'],sort_keys=True).encode()).hexdigest()
    try:verify_summary(bad,contract,sources,names,'train')
    except (AssertionError,TypeError,ValueError):rejected.append(name)
    else:raise AssertionError('Stale identity accepted: '+name)
for name,value in [('minimum_cls_cosine',float('nan')),('minimum_cls_cosine',.99),
 ('checkpoint_complete',False),('feasibility_gate_passed',False),('descriptor_shape',[20,5,768])]:
    bad=copy.deepcopy(summary);bad[name]=value
    try:verify_summary(bad,contract,sources,names,'train')
    except (AssertionError,TypeError,ValueError):rejected.append(name)
    else:raise AssertionError('Invalid summary accepted: '+name)
record={'cpu_only':True,'emd_known_fixtures':3,'invalid_cdf_rejected':True,
 'pooling_fullframe_exclusion_and_photo_order_checks':True,'bootstrap_cluster_fixtures':4,
 'actual_identity_positive_fixture_passed':True,'adverse_identity_or_gate_cases_rejected':len(rejected),
 'real_training_bundle_available_or_evaluated':False,'new_test_permission_issued':False}
(r/'EVALUATION_CORE_CHECKS.json').write_text(json.dumps(record,indent=2)+'\n')
baseline=r/'baseline-private'
(r/'BASELINE_LOCAL_ARRAY_MANIFEST.json').write_text(json.dumps({name:file_sha(baseline/name) for name in
 ['train-features.npy','repeated-oof.npy','per-sample-losses.npy']},indent=2)+'\n')
print(json.dumps(record))
