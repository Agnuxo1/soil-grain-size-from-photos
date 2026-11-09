import hashlib
import json
from pathlib import Path
import numpy as np
from pooling import aggregate
r=Path(__file__).resolve().parent
ids=[f'S{i}' for i in range(24)];groups=np.asarray([ids[i%24] for i in range(127)])
raw=np.zeros((127,5,384),dtype=np.float32);raw[:,0,0]=1;raw[:,1:,1]=1
four=aggregate(raw,groups,ids,(1,2,3,4));five=aggregate(raw,groups,ids,(0,1,2,3,4))
expected4=np.zeros((24,768),dtype=np.float32);expected4[:,1]=1
expected5=np.zeros((24,768),dtype=np.float32);expected5[:,0]=.2;expected5[:,1]=.8;expected5[:,384:386]=.4
np.testing.assert_array_equal(four,expected4);np.testing.assert_allclose(five,expected5,rtol=0,atol=1e-7)
np.testing.assert_allclose(aggregate(raw*7,groups,ids,(0,1,2,3,4)),five,rtol=0,atol=1e-7)
np.testing.assert_allclose(aggregate(raw[::-1],groups[::-1],ids,(0,1,2,3,4)),five,rtol=0,atol=1e-7)
rejected=0
for views in [(0,1),(1,2,2,4),(0,1,2,3,4,5)]:
    try:aggregate(raw,groups,ids,views)
    except AssertionError:rejected+=1
    else:raise AssertionError('Unregistered views accepted')
prior=r.parent/'soil-rescue-20261109';assets=prior/'server-assets'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
contract=json.loads((r/'INPUT_CONTRACT.json').read_text());assert sha(assets/'ASSET_MANIFEST.json')==contract['asset_manifest_sha256']
manifest=json.loads((assets/'ASSET_MANIFEST.json').read_text());assert sha(assets/'train-reference.npz')==manifest['train-reference.npz']['sha256']
baseline=json.loads((r/'BASELINE_REPRODUCTION.json').read_text());labels_path=Path('D:/PROJECTS/183_Machine_Learning/data/Training_labels_updated.csv')
assert sha(labels_path)==baseline['source_and_artifact_sha256'][str(labels_path)]
import pandas as pd
actual_ids=pd.read_csv(labels_path).sample_id.astype(str).tolist()
with np.load(assets/'train-reference.npz',allow_pickle=False) as reference:
    actual4=aggregate(reference['features'],reference['sample_ids'].astype(str),actual_ids,(1,2,3,4))
controls=json.loads((r/'BASELINE_LOCAL_ARRAY_MANIFEST.json').read_text())
assert sha(prior/'baseline-private/train-features.npy')==controls['train-features.npy']
np.testing.assert_array_equal(actual4.astype(np.float64),np.load(prior/'baseline-private/train-features.npy',allow_pickle=False)[:,:768])
members=['evaluate.py','pooling.py','evaluation_core.py','verify_pooling.py','PREREGISTRATION.json',
         'BASELINE_REPRODUCTION.json','BASELINE_LOCAL_ARRAY_MANIFEST.json','INPUT_CONTRACT.json']
seal={name:sha(r/name) for name in members}
assert not (r/'SOURCE_MANIFEST.json').exists();(r/'SOURCE_MANIFEST.json').write_text(json.dumps(seal,indent=2)+'\n')
record={'independent_four_vs_five_orthogonal_vector_fixture_passed':True,'view0_included_only_in_new_candidate':True,
 'scale_and_photo_order_invariant':True,'unregistered_viewconfigurations_rejected':rejected,
 'actual_old4crop_features_exactly_equal_frozen_control':True,'no_test_data_access':True}
(r/'POOLING_CHECKS.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
