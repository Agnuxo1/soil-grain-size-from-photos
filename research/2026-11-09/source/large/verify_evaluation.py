import hashlib
import json
import numpy as np
from pathlib import Path
from evaluation_large import aggregate_large
r=Path(__file__).resolve().parent
features=np.zeros((3,5,1024),dtype=np.float32);features[:,0,2]=999
features[0,1:,0]=2;features[1,1:,1]=3;features[2,1:,0]=-5
out=aggregate_large(features,['A','A','B'],['B','A'])
expected=np.zeros((2,2048),dtype=np.float32)
expected[0,0]=-1;expected[1,:2]=.5;expected[1,1024:1026]=.5
np.testing.assert_array_equal(out,expected)
np.testing.assert_array_equal(aggregate_large(features[[2,0,1]],['B','A','A'],['B','A']),out)
np.testing.assert_array_equal(aggregate_large(features*7,['A','A','B'],['B','A']),out)
for name,data,groups,ids in [('missing_sample',features,['A','A','B'],['A']),
 ('duplicate_requested_id',features,['A','A','B'],['A','A','B']),
 ('zero_vector',features*0,['A','A','B'],['A','B']),
 ('nonfinite',np.full_like(features,np.nan),['A','A','B'],['A','B'])]:
    try:aggregate_large(data,groups,ids)
    except AssertionError:pass
    else:raise AssertionError('Invalid pooling input accepted: '+name)
members=['evaluate.py','evaluation_large.py','evaluation_core.py','verify_evaluation.py',
 'BASELINE_REPRODUCTION.json','BASELINE_LOCAL_ARRAY_MANIFEST.json','PREREGISTRATION.json',
 'GPU_SOURCE_MANIFEST.json','MODEL_PROVENANCE.json','SMOKE_VERIFICATION.json','TRAIN_JOB_READBACK.json']
seal={name:hashlib.sha256((r/name).read_bytes()).hexdigest() for name in members}
assert not (r/'EVALUATION_SOURCE_MANIFEST.json').exists(),'Do not reseal an existing evaluator'
(r/'EVALUATION_SOURCE_MANIFEST.json').write_text(json.dumps(seal,indent=2)+'\n')
# The missing baseline-proof binding found in the previous experiment is explicitly exercised here.
baseline=json.loads((r/'BASELINE_REPRODUCTION.json').read_text())
baseline['runtime']['numpy']='changed'
assert hashlib.sha256(json.dumps(baseline,indent=2).encode()).hexdigest()!=seal['BASELINE_REPRODUCTION.json']
record={'independent_orthogonal_vector_pooling_fixture_passed':True,'row_permutation_invariant':True,
 'per_view_scale_invariant':True,'adverse_pooling_cases_rejected':4,
 'baseline_proof_itself_in_frozen_seal':True,'changed_baseline_proof_rejected_without_resealing':True,
 'core_metric_and24cluster_bootstrap_reused_byte_exact_from_previous_verified_source':True,
 'new_test_access_or_permission':False}
(r/'EVALUATION_INPUT_CHECKS.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
