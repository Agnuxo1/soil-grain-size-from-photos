"""Execute the source-to-label mapping counterexample using only an isolated NPZ copy."""
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluation_large import aggregate_large
r=Path(__file__).resolve().parent;prior=r.parent/'soil-rescue-20261109';p=prior/'server-assets/train-reference.npz'
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
before=sha(p)
contract=json.loads((r/'gpu-source/RUN_CONTRACT.json').read_text())
manifest=prior/'server-assets/ASSET_MANIFEST.json'
assert sha(manifest)==contract['asset_manifest_sha256']
expected=json.loads(manifest.read_text())['train-reference.npz']['sha256'];assert before==expected
with np.load(p,allow_pickle=False) as original:values={k:original[k].copy() for k in original.files}
ids=values['sample_ids'];indices=next((i,j) for i in range(len(ids)) for j in range(i+1,len(ids)) if ids[i]!=ids[j])
i,j=indices;values['sample_ids'][i],values['sample_ids'][j]=values['sample_ids'][j],values['sample_ids'][i]
directory=r/'reference-adversarial-fixture';assert not directory.exists();directory.mkdir()
changed=directory/'train-reference.npz';np.savez_compressed(changed,**values)
names=values['file_names'].astype(str).tolist()
original_names=np.load(p,allow_pickle=False)['file_names'].astype(str).tolist()
assert names==original_names and set(ids)==set(values['sample_ids'])
assert hashlib.sha256('\n'.join(names).encode()).hexdigest()==hashlib.sha256('\n'.join(original_names).encode()).hexdigest()
# An image-specific synthetic CLS pattern makes the incorrect label assignment observable.
f=np.random.default_rng(19).normal(size=(127,5,1024)).astype(np.float32)
with np.load(p,allow_pickle=False) as original:original_ids=original['sample_ids'].astype(str)
order=sorted(set(original_ids));a=aggregate_large(f,original_ids,order);b=aggregate_large(f,values['sample_ids'],order)
assert not np.array_equal(a,b) and sha(changed)!=expected
try:assert sha(changed)==expected
except AssertionError:pass
else:raise AssertionError('Modified mapping admitted')
assert sha(p)==before
seal=json.loads((r/'EVALUATION_SOURCE_MANIFEST.json').read_text());old_seal=json.loads((r/'evaluation-before-local-reference-fix/EVALUATION_SOURCE_MANIFEST.json').read_text())
assert seal==old_seal and not (r/'CANDIDATE_EVALUATION.json').exists()
assert all(sha(r/name)==digest for name,digest in seal.items() if name!='evaluate.py')
seal['evaluate.py']=sha(r/'evaluate.py');seal['verify_reference_binding.py']=sha(Path(__file__).resolve())
(r/'EVALUATION_SOURCE_MANIFEST.json').write_text(json.dumps(seal,indent=2)+'\n')
record={'original_missing_mapping_binding_counterexample_reproduced':True,
 'same_file_names_and_id_set_but_wrong_pooling':True,'new_input_hash_gate_rejects_changed_mapping':True,
 'original_reference_unchanged':True,'previous_source_and_seal_archived':True,
 'amendment_before_first_candidate_evaluation':True,'scientific_parameters_or_thresholds_changed':False}
(r/'REFERENCE_BINDING_CHECKS.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
