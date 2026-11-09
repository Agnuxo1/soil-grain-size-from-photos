"""CPU-only snapshot of completed, identity-checked managed inference results."""
if not __debug__:
    raise RuntimeError('Optimised Python is forbidden: collection assertions must remain active')
import fcntl
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from evaluation_core import verify_summary,file_sha

r=Path(__file__).resolve().parent
expected=json.loads((r/'EXPECTED.json').read_text())
job=json.loads((r/'JOB_READBACK.json').read_text())
stage=expected['stage'];assert stage in ['pilot','train']
assert job['id']==expected['job_id'] and job['state']=='succeeded' and job.get('exit_code')==0
assert job['project']=='soil-grain-size'
assert job['command']==['timeout','15m','/home/ai/venvs/training/bin/python','run_safe.py','--stage',stage]
contract=expected['contract'];sources=expected['source_manifest'];project=Path(contract['project'])
assert project==Path('/mnt/datos/proyectos/soil-grain-size-20261109')
assets=project/'assets';manifest=json.loads((assets/'ASSET_MANIFEST.json').read_text())
assert file_sha(assets/'ASSET_MANIFEST.json')==contract['asset_manifest_sha256']
assert all(file_sha(assets/name)==record['sha256'] for name,record in manifest.items())
with np.load(assets/'train-reference.npz',allow_pickle=False) as ref:names=ref['file_names'].astype(str).tolist()
source=project/stage
lock=(source/'writer.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_SH|fcntl.LOCK_NB)
summary=json.loads((source/'summary.json').read_text())
verify_summary(summary,contract,sources,names,stage)
destination=r/'payload';assert not destination.exists(),'Never overwrite an earlier snapshot'
destination.mkdir()
members=['summary.json','patch-statistics.npy','cls-control.npy','checkpoint.npz']
digests={name:file_sha(source/name) for name in members}
for name in ['patch-statistics.npy','cls-control.npy']:assert digests[name]==summary['outputs_sha256'][name]
for name in members:
    shutil.copy2(source/name,destination/name)
    assert file_sha(destination/name)==digests[name]==file_sha(source/name)
receipt={'job_id':job['id'],'stage':stage,'source_manifest':sources,'files_sha256':digests,
         'verified_after_managed_job_succeeded':True,'shared_reader_lock_acquired':True,
         'config_digest':summary['config_digest'],'no_model_inference_or_test_access':True}
(destination/'COLLECTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'snapshot_complete':True,'job_id':job['id'],'stage':stage,
                  'members':len(members)+1,'bytes':sum(p.stat().st_size for p in destination.iterdir())}))
