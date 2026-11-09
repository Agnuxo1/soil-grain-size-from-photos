"""CPU-only snapshot of completed managed large-CLS outputs."""
if not __debug__:
    raise RuntimeError('Optimised Python is forbidden')
import fcntl
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from admission import admit_smoke,sha
r=Path(__file__).resolve().parent
expected=json.loads((r/'EXPECTED.json').read_text());job=json.loads((r/'JOB_READBACK.json').read_text())
stage=expected['stage'];assert stage in ['smoke','train']
assert job['id']==expected['job_id'] and job['state']=='succeeded' and job.get('exit_code')==0
assert job['project']=='soil-grain-size' and job['command']==['timeout','15m','/home/ai/venvs/training/bin/python','run_safe.py','--stage',stage]
contract=expected['contract'];sources=expected['source_manifest'];project=Path(contract['project'])
assert project==Path('/mnt/datos/proyectos/soil-large-20261109')
assets=Path(contract['assets']);assert sha(assets/'ASSET_MANIFEST.json')==contract['asset_manifest_sha256']
with np.load(assets/'train-reference.npz',allow_pickle=False) as ref:names=ref['file_names'].astype(str).tolist()
count=4 if stage=='smoke' else 127
identity={'experiment':contract['experiment'],'stage':stage,'rows':count,'sources':sources,
 'contract_sha256':expected['contract_sha256'],'runtime':contract['runtime'],'timm_version':contract['timm_version'],
 'row_order_sha256':hashlib.sha256('\n'.join(names[:count]).encode()).hexdigest()}
digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
source=project/stage;lock=(source/'writer.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_SH|fcntl.LOCK_NB)
summary=json.loads((source/'summary.json').read_text())
assert summary['identity']==identity and summary['config_digest']==digest
assert summary['checkpoint_complete'] is True and summary['exclusive_writer_lock'] is True
assert summary['accuracy_evidence'] is False and summary['precision_gate_passed'] is True
assert np.isfinite(summary['minimum_precision_cosine']) and summary['minimum_precision_cosine']>=.9999
assert summary['images']==count and summary['feature_shape']==[count,5,1024]
assert set(summary['outputs_sha256'])=={'cls.npy','checkpoint.npz'}
for name,expected_sha in summary['outputs_sha256'].items():assert sha(source/name)==expected_sha
if stage=='smoke':admit_smoke(source,identity)
cls=np.load(source/'cls.npy',allow_pickle=False)
assert cls.shape==(count,5,1024) and cls.dtype==np.float32 and np.isfinite(cls).all()
assert np.all(np.linalg.norm(cls,axis=-1)>1e-12)
with np.load(source/'checkpoint.npz',allow_pickle=False) as checkpoint:
    assert str(checkpoint['digest'].item())==digest and int(checkpoint['processed'])==count
    assert np.array_equal(checkpoint['features'],cls)
    assert abs(float(checkpoint['minimum_precision_cosine'])-summary['minimum_precision_cosine'])<1e-8
destination=r/'payload';assert not destination.exists();destination.mkdir()
members=['summary.json','cls.npy','checkpoint.npz'];digests={name:sha(source/name) for name in members}
for name in members:
    shutil.copy2(source/name,destination/name);assert sha(destination/name)==digests[name]==sha(source/name)
receipt={'job_id':job['id'],'stage':stage,'source_manifest':sources,'contract_sha256':expected['contract_sha256'],
 'files_sha256':digests,'managed_succeeded_exit0_verified':True,'completed_checkpoint_verified':True,
 'shared_reader_lock':True,'config_digest':digest}
(destination/'COLLECTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'snapshot_complete':True,'stage':stage,'job_id':job['id'],'members':4,
                  'bytes':sum(p.stat().st_size for p in destination.iterdir())}))
