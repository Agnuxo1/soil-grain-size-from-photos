if not __debug__:
    raise RuntimeError('Optimised Python is forbidden')
import hashlib
import json
import shutil
import sys
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parent;sys.path.insert(0,str(r/'gpu-source'))
from admission import admit_smoke,sha
job=json.loads((r/'SMOKE_JOB_READBACK.json').read_text())
assert job['state']=='succeeded' and job['exit_code']==0 and job['project']=='soil-grain-size'
assert job['command']==['timeout','15m','/home/ai/venvs/training/bin/python','run_safe.py','--stage','smoke']
gpu=r/'gpu-source';sources=json.loads((r/'GPU_SOURCE_MANIFEST.json').read_text())
assert all(sha(gpu/name)==digest for name,digest in sources.items())
contract=json.loads((gpu/'RUN_CONTRACT.json').read_text())
with np.load(r.parent/'soil-rescue-20261109/server-assets/train-reference.npz',allow_pickle=False) as ref:
    names=ref['file_names'].astype(str).tolist()
identity={'experiment':contract['experiment'],'stage':'smoke','rows':4,'sources':sources,
 'contract_sha256':sha(gpu/'RUN_CONTRACT.json'),'runtime':contract['runtime'],'timm_version':contract['timm_version'],
 'row_order_sha256':hashlib.sha256('\n'.join(names[:4]).encode()).hexdigest()}
bundle=r/'smoke-download-001/payload'
receipt=json.loads((bundle/'COLLECTION_RECEIPT.json').read_text())
assert receipt['job_id']==job['id'] and receipt['stage']=='smoke' and receipt['source_manifest']==sources
assert receipt['contract_sha256']==identity['contract_sha256']
assert all(sha(bundle/name)==digest for name,digest in receipt['files_sha256'].items())
summary=admit_smoke(bundle,identity)
proof={'job_id':job['id'],'state':'succeeded','source_model_runtime_data_order_bound':True,
 'complete_smoke_artifacts_and_checkpoint_verified':True,'minimum_precision_cosine':summary['minimum_precision_cosine'],
 'peak_allocated_gpu_bytes':summary['peak_allocated_gpu_bytes'],'rows':4,'accuracy_evidence':False,
 'train_extraction_admitted':True,'test_extraction_admitted':False,'config_digest':summary['config_digest']}
(r/'SMOKE_VERIFICATION.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
