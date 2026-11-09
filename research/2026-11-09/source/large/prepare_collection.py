import argparse
import hashlib
import json
import shutil
from pathlib import Path
r=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','train']);a=p.parse_args()
gpu=r/'gpu-source';manifest=json.loads((r/'GPU_SOURCE_MANIFEST.json').read_text())
assert all(hashlib.sha256((gpu/name).read_bytes()).hexdigest()==v for name,v in manifest.items())
job=json.loads((r/(a.stage.upper()+'_JOB_READBACK.json')).read_text())
assert job['state']=='succeeded' and job['exit_code']==0
destination=r/('collect-'+a.stage+'-source');assert not destination.exists();destination.mkdir()
shutil.copy2(r/'collect-source/collect.py',destination/'collect.py')
shutil.copy2(gpu/'admission.py',destination/'admission.py')
shutil.copy2(r/(a.stage.upper()+'_JOB_READBACK.json'),destination/'JOB_READBACK.json')
expected={'stage':a.stage,'job_id':job['id'],'contract':json.loads((gpu/'RUN_CONTRACT.json').read_text()),
 'source_manifest':manifest,'contract_sha256':hashlib.sha256((gpu/'RUN_CONTRACT.json').read_bytes()).hexdigest()}
(destination/'EXPECTED.json').write_text(json.dumps(expected,indent=2)+'\n')
print(json.dumps({'stage':a.stage,'prepared':True,'job_id':job['id']}))
