"""One fixed SAM physical-geometry study, private training photos only.
Author: Francisco Angulo de Lafuente. Assistance: Codex. SPDX-License-Identifier: Apache-2.0
"""
import sys
if sys.flags.optimize:raise SystemExit('Optimized Python forbidden')
sys.dont_write_bytecode=True
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import resource
import time
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source_pins=json.loads((root/'SOURCE_MANIFEST.json').read_text())
actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
assert actual==set(source_pins)|{'SOURCE_MANIFEST.json'},'Unexpected source/importable files'
assert all(sha(root/name)==digest for name,digest in source_pins.items()),'Source pin mismatch'
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','train'],required=True);p.add_argument('--pilot-job-id');p.add_argument('--validate-only',action='store_true');args=p.parse_args()
contract=json.loads((root/'CONTRACT.json').read_text())
project=Path(contract['project']);assets=Path(contract['assets']);out=project/'features';out.mkdir(exist_ok=True)
assert sha(assets/'ASSET_MANIFEST.json')==contract['asset_manifest_sha256']
asset_manifest=json.loads((assets/'ASSET_MANIFEST.json').read_text())
assert all(sha(assets/name)==value['sha256'] for name,value in asset_manifest.items()),'Training input changed'
assert sha(Path(contract['model']['path']))==contract['model']['sha256'],'Model bytes changed'
runtime={name:importlib.metadata.version(name) for name in contract['runtime']}
assert runtime==contract['runtime'],'Runtime changed'
import numpy as np
from PIL import Image,ImageOps
from morphometry import describe_masks
from preprocess import image_index,physical_crops
with np.load(assets/'train-reference.npz',allow_pickle=False) as ref:
    names=ref['file_names'].astype(str).tolist();sample_ids=ref['sample_ids'].astype(str).tolist()
assert len(names)==len(set(names))==127 and len(set(sample_ids))==24
frame=image_index({'paths':names,'sample_ids':sample_ids},assets)
identity={'experiment':contract['experiment'],'sources':source_pins,'model':contract['model'],'runtime':runtime,
          'asset_manifest_sha256':contract['asset_manifest_sha256'],'sam_settings':contract['sam_settings'],
          'geometry':contract['geometry'],'row_order_sha256':hashlib.sha256('\n'.join(names).encode()).hexdigest(),
          'sample_mapping_sha256':hashlib.sha256('\n'.join(sample_ids).encode()).hexdigest(),'rows':127}
digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()

def crop(index):
    row=frame.iloc[index]
    with Image.open(row['path']) as image:rgb=ImageOps.exif_transpose(image).convert('RGB')
    scale=5.0/float(row['estimated_ppm']);rgb=rgb.resize((round(rgb.width*scale),round(rgb.height*scale)),Image.Resampling.LANCZOS)
    assert min(rgb.size)>=518,'Physical image too small; no silent padding'
    value=np.asarray(physical_crops(rgb,518)[2],dtype=np.uint8)
    assert value.shape==(518,518,3)
    return value

if args.validate_only:
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU preflight must mask CUDA'
    for i in (0,126):crop(i)
    sys.path.insert(0,str(root/'vendor'))
    from segment_anything import SamAutomaticMaskGenerator,sam_model_registry
    result={'rows':127,'physical_samples':24,'training_assets_verified':len(asset_manifest),'model_SHA_verified':True,'runtime':runtime,'endpoint_crops_518_RGB_uint8':True,'SAM_import_success':True,'GPU_used':False,'config_digest':digest}
    (out/'CPU_PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));raise SystemExit(0)
lock=(out/'writer.lock').open('a')
try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit(73)
if args.stage=='train':
    assert args.pilot_job_id,'Completed pilot ID required'
    metadata=json.loads((Path('/home/ai/servidor-ia/jobs')/args.pilot_job_id/'job.json').read_text())
    assert metadata['state']=='succeeded' and metadata.get('exit_code')==0 and metadata['project']=='soil-grain-size'
    assert metadata['command'][-2:]==['--stage','pilot'] and metadata['command'][-3]=='extract.py'
    pilot_source=Path('/home/ai/servidor-ia/jobs')/args.pilot_job_id/'work'
    assert all(sha(pilot_source/name)==value for name,value in source_pins.items()),'Pilot source mismatch'
    pilot=json.loads((out/'pilot.json').read_text());assert pilot['config_digest']==digest and pilot['feasibility_gate_passed']
features=np.full((127,44),np.nan,dtype=np.float64);counts=np.full(127,-1,dtype=np.int64);processed=0
checkpoint=out/'checkpoint.npz'
if args.stage=='pilot' and checkpoint.is_file():
    raise SystemExit('Pilot is one attempt: keep interrupted/failed evidence; no timing reset by resume')
if checkpoint.is_file():
    with np.load(checkpoint,allow_pickle=False) as saved:
        assert str(saved['digest'].item())==digest,'Checkpoint identity changed'
        features=saved['features'].copy();counts=saved['counts'].copy();processed=int(saved['processed'])
    assert features.shape==(127,44) and counts.shape==(127,) and 0<=processed<=127
    assert np.isfinite(features[:processed]).all() and (counts[:processed]>=0).all()
stop=3 if args.stage=='pilot' else 127
assert processed<stop,'Stage already complete; do not repeat'
import torch
torch.set_num_threads(4);torch.manual_seed(20261110);np.random.seed(20261110)
assert torch.cuda.is_available(),'Managed CUDA needed; no fallback'
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
sys.path.insert(0,str(root/'vendor'))
from segment_anything import SamAutomaticMaskGenerator,sam_model_registry
model=sam_model_registry['vit_b'](checkpoint=contract['model']['path']).eval().cuda()
generator=SamAutomaticMaskGenerator(model,**contract['sam_settings'])
started=time.monotonic()
print(json.dumps({'stage':args.stage,'status':'inference_started','resume_row':processed,'stop':stop,'config_digest':digest}),flush=True)
with torch.inference_mode():
    for i in range(processed,stop):
        masks=generator.generate(crop(i));features[i],counts[i]=describe_masks(masks)
        temporary=checkpoint.with_suffix('.tmp')
        with temporary.open('wb') as handle:np.savez_compressed(handle,features=features,counts=counts,processed=i+1,digest=digest)
        temporary.replace(checkpoint);processed=i+1
        print(json.dumps({'processed':processed,'rows':127,'accepted_masks':int(counts[i]),'elapsed_seconds':time.monotonic()-started}),flush=True)
assert sha(Path(contract['model']['path']))==contract['model']['sha256'],'Model changed during inference'
assert all(sha(assets/name)==value['sha256'] for name,value in asset_manifest.items()),'Training inputs changed during inference'
peak=float(torch.cuda.max_memory_allocated()/1024**3);rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
summary={'experiment':contract['experiment'],'stage':args.stage,'processed':processed,'identity':identity,'config_digest':digest,
         'seconds':time.monotonic()-started,'accepted_masks_total':int(counts[:processed].sum()),'GPU_peak_GiB':peak,'RSS_peak_GiB':rss,
         'exclusive_writer_lock':True,'source_model_inputs_verified':True,'predictive_accuracy_evidence':False}
if args.stage=='pilot':
    summary['feasibility_gate_passed']=bool(processed==3 and np.isfinite(features[:3]).all() and counts[:3].sum()>0 and summary['seconds']<=120 and peak<=7.5 and rss<=7)
    (out/'pilot.json').write_text(json.dumps(summary,indent=2)+'\n')
    assert summary['feasibility_gate_passed'],'Pilot failed fixed feasibility gate; do not retune'
else:
    assert processed==127 and np.isfinite(features).all()
    np.save(out/'features.npy',features)
    summary['outputs_sha256']={'features.npy':sha(out/'features.npy'),'checkpoint.npz':sha(checkpoint)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='identity'}),flush=True)
