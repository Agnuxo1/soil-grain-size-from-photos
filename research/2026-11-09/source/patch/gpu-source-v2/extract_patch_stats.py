"""Frozen DINO patch-statistics experiment; managed GPU only, no labels/test inputs."""
if not __debug__:
    raise RuntimeError('Optimised Python is forbidden: admission assertions must remain active')
import os
os.environ.update(HF_HOME='/mnt/datos/cache/huggingface',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                  TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','train'],required=True)
    p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parent;contract=json.loads((root/'RUN_CONTRACT.json').read_text())
    sources=json.loads((root/'SOURCE_MANIFEST.json').read_text())
    assert all(sha(root/name)==digest for name,digest in sources.items()),'Source changed after freeze'
    project=Path(contract['project']);assets=project/'assets';model_dir=project/'model'
    assert sha(assets/'ASSET_MANIFEST.json')==contract['asset_manifest_sha256']
    asset_manifest=json.loads((assets/'ASSET_MANIFEST.json').read_text())
    assert all(sha(assets/name)==entry['sha256'] for name,entry in asset_manifest.items()),'Input content mismatch'
    assert sha(model_dir/'model.safetensors')==contract['weights_sha256']
    assert sha(model_dir/'config.json')==contract['model_config_sha256']
    site=project/'site-timm-1.0.30';sys.path.insert(0,str(site))
    runtime={n:importlib.metadata.version(n) for n in contract['runtime']}
    assert runtime==contract['runtime'],'Runtime changed after setup'
    import numpy as np
    import torch
    import torch.nn.functional as F
    import timm
    from torch.utils.data import DataLoader
    from preprocess import image_index,PhysicalCropDataset
    assert timm.__version__==contract['timm_version']
    torch.set_num_threads(4)
    with np.load(assets/'train-reference.npz',allow_pickle=False) as ref:
        file_names=ref['file_names'].copy();sample_ids=ref['sample_ids'].copy();reference=ref['features'].copy()
    assert reference.shape==(127,5,384) and len(set(sample_ids))==24 and len(set(file_names))==127
    count=contract['pilot_rows'] if a.stage=='pilot' else contract['rows']
    frame=image_index({'paths':file_names[:count],'sample_ids':sample_ids[:count]},assets)
    cfg=json.loads((model_dir/'config.json').read_text())['pretrained_cfg']
    assert list(cfg['input_size'])==[3,518,518]
    dataset=PhysicalCropDataset(frame,518,tuple(cfg['mean']),tuple(cfg['std']),'rgb')
    identity={'experiment':contract['experiment'],'stage':a.stage,'rows':count,'sources':sources,
              'asset_manifest_sha256':contract['asset_manifest_sha256'],'weights_sha256':contract['weights_sha256'],
              'model_config_sha256':contract['model_config_sha256'],'runtime':runtime,'timm_version':timm.__version__,
              'row_order_sha256':hashlib.sha256('\n'.join(file_names[:count]).encode()).hexdigest(),
              'batch_images':4,'workers':0,'precision':contract['precision']}
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if a.validate_only:
        for i in [0,count-1]:
            value,index=dataset[i];assert index==i and value.shape==(5,3,518,518) and torch.isfinite(value).all()
        proof={'stage':a.stage,'expected_rows':count,'image_index_rows':len(frame),'source_and_all_input_hashes_verified':True,
               'endpoint_preprocessing_shape_verified':True,'gpu_or_model_inference':False,'config_digest':digest}
        (root/'cpu-preflight.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof));return
    output=project/a.stage;output.mkdir(exist_ok=True)
    lock=(output/'writer.lock').open('a')
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit(73)
    if a.stage=='train':
        pilot=json.loads((project/'pilot/summary.json').read_text())
        assert pilot['feasibility_gate_passed'] and pilot['minimum_cls_cosine']>=contract['minimum_cls_cosine']
        assert pilot['identity']=={**identity,'stage':'pilot','rows':20,'row_order_sha256':hashlib.sha256('\n'.join(file_names[:20]).encode()).hexdigest()}
    cls=np.full((count,5,384),np.nan,dtype=np.float32)
    stats=np.full((count,5,768),np.nan,dtype=np.float32);processed=0
    checkpoint=output/'checkpoint.npz'
    if checkpoint.exists():
        with np.load(checkpoint,allow_pickle=False) as saved:
            assert str(saved['digest'].item())==digest,'Stale checkpoint'
            cls=saved['cls'];stats=saved['stats'];processed=int(saved['processed'])
        assert 0<=processed<=count and cls.shape==(count,5,384) and stats.shape==(count,5,768)
        assert np.isfinite(cls[:processed]).all() and np.isfinite(stats[:processed]).all()
    assert torch.cuda.is_available(),'Managed CUDA allocation required; no CPU/local fallback'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=timm.create_model('local-dir:'+str(model_dir),pretrained=True,num_classes=0).eval().cuda()
    assert model.num_prefix_tokens==5 and model.num_features==384
    remaining=torch.utils.data.Subset(dataset,range(processed,count))
    loader=DataLoader(remaining,batch_size=4,shuffle=False,num_workers=0,pin_memory=True)
    t0=time.monotonic()
    with torch.inference_mode():
        for patches,indices in loader:
            n,views=patches.shape[:2];assert views==5
            patches=patches.flatten(0,1).cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.float16):
                tokens=model.forward_features(patches)
                pooled=model.forward_head(tokens,pre_logits=True)
            assert tokens.shape==(n*5,1374,384) and pooled.shape==(n*5,384)
            local=F.normalize(tokens[:,5:,:].float(),dim=-1)
            descriptor=torch.cat([local.mean(1),local.std(1,correction=0)],dim=1)
            idx=indices.numpy();assert np.array_equal(idx,np.arange(processed,processed+n))
            cls[idx]=pooled.float().cpu().numpy().reshape(n,5,384)
            stats[idx]=descriptor.cpu().numpy().reshape(n,5,768)
            processed+=n
            temporary=output/'checkpoint.tmp.npz'
            with temporary.open('wb') as handle:np.savez_compressed(handle,cls=cls,stats=stats,processed=processed,digest=digest)
            temporary.replace(checkpoint)
            print(json.dumps({'processed_images':processed,'total_images':count,'seconds':round(time.monotonic()-t0,2)}),flush=True)
    assert processed==count and np.isfinite(cls).all() and np.isfinite(stats).all()
    denom=np.linalg.norm(cls,axis=-1)*np.linalg.norm(reference[:count],axis=-1)
    assert np.all(denom>1e-12)
    cos=np.sum(cls*reference[:count],axis=-1)/denom
    np.save(output/'patch-statistics.npy',stats);np.save(output/'cls-control.npy',cls)
    summary={'experiment':contract['experiment'],'author':contract['author'],'stage':a.stage,'images':count,
             'descriptor_shape':list(stats.shape),'minimum_cls_cosine':float(cos.min()),'mean_cls_cosine':float(cos.mean()),
             'feasibility_gate_passed':bool(cos.min()>=contract['minimum_cls_cosine']),
             'accuracy_evidence':False,'seconds_current_chunk':round(time.monotonic()-t0,2),
             'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(),'identity':identity,'config_digest':digest,
             'checkpoint_complete':True,'exclusive_writer_lock':True,
             'outputs_sha256':{name:sha(output/name) for name in ['patch-statistics.npy','cls-control.npy']}}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['stage','images','minimum_cls_cosine','mean_cls_cosine','feasibility_gate_passed','peak_allocated_gpu_bytes','accuracy_evidence']}),flush=True)
    assert summary['feasibility_gate_passed'],'Original CLS compatibility gate failed; do not evaluate new descriptor'

if __name__=='__main__':main()
