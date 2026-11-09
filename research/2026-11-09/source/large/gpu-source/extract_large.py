"""One frozen large-CLS experiment; managed CUDA and training photographs only."""
if not __debug__:
    raise RuntimeError('Optimised Python is forbidden: admission assertions must remain active')
import os
os.environ.update(HF_HOME='/mnt/datos/cache/huggingface',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                  TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
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
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke','train'],required=True)
    p.add_argument('--validate-only',action='store_true');args=p.parse_args()
    root=Path(__file__).resolve().parent;contract=json.loads((root/'RUN_CONTRACT.json').read_text())
    sources=json.loads((root/'SOURCE_MANIFEST.json').read_text())
    assert all(sha(root/name)==digest for name,digest in sources.items())
    project=Path(contract['project']);assets=Path(contract['assets']);model_dir=project/'model'
    assert sha(assets/'ASSET_MANIFEST.json')==contract['asset_manifest_sha256']
    manifest=json.loads((assets/'ASSET_MANIFEST.json').read_text())
    assert all(sha(assets/name)==record['sha256'] for name,record in manifest.items())
    assert sha(model_dir/'model.safetensors')==contract['weights_sha256']
    assert sha(model_dir/'config.json')==contract['model_config_sha256']
    assert sha(root/'preprocess.py')==contract['preprocessing_sha256']
    sys.path.insert(0,contract['timm_site'])
    runtime={n:importlib.metadata.version(n) for n in contract['runtime']}
    assert runtime==contract['runtime']
    import numpy as np
    import torch
    import torch.nn.functional as F
    import timm
    from torch.utils.data import DataLoader,Subset
    from preprocess import image_index,PhysicalCropDataset
    from admission import admit_smoke
    assert timm.__version__==contract['timm_version'];torch.set_num_threads(4)
    with np.load(assets/'train-reference.npz',allow_pickle=False) as reference:
        names=reference['file_names'].astype(str);sample_ids=reference['sample_ids'].astype(str)
    assert len(names)==len(set(names))==127 and len(set(sample_ids))==24
    count=4 if args.stage=='smoke' else 127
    cfg=json.loads((model_dir/'config.json').read_text())
    assert cfg['num_features']==1024 and cfg['pretrained_cfg']['input_size']==[3,518,518]
    frame=image_index({'paths':names[:count],'sample_ids':sample_ids[:count]},assets)
    dataset=PhysicalCropDataset(frame,518,tuple(cfg['pretrained_cfg']['mean']),tuple(cfg['pretrained_cfg']['std']),'rgb')

    def identity_for(stage):
        n=4 if stage=='smoke' else 127
        return {'experiment':contract['experiment'],'stage':stage,'rows':n,'sources':sources,
                'contract_sha256':sha(root/'RUN_CONTRACT.json'),'runtime':runtime,'timm_version':timm.__version__,
                'row_order_sha256':hashlib.sha256('\n'.join(names[:n]).encode()).hexdigest()}
    identity=identity_for(args.stage);digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if args.validate_only:
        for i in [0,count-1]:
            value,index=dataset[i];assert index==i and value.shape==(5,3,518,518) and torch.isfinite(value).all()
        proof={'stage':args.stage,'rows':count,'source_input_model_runtime_hashes_verified':True,
               'endpoint_preprocessing_verified':True,'gpu_compute':False,'config_digest':digest}
        (root/'cpu-preflight.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof));return
    output=project/args.stage;output.mkdir(exist_ok=True)
    lock=(output/'writer.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    smoke=None
    if args.stage=='train':
        smoke=admit_smoke(project/'smoke',identity_for('smoke'),contract['minimum_precision_cosine'])
    features=np.full((count,5,1024),np.nan,dtype=np.float32);processed=0
    checkpoint=output/'checkpoint.npz';minimum_precision_cosine=smoke['minimum_precision_cosine'] if smoke is not None else 1.
    if checkpoint.exists():
        with np.load(checkpoint,allow_pickle=False) as saved:
            assert str(saved['digest'].item())==digest
            features=saved['features'];processed=int(saved['processed']);minimum_precision_cosine=float(saved['minimum_precision_cosine'])
        assert features.shape==(count,5,1024) and 0<=processed<=count
        assert np.isfinite(features[:processed]).all() and np.isfinite(minimum_precision_cosine)
        if smoke is not None:assert abs(minimum_precision_cosine-smoke['minimum_precision_cosine'])<1e-8
    assert torch.cuda.is_available(),'Managed CUDA allocation required'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=timm.create_model('local-dir:'+str(model_dir),pretrained=True,num_classes=0).eval().cuda()
    assert model.num_features==1024 and model.num_prefix_tokens==5
    loader=DataLoader(Subset(dataset,range(processed,count)),batch_size=1,shuffle=False,num_workers=0,pin_memory=True)
    started=time.monotonic()
    with torch.inference_mode():
        for patches,indices in loader:
            patches=patches.flatten(0,1).cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.float16):pooled=model(patches)
            assert pooled.shape==(5,1024) and torch.isfinite(pooled).all()
            if args.stage=='smoke':
                control=model(patches)
                assert control.shape==(5,1024) and torch.isfinite(control).all()
                cos=F.cosine_similarity(pooled.float(),control.float(),dim=-1)
                minimum_precision_cosine=min(minimum_precision_cosine,float(cos.min()))
            idx=indices.numpy();assert np.array_equal(idx,np.asarray([processed]))
            features[idx]=pooled.float().cpu().numpy().reshape(1,5,1024);processed+=1
            temporary=output/'checkpoint.tmp.npz'
            with temporary.open('wb') as handle:
                np.savez_compressed(handle,features=features,processed=processed,digest=digest,
                                    minimum_precision_cosine=minimum_precision_cosine)
            temporary.replace(checkpoint)
            if processed%8==0 or processed==count:print(json.dumps({'processed_images':processed,'total_images':count,'seconds':round(time.monotonic()-started,2)}),flush=True)
    assert processed==count and np.isfinite(features).all()
    if smoke is not None:minimum_precision_cosine=smoke['minimum_precision_cosine']
    np.save(output/'cls.npy',features)
    summary={'experiment':contract['experiment'],'stage':args.stage,'images':count,'feature_shape':list(features.shape),
             'minimum_precision_cosine':minimum_precision_cosine,'precision_gate_passed':bool(minimum_precision_cosine>=.9999),
             'accuracy_evidence':False,'identity':identity,'config_digest':digest,'checkpoint_complete':True,
             'exclusive_writer_lock':True,'outputs_sha256':{'cls.npy':sha(output/'cls.npy'),'checkpoint.npz':sha(checkpoint)},
             'seconds_current_chunk':round(time.monotonic()-started,2),'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated()}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['stage','images','minimum_precision_cosine','precision_gate_passed','peak_allocated_gpu_bytes','accuracy_evidence']}),flush=True)
    assert summary['precision_gate_passed'],'Mixed precision failed compatibility; do not train/evaluate'


if __name__=='__main__':main()
