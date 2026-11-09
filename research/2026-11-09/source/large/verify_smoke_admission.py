"""Execute old counterexample and real corrected admission against synthetic fixtures."""
import ast
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
r=Path(__file__).resolve().parent;sys.path.insert(0,str(r/'gpu-source'))
from admission import admit_smoke,sha
base=r/'admission-fixtures';assert not base.exists();base.mkdir()
expected={'experiment':'SYNTHETIC-ADMISSION-FIXTURE','stage':'smoke','rows':4,'sources':{'fixture':'only'}}
digest=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()
features=np.random.default_rng(7).normal(size=(4,5,1024)).astype(np.float32)
valid=base/'valid';valid.mkdir();np.save(valid/'cls.npy',features)
np.savez_compressed(valid/'checkpoint.npz',features=features,processed=4,digest=digest,minimum_precision_cosine=.99999)
summary={'experiment':expected['experiment'],'stage':'smoke','images':4,'identity':expected,'config_digest':digest,
 'feature_shape':[4,5,1024],'checkpoint_complete':True,'exclusive_writer_lock':True,'accuracy_evidence':False,
 'precision_gate_passed':True,'minimum_precision_cosine':.99999,
 'outputs_sha256':{n:sha(valid/n) for n in ['cls.npy','checkpoint.npz']}}
(valid/'summary.json').write_text(json.dumps(summary))
admit_smoke(valid,expected)
# Execute the exact old train-admission AST on a summary-only fixture; no CUDA/model code.
old=ast.parse((r/'source-before-smoke-guard/extract_large.py').read_text())
main=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='main')
branch=next(n for n in main.body if isinstance(n,ast.If) and ast.unparse(n.test)=="args.stage == 'train'")
orphan=base/'orphan';(orphan/'smoke').mkdir(parents=True)
(orphan/'smoke/summary.json').write_text(json.dumps(summary))
ns={'args':SimpleNamespace(stage='train'),'project':orphan,'json':json,'identity_for':lambda stage:expected}
exec(compile(ast.Module(body=[branch],type_ignores=[]),'<actual-old-admission>','exec'),ns)
assert ns['smoke']['checkpoint_complete'] and not (orphan/'smoke/checkpoint.npz').exists()
rejected=[]
try:admit_smoke(orphan/'smoke',expected)
except (AssertionError,FileNotFoundError):rejected.append('missing_both_artifacts')
else:raise AssertionError('Summary-only admitted')
for kind in ['truncated_cls','truncated_checkpoint','incomplete_checkpoint','wrong_digest','wrong_shape',
             'nonfinite','changed_features','low_cosine','false_completed','wrong_identity']:
    directory=base/kind;shutil.copytree(valid,directory);s=copy.deepcopy(summary)
    f=features.copy();processed=4;d=digest;score=.99999
    if kind=='truncated_cls':(directory/'cls.npy').write_bytes(b'')
    elif kind=='truncated_checkpoint':(directory/'checkpoint.npz').write_bytes(b'')
    elif kind in ['incomplete_checkpoint','wrong_digest','wrong_shape','nonfinite','changed_features']:
        if kind=='incomplete_checkpoint':processed=3
        elif kind=='wrong_digest':d='wrong'
        elif kind=='wrong_shape':f=f[:3]
        elif kind=='nonfinite':f[0,0,0]=np.nan
        elif kind=='changed_features':f[0,0,0]+=1
        if kind in ['wrong_shape','nonfinite']:np.save(directory/'cls.npy',f)
        np.savez_compressed(directory/'checkpoint.npz',features=f,processed=processed,digest=d,minimum_precision_cosine=score)
        s['outputs_sha256']={n:sha(directory/n) for n in ['cls.npy','checkpoint.npz']}
    elif kind=='low_cosine':s['minimum_precision_cosine']=.99
    elif kind=='false_completed':s['checkpoint_complete']=False
    elif kind=='wrong_identity':s['identity']={**expected,'rows':3}
    (directory/'summary.json').write_text(json.dumps(s))
    try:admit_smoke(directory,expected)
    except (AssertionError,FileNotFoundError,ValueError,EOFError):rejected.append(kind)
    else:raise AssertionError('Invalid smoke admitted: '+kind)
# Read and execute the actual pure crop function, without importing model dependencies.
tree=ast.parse((r/'gpu-source/preprocess.py').read_text(encoding='utf-8'))
parts=[]
for n in tree.body:
    if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id in ['LONG_MARGIN','CROP_POSITIONS'] for x in n.targets):parts.append(n)
    if isinstance(n,ast.FunctionDef) and n.name=='physical_crops':parts.append(n)
cropns={'Image':Image};exec(compile(ast.Module(body=parts,type_ignores=[]),'<actual-crop-function>','exec'),cropns)
crops=cropns['physical_crops'](Image.new('RGB',(1200,600)),518)
assert len(crops)==5 and all(c.size==(518,518) for c in crops)
record={'synthetic_cpu_only':True,'actual_old_summary_only_bypass_reproduced':True,
        'corrected_positive_passed':True,'adverse_cases_rejected':rejected,
        'actual_crop_function_gives_five_physical_squares_not_fullframe':True,
        'scientific_indices_1_to_4_unchanged':True,'real_smoke_or_accuracy_result':False}
(r/'SMOKE_ADMISSION_CHECKS.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
