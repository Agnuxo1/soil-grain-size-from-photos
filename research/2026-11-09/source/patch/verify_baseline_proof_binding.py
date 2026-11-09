"""Reproduce the old unbound-proof admission with an isolated filesystem mapping."""
import ast
import copy
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path
r=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
baseline_path=r/'BASELINE_REPRODUCTION.json';baseline=json.loads(baseline_path.read_text())
seal=json.loads((r/'EVALUATION_SOURCE_MANIFEST.json').read_text())
original_sha=sha(baseline_path)
module=Path('D:/PROJECTS/183_Machine_Learning/src/soil_gsd_feature_fusion_audit.py')
module_before=sha(module)
fixture=r/'baseline-proof-adversarial-fixture';assert not fixture.exists();fixture.mkdir()
for name in seal:shutil.copy2(r/name,fixture/name)
shutil.copytree(r/'gpu-source-v2',fixture/'gpu-source-v2')
shutil.copy2(r/'GPU_SOURCE_MANIFEST_V2.json',fixture/'GPU_SOURCE_MANIFEST_V2.json')
shutil.copy2(r/'EVALUATION_SOURCE_MANIFEST.json',fixture/'EVALUATION_SOURCE_MANIFEST.json')
changed=fixture/'changed_historical_module.py'
text=module.read_text(encoding='utf-8');needle='Ridge(alpha=alpha, solver="lsqr")';assert text.count(needle)==1
changed.write_text(text.replace(needle,'Ridge(alpha=alpha * 2, solver="lsqr")'),encoding='utf-8')
modified=copy.deepcopy(baseline);modified['source_and_artifact_sha256'][str(module)]=sha(changed)
(fixture/'BASELINE_REPRODUCTION.json').write_text(json.dumps(modified,indent=2)+'\n')
assert 'BASELINE_REPRODUCTION.json' not in seal
# Execute the exact real admission prefix, stopping before arrays, fitting or output.
tree=ast.parse((r/'evaluate_candidate.py').read_text())
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
prefix=[]
for node in main.body:
    prefix.append(node)
    if isinstance(node,ast.For) and 'source_and_artifact_sha256' in ast.unparse(node.iter):break
else:raise AssertionError('Actual baseline-admission loop not found')
probe=ast.FunctionDef(name='admission_prefix',args=main.args,body=prefix,decorator_list=[])
ast.fix_missing_locations(probe)
def mapped_sha(path):
    return sha(changed) if Path(path)==module else sha(Path(path))
namespace={'__file__':str(fixture/'evaluate_candidate.py'),'Path':Path,'json':json,
           'file_sha':mapped_sha,'importlib':importlib}
exec(compile(ast.Module(body=[probe],type_ignores=[]),'<actual-admission-prefix>','exec'),namespace)
namespace['admission_prefix']()
# Bind the unchanged actual proof as well as the old source seal and result bytes.
complete={**seal,'BASELINE_REPRODUCTION.json':original_sha,
          'EVALUATION_SOURCE_MANIFEST.json':sha(r/'EVALUATION_SOURCE_MANIFEST.json'),
          'CANDIDATE_EVALUATION.json':sha(r/'CANDIDATE_EVALUATION.json')}
assert sha(fixture/'BASELINE_REPRODUCTION.json')!=complete['BASELINE_REPRODUCTION.json']
assert sha(baseline_path)==original_sha and sha(module)==module_before
(r/'EVALUATION_COMPLETE_MANIFEST.json').write_text(json.dumps(complete,indent=2)+'\n')
record={'actual_old_admission_prefix_accepted_changed_module_hash_in_unsealed_proof':True,
 'filesystem_override_used_only_for_isolated_module_copy':True,'original_module_and_proof_unchanged':True,
 'new_complete_manifest_includes_original_baseline_proof':True,'changed_proof_now_rejected':True,
 'real_negative_result_still_requires_recomputation_under_complete_binding':True}
(r/'BASELINE_PROOF_BINDING_CHECKS.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
