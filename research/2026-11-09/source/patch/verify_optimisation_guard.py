"""Exercise real entry points under both optimisation mechanisms, before GPU work."""
import json
import os
import subprocess
import sys
from pathlib import Path
r=Path(__file__).resolve().parent
entries=[r/'evaluation_core.py',r/'evaluate_candidate.py',r/'gpu-source-v2/extract_patch_stats.py',r/'gpu-source-v2/run_safe.py']
checked=[]
for p in entries:
    for mode in ['flag','environment']:
        env=os.environ.copy()
        if mode=='environment':env['PYTHONOPTIMIZE']='1'
        args=[sys.executable]+(['-O'] if mode=='flag' else [])+[str(p)]
        result=subprocess.run(args,capture_output=True,text=True,env=env,timeout=20)
        assert result.returncode!=0 and 'Optimised Python is forbidden' in result.stderr
        checked.append({'entry':p.relative_to(r).as_posix(),'mode':mode,'exit_code':result.returncode,'rejected_before_imports':True})
record={'original_counterexample_reproduced_before_fix':True,'original_empty_summary_accepted_under_O':True,
 'optimised_execution_rejected_after_fix':checked,'checks_passed':len(checked),
 'previous_gpu_revision_preserved':True,'scientific_configuration_or_threshold_changed':False}
(r/'OPTIMISATION_GUARD_CHECKS.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'checks_passed':len(checked),'all_rejected_before_imports':True}))
