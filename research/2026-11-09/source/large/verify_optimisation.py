import json
import os
import subprocess
import sys
from pathlib import Path
r=Path(__file__).resolve().parent;rows=[]
for name in ['extract_large.py','run_safe.py','admission.py']:
    for mode in ['flag','environment']:
        env=os.environ.copy()
        if mode=='environment':env['PYTHONOPTIMIZE']='1'
        process=subprocess.run([sys.executable]+(['-O'] if mode=='flag' else [])+[str(r/'gpu-source'/name)],
                               capture_output=True,text=True,env=env,timeout=20)
        assert process.returncode!=0 and 'Optimised Python is forbidden' in process.stderr
        rows.append({'entry':name,'mode':mode,'rejected_before_imports':True})
(r/'OPTIMISATION_GUARD_CHECKS.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps({'checks_passed':len(rows)}))
