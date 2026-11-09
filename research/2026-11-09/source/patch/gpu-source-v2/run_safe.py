"""Preserve private inference diagnostics; stdout reports only error types."""
if not __debug__:
    raise RuntimeError('Optimised Python is forbidden: admission assertions must remain active')
import json
import runpy
import traceback
from pathlib import Path
try:runpy.run_path(str(Path(__file__).with_name('extract_patch_stats.py')),run_name='__main__')
except Exception as exc:
    frame=traceback.extract_tb(exc.__traceback__)[-1]
    print(json.dumps({'completed':False,'error_type':type(exc).__name__,
                      'code_location':{'file':Path(frame.filename).name,'line':frame.lineno,'function':frame.name}}),flush=True)
    raise SystemExit(1)
