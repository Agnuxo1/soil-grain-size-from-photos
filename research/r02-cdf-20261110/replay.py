"""Replay anonymous sample-cluster statistics only; never a model run."""
import json
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parent
j=json.loads((r/'RESULT.json').read_text())
d=np.asarray(j['per_sample_mean_improvement'],dtype=float)
assert d.shape==(24,) and np.isfinite(d).all()
rng=np.random.default_rng(20261110)
interval=np.quantile(d[rng.integers(0,24,size=(10000,24))].mean(axis=1),[.025,.975])
gain=float(d.mean())
np.testing.assert_allclose(interval,j['paired_cluster_95pct_ci'],rtol=0,atol=1e-12)
assert abs(gain-j['mean_emd_reduction'])<1e-12
gate=bool(gain>=1.0 and interval[0]>0.)
assert gate is False and j['preregistered_gate_passed'] is False
for delta,lower,expected in [(1.,0.,False),(.999999,1.,False),(1.,.000001,True),(-1.,1.,False)]:
    assert bool(delta>=1. and lower>0.) is expected
result={'aggregate_only_statistics_replayed':True,'samples':24,'draws':10000,
        'seed':20261110,'mean_emd_reduction':gain,'ci95':interval.tolist(),
        'promotion_gate_passed':gate,'training_or_test_access':False,
        'official_submission_created':False,'four_strict_boundary_checks':True}
(r/'REPLAY_VERIFIED.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
