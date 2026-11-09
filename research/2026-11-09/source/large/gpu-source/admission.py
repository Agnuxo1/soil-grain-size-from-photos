"""Verify actual completed precision-smoke artefacts before full training extraction."""
if not __debug__:
    raise RuntimeError('Optimised Python is forbidden: admission assertions must remain active')
import hashlib
import json
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def admit_smoke(directory,expected_identity,minimum=.9999):
    summary=json.loads((directory/'summary.json').read_text())
    digest=hashlib.sha256(json.dumps(expected_identity,sort_keys=True).encode()).hexdigest()
    assert expected_identity['stage']=='smoke' and expected_identity['rows']==4
    assert summary['identity']==expected_identity and summary['config_digest']==digest
    assert summary['experiment']==expected_identity['experiment'] and summary['stage']=='smoke'
    assert summary['images']==4 and summary['feature_shape']==[4,5,1024]
    assert summary['checkpoint_complete'] is True and summary['exclusive_writer_lock'] is True
    assert summary['accuracy_evidence'] is False and summary['precision_gate_passed'] is True
    score=summary['minimum_precision_cosine']
    assert np.isfinite(score) and minimum<=score<=1.000001
    assert set(summary['outputs_sha256'])=={'cls.npy','checkpoint.npz'}
    for name,digest_value in summary['outputs_sha256'].items():assert sha(directory/name)==digest_value
    cls=np.load(directory/'cls.npy',allow_pickle=False)
    assert cls.shape==(4,5,1024) and cls.dtype==np.float32 and np.isfinite(cls).all()
    assert np.all(np.linalg.norm(cls,axis=-1)>1e-12)
    with np.load(directory/'checkpoint.npz',allow_pickle=False) as checkpoint:
        assert str(checkpoint['digest'].item())==digest and int(checkpoint['processed'])==4
        assert np.array_equal(checkpoint['features'],cls)
        assert np.isfinite(float(checkpoint['minimum_precision_cosine']))
        assert abs(float(checkpoint['minimum_precision_cosine'])-score)<1e-8
    return summary
