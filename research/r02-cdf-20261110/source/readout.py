"""R02: one fixed direct-CDF readout. No data discovery or file access.

Francisco Angulo de Lafuente, ORCID 0009-0001-1634-7063. MIT.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def project_cdf(raw, weights):
    raw = np.asarray(raw, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 10 or weights.shape != (10,):
        raise ValueError('Expected N x 10 CDF predictions and ten interval weights')
    if not np.isfinite(raw).all() or not np.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError('Nonfinite predictions or invalid weights')
    out = np.empty((len(raw), 11), dtype=np.float64)
    for k, row in enumerate(raw):
        out[k, :10] = IsotonicRegression(increasing=True, y_min=0., y_max=100.).fit_transform(
            np.arange(10, dtype=np.float64), row, sample_weight=weights)
    out[:, 10] = 100.
    return out


def fold_predict(x_train, y_train, x_valid, weights):
    x_train = np.asarray(x_train, dtype=np.float64)
    x_valid = np.asarray(x_valid, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    if x_train.ndim != 2 or x_valid.ndim != 2 or x_train.shape[1] != x_valid.shape[1]:
        raise ValueError('Feature shape mismatch')
    if y_train.shape != (len(x_train), 11) or len(x_train) < 2:
        raise ValueError('Target shape mismatch')
    if not all(np.isfinite(a).all() for a in (x_train, y_train, x_valid)):
        raise ValueError('Nonfinite inputs')
    if not ((y_train >= 0).all() and (y_train <= 100).all()
            and (np.diff(y_train, axis=1) >= -1e-7).all()
            and np.allclose(y_train[:, -1], 100., rtol=0, atol=1e-7)):
        raise ValueError('Training targets are not valid CDFs')
    scaler = StandardScaler().fit(x_train)
    model = Ridge(alpha=1., solver='lsqr').fit(scaler.transform(x_train), y_train[:, :10])
    return project_cdf(model.predict(scaler.transform(x_valid)), weights)
