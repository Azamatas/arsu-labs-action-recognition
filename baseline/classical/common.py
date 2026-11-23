"""Shared part of the classical baselines: input features, standardisation, fit and predict.

Feature sets (computed on (n, 3, T) accelerometer windows):
  raw_y        the raw y axis of each window (Ignatov's 1-NN baseline)
  statistical  Ignatov's 40 statistical features
  temporal     the 33 temporal features (11 per axis, bifin.targets.TEMPORAL)
"""
import numpy as np
from sklearn.preprocessing import StandardScaler

from bifin import features, targets

FEATURES = ("raw_y", "statistical", "temporal")


def compute(windows, name, max_lag):
    if name == "raw_y":
        return windows[:, 1].astype(np.float64)
    if name == "statistical":
        return features.sensor40(windows[:, :3])
    return targets.temporal33(windows[:, :3], max_lag)


def fit_predict(make, scale, xtr, ytr, tests, seed, shuffle=True):
    """make(seed) -> sklearn classifier. scale: standardise with the training statistics. shuffle: permute the
    training windows with the seed first (the only randomness of deterministic models such as 1-NN).
    tests: {name: x}. Returns {name: predictions}."""
    if shuffle:
        order = np.random.default_rng(seed).permutation(len(ytr))
        xtr, ytr = xtr[order], ytr[order]
    if scale:
        scaler = StandardScaler().fit(xtr)
        xtr, tests = scaler.transform(xtr), {k: scaler.transform(v) for k, v in tests.items()}
    clf = make(seed).fit(xtr, ytr)
    return {k: clf.predict(v) for k, v in tests.items()}
