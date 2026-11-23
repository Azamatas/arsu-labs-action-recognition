"""The code still computes exactly what the paper's code did: windows, temporal (imitation) features and the 40
statistical features are checked against SHA-256 fingerprints taken from the original implementation (identical
arrays, 2026-11-24); plus the spectral form of the circular FIN and the parameter counts.

    python tests/test_equivalence.py
"""
import hashlib
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bifin import config, cost, features, model, spectral, targets  # noqa: E402
from data_processing import daghar, motionsense, wisdm  # noqa: E402

REFERENCE = {"wisdm": "eb0ccdd6680eb643", "motionsense": "4b3f1514597703a3", "temporal": "a5a25895aef00c7b",
             "sensor40": "890fb9ae24cdc6db"}


def fingerprint(*arrays):
    return hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in arrays)).hexdigest()[:16]


def test_windows_match_paper():
    assert fingerprint(*wisdm.windows()) == REFERENCE["wisdm"]
    assert fingerprint(*motionsense.windows()) == REFERENCE["motionsense"]


def test_targets_and_features_match_paper():
    x = wisdm.windows()[0][:300]
    t = targets.temporal(x[:, 0].astype(np.float64), max_lag=60)
    assert fingerprint(*[t[k] for k in targets.TEMPORAL]) == REFERENCE["temporal"]
    f = features.sensor40(x)
    assert fingerprint(f) == REFERENCE["sensor40"]
    assert np.array_equal(features.compute(x, "acc7"), f[:, [0, 1, 2, 3, 4, 5, 9]])


def test_daghar_shapes():
    x, y = daghar.read("UCI", "test")
    assert x.shape[1:] == (6, 60) and set(np.unique(y)) <= set(range(6))


def test_circular_fin_is_spectral():
    for t, kernel in [(60, 31), (200, 61)]:
        torch.manual_seed(0)
        fin = model.FIN(32, kernel, 64, circular=True).double()
        x = np.random.default_rng(0).normal(size=(200, t))
        with torch.no_grad():
            conv = fin.embed(torch.as_tensor(x)).numpy()
        lmat, c = spectral.matrix(fin, t)
        assert np.abs(conv - spectral.embed(x, lmat, c)).max() / np.abs(conv).max() < 1e-9


def test_parameter_counts():
    # the paper's FIN + 40 (conv form, inference) and the smallest BiTFIN on DAGHAR, as in the old result tables
    m = model.build(config.BIFIN, 3, 6)
    assert cost.count(m) - cost.count(m.fin.head) == cost.conv_form(config.BIFIN, 3, 200, 6)[0] == 42566
    assert cost.conv_form(config.BITFIN, 6, 60, 6)[0] == 31686


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
