"""Parameters and FLOPs (2 per multiply-add) of one window, for the convolutional and the spectral form.

Statistical features are counted roughly: about 4 operations per sample and axis for the 7- and 10-feature sets
(sums for the mean, squares for the standard deviation, the magnitude), 8 for the 40 (adds the histograms).
"""
import numpy as np

from . import features


def _feature_flops(name, t):
    return sum(0 if n == 0 else 3 * t * (8 if n == 40 else 4) for n in features.SETS[name])


def head(cfg, channels, n_classes):
    per_sensor = {"concat": 3, "mean": 1, "meanmax": 2}[cfg.pooling]
    n_in = (channels // 3) * per_sensor * cfg.emb + features.size(cfg.features)
    params = n_in * cfg.head + cfg.head + cfg.head * n_classes + n_classes
    return params, 2 * (n_in * cfg.head + cfg.head * n_classes)


def conv_form(cfg, channels, t, n_classes):
    """(parameters, FLOPs): two convolutions, product, mean and dense layer per channel, then the head."""
    params = 2 * cfg.width * cfg.kernel + cfg.width * cfg.emb + cfg.emb
    fin = channels * (2 * 2 * cfg.width * cfg.kernel * t + 2 * cfg.width * t + 2 * cfg.width * cfg.emb)
    hp, hf = head(cfg, channels, n_classes)
    return params + hp, fin + hf + _feature_flops(cfg.features, t)


def spectral_form(cfg, channels, t, n_classes):
    """(parameters, FLOPs) as deployed: z-score, real FFT, |X|^2, matrix L, then the head (circular FIN only)."""
    f = t // 2 + 1
    params = f * cfg.emb + cfg.emb
    fin = channels * (3 * t + 2.5 * t * np.log2(t) + 3 * f + 2 * f * cfg.emb)
    hp, hf = head(cfg, channels, n_classes)
    return params + hp, fin + hf + _feature_flops(cfg.features, t)


def count(model):
    return sum(p.numel() for p in model.parameters())
