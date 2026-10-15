
import numpy as np


def extract_basic_features(x, y, z):
    n = len(x)
    features = [x.mean(), y.mean(), z.mean(), x.std(ddof=1), y.std(ddof=1), z.std(ddof=1)]
    features += [
        np.mean(np.abs(x - x.mean())),
        np.mean(np.abs(y - y.mean())),
        np.mean(np.abs(z - z.mean())),
    ]
    features.append(np.mean(np.sqrt(x**2 + y**2 + z**2)))
    for axis in (x, y, z):
        # matches MATLAB hist(axis, 10): bin edges padded by half a bin width
        # beyond min/max, unlike numpy's default edges-at-min/max histogram.
        lo, hi = axis.min(), axis.max()
        binwidth = (hi - lo) / 10 or 1.0
        edges = np.linspace(lo - binwidth / 2, hi + binwidth / 2, 11)
        counts, _ = np.histogram(axis, bins=edges)
        features.extend(counts / n)
    return np.array(features)
