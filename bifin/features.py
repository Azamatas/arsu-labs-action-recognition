"""Statistical features fed to the head next to the FIN embeddings.

The 40 of Ignatov (2018) per 3-axis sensor, column order:
    0-2 mean, 3-5 standard deviation, 6-8 mean absolute deviation (x, y, z), 9 mean magnitude,
    10-39 normalised 10-bin histograms of x, y and z over each window's own range.
Named sets take columns of these per sensor (accelerometer = channels 0-2, gyroscope = channels 3-5).
"""
import numpy as np

COLUMNS = {40: list(range(40)), 10: list(range(10)), 7: [0, 1, 2, 3, 4, 5, 9], 4: [0, 1, 2, 9], 0: []}
SETS = {  # name: (accelerometer columns, gyroscope columns)
    "acc40": (40, 0), "acc10": (10, 0), "acc7": (7, 0), "acc4": (4, 0), "none": (0, 0),
    "acc10_gyro10": (10, 10), "acc7_gyro7": (7, 7),
}


def ignatov40(x, y, z):
    """The 40 features of one window (three 1-d axes, float64)."""
    n = len(x)
    f = [x.mean(), y.mean(), z.mean(), x.std(ddof=1), y.std(ddof=1), z.std(ddof=1)]
    f += [np.mean(np.abs(a - a.mean())) for a in (x, y, z)]
    f.append(np.mean(np.sqrt(x ** 2 + y ** 2 + z ** 2)))
    for a in (x, y, z):
        lo, hi = a.min(), a.max()
        width = (hi - lo) / 10 or 1.0
        edges = np.linspace(lo - width / 2, hi + width / 2, 11)
        f.extend(np.histogram(a, bins=edges)[0] / n)
    return np.array(f)


def sensor40(windows):
    """(n, 40) for (n, 3, T) windows of one 3-axis sensor."""
    return np.stack([ignatov40(*w) for w in windows.astype(np.float64)])


def compute(windows, name):
    """(n, k) features of the named set for (n, C, T) windows (C = 3, or 6 with a gyroscope)."""
    n_acc, n_gyro = SETS[name]
    parts = [] if n_acc == 0 else [sensor40(windows[:, :3])[:, COLUMNS[n_acc]]]
    if n_gyro:
        parts.append(sensor40(windows[:, 3:6])[:, COLUMNS[n_gyro]])
    return np.concatenate(parts, axis=1) if parts else np.zeros((len(windows), 0))


def size(name):
    n_acc, n_gyro = SETS[name]
    return len(COLUMNS[n_acc]) + len(COLUMNS[n_gyro])


class Standardiser:
    """z-score with the statistics of the training windows; constant columns are left unscaled."""

    def __init__(self, train):
        self.mu, sd = train.mean(0), train.std(0)
        self.sd = np.where(sd < 1e-8, 1.0, sd)

    def __call__(self, f):
        return ((f - self.mu) / self.sd).astype(np.float32)
