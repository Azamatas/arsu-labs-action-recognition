"""Windows are cut once and kept in data_processing/cache/<name>.npz (not in git); delete a file to rebuild it."""
import os

import numpy as np

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def cached(name, build, rebuild=False):
    """build() returns a dict of arrays; it runs only when data_processing/cache/<name>.npz is missing."""
    path = os.path.join(DIR, f"{name}.npz")
    if not rebuild and os.path.exists(path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    arrays = build()
    os.makedirs(DIR, exist_ok=True)
    np.savez(path, **arrays)
    return arrays
