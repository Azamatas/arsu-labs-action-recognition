"""Augmentations of real single-axis windows, used to pretrain the FIN."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Feature-Imitating-Network

from config import SEGMENT


def warped(rng, pool):
    """A real single-axis window sped up or slowed down, maybe flipped or reversed, plus noise."""
    x = np.pad(pool[rng.integers(len(pool))].astype(np.float64), (0, SEGMENT), mode="reflect")
    length = SEGMENT / rng.uniform(0.75, 1.35)  # >1 slows the rhythm down, <1 speeds it up
    pos = rng.uniform(0, len(x) - length - 1) + np.arange(SEGMENT) * length / SEGMENT
    x = np.interp(pos, np.arange(len(x)), x)
    x = x.mean() + rng.choice([-1, 1]) * (x - x.mean())  # flip the swings, keep the level (gravity)
    x = x[::-1] if rng.random() < 0.5 else x
    return x + rng.uniform(0, 0.3) * x.std() * rng.standard_normal(SEGMENT)


def make(n, seed, pool):
    """n warped copies of single-axis windows from the pool."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, SEGMENT), np.float32)
    for i in range(n):
        out[i] = warped(rng, pool)
    return out
