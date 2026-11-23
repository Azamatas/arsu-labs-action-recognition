"""Data-scarcity protocol: few labelled WISDM users (10 s windows at 20 Hz, accelerometer).

For each seed, N users (default 7) are drawn for training and 2 for validation from the 35 WISDM users other than
user 30 (np.random.default_rng(seed).choice, as in the paper); the remaining users are the test set. 150 epochs,
batch 256, no augmentation; scored after the last epoch and at the epoch with the best validation accuracy; test
accuracy recorded after every epoch.

A pretrained FIN is pretrained on the training users' single axes plus 30,000 warped copies of them (sped up or
slowed down, maybe flipped or reversed, plus noise); a pretrained FIN WITHOUT statistical features on the warped
copies only, as in the paper.
"""
import numpy as np

from data_processing import wisdm
from data_processing.activities import PAPER

from . import base
from .models import Context

NAME = "scarcity"
EPOCHS, LR, N_VAL, N_WARPED, EXCLUDED = 150, 1e-3, 2, 30000, 30


def add_args(parser):
    parser.add_argument("--users", type=int, nargs="+", default=[7], help="labelled training users")


def warped(rng, pool):
    """A real single-axis window sped up or slowed down, maybe flipped or reversed, plus noise."""
    t = pool.shape[1]
    x = np.pad(pool[rng.integers(len(pool))].astype(np.float64), (0, t), mode="reflect")
    length = t / rng.uniform(0.75, 1.35)
    pos = rng.uniform(0, len(x) - length - 1) + np.arange(t) * length / t
    x = np.interp(pos, np.arange(len(x)), x)
    x = x.mean() + rng.choice([-1, 1]) * (x - x.mean())  # flip the swings, keep the level (gravity)
    x = x[::-1] if rng.random() < 0.5 else x
    return x + rng.uniform(0, 0.3) * x.std() * rng.standard_normal(t)


def augmented(n, seed, pool):
    rng = np.random.default_rng(seed)
    return np.stack([warped(rng, pool) for _ in range(n)]).astype(np.float32)


def pretrain_axes(xtr, seed, spec):
    real = xtr.reshape(-1, xtr.shape[-1])
    warp = augmented(N_WARPED, seed, real)
    with_features = getattr(spec, "cfg", None) is not None and spec.cfg.features != "none"
    return np.concatenate([real, warp]) if with_features else warp


CONTEXT = Context(max_lag=60, pretrain_axes=pretrain_axes)


def split(users, n_train, seed):
    candidates = np.setdiff1d(np.unique(users), [EXCLUDED])
    picked = np.random.default_rng(seed).choice(candidates, n_train + N_VAL, replace=False)
    tr, va = np.isin(users, picked[:n_train]), np.isin(users, picked[n_train:])
    return tr, va, ~(tr | va), picked


def run(spec, args, seed):
    x, y, u = wisdm.windows()
    for n in args.users:
        variant = f"users{n}"
        if base.done(NAME, variant, spec, seed) and not args.force:
            print(f"skip {NAME}/{variant}/{spec.id()}/seed{seed} (exists)")
            continue
        tr, va, te, picked = split(u, n, seed)
        tests = {"unseen users": (x[te], y[te])}
        if spec.classical:
            result, arrays = base.classical(spec, x[tr], y[tr], tests, CONTEXT, seed, len(PAPER))
        else:
            result, arrays = base.neural_epochs(spec, x[tr], y[tr], tests, CONTEXT, seed, len(PAPER), EPOCHS, LR,
                                                val=(x[va], y[va]))
        result.update(train_users=sorted(picked[:n].tolist()), val_users=sorted(picked[n:].tolist()))
        print(base.save(NAME, variant, spec, seed, result, arrays), "|",
              ", ".join(f"{t} {s['accuracy']:.3f}" for t, s in result["scores"].items()), flush=True)
