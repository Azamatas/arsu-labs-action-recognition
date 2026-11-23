"""Cross-dataset protocol of the paper (10 s windows at 20 Hz, accelerometer).

  w2m  train on WISDM users 1-26; test on all 24 MotionSense users and on WISDM users 27-36
  m2w  train on MotionSense users 1-18; test on all 36 WISDM users and on MotionSense users 19-24

Every training window is used twice, as recorded and rotated 180 degrees about y (x and z change sign). Neural
models: 60 epochs, batch 256, Adam 1e-3 (the Ignatov CNN 5e-4), scored after the last epoch, test accuracy recorded
after every epoch. --rates also tests on copies of the test windows decimated to 10 / 5 / 2 / 1 Hz (polyphase, with
the anti-alias low-pass) and interpolated back to 20 Hz, as a device sampling slower would see them.
"""
import numpy as np
from scipy.signal import resample_poly

from data_processing import motionsense, wisdm
from data_processing.activities import PAPER

from . import base
from .models import Context

NAME = "cross_dataset"
VARIANTS = ("w2m", "m2w")
FLIP = np.array([-1, 1, -1], np.float32)[:, None]
EPOCHS, LR = 60, 1e-3
RATES = (10, 5, 2, 1)
CONTEXT = Context(max_lag=60)


def add_args(parser):
    parser.add_argument("--direction", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--rates", action="store_true", help="also test at 10 / 5 / 2 / 1 Hz")


def degrade(x, rate, fs=20):
    low = resample_poly(x, rate, fs, axis=-1)
    return resample_poly(low, fs, rate, axis=-1)[..., :x.shape[-1]].astype(np.float32)


def data(direction, rates=False):
    """(training windows with the flipped copies, labels, {test name: (x, y)})."""
    wx, wy, wu = wisdm.windows()
    mx, my, mu = motionsense.windows()
    if direction == "w2m":
        train = wu <= 26
        x, y = wx[train], wy[train]
        tests = {"MotionSense": (mx, my), "WISDM 27-36": (wx[~train], wy[~train])}
    else:
        held = mu >= 19
        x, y = mx[~held], my[~held]
        tests = {"WISDM": (wx, wy), "MotionSense 19-24": (mx[held], my[held])}
    if rates:
        for name, (xt, yt) in list(tests.items()):
            for r in RATES:
                tests[f"{name} @ {r} Hz"] = (degrade(xt, r), yt)
    return np.concatenate([x, x * FLIP]), np.concatenate([y, y]), tests


def run(spec, args, seed):
    for direction in args.direction:
        variant = direction + ("-rates" if args.rates else "")
        if base.done(NAME, variant, spec, seed) and not args.force:
            print(f"skip {NAME}/{variant}/{spec.id()}/seed{seed} (exists)")
            continue
        xtr, ytr, tests = data(direction, args.rates)
        if spec.classical:
            result, arrays = base.classical(spec, xtr, ytr, tests, CONTEXT, seed, len(PAPER))
        else:
            result, arrays = base.neural_epochs(spec, xtr, ytr, tests, CONTEXT, seed, len(PAPER), EPOCHS, LR)
        print(base.save(NAME, variant, spec, seed, result, arrays), "|",
              ", ".join(f"{t} {s['accuracy']:.3f}" for t, s in result["scores"].items()), flush=True)
