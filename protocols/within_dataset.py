"""Within-dataset protocol (10 s windows at 20 Hz, accelerometer, no augmentation):

  wisdm        train on WISDM users 1-26, test on users 27-36 (Ignatov's split)
  motionsense  train on MotionSense users 1-17, test on users 18-24

Neural models as in the cross-dataset protocol: 60 epochs, batch 256, Adam 1e-3 (CNN 5e-4), last epoch.
"""
from data_processing import motionsense, wisdm
from data_processing.activities import PAPER

from . import base
from .cross_dataset import CONTEXT, EPOCHS, LR

NAME = "within_dataset"
VARIANTS = ("wisdm", "motionsense")


def add_args(parser):
    parser.add_argument("--dataset", nargs="+", choices=VARIANTS, default=list(VARIANTS))


def data(dataset):
    if dataset == "wisdm":
        x, y, u = wisdm.windows()
        train = u <= 26
        return x[train], y[train], {"WISDM 27-36": (x[~train], y[~train])}
    x, y, u = motionsense.windows()
    train = u <= 17
    return x[train], y[train], {"MotionSense 18-24": (x[~train], y[~train])}


def run(spec, args, seed):
    for dataset in args.dataset:
        if base.done(NAME, dataset, spec, seed) and not args.force:
            print(f"skip {NAME}/{dataset}/{spec.id()}/seed{seed} (exists)")
            continue
        xtr, ytr, tests = data(dataset)
        if spec.classical:
            result, arrays = base.classical(spec, xtr, ytr, tests, CONTEXT, seed, len(PAPER))
        else:
            result, arrays = base.neural_epochs(spec, xtr, ytr, tests, CONTEXT, seed, len(PAPER), EPOCHS, LR)
        print(base.save(NAME, dataset, spec, seed, result, arrays), "|",
              ", ".join(f"{t} {s['accuracy']:.3f}" for t, s in result["scores"].items()), flush=True)
