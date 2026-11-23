"""DAGHAR leave-one-dataset-out protocol (Napoli & Borin, ESANN 2026): 3 s windows at 20 Hz, accelerometer +
gyroscope (6 x 60), six domains.

For every target domain: train on the train + validation splits of the five other domains, select the model on their
test splits, report accuracy on all splits of the target. Domain-balanced batches of 60 (12 per source), 5000 steps,
Adam; every 100 steps the accuracy on the source test splits is checked and the best weights kept. With several
learning rates (--lrs) the one with the better source accuracy is kept; lr 1e-3 won in all 96 folds we ran with
1e-3 and 1e-4, so it is the default. Statistical features and input statistics come from all source training windows;
the FIN is pretrained on all their single axes (accelerometer and gyroscope).
"""
import numpy as np

from data_processing import daghar
from data_processing.activities import DAGHAR

from . import base
from .models import Context
from bifin import train

NAME = "daghar_lodo"
STEPS, PER_DOMAIN, EVAL_EVERY = 5000, 12, 100
CONTEXT = Context(max_lag=30)


def add_args(parser):
    parser.add_argument("--targets", nargs="+", choices=daghar.DOMAINS, default=list(daghar.DOMAINS))
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-3])
    parser.add_argument("--accel-only", action="store_true", help="drop the gyroscope channels")


def data(target, accel_only=False):
    sources = [d for d in daghar.DOMAINS if d != target]
    train_sets = [daghar.read_many([d], ("train", "validation"), accel_only) for d in sources]
    val = daghar.read_many(sources, ("test",), accel_only)
    test = daghar.read_many([target], daghar.SPLITS, accel_only)
    return sources, train_sets, val, test


def run(spec, args, seed):
    n_classes = len(DAGHAR)
    for target in args.targets:
        variant = target + ("-accel" if args.accel_only else "")
        if base.done(NAME, variant, spec, seed) and not args.force:
            print(f"skip {NAME}/{variant}/{spec.id()}/seed{seed} (exists)")
            continue
        sources, train_sets, (xv, yv), (xt, yt) = data(target, args.accel_only)
        xall = np.concatenate([x for x, _ in train_sets])
        yall = np.concatenate([y for _, y in train_sets])
        tests = {target: (xt, yt)}
        if spec.classical:
            result, arrays = base.classical(spec, xall, yall, tests, CONTEXT, seed, n_classes)
        else:
            spec.setup(xall, CONTEXT)
            domains = [(train.tensors(*spec.inputs(x)), train.tensors(y.astype(np.int64))[0]) for x, y in train_sets]
            val = (train.tensors(*spec.inputs(xv)), yv)
            test_in = train.tensors(*spec.inputs(xt))
            best = None
            for lr in args.lrs:
                model = spec.build(xall.shape[1], xall.shape[2], n_classes, CONTEXT, seed, xall)
                val_acc, step = train.fit_steps(model, domains, STEPS, lr, seed, PER_DOMAIN, val, EVAL_EVERY,
                                                frozen=spec.frozen)
                pred = train.predict(model, test_in)
                print(f"  {target} lr {lr:g}: source validation {val_acc:.4f} at step {step}, "
                      f"target {np.mean(pred == yt):.4f}", flush=True)
                if best is None or val_acc > best[0]:
                    best = (val_acc, step, lr, pred)
            val_acc, step, lr, pred = best
            result = {"lr": lr, "lrs": args.lrs, "steps": STEPS, "best_step": step, "source_val_accuracy": val_acc,
                      "train_windows": int(len(yall)), "scores": {target: base.scores(yt, pred, n_classes)}}
            arrays = {f"truth|{target}": yt, f"pred|{target}": pred}
        result["sources"] = sources
        print(base.save(NAME, variant, spec, seed, result, arrays), "|",
              f"{target} {result['scores'][target]['accuracy']:.3f}", flush=True)
