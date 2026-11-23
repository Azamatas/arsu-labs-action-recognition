"""Shared by the protocols: scores, result files, and training one neural model on fixed train / test sets.

Every run writes results/<protocol>/<variant>/<model id>/seed<s>.json (configuration, scores per test set, cost,
training details) and a .npz next to it (predictions, truths and accuracy curves).
"""
import json
import os
import time

import numpy as np
from sklearn.metrics import f1_score

from bifin import train

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def scores(y, p, n_classes):
    return {"accuracy": float(np.mean(p == y)), "macro_f1": float(f1_score(y, p, average="macro")),
            "recall": [float(np.mean(p[y == c] == c)) if np.any(y == c) else None for c in range(n_classes)],
            "windows": int(len(y))}


def path(protocol, variant, model_id, seed):
    return os.path.join(RESULTS, protocol, variant, model_id, f"seed{seed}")


def save(protocol, variant, spec, seed, result, arrays):
    base = path(protocol, variant, spec.id(), seed)
    os.makedirs(os.path.dirname(base), exist_ok=True)
    result = {"protocol": protocol, "variant": variant, "model": spec.id(), "seed": seed,
              "config": spec.config(), **{k: v for k, v in spec.info.items()}, **result,
              "date": time.strftime("%Y-%m-%d %H:%M")}
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    np.savez_compressed(base + ".npz", **arrays)
    return base + ".json"


def done(protocol, variant, spec, seed):
    return os.path.exists(path(protocol, variant, spec.id(), seed) + ".json")


def neural_epochs(spec, xtr, ytr, tests, ctx, seed, n_classes, epochs, lr, val=None, curves=True):
    """Train a neural spec for a number of epochs on (xtr, ytr) and score it on every test set.

    tests: {name: (x, y)}. val: optional (x, y) whose best epoch is also reported ('best' scores).
    Returns (result dict, arrays dict)."""
    spec.setup(xtr, ctx)
    model = spec.build(xtr.shape[1], xtr.shape[2], n_classes, ctx, seed, xtr)
    sets = {name: (train.tensors(*spec.inputs(x)), y) for name, (x, y) in tests.items()}
    if val is not None:
        sets["validation"] = (train.tensors(*spec.inputs(val[0])), val[1])
    out = train.fit_epochs(model, train.tensors(*spec.inputs(xtr)), ytr, epochs, spec.lr or lr, seed,
                           evals=sets if curves or val is not None else {},
                           select="validation" if val is not None else None, frozen=spec.frozen)
    last = out["last"] or {name: train.predict(model, xs) for name, (xs, _) in sets.items()}
    result = {"epochs": epochs, "lr": spec.lr or lr, "train_windows": int(len(ytr)), "train_seconds": out["seconds"],
              "scores": {name: scores(y, last[name], n_classes) for name, (_, y) in tests.items()}}
    arrays = {f"truth|{name}": y for name, (_, y) in tests.items()}
    arrays.update({f"pred|{name}": last[name] for name in tests})
    arrays.update({f"curve|{name}": np.array(c) for name, c in out["curves"].items()})
    if val is not None:
        result["best_epoch"] = out["best_epoch"]
        result["scores_best"] = {name: scores(y, out["best"][name], n_classes) for name, (_, y) in tests.items()}
        arrays.update({f"pred_best|{name}": out["best"][name] for name in tests})
    return result, arrays


def classical(spec, xtr, ytr, tests, ctx, seed, n_classes):
    preds = spec.fit_predict(xtr, ytr, {name: x for name, (x, _) in tests.items()}, seed, ctx)
    result = {"train_windows": int(len(ytr)),
              "scores": {name: scores(y, preds[name], n_classes) for name, (_, y) in tests.items()}}
    arrays = {f"truth|{name}": y for name, (_, y) in tests.items()}
    arrays.update({f"pred|{name}": preds[name] for name in tests})
    return result, arrays
