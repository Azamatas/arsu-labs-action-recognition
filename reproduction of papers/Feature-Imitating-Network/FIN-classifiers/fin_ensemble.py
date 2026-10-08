import argparse
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "ensemble_results")
sys.path.insert(0, os.path.dirname(BASE_DIR))

import keras
import numpy as np
from keras import layers, ops
from sklearn.metrics import f1_score

import wisdm_windowing
from config import (AXES, BATCH_SIZE, FEATURES, FIN_EPOCHS, FIN_LR, FREEZE_EPOCHS, HEAD_EPOCHS, HEAD_LR,
                    HEAD_WIDTH, N_CLASSES, SEED, SEGMENT)

import conv_fin
import flat_fin
from conv_fin import ConvFIN
from fin_classifier import Ensemble
from flat_fin import FLAT_TOPOLOGIES, FlatFIN


compute_feature = wisdm_windowing.compute_feature
load_splits = wisdm_windowing.load_splits


def extra_features(windows):
    w = windows.astype(np.float64)
    out = [np.sqrt((w**2).sum(axis=1)).mean(axis=1)]

    for a in range(len(AXES)):
        x = w[:, a, :]
        lo = x.min(axis=1, keepdims=True)
        width = (x.max(axis=1, keepdims=True) - lo) / 10
        width = np.where(width == 0, 1.0, width)
        idx = np.clip(((x - (lo - width / 2)) / width).astype(int), 0, 9)
        counts = np.zeros((len(x), 10))
        np.add.at(counts, (np.arange(len(x))[:, None], idx), 1)
        out += list((counts / x.shape[1]).T)

    return np.stack(out, axis=1).astype(np.float32)


def make_fin(bank, topos, feature, truncate):
    return ConvFIN(truncate=truncate) if bank == "conv" else FlatFIN(topos[feature], truncate)


def build_bank(bank, topos, truncate, init, x_axes_train, x_axes_test, seed):
    module = conv_fin if bank == "conv" else flat_fin
    return module.build_bank(x_axes_train, x_axes_test, seed, init, truncate)


class TestTracker(keras.callbacks.Callback):
    def __init__(self, x, y):
        super().__init__()
        self.x, self.y, self.history = x, y, []

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.x, batch_size=1024, verbose=0).argmax(axis=1)
        self.history.append(float((pred == self.y).mean()))


def run_regime(fins, head_layers, regime, data, seed, extras=None, track_test=False):
    keras.utils.set_random_seed(seed)
    model = Ensemble(fins, head_layers, use_extra=extras is not None)
    (xtr, ytr), (xva, yva), (xte, yte) = data["train"], data["val"], data["test"]

    def inp(split, x):
        return [x, extras[split]] if extras is not None else x

    cbs = [TestTracker(inp("test", xte), yte)] if track_test else []

    def compile_with(trainable, lr):
        model.set_fin_trainable(trainable)
        model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    if regime == "frozen":
        compile_with(False, HEAD_LR)
        model.fit(inp("train", xtr), ytr, epochs=HEAD_EPOCHS, batch_size=BATCH_SIZE,
                  callbacks=cbs, verbose=0)
    elif regime == "gradual":
        compile_with(False, HEAD_LR)
        model.fit(inp("train", xtr), ytr, epochs=FREEZE_EPOCHS, batch_size=BATCH_SIZE,
                  callbacks=cbs, verbose=0)
        compile_with(True, HEAD_LR)
        model.fit(inp("train", xtr), ytr, epochs=HEAD_EPOCHS - FREEZE_EPOCHS,
                  batch_size=BATCH_SIZE, callbacks=cbs, verbose=0)
    else:
        compile_with(True, HEAD_LR)
        model.fit(inp("train", xtr), ytr, epochs=HEAD_EPOCHS, batch_size=BATCH_SIZE,
                  callbacks=cbs, verbose=0)

    def score(split, x, y):
        pred = model.predict(inp(split, x), batch_size=1024, verbose=0).argmax(axis=1)
        return float((pred == y).mean()), f1_score(y, pred, average="macro"), pred

    val_acc, _, _ = score("val", xva, yva)
    test_acc, test_f1, pred = score("test", xte, yte)
    out = {"val_acc": val_acc, "test_acc": test_acc, "test_f1": test_f1, "pred": pred}
    if track_test:
        out["max_test"] = max(cbs[0].history)
    return out




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", choices=["flat", "conv"], default="flat")
    parser.add_argument("--protocol", choices=["clean", "ignatov"], default="clean",
                        help="ignatov: train on users 1-26 (no val group) and report the best "
                             "test reading seen during training, as cnn_wisdm.py does")
    parser.add_argument("--extras", action="store_true",
                        help="concatenate the 31 handcrafted features the FINs do not supply")
    parser.add_argument("--init", choices=["fin", "random"], default="fin")
    parser.add_argument("--truncate", type=int, choices=[1, 2], default=1)
    parser.add_argument("--regimes", nargs="+", default=["frozen", "gradual", "unfrozen"])
    parser.add_argument("--head-layers", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[SEED])
    args = parser.parse_args()

    data = load_splits()
    if args.protocol == "ignatov":
        data["train"] = (np.concatenate([data["train"][0], data["val"][0]]),
                         np.concatenate([data["train"][1], data["val"][1]]))
    x_axes_train = data["train"][0].reshape(-1, SEGMENT)
    x_axes_test = data["test"][0].reshape(-1, SEGMENT)
    topos = FLAT_TOPOLOGIES if args.bank == "flat" else {}

    extras = None
    if args.extras:
        raw = {k: extra_features(v[0]) for k, v in data.items()}
        mu, sd = raw["train"].mean(0), raw["train"].std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        extras = {k: ((v - mu) / sd).astype(np.float32) for k, v in raw.items()}

    rows = []
    for seed in args.seeds:
        fins, imitation = build_bank(
            args.bank, topos, args.truncate, args.init, x_axes_train, x_axes_test, seed
        )
        if args.init == "fin":
            print(f"seed {seed}: imitation R² "
                  + ", ".join(f"{k} {v:.4f}" for k, v in imitation.items()), flush=True)
        for regime in args.regimes:
            for hl in args.head_layers:
                r = run_regime(fins, hl, regime, data, seed, extras,
                               track_test=args.protocol == "ignatov")
                rows.append((seed, regime, hl, r))
                best_test = f", best test reading {r['max_test']:.4f}" if "max_test" in r else ""
                print(f"seed {seed}, {regime}, {hl}-layer head: accuracy {r['test_acc']:.4f}, "
                      f"macro F1 {r['test_f1']:.4f}{best_test}", flush=True)

    if len(args.seeds) > 1:
        for regime in args.regimes:
            for hl in args.head_layers:
                sel = [r for _, rg, h, r in rows if rg == regime and h == hl]
                print(f"mean over seeds, {regime}, {hl}-layer head: "
                      f"accuracy {np.mean([r['test_acc'] for r in sel]):.4f}, "
                      f"macro F1 {np.mean([r['test_f1'] for r in sel]):.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"{args.bank}_{args.init}_truncate{args.truncate}" + ("_extras" if args.extras else "")
    path = os.path.join(OUT_DIR, f"{tag}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([{"seed": s, "regime": rg, "head_layers": h, "val_acc": r["val_acc"],
                    "test_acc": r["test_acc"], "test_f1": r["test_f1"]}
                   for s, rg, h, r in rows], fh, indent=2)


if __name__ == "__main__":
    main()
