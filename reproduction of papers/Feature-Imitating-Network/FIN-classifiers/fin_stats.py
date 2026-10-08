import os
import sys

os.environ["KERAS_BACKEND"] = "torch"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(HERE, "ensemble_results")
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "ClassicML", "data_processing"))

import argparse
import glob
import json

import keras
import numpy as np
from keras import layers, ops
from sklearn.metrics import f1_score

import config as cfg
import fin_ensemble as fe
import generate_wisdm_data as ignatov

FIN_COLUMNS = 9
WITH_FIN = ("none", "concat", "after")
REGIMES = ("frozen", "gradual", "unfrozen")
HEADS = (1, 2, 3, 4)


def ignatov_stats(windows):
    w = windows.astype(np.float64)
    full = np.stack([ignatov.extract_basic_features(x[0], x[1], x[2]) for x in w])
    return full[:, FIN_COLUMNS:]


class StatsEnsemble(keras.Model):

    def __init__(self, fins, head_layers, where, **kw):
        super().__init__(**kw)
        self.fins = [] if where == "only" else list(fins.values())
        self.where = where
        self.hidden = [layers.Dense(cfg.HEAD_WIDTH, activation="relu")
                       for _ in range(head_layers - 1)]
        self.out = layers.Dense(cfg.N_CLASSES, activation="softmax")

    def set_fin_trainable(self, flag):
        for fin in self.fins:
            fin.trainable = flag

    def call(self, inputs):
        x, stats = inputs
        parts = [fin.embed(x[:, i, :]) for i in range(len(cfg.AXES)) for fin in self.fins]
        if self.where in ("concat", "only"):
            parts.append(stats)
        h = parts[0] if len(parts) == 1 else ops.concatenate(parts, axis=1)
        for j, layer in enumerate(self.hidden):
            h = layer(h)
            if j == 0 and self.where == "after":
                h = ops.concatenate([h, stats], axis=1)
        return self.out(h)


def build_bank(bank, topos, x_train, x_test, seed):
    fins, imitation = {}, {}
    for feature in cfg.FEATURES:
        keras.utils.set_random_seed(seed)
        model = fe.make_fin(bank, topos, feature, 1)
        y = fe.compute_feature(x_train, feature)
        mu, sd = y.mean(), y.std()
        model.compile(optimizer=keras.optimizers.Adam(cfg.FIN_LR), loss="mse")
        model.fit(x_train, ((y - mu) / sd).astype(np.float32),
                  epochs=cfg.FIN_EPOCHS, batch_size=cfg.BATCH_SIZE, verbose=0)
        pred = model.predict(x_test, batch_size=1024, verbose=0).ravel().astype(np.float64)
        pred = pred * sd + mu
        true = fe.compute_feature(x_test, feature)
        imitation[feature] = float(1 - ((true - pred) ** 2).sum()
                                   / ((true - true.mean()) ** 2).sum())
        fins[feature] = model
    return fins, imitation


def run(fins, snapshot, where, head_layers, regime, data, stats, seed):
    for feature, model in fins.items():
        model.set_weights(snapshot[feature])
    keras.utils.set_random_seed(seed)
    model = StatsEnsemble(fins, head_layers, where)
    (xtr, ytr), (xva, yva), (xte, yte) = data["train"], data["val"], data["test"]
    curve = []

    def fit(epochs, trainable):
        model.set_fin_trainable(trainable)
        model.compile(optimizer=keras.optimizers.Adam(cfg.HEAD_LR),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        hist = model.fit([xtr, stats["train"]], ytr, epochs=epochs,
                         batch_size=cfg.BATCH_SIZE, verbose=0)
        curve.extend(float(v) for v in hist.history["loss"])

    if regime == "frozen":
        fit(cfg.HEAD_EPOCHS, False)
    elif regime == "gradual":
        fit(cfg.FREEZE_EPOCHS, False)
        fit(cfg.HEAD_EPOCHS - cfg.FREEZE_EPOCHS, True)
    else:
        fit(cfg.HEAD_EPOCHS, True)

    def predict(split, x):
        return model.predict([x, stats[split]], batch_size=1024, verbose=0).argmax(axis=1)

    pva, pte = predict("val", xva), predict("test", xte)
    return {"val_acc": float((pva == yva).mean()), "test_acc": float((pte == yte).mean()),
            "test_f1": float(f1_score(yte, pte, average="macro")), "loss_curve": curve}


def train(seed, banks):
    data = fe.load_splits()
    raw = {k: ignatov_stats(v[0]) for k, v in data.items()}
    mu, sd = raw["train"].mean(0), raw["train"].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    stats = {k: ((v - mu) / sd).astype(np.float32) for k, v in raw.items()}
    x_axes_train = data["train"][0].reshape(-1, cfg.SEGMENT)
    x_axes_test = data["test"][0].reshape(-1, cfg.SEGMENT)
    topos = fe.FLAT_TOPOLOGIES

    rows, imitation = [], {}

    def record(bank, where, regime, hl, r):
        rows.append({"seed": seed, "bank": bank, "where": where, "regime": regime,
                     "head_layers": hl, **r})
        print(f"seed {seed}, {label(bank, where, regime)}, {hl}-layer head: "
              f"accuracy {r['test_acc']:.4f}, macro F1 {r['test_f1']:.4f}", flush=True)

    for bank in banks:
        fins, imitation[bank] = build_bank(bank, topos, x_axes_train, x_axes_test, seed)
        snapshot = {f: [w.copy() for w in m.get_weights()] for f, m in fins.items()}
        print(f"seed {seed}, {bank}: imitation R² "
              + ", ".join(f"{k} {v:.4f}" for k, v in imitation[bank].items()), flush=True)
        for where in WITH_FIN:
            for regime in REGIMES:
                for hl in HEADS:
                    if where == "after" and hl < 2:
                        continue
                    record(bank, where, regime, hl,
                           run(fins, snapshot, where, hl, regime, data, stats, seed))

    for hl in HEADS:
        record("-", "only", "-", hl, run({}, {}, "only", hl, "unfrozen", data, stats, seed))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"stats_seed{seed}.json"), "w", encoding="utf-8") as fh:
        json.dump({"seed": seed, "imitation": imitation, "rows": rows}, fh, indent=1)


def label(bank, where, regime):
    return "stats only" if where == "only" else f"{bank} {where} {regime}"


def summary():
    rows = []
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "stats_seed*.json"))):
        with open(path, encoding="utf-8") as fh:
            rows += json.load(fh)["rows"]
    if not rows:
        raise SystemExit("no ensemble_results/stats_seed*.json yet")

    def mean_sd(v):
        return f"{np.mean(v):.4f}" + (f" ± {np.std(v, ddof=1):.4f}" if len(v) > 1 else "")

    cells = [(b, w, rg, hl) for b in ("flat", "conv") for w in WITH_FIN for rg in REGIMES for hl in HEADS]
    cells += [("-", "only", "-", hl) for hl in HEADS]
    for bank, where, regime, hl in cells:
        sel = [r for r in rows if r["bank"] == bank and r["where"] == where
               and r["regime"] == regime and r["head_layers"] == hl]
        if sel:
            print(f"{label(bank, where, regime)}, {hl}-layer head, {len(sel)} seeds: "
                  f"accuracy {mean_sd([r['test_acc'] for r in sel])}, "
                  f"macro F1 {mean_sd([r['test_f1'] for r in sel])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--banks", nargs="+", choices=["flat", "conv"], default=["flat", "conv"])
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        summary()
    else:
        train(args.seed, args.banks)


if __name__ == "__main__":
    main()
