import os
import sys

os.environ["KERAS_BACKEND"] = "torch"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "ensemble_results")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ClassicML", "data_processing"))

import argparse
import glob
import json
import time

import keras
import numpy as np
from keras import layers, ops
from sklearn.metrics import f1_score

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
        self.hidden = [layers.Dense(fe.HEAD_WIDTH, activation="relu")
                       for _ in range(head_layers - 1)]
        self.out = layers.Dense(fe.N_CLASSES, activation="softmax")

    def set_fin_trainable(self, flag):
        for fin in self.fins:
            fin.trainable = flag

    def call(self, inputs):
        x, stats = inputs
        parts = [fin.embed(x[:, i, :]) for i in range(len(fe.AXES)) for fin in self.fins]
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
    for feature in fe.FEATURES:
        keras.utils.set_random_seed(seed)
        model = fe.make_fin(bank, topos, feature, 1)
        y = fe.compute_feature(x_train, feature)
        mu, sd = y.mean(), y.std()
        model.compile(optimizer=keras.optimizers.Adam(fe.FIN_LR), loss="mse")
        model.fit(x_train, ((y - mu) / sd).astype(np.float32),
                  epochs=fe.FIN_EPOCHS, batch_size=fe.BATCH_SIZE, verbose=0)
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
        model.compile(optimizer=keras.optimizers.Adam(fe.HEAD_LR),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        hist = model.fit([xtr, stats["train"]], ytr, epochs=epochs,
                         batch_size=fe.BATCH_SIZE, verbose=0)
        curve.extend(float(v) for v in hist.history["loss"])

    if regime == "frozen":
        fit(fe.HEAD_EPOCHS, False)
    elif regime == "gradual":
        fit(fe.FREEZE_EPOCHS, False)
        fit(fe.HEAD_EPOCHS - fe.FREEZE_EPOCHS, True)
    else:
        fit(fe.HEAD_EPOCHS, True)

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
    x_axes_train = data["train"][0].reshape(-1, fe.SEGMENT)
    x_axes_test = data["test"][0].reshape(-1, fe.SEGMENT)
    topos, _ = fe.winning_topologies()

    print(f"backend={keras.backend.backend()}  seed={seed}  statistics={stats['train'].shape[1]}"
          f"  train {len(data['train'][0])} / val {len(data['val'][0])} / test "
          f"{len(data['test'][0])}", flush=True)

    rows, imitation = [], {}

    def record(bank, where, regime, hl, r, t):
        rows.append({"seed": seed, "bank": bank, "where": where, "regime": regime,
                     "head_layers": hl, **r})
        print(f"  {bank:<4} {where:<6} {regime:<8} head={hl}L  val={r['val_acc']:.4f}  "
              f"test={r['test_acc']:.4f}  F1={r['test_f1']:.4f}  [{time.time() - t:.0f}s]",
              flush=True)

    for bank in banks:
        t0 = time.time()
        fins, imitation[bank] = build_bank(bank, topos, x_axes_train, x_axes_test, seed)
        snapshot = {f: [w.copy() for w in m.get_weights()] for f, m in fins.items()}
        print(f"{bank} FINs pretrained in {time.time() - t0:.0f}s  R2 "
              + "  ".join(f"{k}={v:.4f}" for k, v in imitation[bank].items()), flush=True)
        for where in WITH_FIN:
            for regime in REGIMES:
                for hl in HEADS:
                    if where == "after" and hl < 2:
                        continue
                    t = time.time()
                    record(bank, where, regime, hl,
                           run(fins, snapshot, where, hl, regime, data, stats, seed), t)

    for hl in HEADS:
        t = time.time()
        record("-", "only", "-", hl, run({}, {}, "only", hl, "unfrozen", data, stats, seed), t)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"stats_seed{seed}.json"), "w", encoding="utf-8") as fh:
        json.dump({"seed": seed, "imitation": imitation, "rows": rows}, fh, indent=1)
    print(f"wrote ensemble_results/stats_seed{seed}.json", flush=True)


def summary():
    rows = []
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "stats_seed*.json"))):
        with open(path, encoding="utf-8") as fh:
            rows += json.load(fh)["rows"]
    if not rows:
        raise SystemExit("no ensemble_results/stats_seed*.json yet")
    seeds = sorted({r["seed"] for r in rows})

    def pick(bank, where, regime, heads, key="test_acc"):
        return [r[key] for r in rows if r["bank"] == bank and r["where"] == where
                and r["regime"] == regime and r["head_layers"] in heads]

    print(f"seeds {seeds}; each cell is the mean over head depths 2-4 and seeds, the depths "
          f"where all three placements exist\n")
    for bank in ("flat", "conv"):
        if not any(r["bank"] == bank for r in rows):
            continue
        print(f"### {bank} bank\n")
        print("| regime | none | concat | after | concat − none | after − none |")
        print("|---|---:|---:|---:|---:|---:|")
        for regime in REGIMES:
            m = {w: np.mean(pick(bank, w, regime, (2, 3, 4))) for w in WITH_FIN}
            print(f"| {regime} | {m['none']:.4f} | {m['concat']:.4f} | {m['after']:.4f} "
                  f"| {m['concat'] - m['none']:+.4f} | {m['after'] - m['none']:+.4f} |")
        print("\nmacro F1, same cells\n")
        print("| regime | none | concat | after |")
        print("|---|---:|---:|---:|")
        for regime in REGIMES:
            m = {w: np.mean(pick(bank, w, regime, (2, 3, 4), "test_f1")) for w in WITH_FIN}
            print(f"| {regime} | {m['none']:.4f} | {m['concat']:.4f} | {m['after']:.4f} |")

        # paired differences cell by cell: which placement wins how often
        wins = {"concat": 0, "after": 0}
        total = 0
        for regime in REGIMES:
            for hl in (2, 3, 4):
                for s in seeds:
                    base = [r["test_acc"] for r in rows if r["bank"] == bank
                            and r["where"] == "none" and r["regime"] == regime
                            and r["head_layers"] == hl and r["seed"] == s]
                    if not base:
                        continue
                    total += 1
                    for w in wins:
                        other = [r["test_acc"] for r in rows if r["bank"] == bank
                                 and r["where"] == w and r["regime"] == regime
                                 and r["head_layers"] == hl and r["seed"] == s]
                        wins[w] += int(other and other[0] > base[0])
        print(f"\ncells where adding the statistics beats `none`: "
              f"concat {wins['concat']}/{total}, after {wins['after']}/{total}")
        h1 = {w: np.mean(pick(bank, w, regime, (1,))) for w in ("none", "concat")
              for regime in ("frozen",)}
        print(f"head 1 layer, frozen: none {h1['none']:.4f}, concat {h1['concat']:.4f} "
              f"(`after` has no hidden layer to follow here)\n")

    only = [r for r in rows if r["where"] == "only"]
    if only:
        print("### 31 statistical features alone, no FIN\n")
        print("| head layers | " + " | ".join(str(h) for h in HEADS) + " |")
        print("|---|" + "---:|" * len(HEADS))
        print("| test acc | " + " | ".join(
            f"{np.mean([r['test_acc'] for r in only if r['head_layers'] == h]):.4f}"
            for h in HEADS) + " |")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=fe.SEED)
    parser.add_argument("--banks", nargs="+", choices=["flat", "conv"], default=["flat", "conv"])
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        summary()
    else:
        train(args.seed, args.banks)


if __name__ == "__main__":
    main()
