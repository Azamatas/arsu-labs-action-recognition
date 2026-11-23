"""FIN + 40 with the gyroscope through ONE shared FIN, in the DAGHAR leave-one-dataset-out protocol
(Napoli & Borin, ESANN 2026).

Identical to fin_daghar_acc_only.py (data, splits, domain-balanced batches, 5000 steps, lr 1e-3 / 1e-4 chosen
on the sources' test splits, evaluation on all splits of the target, kernel 31, autocorrelation lags 5-30) except
that the model also sees the gyroscope:
  * one FIN, pretrained on the three accelerometer and the three gyroscope axes together (same three target
    features: spectral entropy, height of the autocorrelation peak, spectral peak; per-axis z-score inside the FIN)
  * frozen; head: 6 embeddings + the 40 statistical features (accelerometer, as in the paper)
    -> Dense 128 -> dropout 0.5 -> softmax

    python daghar_benchmark/fin_daghar_shared.py --seeds 0 --folds UCI     # results_shared/<fold>_seed<s>.json and .md
"""
import json
import os
import sys

os.environ["KERAS_BACKEND"] = "torch"  # before the first keras import
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "data-processing")]

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from keras import layers, ops  # noqa: E402

import config as cfg  # noqa: E402
import fin_daghar_acc_only as fd  # noqa: E402  (sets kernel 31 and lags 5-30 on import)

OUT = os.path.join(HERE, "results_shared")


def read(domain, split):
    df = pd.read_csv(os.path.join(fd.DATA, domain, f"{split}.csv"))
    x = np.stack([df[[f"{s}-{a}-{i}" for i in range(fd.WINDOW)]].to_numpy(np.float32)
                  for s in ("accel", "gyro") for a in "xyz"], axis=1)  # (n, 6, 60): accel x y z, gyro x y z
    return x, df["standard activity code"].to_numpy(int)


class Ensemble(keras.Model):
    def __init__(self, fin, **kw):
        super().__init__(**kw)
        self.fin = fin
        self.hidden = layers.Dense(cfg.HEAD_WIDTH, activation="relu")
        self.drop = layers.Dropout(0.5)
        self.out = layers.Dense(fd.N_CLASSES, activation="softmax")

    def call(self, inputs, training=None):
        x, stats = inputs
        h = ops.concatenate([self.fin.embed(x[:, i, :]) for i in range(6)] + [stats], axis=1)
        return self.out(self.drop(self.hidden(h), training=training))


def pretrain(axes, seed):
    keras.utils.set_random_seed(seed)
    fin = fd.bfin.RhythmFIN()
    y = fd.bfin.targets(axes)
    fin.compile(optimizer=keras.optimizers.Adam(cfg.FIN_LR), loss="mse")
    fin.fit(axes, ((y - y.mean(0)) / y.std(0)).astype(np.float32), epochs=cfg.FIN_EPOCHS,
            batch_size=cfg.BATCH_SIZE, verbose=0)
    fin.trainable = False
    return fin


def features(x):
    """Statistical features of (n, 6, 60) windows: the 40 of the paper on the accelerometer (fin_daghar_features.py swaps it)."""
    return fd.bfin.ignatov_features(x[:, :3])


def fold(target, seed):
    sources = [d for d in fd.DOMAINS if d != target]
    train = {d: [read(d, s) for s in ("train", "validation")] for d in sources}
    train = {d: (np.concatenate([a for a, _ in v]), np.concatenate([b for _, b in v])) for d, v in train.items()}
    val = [read(d, "test") for d in sources]
    xv, yv = np.concatenate([a for a, _ in val]), np.concatenate([b for _, b in val])
    tgt = [read(target, s) for s in fd.SPLITS]
    xt, yt = np.concatenate([a for a, _ in tgt]), np.concatenate([b for _, b in tgt])

    xall = np.concatenate([x for x, _ in train.values()])
    f_all = features(xall)
    mu, sd = f_all.mean(0), f_all.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    feats = lambda x: ((features(x) - mu) / sd).astype(np.float32)
    ftrain = {d: feats(x) for d, (x, _) in train.items()}
    fv, ft = feats(xv), feats(xt)

    fin = pretrain(np.concatenate([xall[:, :3].reshape(-1, fd.WINDOW), xall[:, 3:].reshape(-1, fd.WINDOW)]), seed)

    best = None
    for lr in fd.LRS:
        keras.utils.set_random_seed(seed)
        rng = np.random.default_rng(seed)
        model = Ensemble(fin)
        model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy")
        run_best = (-1.0, None, 0)
        for step in range(1, fd.STEPS + 1):
            idx = {d: rng.integers(0, len(train[d][1]), fd.PER_DOMAIN) for d in sources}
            xb = np.concatenate([train[d][0][i] for d, i in idx.items()])
            fb = np.concatenate([ftrain[d][i] for d, i in idx.items()])
            yb = np.concatenate([train[d][1][i] for d, i in idx.items()])
            model.train_on_batch([xb, fb], yb)
            if step % fd.EVAL_EVERY == 0:
                acc = float(np.mean(model.predict([xv, fv], batch_size=2048, verbose=0).argmax(1) == yv))
                if acc > run_best[0]:
                    run_best = (acc, [w.copy() for w in model.get_weights()], step)
        model.set_weights(run_best[1])
        pred = model.predict([xt, ft], batch_size=2048, verbose=0).argmax(1)
        result = {"lr": lr, "val_acc": run_best[0], "best_step": run_best[2], "target_acc": float(np.mean(pred == yt)),
                  "recall": {int(c): float(np.mean(pred[yt == c] == c)) for c in np.unique(yt)}}
        print(f"  {target} lr {lr}: source-val {run_best[0]:.4f} at step {run_best[2]}, target {result['target_acc']:.4f}",
              flush=True)
        if best is None or result["val_acc"] > best["val_acc"]:
            best = result
    best.update(target=target, seed=seed, target_windows=int(len(yt)))
    return best


def report():
    rows = [json.load(open(os.path.join(OUT, n), encoding="utf-8")) for n in sorted(os.listdir(OUT)) if n.endswith(".json")]
    lines = ["# FIN + 40 with one FIN shared by accelerometer and gyroscope axes, DAGHAR leave-one-dataset-out", "",
             "Written by `daghar_benchmark/fin_daghar_shared.py`; same protocol as `daghar_benchmark/fin_daghar_acc_only.py` "
             "(accelerometer only: 70.7 ± 0.3 % over seeds 0-2).", "",
             "| target | seed | windows | accuracy | lr | best step | recall sit / stand / walk / up / down / run |",
             "|---|---|---:|---:|---:|---:|---|"]
    for r in sorted(rows, key=lambda r: (fd.DOMAINS.index(r["target"]), r["seed"])):
        rec = " / ".join(f"{r['recall'][str(c)]:.2f}" if str(c) in r["recall"] else "-" for c in range(fd.N_CLASSES))
        lines.append(f"| {r['target']} | {r['seed']} | {r['target_windows']} | {100 * r['target_acc']:.1f} % | "
                     f"{r['lr']:g} | {r['best_step']} | {rec} |")
    if rows:
        lines += ["", f"Mean over the targets above: {100 * np.mean([r['target_acc'] for r in rows]):.1f} %"]
    try:  # parallel processes may race on this file; the last one rewrites it with every fold anyway
        with open(os.path.join(OUT, "fin_daghar_shared.md"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    except OSError:
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--folds", nargs="+", choices=fd.DOMAINS, default=fd.DOMAINS)
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for seed in args.seeds:
        for target in args.folds:
            print(f"fold: target {target}, seed {seed}", flush=True)
            r = fold(target, seed)
            with open(os.path.join(OUT, f"{target}_seed{seed}.json"), "w", encoding="utf-8") as fh:
                json.dump(r, fh, indent=1)
            print(f"seed {seed}, target {target}: {100 * r['target_acc']:.1f} %", flush=True)
            report()
