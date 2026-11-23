"""FIN + 40 in the DAGHAR leave-one-dataset-out protocol of Napoli & Borin (ESANN 2026), for comparison with their
published numbers (mean target accuracy: ERM 65.7 %, ERM + TF-C pretraining 66.8 %, 3 seeds, 4 backbones).

Their protocol, followed here:
  * DAGHAR standardized view (datasets/DAGHAR/standardized_view): 20 Hz, 3 s windows (60 samples), m/s^2, gravity
    removed; six domains KuHar, MotionSense, RealWorld_thigh, RealWorld_waist, UCI, WISDM (WISDM 2019)
  * leave one dataset out: train on the train + validation splits of the five sources, select the model on their test
    splits, report accuracy on all splits of the held-out target
  * domain-balanced mini-batches of 60 (12 per source), 5000 steps, Adam with lr 1e-3 and 1e-4, the better one kept
    (by the source validation accuracy), the checkpoint with the highest validation accuracy evaluated
Our model, unchanged except for the window length (fixed before looking at any target result):
  * FIN on the accelerometer only (DAGHAR also has the gyroscope; our model never used it), shared by the three axes,
    convolution kernels of 31 samples instead of 61 (half the window, as 61 was about a third of 200), imitating
    spectral entropy, height of the autocorrelation peak (lags 5-30 instead of 5-60) and spectral peak; pretrained
    without labels on the source training windows (60 epochs, batch 256), then frozen
  * head: three FIN embeddings + the 40 statistical features -> Dense 128 -> dropout 0.5 -> softmax (6 classes)
  * no flip augmentation (not part of their protocol)

    python daghar_benchmark/fin_daghar_acc_only.py --seeds 0              # all six folds
    python daghar_benchmark/fin_daghar_acc_only.py --seeds 0 --folds UCI  # some folds; results_fin/<fold>.json, fin_daghar.md
"""
import json
import os
import sys

os.environ["KERAS_BACKEND"] = "torch"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "data-processing")]

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from keras import layers, ops  # noqa: E402

import bilin_fin_temporal_plus_original as bfin  # noqa: E402
import config as cfg  # noqa: E402
import wisdm_windowing as ww  # noqa: E402

DATA = os.path.join(ROOT, "datasets", "DAGHAR", "standardized_view")
OUT = os.path.join(HERE, "results_fin")
DOMAINS = ["KuHar", "MotionSense", "RealWorld_thigh", "RealWorld_waist", "UCI", "WISDM"]
SPLITS = ("train", "validation", "test")
WINDOW = 60
bfin.KERNEL = 31  # read when RhythmFIN builds its convolutions
ww.MAX_LAG = 30   # read by temporal_axis for the autocorrelation peak
STEPS, PER_DOMAIN, LRS, EVAL_EVERY = 5000, 12, (1e-3, 1e-4), 100
N_CLASSES = 6     # DAGHAR codes: 0 sit, 1 stand, 2 walk, 3 upstairs, 4 downstairs, 5 run


def read(domain, split):
    df = pd.read_csv(os.path.join(DATA, domain, f"{split}.csv"))
    x = np.stack([df[[f"accel-{a}-{i}" for i in range(WINDOW)]].to_numpy(np.float32) for a in "xyz"], axis=1)
    return x, df["standard activity code"].to_numpy(int)


class Ensemble(keras.Model):
    def __init__(self, fin, **kw):
        super().__init__(**kw)
        self.fin = fin
        self.hidden = layers.Dense(cfg.HEAD_WIDTH, activation="relu")
        self.drop = layers.Dropout(0.5)
        self.out = layers.Dense(N_CLASSES, activation="softmax")

    def call(self, inputs, training=None):
        x, stats = inputs
        h = ops.concatenate([self.fin.embed(x[:, i, :]) for i in range(3)] + [stats], axis=1)
        return self.out(self.drop(self.hidden(h), training=training))


def fold(target, seed):
    sources = [d for d in DOMAINS if d != target]
    train = {d: [read(d, s) for s in ("train", "validation")] for d in sources}
    train = {d: (np.concatenate([a for a, _ in v]), np.concatenate([b for _, b in v])) for d, v in train.items()}
    val = [read(d, "test") for d in sources]
    xv, yv = np.concatenate([a for a, _ in val]), np.concatenate([b for _, b in val])
    tgt = [read(target, s) for s in SPLITS]
    xt, yt = np.concatenate([a for a, _ in tgt]), np.concatenate([b for _, b in tgt])

    xall = np.concatenate([x for x, _ in train.values()])
    f_all = bfin.ignatov_features(xall)
    mu, sd = f_all.mean(0), f_all.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    feats = lambda x: ((bfin.ignatov_features(x) - mu) / sd).astype(np.float32)
    ftrain = {d: feats(x) for d, (x, _) in train.items()}
    fv, ft = feats(xv), feats(xt)

    # FIN pretraining on the source training windows (no labels), then frozen
    keras.utils.set_random_seed(seed)
    fin = bfin.RhythmFIN()
    axes = xall.reshape(-1, WINDOW)
    y = bfin.targets(axes)
    fin.compile(optimizer=keras.optimizers.Adam(cfg.FIN_LR), loss="mse")
    fin.fit(axes, ((y - y.mean(0)) / y.std(0)).astype(np.float32), epochs=cfg.FIN_EPOCHS,
            batch_size=cfg.BATCH_SIZE, verbose=0)
    fin.trainable = False

    best = None
    for lr in LRS:
        keras.utils.set_random_seed(seed)
        rng = np.random.default_rng(seed)
        model = Ensemble(fin)
        model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy")
        run_best = (-1.0, None, 0)
        for step in range(1, STEPS + 1):
            idx = {d: rng.integers(0, len(train[d][1]), PER_DOMAIN) for d in sources}
            xb = np.concatenate([train[d][0][i] for d, i in idx.items()])
            fb = np.concatenate([ftrain[d][i] for d, i in idx.items()])
            yb = np.concatenate([train[d][1][i] for d, i in idx.items()])
            model.train_on_batch([xb, fb], yb)
            if step % EVAL_EVERY == 0:
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
    rows = []
    for name in sorted(os.listdir(OUT)):
        if name.endswith(".json"):
            with open(os.path.join(OUT, name), encoding="utf-8") as fh:
                rows.append(json.load(fh))
    lines = ["# FIN + 40 in the DAGHAR leave-one-dataset-out protocol", "",
             "Written by `daghar_benchmark/fin_daghar_acc_only.py`. Accuracy on all splits of the held-out target; learning rate "
             "and checkpoint chosen on the sources' test splits. Napoli & Borin (ESANN 2026), same protocol, mean over "
             "4 backbones x 6 targets x 3 seeds: ERM 65.7 %, ERM + TF-C 66.8 %.", "",
             "| target | seed | windows | accuracy | lr | best step | recall sit / stand / walk / up / down / run |",
             "|---|---|---:|---:|---:|---:|---|"]
    for r in sorted(rows, key=lambda r: (DOMAINS.index(r["target"]), r["seed"])):
        rec = " / ".join(f"{r['recall'][str(c)]:.2f}" if str(c) in r["recall"] else "-" for c in range(N_CLASSES))
        lines.append(f"| {r['target']} | {r['seed']} | {r['target_windows']} | {100 * r['target_acc']:.1f} % | "
                     f"{r['lr']:g} | {r['best_step']} | {rec} |")
    if rows:
        lines += ["", f"Mean over the targets above: {100 * np.mean([r['target_acc'] for r in rows]):.1f} %"]
    with open(os.path.join(OUT, "fin_daghar.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--folds", nargs="+", choices=DOMAINS, default=DOMAINS)
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
