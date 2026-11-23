"""BiTFIN on DAGHAR with the per-axis embeddings pooled over the axes instead of concatenated, so that the head no longer
knows which axis carried which rhythm (invariant to how the device is held). Accelerometer and gyroscope are pooled
separately. Everything else is fin_daghar_features.py acc7 (circular FIN w32 k31 e64 shared by the 6 axes, head 64, the
7 accelerometer features):

  concat   3 + 3 embeddings side by side, 6 x 64 + 7 head inputs  (= results_features/acc7, not rerun)
  mean     mean over the 3 accelerometer and over the 3 gyroscope embeddings, 2 x 64 + 7 inputs
  meanmax  mean and max over the axes of each sensor, 4 x 64 + 7 inputs

Faster than fin_daghar_features.py, with the same results for the same seed:
  * only lr 1e-3: it had the higher source validation accuracy in all 96 earlier DAGHAR folds (results_small,
    results_shared, results_features), so the lr 1e-4 run never changed a reported number
  * one process per target fold; the FIN is pretrained once per fold and seed and shared by the variants

    python daghar_benchmark/fin_daghar_pooling.py --folds UCI --seeds 0     # results_pooling/<variant>/<fold>_seed<s>.json
    python daghar_benchmark/fin_daghar_pooling.py --report
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]

import fin_daghar_features as ff  # noqa: E402  (sets the keras backend, kernel 31, lags 5-30)
import keras  # noqa: E402
import numpy as np  # noqa: E402
from keras import layers, ops  # noqa: E402

s, fs, fd, cfg = ff.s, ff.fs, ff.fd, ff.cfg
OUT = os.path.join(HERE, "results_pooling")
CONCAT = os.path.join(HERE, "results_features", "acc7")
VARIANTS = ("mean", "meanmax")
N_FEATURES = 7


def make_head(mode):
    class Pooled(keras.Model):
        def __init__(self, fin, **kw):
            super().__init__(**kw)
            self.fin = fin
            self.hidden = layers.Dense(cfg.HEAD_WIDTH, activation="relu")
            self.drop = layers.Dropout(0.5)
            self.out = layers.Dense(fd.N_CLASSES, activation="softmax")

        def call(self, inputs, training=None):
            x, stats = inputs
            emb = [self.fin.embed(x[:, i, :]) for i in range(6)]
            parts = []
            for group in (emb[:3], emb[3:]):  # accelerometer, gyroscope
                g = ops.stack(group, axis=1)
                parts.append(ops.mean(g, axis=1))
                if mode == "meanmax":
                    parts.append(ops.max(g, axis=1))
            h = ops.concatenate(parts + [stats], axis=1)
            return self.out(self.drop(self.hidden(h), training=training))
    return Pooled


def head_params(mode):
    inputs = {"concat": 6, "mean": 2, "meanmax": 4}[mode] * ff.EMB + N_FEATURES
    return inputs * ff.HEAD + ff.HEAD + ff.HEAD * fd.N_CLASSES + fd.N_CLASSES


def report():
    def load(d):
        rows = {}
        for f in glob.glob(os.path.join(d, "*_seed*.json")):
            r = json.load(open(f, encoding="utf-8"))
            rows.setdefault(r["target"], []).append(r["target_acc"])
        return rows

    fin_params = (fd.WINDOW // 2 + 1) * ff.EMB + ff.EMB  # L and c of the spectral form
    lines = ["# BiTFIN on DAGHAR leave-one-dataset-out: concatenated vs pooled axis embeddings", "",
             "Written by `daghar_benchmark/fin_daghar_pooling.py`. 7 accelerometer features; parameters of the deployed "
             "form (matrix L + bias + head); target accuracy, %.", "",
             "| head input | parameters | " + " | ".join(fd.DOMAINS) + " | mean |",
             "|---|---:|" + "---:|" * (len(fd.DOMAINS) + 1)]
    for mode, d in [("concat", CONCAT)] + [(v, os.path.join(OUT, v)) for v in VARIANTS]:
        rows = load(d)
        if not rows:
            continue
        cells = [f"{100 * np.mean(rows[t]):.1f}" if t in rows else "-" for t in fd.DOMAINS]
        mean = f"**{100 * np.mean([np.mean(rows[t]) for t in fd.DOMAINS]):.1f}**" if len(rows) == 6 else "-"
        lines.append(f"| {mode} | {fin_params + head_params(mode):,} | " + " | ".join(cells) + f" | {mean} |")
    with open(os.path.join(OUT, "pooling.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--folds", nargs="+", choices=fd.DOMAINS, default=fd.DOMAINS)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not args.report:
        ff.setup("acc7")
        fd.LRS = (1e-3,)
        cache = {}

        def pretrain_once(axes, seed):
            key = (seed, axes.shape)
            if key not in cache:
                cache[key] = (fs.pretrain(axes, seed), dict(fs._last_mse))
            fs._last_mse.update(cache[key][1])
            return cache[key][0]

        s.pretrain = pretrain_once
        for seed in args.seeds:
            for target in args.folds:
                for mode in args.variants:
                    s.Ensemble = make_head(mode)
                    print(f"fold: target {target}, seed {seed}, {mode}", flush=True)
                    r = s.fold(target, seed)
                    r.update(variant=mode, imitation_mse=fs._last_mse["mse"],
                             spectral_err=fs._last_mse["spectral_err"])
                    out = os.path.join(OUT, mode)
                    os.makedirs(out, exist_ok=True)
                    with open(os.path.join(out, f"{target}_seed{seed}.json"), "w", encoding="utf-8") as fh:
                        json.dump(r, fh, indent=1)
                    print(f"seed {seed}, target {target}, {mode}: {100 * r['target_acc']:.1f} %", flush=True)
    report()
