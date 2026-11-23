"""Smaller shared FIN (accelerometer + gyroscope) in the DAGHAR leave-one-dataset-out protocol: the pipeline of
fin_daghar_shared.py with the size made a parameter
  * --width   filters per convolution branch (64 in fin_daghar_shared.py)
  * --kernel  convolution length in samples (31)
  * --emb     embedding size of the FIN's dense layer (64; tied to the width there)
  * --head    hidden units of the classification head (128)
  * --circular  wrap-around instead of zero padding (bilin_fin_circular.py): the FIN is then exactly a linear map of
                the power spectrum; the largest relative difference between the convolutional and the spectral
                embedding (efficiency/spectral_equivalence.py) on 5000 training axes is stored with each fold
Everything else (data, pretraining targets, epochs, lr choice, steps, evaluation) is unchanged. The final imitation MSE
of the pretraining (on z-scored targets) is stored with each fold.

    python daghar_benchmark/fin_daghar_small.py --width 32 --kernel 31 --emb 16 --head 32 --seeds 0
    python daghar_benchmark/fin_daghar_small.py --summary          # results_small/summary.md over all configurations
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "efficiency")]

import fin_daghar_shared as s  # noqa: E402  (sets the keras backend, kernel 31, lags 5-30)

import keras  # noqa: E402
import numpy as np  # noqa: E402
from keras import layers  # noqa: E402

fd, cfg, bfin = s.fd, s.cfg, s.fd.bfin
OUT_ROOT = os.path.join(HERE, "results_small")
CHANNELS, T = 6, fd.WINDOW
_last_mse = {}
CIRCULAR = False


def name(w, k, e, h, circ=False):
    return f"w{w}_k{k}_e{e}_h{h}" + ("_circ" if circ else "")


def cost(w, k, e, h):
    """Parameters and FLOPs (2 per multiply-add) of one 6 x 60 window; the 40 statistical features not counted."""
    fin_params = 2 * w * k + w * e + e
    head_params = (CHANNELS * e + 40) * h + h + h * fd.N_CLASSES + fd.N_CLASSES
    flops = CHANNELS * (2 * 2 * w * k * T + w * T + w * T + 2 * w * e) + 2 * ((CHANNELS * e + 40) * h + h * fd.N_CLASSES)
    return fin_params + head_params, flops


def make_fin(emb):
    class SmallFIN(bfin.RhythmFIN):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.proj = layers.Dense(emb, activation="relu")
    return SmallFIN


def pretrain(axes, seed):
    keras.utils.set_random_seed(seed)
    fin = bfin.RhythmFIN()
    y = bfin.targets(axes)
    fin.compile(optimizer=keras.optimizers.Adam(cfg.FIN_LR), loss="mse")
    hist = fin.fit(axes, ((y - y.mean(0)) / y.std(0)).astype(np.float32), epochs=cfg.FIN_EPOCHS,
                   batch_size=cfg.BATCH_SIZE, verbose=0)
    _last_mse["mse"] = float(hist.history["loss"][-1])
    if CIRCULAR:
        import spectral_equivalence as se
        x = axes[np.random.default_rng(seed).choice(len(axes), 5000, replace=False)]
        wts = se.weights(fin)
        conv_emb = keras.ops.convert_to_numpy(fin.embed(keras.ops.convert_to_tensor(x.astype(np.float32))))
        spec_emb = se.spectral_embed(x, wts, se.spectral_matrix(wts, x.shape[1]))
        _last_mse["spectral_err"] = float(np.abs(conv_emb - spec_emb).max() / np.abs(conv_emb).max())
    fin.trainable = False
    return fin


def report(out, w, k, e, h):
    rows = [json.load(open(os.path.join(out, n), encoding="utf-8")) for n in sorted(os.listdir(out)) if n.endswith(".json")]
    params, flops = cost(w, k, e, h)
    lines = [f"# Shared FIN, width {w}, kernel {k}, embedding {e}, head {h}: DAGHAR leave-one-dataset-out", "",
             f"Written by `daghar_benchmark/fin_daghar_small.py`. Parameters {params:,}, {flops / 1e6:.2f} MFLOPs per window.", "",
             "| target | seed | accuracy | lr | best step | imitation MSE |", "|---|---|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: (fd.DOMAINS.index(r["target"]), r["seed"])):
        lines.append(f"| {r['target']} | {r['seed']} | {100 * r['target_acc']:.1f} % | {r['lr']:g} | {r['best_step']} | "
                     f"{r['imitation_mse']:.3f} |")
    if rows:
        lines += ["", f"Mean over the targets above: {100 * np.mean([r['target_acc'] for r in rows]):.1f} %"]
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def summary():
    lines = ["# Smaller shared FIN, DAGHAR leave-one-dataset-out, mean accuracy over the six targets", "",
             "Written by `daghar_benchmark/fin_daghar_small.py --summary`. w64_k31_e64_h128 = fin_daghar_shared.py.", "",
             "| configuration | parameters | MFLOPs | folds | mean accuracy | mean imitation MSE |",
             "|---|---:|---:|---:|---:|---:|"]
    for d in sorted(os.listdir(OUT_ROOT)):
        p = os.path.join(OUT_ROOT, d)
        if not os.path.isdir(p) or not d.startswith("w"):
            continue
        w, k, e, h = (int(x[1:]) for x in d.split("_")[:4])
        rows = [json.load(open(os.path.join(p, n), encoding="utf-8")) for n in os.listdir(p) if n.endswith(".json")]
        if not rows:
            continue
        params, flops = cost(w, k, e, h)
        lines.append(f"| {d} | {params:,} | {flops / 1e6:.2f} | {len(rows)} | "
                     f"{100 * np.mean([r['target_acc'] for r in rows]):.1f} % | "
                     f"{np.mean([r['imitation_mse'] for r in rows]):.3f} |")
    with open(os.path.join(OUT_ROOT, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--kernel", type=int, default=31)
    parser.add_argument("--emb", type=int, default=64)
    parser.add_argument("--head", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--folds", nargs="+", choices=fd.DOMAINS, default=fd.DOMAINS)
    parser.add_argument("--circular", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    os.makedirs(OUT_ROOT, exist_ok=True)
    if args.summary:
        summary()
        sys.exit()
    bfin.WIDTH, bfin.KERNEL, cfg.HEAD_WIDTH = args.width, args.kernel, args.head
    bfin.RhythmFIN = make_fin(args.emb)
    if args.circular:
        from bilin_fin_circular import circular
        bfin.RhythmFIN, CIRCULAR = circular(bfin.RhythmFIN), True
    s.pretrain = pretrain
    out = os.path.join(OUT_ROOT, name(args.width, args.kernel, args.emb, args.head, args.circular))
    os.makedirs(out, exist_ok=True)
    for seed in args.seeds:
        for target in args.folds:
            print(f"fold: target {target}, seed {seed}", flush=True)
            r = s.fold(target, seed)
            r["imitation_mse"] = _last_mse["mse"]
            if CIRCULAR:
                r["spectral_err"] = _last_mse["spectral_err"]
            with open(os.path.join(out, f"{target}_seed{seed}.json"), "w", encoding="utf-8") as fh:
                json.dump(r, fh, indent=1)
            print(f"seed {seed}, target {target}: {100 * r['target_acc']:.1f} %", flush=True)
            report(out, args.width, args.kernel, args.emb, args.head)
