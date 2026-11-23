"""BiTFIN (circular, 32 filters, kernel 31, embedding 64, head 64; FFT + one matrix on the device) in the DAGHAR
leave-one-dataset-out protocol with the 40 accelerometer features of the paper replaced by short per-sensor sets.
Everything else is fin_daghar_small.py --width 32 --kernel 31 --emb 64 --head 64 --circular (its result with the 40
features: results_small/w32_k31_e64_h64_circ).

Per sensor (accelerometer, gyroscope), columns of statistical_features.extract_basic_features:
  10: mean, std and mean absolute deviation per axis, mean magnitude
   7: mean and std per axis, mean magnitude

  acc10        10 accelerometer features
  acc10_gyro10 10 accelerometer + 10 gyroscope features
  acc7         7 accelerometer features
  acc7_gyro7   7 accelerometer + 7 gyroscope features

    python daghar_benchmark/fin_daghar_features.py acc10_gyro10 --seeds 0
    python daghar_benchmark/fin_daghar_features.py --report
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]

import fin_daghar_small as fs  # noqa: E402  (sets the keras backend, kernel 31, lags 5-30)
import numpy as np  # noqa: E402

s, fd, cfg, bfin = fs.s, fs.fd, fs.cfg, fs.bfin
OUT = os.path.join(HERE, "results_features")
REFERENCE = os.path.join(HERE, "results_small", "w32_k31_e64_h64_circ")
COLS = {10: list(range(10)), 7: [0, 1, 2, 3, 4, 5, 9]}
VARIANTS = {"acc10": (10, 0), "acc10_gyro10": (10, 10), "acc7": (7, 0), "acc7_gyro7": (7, 7)}
WIDTH, KERNEL, EMB, HEAD = 32, 31, 64, 64


def make_features(n_acc, n_gyro):
    def features(x):
        f = [bfin.ignatov_features(x[:, :3])[:, COLS[n_acc]]]
        if n_gyro:
            f.append(bfin.ignatov_features(x[:, 3:])[:, COLS[n_gyro]])
        return np.concatenate(f, axis=1)
    return features


def setup(variant):
    bfin.WIDTH, bfin.KERNEL, cfg.HEAD_WIDTH = WIDTH, KERNEL, HEAD
    from bilin_fin_circular import circular
    bfin.RhythmFIN = circular(fs.make_fin(EMB))
    fs.CIRCULAR = True
    s.pretrain = fs.pretrain
    s.features = make_features(*VARIANTS[variant])


def report():
    def load(d):
        return {(r["target"], r["seed"]): r for r in
                (json.load(open(f, encoding="utf-8")) for f in glob.glob(os.path.join(d, "*_seed*.json")))}

    runs = {"acc40 (paper, results_small)": (40, load(REFERENCE))}
    for v, (a, g) in VARIANTS.items():
        runs[v] = (a + g, load(os.path.join(OUT, v)))
    lines = ["# BiTFIN on DAGHAR leave-one-dataset-out: which statistical features", "",
             "Written by `daghar_benchmark/fin_daghar_features.py`. Circular FIN w32 k31 e64, head 64; target accuracy, "
             "%; mean over seeds where several.", "",
             "| features | n | " + " | ".join(fd.DOMAINS) + " | mean | seeds |", "|---|---:|" + "---:|" * (len(fd.DOMAINS) + 2)]
    for name, (n, rows) in runs.items():
        if not rows:
            continue
        per = [[r["target_acc"] for (t, _), r in rows.items() if t == d] for d in fd.DOMAINS]
        cells = [f"{100 * np.mean(p):.1f}" if p else "-" for p in per]
        done = all(per)
        mean = f"{100 * np.mean([np.mean(p) for p in per]):.1f}" if done else "-"
        seeds = sorted({sd for _, sd in rows})
        lines.append(f"| {name} | {n} | " + " | ".join(cells) + f" | **{mean}** | {','.join(map(str, seeds))} |")
    with open(os.path.join(OUT, "features.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("variant", nargs="?", choices=list(VARIANTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--folds", nargs="+", choices=fd.DOMAINS, default=fd.DOMAINS)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not args.report:
        setup(args.variant)
        out = os.path.join(OUT, args.variant)
        os.makedirs(out, exist_ok=True)
        for seed in args.seeds:
            for target in args.folds:
                print(f"fold: target {target}, seed {seed}", flush=True)
                r = s.fold(target, seed)
                r.update(imitation_mse=fs._last_mse["mse"], spectral_err=fs._last_mse["spectral_err"],
                         variant=args.variant)
                with open(os.path.join(out, f"{target}_seed{seed}.json"), "w", encoding="utf-8") as fh:
                    json.dump(r, fh, indent=1)
                print(f"seed {seed}, target {target}: {100 * r['target_acc']:.1f} %", flush=True)
    report()
