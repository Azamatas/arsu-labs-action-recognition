"""Which of the 40 statistical features does the pretrained BiFIN need? Paper cross-dataset protocol (cross_dataset.py,
10 s windows, 60 epochs, frozen pretrained FIN + head with dropout), with the 40 features replaced by a subset:

  A  all 40
  B  10: mean, std, mean absolute deviation per axis, mean resultant acceleration (no histograms)
  C  7:  mean and std per axis, mean resultant acceleration
  D  4:  mean per axis (orientation), mean resultant acceleration
  E  0:  no features (one constant column, so that the head keeps its two inputs)

Column order of statistical_features.extract_basic_features: 0-2 mean, 3-5 std, 6-8 MAD, 9 resultant, 10-39 histograms.

    python cross-dataset/cross_dataset_features.py w2m A B C --seed 0
    python cross-dataset/cross_dataset_features.py --report
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]

import numpy as np  # noqa: E402

import cross_dataset as cd  # noqa: E402  (sets the keras backend)

OUT = os.path.join(HERE, "results_features")
SUBSETS = {"A": list(range(40)), "B": list(range(10)), "C": [0, 1, 2, 3, 4, 5, 9], "D": [0, 1, 2, 9], "E": []}
LABELS = {"A": "all 40", "B": "10, no histograms", "C": "7: mean, std, resultant", "D": "4: mean, resultant",
          "E": "none"}


def run(direction, variant, seed):
    from sklearn.metrics import f1_score

    import bilin_fin_temporal_plus_original as bfin

    full = bfin.ignatov_features.__wrapped__ if hasattr(bfin.ignatov_features, "__wrapped__") else bfin.ignatov_features
    cols = SUBSETS[variant]

    def subset(windows):
        f = full(windows)
        return f[:, cols] if cols else np.zeros((len(f), 1))

    subset.__wrapped__ = full
    bfin.ignatov_features = subset
    cd.MODELS = {"fin_frozen_dropout": cd.MODELS["fin_frozen_dropout"]}
    cd.OUT = os.path.join(OUT, "npz", variant)
    data = np.load(cd.train(direction, [seed], ["fin_frozen_dropout"]))
    res = {"direction": direction, "variant": variant, "n_features": len(cols), "seed": seed}
    for t in data["tests"].tolist():
        y, p = data[f"truth|{t}"], data[f"pred|fin_frozen_dropout|{t}"][0]
        res[t] = [float(np.mean(p == y)), float(f1_score(y, p, average="macro"))]
    with open(os.path.join(OUT, f"{direction}_{variant}_seed{seed}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(res, flush=True)


def report():
    rows = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(OUT, "*_seed*.json")))]
    lines = ["# Pretrained BiFIN + a subset of the 40 statistical features (paper cross-dataset protocol)", "",
             "Written by `cross-dataset/cross_dataset_features.py`. Accuracy / macro F1, one seed.", ""]
    for direction in ("w2m", "m2w"):
        rs = sorted([r for r in rows if r["direction"] == direction], key=lambda r: r["variant"])
        if not rs:
            continue
        tests = [k for k in rs[0] if k not in ("direction", "variant", "n_features", "seed")]
        lines += [f"## {cd.DIRECTIONS[direction]}", "", "| features | seed | " + " | ".join(tests) + " |",
                  "|---|---:|" + "---:|" * len(tests)]
        lines += [f"| {r['variant']}: {LABELS[r['variant']]} | {r['seed']} | "
                  + " | ".join(f"{r[t][0]:.3f} / {r[t][1]:.3f}" for t in tests) + " |" for r in rs]
        lines.append("")
    with open(os.path.join(OUT, "features.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("direction", nargs="?", choices=list(cd.DIRECTIONS))
    parser.add_argument("variants", nargs="*", choices=list(SUBSETS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not args.report:
        for v in args.variants:
            run(args.direction, v, args.seed)
    report()
