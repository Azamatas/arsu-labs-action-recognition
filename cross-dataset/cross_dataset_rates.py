"""Sampling-rate robustness, cross-rate: models trained at 20 Hz in the paper's cross-dataset protocol (cross_dataset.py,
10 s windows) and tested on the same test windows recorded at a LOWER rate, as on a device that samples slower.

Each test window (3 x 200 at 20 Hz) is decimated to r Hz (r = 10, 5, 2, 1; polyphase resampling with its anti-alias
low-pass, scipy.signal.resample_poly), which removes everything above r / 2 Hz, and interpolated back to 20 Hz the same
way so that every model gets its usual 3 x 200 input. Training is unchanged (20 Hz). For a circular FIN this is the
same as running its spectral form natively at r Hz on the first 10 r / 2 + 1 frequency columns of L (the columns
above the new Nyquist frequency see zero power), at a cost that shrinks with r.

    python cross-dataset/cross_dataset_rates.py fin w2m        # pretrained FIN + 40 and CNN + 40 (cross_dataset.py)
    python cross-dataset/cross_dataset_rates.py circular w2m   # pretrained FIN + 40 with circular convolutions
    python cross-dataset/cross_dataset_rates.py tinyhar w2m    # TinyHAR (pad 0), TinierHAR: tinierhar
    python cross-dataset/cross_dataset_rates.py --report       # results_rates/rates.md
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "baseline", "tinyhar"), os.path.join(ROOT, "baseline", "tinierhar")]

import numpy as np  # noqa: E402

import cross_dataset as cd  # noqa: E402  (sets the keras backend)

OUT = os.path.join(HERE, "results_rates")
RATES = (10, 5, 2, 1)
SEED = 42
LABELS = {"fin_frozen_dropout": "FIN + 40, pretrained", "ignatov_cnn": "CNN + 40", "circular": "FIN + 40, circular",
          "tinyhar": "TinyHAR", "tinierhar": "TinierHAR"}


def degrade(x, rate):
    from scipy.signal import resample_poly

    low = resample_poly(x, rate, 20, axis=-1)
    return resample_poly(low, 20, rate, axis=-1)[..., :x.shape[-1]].astype(np.float32)


_load = cd.load


def load_with_rates(direction):
    x, y, tests = _load(direction)
    for name, (xt, yt) in list(tests.items()):
        for r in RATES:
            tests[f"{name} @ {r} Hz"] = (degrade(xt, r), yt)
    return x, y, tests


def scores(preds, tests):
    from sklearn.metrics import f1_score

    return {t: [float(np.mean(p == tests[t][1])), float(f1_score(tests[t][1], p, average="macro"))]
            for t, p in preds.items()}


def save(model, direction, res):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{direction}_{model}_seed{SEED}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


def run_keras(direction, circular):
    cd.load = load_with_rates
    names = ["fin_frozen_dropout"] if circular else ["fin_frozen_dropout", "ignatov_cnn"]
    if circular:
        import bilin_fin_temporal_plus_original as bfin
        from bilin_fin_circular import circular as wrap

        bfin.RhythmFIN = wrap(bfin.RhythmFIN)
    cd.MODELS = {m: cd.MODELS[m] for m in names}
    cd.OUT = os.path.join(OUT, "circular" if circular else "fin_cnn")
    path = cd.train(direction, [SEED], names)
    data = np.load(path)
    tests = {t: (None, data[f"truth|{t}"]) for t in data["tests"].tolist()}
    for m in names:
        res = scores({t: data[f"pred|{m}|{t}"][0] for t in tests}, tests)
        save("circular" if circular else m, direction, res)


def run_torch(direction, model):
    import tinyhar as th

    cd.load = load_with_rates
    if model == "tinyhar":
        base = th.TinyHAR
        th.TinyHAR = lambda c, t, k: base(c, t, k, pad=0)
    else:
        from tinierhar import TinierHAR

        th.TinyHAR = lambda c, t, k: TinierHAR(c, t, k)
    preds, tests, _ = th.run(direction, SEED)
    save(model, direction, scores(preds, tests))


def report():
    rows = {}
    for f in sorted(glob.glob(os.path.join(OUT, "*_seed*.json"))):
        direction, model = os.path.basename(f).split("_seed")[0].split("_", 1)
        rows[direction, model] = json.load(open(f, encoding="utf-8"))
    lines = ["# Trained at 20 Hz, tested at lower sampling rates (paper protocol, seed 42)", "",
             "Written by `cross-dataset/cross_dataset_rates.py`. Test windows decimated to r Hz with an anti-alias "
             "filter and interpolated back to 20 Hz. Accuracy / macro F1.", ""]
    for direction in ("w2m", "m2w"):
        models = [m for m in LABELS if (direction, m) in rows]
        if not models:
            continue
        bases = [t for t in rows[direction, models[0]] if " @ " not in t]
        for base in bases:
            lines += [f"## {cd.DIRECTIONS[direction]}: {base}", "",
                      "| model | 20 Hz | " + " | ".join(f"{r} Hz" for r in RATES) + " |",
                      "|---|" + "---:|" * (len(RATES) + 1)]
            for m in models:
                res = rows[direction, m]
                cells = [res[base]] + [res[f"{base} @ {r} Hz"] for r in RATES]
                lines.append(f"| {LABELS[m]} | " + " | ".join(f"{a:.3f} / {f:.3f}" for a, f in cells) + " |")
            lines.append("")
    with open(os.path.join(OUT, "rates.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", choices=["fin", "circular", "tinyhar", "tinierhar"])
    parser.add_argument("direction", nargs="?", choices=list(cd.DIRECTIONS))
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        report()
    elif args.model in ("fin", "circular"):
        run_keras(args.direction, args.model == "circular")
    else:
        run_torch(args.direction, args.model)
