"""TinyHAR (baseline/tinyhar/tinyhar.py, F = 20, unpadded convolutions as in the authors' code) in the DAGHAR
leave-one-dataset-out protocol, exactly as fin_daghar_acc_only.py runs FIN:
  * input: accelerometer + gyroscope, 6 x 60 (20 Hz, 3 s) from datasets/DAGHAR/standardized_view
  * train on the train + validation splits of the five sources, batches of 60 (12 per source), 5000 steps,
    Adam lr 1e-3 and 1e-4; every 100 steps accuracy on the sources' test splits, best checkpoint kept;
    the lr with the better source accuracy is kept; evaluation on all splits of the target
  * input standardised per channel with the training statistics (as in the TinyHAR pipeline)

    python daghar_benchmark/tinyhar_daghar.py --seeds 0 --folds UCI WISDM   # results_tinyhar/<fold>_seed<s>.json, .md
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "baseline", "tinyhar")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from tinyhar import DEV, TinyHAR  # noqa: E402

DATA = os.path.join(ROOT, "datasets", "DAGHAR", "standardized_view")
OUT = os.path.join(HERE, "results_tinyhar")
DOMAINS = ["KuHar", "MotionSense", "RealWorld_thigh", "RealWorld_waist", "UCI", "WISDM"]
SPLITS = ("train", "validation", "test")
WINDOW, N_CLASSES = 60, 6
STEPS, PER_DOMAIN, LRS, EVAL_EVERY = 5000, 12, (1e-3, 1e-4), 100


def read(domain, split):
    df = pd.read_csv(os.path.join(DATA, domain, f"{split}.csv"))
    x = np.stack([df[[f"{s}-{a}-{i}" for i in range(WINDOW)]].to_numpy(np.float32)
                  for s in ("accel", "gyro") for a in "xyz"], axis=1)  # (n, 6, 60)
    return x, df["standard activity code"].to_numpy(int)


def predict(model, x):
    model.eval()
    with torch.no_grad():
        p = torch.cat([model(x[i:i + 4096].to(DEV)).argmax(1).cpu() for i in range(0, len(x), 4096)]).numpy()
    model.train()
    return p


def fold(target, seed):
    sources = [d for d in DOMAINS if d != target]
    train = {}
    for d in sources:
        parts = [read(d, s) for s in ("train", "validation")]
        train[d] = (np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts]))
    xall = np.concatenate([x for x, _ in train.values()])
    mu, sd = xall.mean(axis=(0, 2), keepdims=True), xall.std(axis=(0, 2), keepdims=True) + 1e-8
    t = lambda a: torch.as_tensor(((a - mu) / sd).astype(np.float32))
    train = {d: (t(x), torch.as_tensor(y)) for d, (x, y) in train.items()}
    val = [read(d, "test") for d in sources]
    xv, yv = t(np.concatenate([a for a, _ in val])), np.concatenate([b for _, b in val])
    tgt = [read(target, s) for s in SPLITS]
    xt, yt = t(np.concatenate([a for a, _ in tgt])), np.concatenate([b for _, b in tgt])

    best = None
    for lr in LRS:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = TinyHAR(6, WINDOW, N_CLASSES, pad=0).to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        model.train()
        run_best = (-1.0, None, 0)
        for step in range(1, STEPS + 1):
            idx = {d: torch.as_tensor(rng.integers(0, len(train[d][1]), PER_DOMAIN)) for d in sources}
            xb = torch.cat([train[d][0][i] for d, i in idx.items()]).to(DEV)
            yb = torch.cat([train[d][1][i] for d, i in idx.items()]).to(DEV)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
            if step % EVAL_EVERY == 0:
                acc = float(np.mean(predict(model, xv) == yv))
                if acc > run_best[0]:
                    run_best = (acc, {k: w.detach().clone() for k, w in model.state_dict().items()}, step)
        model.load_state_dict(run_best[1])
        pred = predict(model, xt)
        result = {"lr": lr, "val_acc": run_best[0], "best_step": run_best[2], "target_acc": float(np.mean(pred == yt)),
                  "recall": {int(c): float(np.mean(pred[yt == c] == c)) for c in np.unique(yt)}}
        print(f"  {target} lr {lr}: source-val {run_best[0]:.4f} at step {run_best[2]}, "
              f"target {result['target_acc']:.4f}", flush=True)
        if best is None or result["val_acc"] > best["val_acc"]:
            best = result
    best.update(target=target, seed=seed, target_windows=int(len(yt)),
                params=sum(p.numel() for p in model.parameters()))
    return best


def report():
    rows = [json.load(open(os.path.join(OUT, n), encoding="utf-8")) for n in sorted(os.listdir(OUT)) if n.endswith(".json")]
    lines = ["# TinyHAR, DAGHAR leave-one-dataset-out (accelerometer + gyroscope, 6 x 60)", "",
             "Written by `daghar_benchmark/tinyhar_daghar.py`; same protocol as `daghar_benchmark/fin_daghar_acc_only.py`.", "",
             "| target | seed | windows | accuracy | lr | best step | recall sit / stand / walk / up / down / run |",
             "|---|---|---:|---:|---:|---:|---|"]
    for r in sorted(rows, key=lambda r: (DOMAINS.index(r["target"]), r["seed"])):
        rec = " / ".join(f"{r['recall'][str(c)]:.2f}" if str(c) in r["recall"] else "-" for c in range(N_CLASSES))
        lines.append(f"| {r['target']} | {r['seed']} | {r['target_windows']} | {100 * r['target_acc']:.1f} % | "
                     f"{r['lr']:g} | {r['best_step']} | {rec} |")
    if rows:
        lines += ["", f"Mean over the targets above: {100 * np.mean([r['target_acc'] for r in rows]):.1f} %",
                  f"Parameters: {rows[0]['params']:,}"]
    with open(os.path.join(OUT, "tinyhar_daghar.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--folds", nargs="+", choices=DOMAINS, default=DOMAINS)
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for seed in args.seeds:
        for target in args.folds:
            r = fold(target, seed)
            with open(os.path.join(OUT, f"{target}_seed{seed}.json"), "w", encoding="utf-8") as fh:
                json.dump(r, fh, indent=1)
            print(f"seed {seed}, target {target}: {100 * r['target_acc']:.1f} %", flush=True)
            report()
