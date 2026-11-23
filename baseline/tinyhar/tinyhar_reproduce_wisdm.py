"""Check of the TinyHAR re-implementation against the paper (Zhou et al., ISWC 2022, Fig. 2c): WISDM, macro F1
83.47 % +- 0.89 % (5 runs), 16,184 parameters.

Their protocol on WISDM v1.1 (datasets/WISDM_ar_v1.1), as stated in the paper:
  * 20 Hz, 3 accelerometer channels, 5 s windows (100 samples); training windows with 50 % overlap, test windows
    slid by one sample; leave-one-subject-out over the 36 users; macro F1 per test user, averaged over the users
  * Adam, lr 1e-4, divided by 10 after 5 epochs without validation improvement, at most 150 epochs with early
    stopping, batch 256; F = 14 (16,025 parameters here against their 16,184)
Not stated in the paper, chosen here: validation = a random 10 % of the training windows, early-stopping patience 15
(the value in their fine-tuning code), per-channel standardisation with the training statistics, best-validation
checkpoint (as their experiment.py does).

    python baseline/tinyhar/tinyhar_reproduce_wisdm.py --seed 1 --groups 0 1 2    # results_reproduce_groups/group<g>_seed<s>.json
    python baseline/tinyhar/tinyhar_reproduce_wisdm.py --report                   # mean over the folds done so far
"""
import json
import os
import sys

os.environ["KERAS_BACKEND"] = "torch"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # baseline/tinyhar -> repository root
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "data-processing")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

import wisdm_windowing as ww  # noqa: E402
from config import ACTIVITIES  # noqa: E402
from tinyhar import DEV, TinyHAR  # noqa: E402

OUT = os.path.join(HERE, "results_reproduce_groups")
GROUPS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24],
          [25, 26, 27], [28, 29, 30], [31, 32, 33], [34, 35, 36]]  # LOCV_keys of dataloader_WISDM_har.py
WIN, F_PAPER = 100, 14


def segments():
    """Contiguous stretches of one user and one activity: list of (user, label, xyz (time, 3))."""
    df = pd.read_csv(ww.read_raw(), header=None, names=["user", "activity", "time", "x", "y", "z"],
                     usecols=[0, 1, 2, 3, 4, 5], engine="python", on_bad_lines="skip")
    df = df[df["activity"].isin(ACTIVITIES)]
    df[["x", "y", "z"]] = df[["x", "y", "z"]].fillna(0.0)
    key = ((df["activity"] != df["activity"].shift()) | (df["user"] != df["user"].shift())).cumsum()
    return [(int(g["user"].iloc[0]), ACTIVITIES.index(g["activity"].iloc[0]), g[["x", "y", "z"]].to_numpy(np.float32))
            for _, g in df.groupby(key) if len(g) >= WIN]


def windows(segs, step):
    x, y = [], []
    for _, lab, xyz in segs:
        for s in range(0, len(xyz) - WIN + 1, step):
            x.append(xyz[s:s + WIN].T)
            y.append(lab)
    return np.stack(x), np.array(y)


def fold(segs, test_users, seed):
    xtr, ytr = windows([s for s in segs if s[0] not in test_users], WIN // 2)
    xte, yte = windows([s for s in segs if s[0] in test_users], 1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ytr))
    n_val = len(ytr) // 10
    va, tr = perm[:n_val], perm[n_val:]
    mu, sd = xtr[tr].mean(axis=(0, 2), keepdims=True), xtr[tr].std(axis=(0, 2), keepdims=True) + 1e-8
    t = lambda a: torch.as_tensor(((a - mu) / sd).astype(np.float32))
    x_tr, y_tr = t(xtr[tr]), torch.as_tensor(ytr[tr])
    x_va, y_va = t(xtr[va]), torch.as_tensor(ytr[va])

    torch.manual_seed(seed)
    model = TinyHAR(3, WIN, len(ACTIVITIES), f=F_PAPER).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=5)
    loss_fn = nn.CrossEntropyLoss()
    best, best_state, wait = np.inf, None, 0
    for epoch in range(150):
        model.train()
        for idx in np.array_split(rng.permutation(len(y_tr)), int(np.ceil(len(y_tr) / 256))):
            opt.zero_grad()
            loss_fn(model(x_tr[idx].to(DEV)), y_tr[idx].to(DEV)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(sum(loss_fn(model(x_va[i:i + 2048].to(DEV)), y_va[i:i + 2048].to(DEV)) * len(y_va[i:i + 2048])
                          for i in range(0, len(y_va), 2048)) / len(y_va))
        sched.step(v)
        if v < best:
            best, wait, best_state = v, 0, {k: w.detach().clone() for k, w in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 15:
                break
    model.load_state_dict(best_state)
    model.eval()
    x_te = t(xte)
    with torch.no_grad():
        pred = torch.cat([model(x_te[i:i + 4096].to(DEV)).argmax(1).cpu() for i in range(0, len(x_te), 4096)]).numpy()
    return {"users": list(test_users), "seed": seed, "epochs": epoch + 1, "test_windows": int(len(yte)),
            "macro_f1": float(f1_score(yte, pred, average="macro")), "accuracy": float(np.mean(pred == yte))}


def report():
    rows = [json.load(open(os.path.join(OUT, n))) for n in sorted(os.listdir(OUT)) if n.endswith(".json")]
    f1 = np.array([r["macro_f1"] for r in rows])
    lines = ["# TinyHAR re-implementation vs paper, WISDM cross-validation over their 10 user groups", "",
             f"Written by `baseline/tinyhar/tinyhar_reproduce_wisdm.py`. Folds done: {len(rows)} / 10, seeds "
             f"{sorted({r['seed'] for r in rows})}.", "",
             f"Mean macro F1 over folds: **{100 * f1.mean():.2f} %** (paper: 83.47 % +- 0.89 % over 5 runs).", "",
             "| test users | test windows | epochs | macro F1 | accuracy |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {r['users']} | {r['test_windows']} | {r['epochs']} | {r['macro_f1']:.4f} | {r['accuracy']:.4f} |"
              for r in sorted(rows, key=lambda r: r["users"][0])]
    with open(os.path.join(HERE, "results", "tinyhar_reproduce_wisdm.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--groups", type=int, nargs="+", default=list(range(len(GROUPS))))
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    if not args.report:
        segs = segments()
        for gi in args.groups:
            r = fold(segs, GROUPS[gi], args.seed)
            with open(os.path.join(OUT, f"group{gi}_seed{args.seed}.json"), "w") as fh:
                json.dump(r, fh)
            print(f"group {gi} {GROUPS[gi]}: macro F1 {r['macro_f1']:.4f}, {r['epochs']} epochs, "
                  f"{r['test_windows']} test windows", flush=True)
    report()
