"""Check of the TinierHAR re-implementation (tinierhar.py) on WISDM v1.1, in the protocol of the TinierHAR paper
(its code is a fork of the TinyHAR code, so the WISDM data and user groups are those of
baseline/tinyhar/tinyhar_reproduce_wisdm.py):
  * 20 Hz, 3 accelerometer channels, 4 s windows (80 samples), training windows with 50 % overlap, test windows slid by
    one sample; cross-validation over the 10 user groups of the TinyHAR / TinierHAR data loader; macro F1 per fold
  * AdamW, lr 1e-3, divided by 10 after 7 epochs without validation improvement, at most 150 epochs, early stopping
    with patience 15 (paper, Sec. 4); M = 4, N = 16
Not stated in the paper, chosen as in the TinyHAR check: batch 256, validation = a random 10 % of the training windows,
per-channel standardisation with the training statistics, best-validation-loss checkpoint.

    python baseline/tinierhar/tinierhar_reproduce_wisdm.py --seed 1 --groups 0 1 2   # results_reproduce/group<g>_seed<s>.json
    python baseline/tinierhar/tinierhar_reproduce_wisdm.py --report
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(ROOT, "baseline", "tinyhar")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

import tinyhar_reproduce_wisdm as trw  # noqa: E402
from tinierhar import DEV, TinierHAR  # noqa: E402

OUT = os.path.join(HERE, "results_reproduce")
trw.WIN = 80  # 4 s at 20 Hz, read by trw.segments and trw.windows


def fold(segs, test_users, seed):
    xtr, ytr = trw.windows([s for s in segs if s[0] not in test_users], trw.WIN // 2)
    xte, yte = trw.windows([s for s in segs if s[0] in test_users], 1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ytr))
    va, tr = perm[:len(ytr) // 10], perm[len(ytr) // 10:]
    mu, sd = xtr[tr].mean(axis=(0, 2), keepdims=True), xtr[tr].std(axis=(0, 2), keepdims=True) + 1e-8
    t = lambda a: torch.as_tensor(((a - mu) / sd).astype(np.float32))
    x_tr, y_tr, x_va, y_va = t(xtr[tr]), torch.as_tensor(ytr[tr]), t(xtr[va]), torch.as_tensor(ytr[va])

    torch.manual_seed(seed)
    model = TinierHAR(3, trw.WIN, len(trw.ACTIVITIES)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=7)
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
    lines = ["# TinierHAR re-implementation, WISDM cross-validation over the 10 user groups", "",
             f"Written by `baseline/tinierhar/tinierhar_reproduce_wisdm.py`. Folds done: {len(rows)} / 10, seeds "
             f"{sorted({r['seed'] for r in rows})}. Parameters: 4,981 (the authors' Run_logs/macs_params.txt lists "
             "4,981 for TinierHAR on WISDM).", "",
             f"Mean macro F1 over folds: **{100 * f1.mean():.2f} %**.", "",
             "| test users | test windows | epochs | macro F1 | accuracy |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {r['users']} | {r['test_windows']} | {r['epochs']} | {r['macro_f1']:.4f} | {r['accuracy']:.4f} |"
              for r in sorted(rows, key=lambda r: r["users"][0])]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "tinierhar_reproduce_wisdm.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--groups", type=int, nargs="+", default=list(range(len(trw.GROUPS))))
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not args.report:
        segs = trw.segments()
        for gi in args.groups:
            r = fold(segs, trw.GROUPS[gi], args.seed)
            with open(os.path.join(OUT, f"group{gi}_seed{args.seed}.json"), "w") as fh:
                json.dump(r, fh)
            print(f"group {gi} {trw.GROUPS[gi]}: macro F1 {r['macro_f1']:.4f}, {r['epochs']} epochs", flush=True)
    report()
