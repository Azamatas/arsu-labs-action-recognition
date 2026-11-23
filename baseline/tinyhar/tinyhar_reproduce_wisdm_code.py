"""TinyHAR check on WISDM with the settings of the authors' CODE instead of the paper (tinyhar_reproduce_wisdm.py):
read from github.com/teco-kit/ISWC22-HAR (dataloaders/dataloader_base.py, models/TinyHAR.py) and the training script
of its fork github.com/zhaxidele/TinierHAR (train.py); nothing copied.
  * convolutions without padding (models/TinyHAR.py)
  * windows over each user's whole recording, across activity changes, labelled by the most frequent activity
    (drop_transition = False); training step 50 %, test step 10 % of the window
  * Adam, lr 1e-3, batch 32, at most 80 epochs, lr / 10 after 7 epochs without validation improvement, early stopping
    after 15; validation = a random 10 % of the training windows; per-channel standardisation
Kept from the paper: 5 s windows (100 samples, configs/data.yaml), F = 14, the paper's TinyHAR blocks.

    python baseline/tinyhar/tinyhar_reproduce_wisdm_code.py --seed 1 --groups 0 1 2
    python baseline/tinyhar/tinyhar_reproduce_wisdm_code.py --report
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

import tinyhar_reproduce_wisdm as trw  # noqa: E402
from tinyhar import DEV, TinyHAR  # noqa: E402

OUT = os.path.join(HERE, "results_reproduce_code")
WIN = 100


def streams():
    """Each user's recording as one stream (segments in file order), with a label per sample."""
    trw.WIN = 1  # keep every segment
    users = {}
    for user, label, xyz in trw.segments():
        users.setdefault(user, []).append((xyz, np.full(len(xyz), label)))
    return {u: (np.concatenate([a for a, _ in s]), np.concatenate([b for _, b in s])) for u, s in users.items()}


def windows(data, users, step):
    x, y = [], []
    for u in users:
        xyz, lab = data[u]
        for s in range(0, len(xyz) - WIN + 1, step):
            x.append(xyz[s:s + WIN].T)
            y.append(np.bincount(lab[s:s + WIN]).argmax())
    return np.stack(x), np.array(y)


def fold(data, test_users, seed):
    train_users = [u for u in data if u not in test_users]
    xtr, ytr = windows(data, train_users, WIN // 2)
    xte, yte = windows(data, [u for u in test_users if u in data], WIN // 10)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ytr))
    va, tr = perm[:len(ytr) // 10], perm[len(ytr) // 10:]
    mu, sd = xtr[tr].mean(axis=(0, 2), keepdims=True), xtr[tr].std(axis=(0, 2), keepdims=True) + 1e-8
    t = lambda a: torch.as_tensor(((a - mu) / sd).astype(np.float32))
    x_tr, y_tr, x_va, y_va = t(xtr[tr]), torch.as_tensor(ytr[tr]), t(xtr[va]), torch.as_tensor(ytr[va])

    torch.manual_seed(seed)
    model = TinyHAR(3, WIN, len(trw.ACTIVITIES), f=trw.F_PAPER, pad=0).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=7)
    loss_fn = nn.CrossEntropyLoss()
    best, best_state, wait = np.inf, None, 0
    for epoch in range(80):
        model.train()
        for idx in np.array_split(rng.permutation(len(y_tr)), int(np.ceil(len(y_tr) / 32))):
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
    lines = ["# TinyHAR on WISDM with the settings of the authors' code (10 user groups)", "",
             f"Written by `baseline/tinyhar/tinyhar_reproduce_wisdm_code.py`. Folds done: {len(rows)} / 10, seeds "
             f"{sorted({r['seed'] for r in rows})}.", "",
             f"Mean macro F1 over folds: **{100 * f1.mean():.2f} %** (paper: 83.47 % +- 0.89 %; paper settings in "
             "tinyhar_reproduce_wisdm.py: 80.85 %).", "",
             "| test users | test windows | epochs | macro F1 | accuracy |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {r['users']} | {r['test_windows']} | {r['epochs']} | {r['macro_f1']:.4f} | {r['accuracy']:.4f} |"
              for r in sorted(rows, key=lambda r: r["users"][0])]
    with open(os.path.join(HERE, "results", "tinyhar_reproduce_wisdm_code.md"), "w", encoding="utf-8") as fh:
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
        data = streams()
        for gi in args.groups:
            r = fold(data, trw.GROUPS[gi], args.seed)
            with open(os.path.join(OUT, f"group{gi}_seed{args.seed}.json"), "w") as fh:
                json.dump(r, fh)
            print(f"group {gi} {trw.GROUPS[gi]}: macro F1 {r['macro_f1']:.4f}, {r['epochs']} epochs", flush=True)
    report()
