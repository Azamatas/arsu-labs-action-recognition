"""TinyHAR (baseline/tinyhar/tinyhar.py) in the data-scarcity protocol of data-scarcity/data_scarcity.ipynb (Table 2):
for each seed 0-4, 7 random WISDM users train, 2 validate (unused here, as the paper reports the last epoch) and the
other 27 test; user 30 excluded from the draw exactly as in the notebook; 150 epochs, batch 256, no flip; accuracy and
macro F1 after the last epoch, and the test accuracy after every epoch.

    python data-scarcity/tinyhar_scarcity.py       # writes data-scarcity/tinyhar_scarcity.md and .npz
"""
import os
import sys

os.environ["KERAS_BACKEND"] = "torch"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "baseline", "tinyhar"), ROOT, os.path.join(ROOT, "data-processing")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

import config as cfg  # noqa: E402
import wisdm_windowing as ww  # noqa: E402
from tinyhar import DEV, TinyHAR  # noqa: E402

OUT = HERE

SEEDS, N_TRAIN, N_VAL, EPOCHS = (0, 1, 2, 3, 4), 7, 2, 150


def predict(model, x):
    with torch.no_grad():
        return torch.cat([model(x[i:i + 2048].to(DEV)).argmax(1).cpu() for i in range(0, len(x), 2048)]).numpy()


def main():
    windows, users, labels = ww.window_raw()
    candidates = np.setdiff1d(np.unique(users), [30])
    rows, curves = [], []
    for seed in SEEDS:
        picked = np.random.default_rng(seed).choice(candidates, N_TRAIN + N_VAL, replace=False)
        tr = np.isin(users, picked[:N_TRAIN])
        te = ~(tr | np.isin(users, picked[N_TRAIN:]))
        xtr = windows[tr]
        mu, sd = xtr.mean(axis=(0, 2), keepdims=True), xtr.std(axis=(0, 2), keepdims=True) + 1e-8
        norm = lambda x: torch.as_tensor(((x - mu) / sd).astype(np.float32))
        x, y = norm(xtr), torch.as_tensor(labels[tr], dtype=torch.long)
        xt, yt = norm(windows[te]), labels[te]

        torch.manual_seed(seed)
        model = TinyHAR(len(cfg.AXES), cfg.SEGMENT, cfg.N_CLASSES, pad=0).to(DEV)  # unpadded, as the authors
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        rng = np.random.default_rng(seed)
        curve = []
        for _ in range(EPOCHS):
            model.train()
            for idx in np.array_split(rng.permutation(len(y)), int(np.ceil(len(y) / cfg.BATCH_SIZE))):
                opt.zero_grad()
                loss_fn(model(x[idx].to(DEV)), y[idx].to(DEV)).backward()
                opt.step()
            model.eval()
            curve.append(float(np.mean(predict(model, xt) == yt)))
        p = predict(model, xt)
        rows.append((np.mean(p == yt), f1_score(yt, p, average="macro")))
        curves.append(curve)
        print(f"seed {seed}: train {sorted(picked[:N_TRAIN].tolist())}, test {len(yt)} windows, "
              f"accuracy {rows[-1][0]:.4f}, macro F1 {rows[-1][1]:.4f}", flush=True)

    a, f = np.array(rows).T
    lines = ["# TinyHAR, data scarcity: 7 labelled WISDM users, 27 unseen test users", "",
             "Written by `data-scarcity/tinyhar_scarcity.py`; same users per seed as `data-scarcity/data_scarcity.ipynb`, "
             "150 epochs, batch 256, scored after the last epoch.", "",
             "| seed | accuracy | macro F1 |", "|---|---:|---:|"]
    lines += [f"| {s} | {r[0]:.4f} | {r[1]:.4f} |" for s, r in zip(SEEDS, rows)]
    lines += [f"| mean ± s.d. | {a.mean():.4f} ± {a.std(ddof=1):.4f} | {f.mean():.4f} ± {f.std(ddof=1):.4f} |"]
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "tinyhar_scarcity.npz"), seeds=np.array(SEEDS), tinyhar=np.array(curves))
    with open(os.path.join(OUT, "tinyhar_scarcity.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
