import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
FIN = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path[:0] = [FIN, os.path.join(FIN, "data-processing")]

import statistical_features as ignatov
import wisdm_windowing as ww
from config import TEMPORAL, TOP3
from wisdm_windowing import temporal_axis

SEEDS = (42, 0, 1, 2, 3)
TREES = 300


def features(windows):
    w = windows.astype(np.float64)
    basic = np.stack([ignatov.extract_basic_features(x[0], x[1], x[2]) for x in w])
    axes = [temporal_axis(w[:, a, :]) for a in range(3)]
    temporal = np.stack([axes[a][f] for f in TEMPORAL for a in range(3)], axis=1)
    return np.concatenate([basic, temporal], axis=1)


def columns():
    basic = list(range(40))
    temporal = list(range(40, 40 + 3 * len(TEMPORAL)))
    top3 = [40 + 3 * TEMPORAL.index(f) + a for f in TOP3 for a in range(3)]
    return {
        "Ignatov 40": basic,
        "temporal 33": temporal,
        "40 + temporal 33": basic + temporal,
        "40 + top 3 families": basic + top3,
    }


def scores(y, pred):
    return float((pred == y).mean()), float(f1_score(y, pred, average="macro"))


def forest(seed):
    return RandomForestClassifier(n_estimators=TREES, random_state=seed, n_jobs=-1)


def summary(r):
    acc, f1 = np.array(r).T
    return f"accuracy {acc.mean():.4f} ± {acc.std(ddof=1):.4f}, macro F1 {f1.mean():.4f} ± {f1.std(ddof=1):.4f}"


def main():
    windows, users, labels = ww.window_raw()
    X = features(windows)
    train, test = users <= 26, users >= 27

    for name, cols in columns().items():
        r = [scores(labels[test], forest(s).fit(X[train][:, cols], labels[train]).predict(X[test][:, cols]))
             for s in SEEDS]
        print(f"users 1-26 / 27-36, {name}: {summary(r)}", flush=True)

    for name, cols in columns().items():
        r = []
        for s in SEEDS:
            pred = np.empty_like(labels)
            for u in np.unique(users):
                te = users == u
                pred[te] = forest(s).fit(X[~te][:, cols], labels[~te]).predict(X[te][:, cols])
            r.append(scores(labels, pred))
        print(f"LOSO, {name}: {summary(r)}", flush=True)


if __name__ == "__main__":
    main()
