"""FIN imitation pretrained on MotionSense, then fine-tuned with the head on WISDM and tested on unseen WISDM users.

--split scarcity (default): fine-tuned on 29 users, tested on the 7 left out, users 12, 18, 19, 20, 31, 34, 36 (the
7 training users of the data-scarcity notebooks). --split ignatov: fine-tuned on users 1-26, tested on 27-36.

The FIN imitates spectral entropy, autocorrelation peak and spectral peak on all MotionSense windows (no labels).
Then RhythmEnsemble (FIN embedding of each axis + Ignatov's 40 features) is trained on the 29 users' labels, the FIN
frozen for the first FREEZE_EPOCHS and fine-tuned with the head afterwards, as in
bilin_fin_temporal_plus_original.py. Every training set is used with the flip: each window once more rotated 180
degrees about y (x and z change sign), MotionSense included.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIN = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, FIN, os.path.join(FIN, "FIN-classifiers"), os.path.join(FIN, "data-processing")]

import argparse

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import config as cfg
import motionsense_windowing as ms
import wisdm_windowing as ww
from bilin_fin_temporal_plus_original import RhythmEnsemble, RhythmFIN, ignatov_features, targets

TEST_USERS = {"scarcity": [12, 18, 19, 20, 31, 34, 36], "ignatov": list(range(27, 37))}
FLIP = np.array([-1, 1, -1], np.float32)[:, None]  # 180 degrees about y


def run(xtr, ytr, xte, ftr, fte, motionsense, seed):
    keras.utils.set_random_seed(seed)
    fin = RhythmFIN()
    y = targets(motionsense)
    fin.compile(optimizer=keras.optimizers.Adam(cfg.FIN_LR), loss="mse")
    fin.fit(motionsense, ((y - y.mean(0)) / y.std(0)).astype(np.float32),
            epochs=cfg.FIN_EPOCHS, batch_size=cfg.BATCH_SIZE, verbose=0)

    keras.utils.set_random_seed(seed)
    model = RhythmEnsemble(fin)
    for epochs, trainable in ((cfg.FREEZE_EPOCHS, False), (cfg.HEAD_EPOCHS - cfg.FREEZE_EPOCHS, True)):
        fin.trainable = trainable
        model.compile(optimizer=keras.optimizers.Adam(cfg.HEAD_LR), loss="sparse_categorical_crossentropy")
        model.fit([xtr, ftr], ytr, epochs=epochs, batch_size=cfg.BATCH_SIZE, verbose=0)
    return model.predict([xte, fte], batch_size=1024, verbose=0).argmax(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(cfg.SEEDS))
    parser.add_argument("--split", choices=list(TEST_USERS), default="scarcity")
    args = parser.parse_args()
    seeds, test_users = args.seeds, TEST_USERS[args.split]

    windows, users, labels = ww.window_raw()
    test = np.isin(users, test_users)
    xtr, ytr, xte, yte = windows[~test], labels[~test], windows[test], labels[test]
    xtr, ytr = np.concatenate([xtr, xtr * FLIP]), np.concatenate([ytr, ytr])
    ms_windows = ms.window_raw()[0]
    motionsense = np.concatenate([ms_windows, ms_windows * FLIP]).reshape(-1, cfg.SEGMENT)
    print(f"FIN pretrained on {len(motionsense)} single-axis MotionSense windows (with the flipped copies); "
          f"fine-tuned on {len(xtr)} windows of {36 - len(test_users)} WISDM users (with the flipped copies); "
          f"tested on {len(xte)} windows of users {test_users}", flush=True)

    ftr, fte = ignatov_features(xtr), ignatov_features(xte)
    mu, sd = ftr.mean(0), ftr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    ftr, fte = ((ftr - mu) / sd).astype(np.float32), ((fte - mu) / sd).astype(np.float32)

    rows = []
    for seed in seeds:
        pred = run(xtr, ytr, xte, ftr, fte, motionsense, seed)
        rows.append({"seed": seed, "accuracy": np.mean(pred == yte), "macro F1": f1_score(yte, pred, average="macro"),
                     **{a: np.mean(pred[yte == i] == i) for i, a in enumerate(cfg.ACTIVITIES)}})
        print(f"seed {seed}: accuracy {rows[-1]['accuracy']:.4f}, macro F1 {rows[-1]['macro F1']:.4f}", flush=True)

    results = pd.DataFrame(rows).set_index("seed")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(f"\nmean over seeds {seeds}: accuracy {results['accuracy'].mean():.4f} ± {results['accuracy'].std():.4f}, "
          f"macro F1 {results['macro F1'].mean():.4f} ± {results['macro F1'].std():.4f}")
    print("recall per activity, mean over seeds")
    print(results[list(cfg.ACTIVITIES)].mean().round(3).to_string())


if __name__ == "__main__":
    main()
