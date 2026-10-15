"""Reverse cross-dataset test: the full bilinear FIN classifier trained on all 24 MotionSense users, tested on
WISDM users 27-36 (no WISDM window is used for training).

Same model and training as FIN-classifiers/bilin_fin_temporal_plus_original.py, with and without the "phone turned over in the
pocket" augmentation (every training window also rotated 180 degrees about y).
"""
import os
import sys

os.environ["KERAS_BACKEND"] = "torch"

HERE = os.path.dirname(os.path.abspath(__file__))
FIN = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, FIN, os.path.join(FIN, "FIN-classifiers"), os.path.join(FIN, "data-processing")]

import argparse

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

import config as cfg
import motionsense_windowing as ms
import wisdm_windowing as ww
from bilin_fin_temporal_plus_original import RhythmEnsemble, RhythmFIN, ignatov_features, targets

FLIP = np.array([-1, 1, -1], np.float32)[:, None]  # 180 degrees about y


def train_predict(xtr, ytr, tests, seed):
    """Imitation pretraining of the FIN on xtr, then RhythmEnsemble (FIN embeddings + Ignatov's 40 features),
    the FIN frozen for the first FREEZE_EPOCHS. Returns the predicted labels of every test set."""
    ftr = ignatov_features(xtr)
    mu, sd = ftr.mean(0), ftr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    standardise = lambda f: ((f - mu) / sd).astype(np.float32)

    keras.utils.set_random_seed(seed)
    fin = RhythmFIN()
    axes = xtr.reshape(-1, cfg.SEGMENT)
    y = targets(axes)
    fin.compile(optimizer=keras.optimizers.Adam(cfg.FIN_LR), loss="mse")
    fin.fit(axes, ((y - y.mean(0)) / y.std(0)).astype(np.float32),
            epochs=cfg.FIN_EPOCHS, batch_size=cfg.BATCH_SIZE, verbose=0)

    keras.utils.set_random_seed(seed)
    model = RhythmEnsemble(fin)
    for epochs, trainable in ((cfg.FREEZE_EPOCHS, False), (cfg.HEAD_EPOCHS - cfg.FREEZE_EPOCHS, True)):
        fin.trainable = trainable
        model.compile(optimizer=keras.optimizers.Adam(cfg.HEAD_LR), loss="sparse_categorical_crossentropy")
        model.fit([xtr, standardise(ftr)], ytr, epochs=epochs, batch_size=cfg.BATCH_SIZE, verbose=0)
    return {name: model.predict([x, standardise(ignatov_features(x))], batch_size=1024, verbose=0).argmax(axis=1)
            for name, (x, _) in tests.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    seed = parser.parse_args().seed

    xtr, _, ytr = ms.window_raw()
    tests = {"WISDM 27-36": ww.load_splits()["test"]}
    truth = tests["WISDM 27-36"][1]
    variants = {"as is": (xtr, ytr),
                "+ turned over": (np.concatenate([xtr, xtr * FLIP]), np.concatenate([ytr, ytr]))}

    rows = []
    for variant, (x, y) in variants.items():
        pred = train_predict(x, y, tests, seed)["WISDM 27-36"]
        rows.append({"training": variant, "accuracy": np.mean(pred == truth),
                     "macro F1": f1_score(truth, pred, average="macro"),
                     **{a: np.mean(pred[truth == i] == i) for i, a in enumerate(cfg.ACTIVITIES)}})
        print(f"\n{variant}: confusion on WISDM 27-36 (rows: true, columns: predicted)")
        print(pd.DataFrame(confusion_matrix(truth, pred, labels=range(cfg.N_CLASSES)),
                           index=cfg.ACTIVITIES, columns=cfg.ACTIVITIES), flush=True)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(f"\nseed {seed}, trained on all MotionSense ({len(xtr)} windows), tested on WISDM 27-36 ({len(truth)} windows)")
    print(pd.DataFrame(rows).set_index("training").round(3))


if __name__ == "__main__":
    main()
