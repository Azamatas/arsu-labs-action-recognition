"""The bilinear FIN classifier of bilin_fin_temporal_plus_original.py, trained and tested on MotionSense.

Same pipeline as its main(): the FIN imitates the rhythm features on the training users' windows, then
RhythmEnsemble (FIN embedding of each axis + Ignatov's 40 features) is trained, the FIN frozen for the first
FREEZE_EPOCHS. Split by user like WISDM's 26/10: users 1-17 train, 18-24 test.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIN = os.path.dirname(HERE)
sys.path[:0] = [HERE, FIN, os.path.join(FIN, "data-processing")]

import argparse

import keras
import numpy as np
from sklearn.metrics import f1_score

import config as cfg
import motionsense_windowing as ms
from bilin_fin_temporal_plus_original import RhythmEnsemble, RhythmFIN, ignatov_features, targets

TRAIN_USERS = range(1, 18)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    seed = parser.parse_args().seed

    windows, users, labels = ms.window_raw()
    train = np.isin(users, list(TRAIN_USERS))
    xtr, ytr, xte, yte = windows[train], labels[train], windows[~train], labels[~train]
    print(f"train {len(xtr)} windows of users 1-17, test {len(xte)} windows of users 18-24")

    ftr, fte = ignatov_features(xtr), ignatov_features(xte)
    mu, sd = ftr.mean(0), ftr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    ftr, fte = ((ftr - mu) / sd).astype(np.float32), ((fte - mu) / sd).astype(np.float32)

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
        model.fit([xtr, ftr], ytr, epochs=epochs, batch_size=cfg.BATCH_SIZE, verbose=0)

    pred = model.predict([xte, fte], batch_size=1024, verbose=0).argmax(axis=1)
    print(f"seed {seed}: accuracy {np.mean(pred == yte):.4f}, macro F1 {f1_score(yte, pred, average='macro'):.4f}")
    for i, activity in enumerate(cfg.ACTIVITIES):
        print(f"  {activity:10s} recall {np.mean(pred[yte == i] == i):.3f}")


if __name__ == "__main__":
    main()
