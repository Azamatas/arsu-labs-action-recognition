"""Cross-dataset test: the bilinear FIN classifier trained on WISDM users 1-26, tested on all 24 MotionSense users.

Same training as main() of bilin_fin_temporal_plus_original.py; the 40 Ignatov features are standardised with
WISDM's training statistics. The WISDM test users 27-36 are scored too, as the in-dataset reference.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIN = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, FIN, os.path.join(FIN, "FIN-classifiers"), os.path.join(FIN, "data-processing")]

import argparse

import keras
import numpy as np
from sklearn.metrics import f1_score

import config as cfg
import motionsense_windowing as ms
import wisdm_windowing as ww
from bilin_fin_temporal_plus_original import RhythmEnsemble, RhythmFIN, ignatov_features, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    seed = parser.parse_args().seed

    data = ww.load_splits()
    xtr = np.concatenate([data["train"][0], data["val"][0]])  # WISDM users 1-26
    ytr = np.concatenate([data["train"][1], data["val"][1]])
    tests = {"WISDM users 27-36": data["test"], "MotionSense, all 24 users": ms.window_raw()[::2]}

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

    for name, (x, yte) in tests.items():
        pred = model.predict([x, standardise(ignatov_features(x))], batch_size=1024, verbose=0).argmax(axis=1)
        print(f"seed {seed}, {name}: accuracy {np.mean(pred == yte):.4f}, "
              f"macro F1 {f1_score(yte, pred, average='macro'):.4f}")
        for i, activity in enumerate(cfg.ACTIVITIES):
            print(f"  {activity:10s} recall {np.mean(pred[yte == i] == i):.3f}")


if __name__ == "__main__":
    main()
