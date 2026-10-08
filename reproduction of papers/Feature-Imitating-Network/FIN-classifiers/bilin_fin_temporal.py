import os
import sys

os.environ["KERAS_BACKEND"] = "torch"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # silences tf logs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "ClassicML", "data_processing"))

import argparse

import keras
import numpy as np
from keras import layers, ops
from sklearn.metrics import f1_score

import config as cfg
import fin_ensemble as fe
import generate_wisdm_data as ignatov # retrieve 40 statistical features extractor
import wisdm_windowing as ww # temporal feature calc func for one axis

TARGETS = ("spec_entropy", "ac_peak", "spec_peak")
WIDTH = 64
KERNEL = 61

class RhythmFIN(keras.Model):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.a = layers.Conv1D(WIDTH, KERNEL, padding="same", use_bias=False)
        self.b = layers.Conv1D(WIDTH, KERNEL, padding="same", use_bias=False)
        self.proj = layers.Dense(WIDTH, activation="relu")
        self.head = layers.Dense(len(TARGETS))

    def embed(self, x):
        # z-normalization of each window
        # erases amplitude and gravity, but features predicted are unaffected by them
        x = (x - ops.mean(x, axis=1, keepdims=True)) / (ops.std(x, axis=1, keepdims=True) + 1e-6)
        h = ops.expand_dims(x, axis=-1)
        # self.a(h) * self.b(h) - bilinearity
        return self.proj(ops.mean(self.a(h) * self.b(h), axis=1))

    def call(self, x):
        return self.head(self.embed(x))


class RhythmEnsemble(keras.Model):
    def __init__(self, fin, **kw):
        super().__init__(**kw)
        self.fin = fin
        self.hidden = [layers.Dense(cfg.HEAD_WIDTH, activation="relu")
                       for _ in range(cfg.HEAD_LAYERS - 1)]
        self.out = layers.Dense(cfg.N_CLASSES, activation="softmax")

    def call(self, inputs):
        x, stats = inputs
        h = ops.concatenate([self.fin.embed(x[:, i, :]) for i in range(len(cfg.AXES))] + [stats],
                            axis=1)
        for layer in self.hidden:
            h = layer(h)
        return self.out(h)


def targets(axes):
    t = ww.temporal_axis(axes.astype(np.float64))
    return np.stack([t[name] for name in TARGETS], axis=1)


def ignatov_features(windows):
    return np.stack([ignatov.extract_basic_features(*w) for w in windows.astype(np.float64)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    seed = parser.parse_args().seed

    data = fe.load_splits()
    xtr = np.concatenate([data["train"][0], data["val"][0]])  # users 1-26
    ytr = np.concatenate([data["train"][1], data["val"][1]])
    xte, yte = data["test"]                                     # users 27-36

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
        model.compile(optimizer=keras.optimizers.Adam(cfg.HEAD_LR),
                      loss="sparse_categorical_crossentropy")
        model.fit([xtr, ftr], ytr, epochs=epochs, batch_size=cfg.BATCH_SIZE, verbose=0)

    pred = model.predict([xte, fte], batch_size=1024, verbose=0).argmax(axis=1)
    print(f"seed {seed}: accuracy {np.mean(pred == yte):.4f}, "
          f"macro F1 {f1_score(yte, pred, average='macro'):.4f}")


if __name__ == "__main__":
    main()
