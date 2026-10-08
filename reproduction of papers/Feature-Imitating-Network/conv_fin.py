import keras
from keras import layers, ops
import numpy as np

from config import BATCH_SIZE, FEATURES, FIN_EPOCHS, FIN_LR, SEGMENT
from wisdm_windowing import compute_feature


class ConvFIN(keras.Model):
    def __init__(self, width=64, truncate=1, **kw):
        super().__init__(**kw)
        self.c1 = layers.Conv1D(width, 1, activation="relu")
        self.c2 = layers.Conv1D(width, 1, activation="relu")
        self.proj = layers.Dense(width, activation="relu")
        self.head = layers.Dense(1)
        self.truncate = truncate
        self.embed_width = width

    def _pooled(self, x):
        h = self.c2(self.c1(ops.expand_dims(x, axis=-1)))
        return ops.mean(h, axis=1)

    def embed(self, x):
        h = self._pooled(x)
        return h if self.truncate == 2 else self.proj(h)

    def call(self, x):
        return self.head(self.proj(self._pooled(x)))


def build_bank(x_axes_train, x_axes_test, seed, init="fin", truncate=1):
    fins, imitation = {}, {}
    for feature in FEATURES:
        keras.utils.set_random_seed(seed)
        model = ConvFIN(truncate=truncate)
        if init == "random":
            model(np.zeros((1, SEGMENT), np.float32))
            imitation[feature] = float("nan")
            fins[feature] = model
            continue

        y = compute_feature(x_axes_train, feature)
        y_mean, y_std = y.mean(), y.std()
        model.compile(optimizer=keras.optimizers.Adam(FIN_LR), loss="mse")
        model.fit(x_axes_train, ((y - y_mean) / y_std).astype(np.float32), epochs=FIN_EPOCHS,
                  batch_size=BATCH_SIZE, verbose=0)
        true = compute_feature(x_axes_test, feature)
        pred = model.predict(x_axes_test, batch_size=1024, verbose=0).ravel()
        pred = pred.astype(np.float64) * y_std + y_mean
        imitation[feature] = 1.0 - ((true - pred) ** 2).sum() / ((true - true.mean()) ** 2).sum()
        fins[feature] = model
    return fins, imitation
