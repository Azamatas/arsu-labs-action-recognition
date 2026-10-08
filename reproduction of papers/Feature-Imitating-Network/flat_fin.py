import keras
from keras import layers
import numpy as np

from config import BATCH_SIZE, FEATURES, FIN_EPOCHS, FIN_LR, SEGMENT
from wisdm_windowing import compute_feature

# winners of the random topology search on 1337
FLAT_TOPOLOGIES = {
    "mean": {"widths": [256, 512, 32, 64], "bn": [True, False, False, False]},
    "std": {"widths": [1024, 512, 1280, 64], "bn": [False, True, True, False]},
    "mad": {"widths": [1024, 512, 1280, 64], "bn": [False, True, True, False]},
}


class FlatFIN(keras.Model):
    def __init__(self, topology, truncate=1, **kw):
        super().__init__(**kw)
        self.blocks = []
        for width, bn in zip(topology["widths"], topology["bn"]):
            block = [layers.Dense(width, activation="relu")]
            if bn:
                block.append(layers.BatchNormalization())
            self.blocks.append(block)
        self.head = layers.Dense(1)
        self.kept = len(self.blocks) - (truncate - 1)
        self.embed_width = topology["widths"][self.kept - 1]

    def embed(self, x):
        for block in self.blocks[: self.kept]:
            for layer in block:
                x = layer(x)
        return x

    def call(self, x):
        for block in self.blocks:
            for layer in block:
                x = layer(x)
        return self.head(x)


def build_bank(x_axes_train, x_axes_test, seed, init="fin", truncate=1):
    fins, imitation = {}, {}
    for feature in FEATURES:
        keras.utils.set_random_seed(seed)
        model = FlatFIN(FLAT_TOPOLOGIES[feature], truncate)
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
