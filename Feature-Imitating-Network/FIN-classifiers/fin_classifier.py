import keras
from keras import layers, ops

from config import AXES, HEAD_WIDTH, N_CLASSES


class Ensemble(keras.Model):
    def __init__(self, fins, head_layers, use_extra=False, **kw):
        super().__init__(**kw)
        self.fins = list(fins.values())
        self.use_extra = use_extra
        self.hidden = [layers.Dense(HEAD_WIDTH, activation="relu") for _ in range(head_layers - 1)]
        self.out = layers.Dense(N_CLASSES, activation="softmax")

    def set_fin_trainable(self, flag):
        for f in self.fins:
            f.trainable = flag

    def call(self, inputs):
        x, extra = inputs if self.use_extra else (inputs, None)
        parts = [fin.embed(x[:, i, :]) for i in range(len(AXES)) for fin in self.fins]
        if extra is not None:
            parts.append(extra)
        h = ops.concatenate(parts, axis=1)
        for layer in self.hidden:
            h = layer(h)
        return self.out(h)
