"""Bilinear FIN with CIRCULAR convolutions (wrap-around padding instead of zeros).

With circular convolutions the FIN is exactly a linear map of the power spectrum of the z-scored axis (Parseval):
emb = ReLU(L P + c), checked in efficiency/spectral_equivalence.py, so after training it can run as one real FFT and
one (T // 2 + 1) x emb matrix per axis instead of two convolutions.

    from bilin_fin_circular import circular
    bfin.RhythmFIN = circular(bfin.RhythmFIN)     # every FIN built afterwards wraps around
"""
from keras import layers, ops


def circular(base):
    class CircularFIN(base):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.a = layers.Conv1D(self.a.filters, self.a.kernel_size, padding="valid", use_bias=False)
            self.b = layers.Conv1D(self.b.filters, self.b.kernel_size, padding="valid", use_bias=False)

        def embed(self, x):
            x = (x - ops.mean(x, axis=1, keepdims=True)) / (ops.std(x, axis=1, keepdims=True) + 1e-6)
            k = self.a.kernel_size[0]
            left, right = (k - 1) // 2, k - 1 - (k - 1) // 2  # same alignment as Keras 'same'
            h = ops.expand_dims(x, axis=-1)
            h = ops.concatenate([h[:, h.shape[1] - left:], h, h[:, :right]], axis=1)
            return self.proj(ops.mean(self.a(h) * self.b(h), axis=1))

    return CircularFIN
