"""Two deployment checks for the paper's models, on the paper's data only.

1. int8: FIN + 40 (pretrained, frozen) and the CNN (+40 stat.) are trained exactly as in
   cross-dataset/cross_dataset.py (seed 42) for w2m (WISDM users 1-26 -> WISDM 27-36 and MotionSense) and m2w
   (MotionSense users 1-18 -> all WISDM users and MotionSense 19-24). Their forward passes are re-implemented in
   PyTorch from the trained weights (checked against Keras) and run in float32, with int8 weights, and with int8
   weights and activations: symmetric post-training quantization, per output channel for weights, per tensor for
   activations, scales calibrated on 2000 training windows (max |value|). This simulates int8 arithmetic
   (fake quantization), as an int8 runtime such as TFLite Micro or CMSIS-NN would compute it.
2. streaming FIN: the time average of the product of the two convolutions accumulated sample by sample, so the two
   200 x 64 feature maps are never stored; checked against the batch computation.

    python efficiency/int8_and_streaming.py       # writes efficiency/results_int8.md
"""
import os
import sys

os.environ["KERAS_BACKEND"] = "torch"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "data-processing"), os.path.join(ROOT, "cross-dataset")]

import keras  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from keras import layers, ops  # noqa: E402

import config as cfg  # noqa: E402
import cross_dataset as cd  # noqa: E402
from bilin_fin_temporal_plus_original import RhythmEnsemble, RhythmFIN, ignatov_features, targets  # noqa: E402

SEED = 42
DEV = "cpu"


# ---------------------------------------------------------------- training, as in cross_dataset.train

class DropoutEnsemble(RhythmEnsemble):
    def __init__(self, fin, **kw):
        super().__init__(fin, **kw)
        self.drop = layers.Dropout(cd.HEAD_DROPOUT)

    def call(self, inputs, training=None):
        x, stats = inputs
        h = ops.concatenate([self.fin.embed(x[:, i, :]) for i in range(len(cfg.AXES))] + [stats], axis=1)
        for layer in self.hidden:
            h = self.drop(layer(h), training=training)
        return self.out(h)


def ignatov_cnn(n_features):
    init = dict(kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.01),
                bias_initializer=keras.initializers.Constant(0.01))
    l2 = keras.regularizers.L2(5e-4 / 2 / 200)
    signal = keras.Input((cfg.SEGMENT, len(cfg.AXES)))
    features = keras.Input((n_features,))
    h = layers.Conv1D(196, 12, padding="same", activation="relu", **init)(signal)
    h = layers.MaxPooling1D(4, padding="same")(h)
    h = layers.Concatenate()([layers.Flatten()(h), features])
    h = layers.Dense(1024, activation="relu", kernel_regularizer=l2, bias_regularizer=l2, **init)(h)
    h = layers.Dropout(0.85)(h)
    return keras.Model([signal, features], layers.Dense(cfg.N_CLASSES, activation="softmax", **init)(h))


def centred(x):
    x = x - x.mean(axis=2, keepdims=True)
    return x.reshape(len(x), cfg.SEGMENT, len(cfg.AXES)).astype(np.float32)


def train(direction):
    xtr, ytr, tests = cd.load(direction)
    ftr = ignatov_features(xtr)
    mu, sd = ftr.mean(0), ftr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    std = lambda f: ((f - mu) / sd).astype(np.float32)
    ftr = std(ftr)
    ftests = {t: std(ignatov_features(x)) for t, (x, _) in tests.items()}

    keras.utils.set_random_seed(SEED)
    fin = RhythmFIN()
    axes = xtr.reshape(-1, cfg.SEGMENT)
    y = targets(axes)
    fin.compile(optimizer=keras.optimizers.Adam(cfg.FIN_LR), loss="mse")
    fin.fit(axes, ((y - y.mean(0)) / y.std(0)).astype(np.float32), epochs=cfg.FIN_EPOCHS,
            batch_size=cfg.BATCH_SIZE, verbose=0)
    keras.utils.set_random_seed(SEED)
    fin40 = DropoutEnsemble(fin)
    fin.trainable = False
    fin40.compile(optimizer=keras.optimizers.Adam(cfg.HEAD_LR), loss="sparse_categorical_crossentropy")
    fin40.fit([xtr, ftr], ytr, epochs=cfg.HEAD_EPOCHS, batch_size=cfg.BATCH_SIZE, verbose=0)

    keras.utils.set_random_seed(SEED)
    cnn = ignatov_cnn(ftr.shape[1])
    cnn.compile(optimizer=keras.optimizers.Adam(cd.CNN_LR), loss="sparse_categorical_crossentropy")
    cnn.fit([centred(xtr), ftr], ytr, epochs=cfg.HEAD_EPOCHS, batch_size=cfg.BATCH_SIZE, verbose=0)
    return fin40, cnn, (xtr, ftr), tests, ftests


# ---------------------------------------------------------------- PyTorch re-implementation with fake quantization

def t(a):
    return torch.as_tensor(np.asarray(a, dtype=np.float32), device=DEV)


def q_weight(w, axis_out=-1):
    """Symmetric int8 per output channel; w has the output channels on `axis_out`."""
    dims = [d for d in range(w.dim()) if d != (axis_out % w.dim())]
    s = w.abs().amax(dim=dims, keepdim=True).clamp_min(1e-12) / 127
    return torch.clamp(torch.round(w / s), -127, 127) * s


class Quant:
    """Records max |activation| per point during calibration, then fake-quantizes to int8 with those scales."""

    def __init__(self):
        self.mode, self.maxabs = "float", {}

    def __call__(self, name, x):
        if self.mode == "calibrate":
            self.maxabs[name] = max(self.maxabs.get(name, 0.0), float(x.abs().max()))
        elif self.mode == "int8":
            s = max(self.maxabs[name], 1e-12) / 127
            x = torch.clamp(torch.round(x / s), -127, 127) * s
        return x


def fin_params(model, quant_weights):
    fin = model.fin
    p = {"a": t(fin.a.kernel.numpy()).permute(2, 1, 0), "b": t(fin.b.kernel.numpy()).permute(2, 1, 0),  # (64, 1, 61)
         "P": t(fin.proj.kernel.numpy()), "pb": t(fin.proj.bias.numpy()),
         "H": t(model.hidden[0].kernel.numpy()), "hb": t(model.hidden[0].bias.numpy()),
         "O": t(model.out.kernel.numpy()), "ob": t(model.out.bias.numpy())}
    if quant_weights:
        p["a"], p["b"] = q_weight(p["a"], 0), q_weight(p["b"], 0)
        for k in ("P", "H", "O"):
            p[k] = q_weight(p[k], -1)
    return p


def fin_forward(p, x, f, q):
    embs = []
    for i in range(3):
        s = x[:, i, :]
        z = (s - s.mean(1, keepdim=True)) / (s.std(1, unbiased=False, keepdim=True) + 1e-6)
        z = q("z", z)[:, None, :]
        a = q("a", F.conv1d(z, p["a"], padding=30))
        b = q("b", F.conv1d(z, p["b"], padding=30))
        m = q("m", (a * b).mean(2))
        embs.append(q("emb", torch.relu(m @ p["P"] + p["pb"])))
    h = q("h", torch.relu(torch.cat(embs + [q("stats", f)], 1) @ p["H"] + p["hb"]))
    return h @ p["O"] + p["ob"]


def cnn_params(model, quant_weights):
    conv, d1, d2 = [l for l in model.layers if l.weights]
    p = {"c": t(conv.kernel.numpy()).permute(2, 1, 0), "cb": t(conv.bias.numpy()),  # (196, 3, 12)
         "D": t(d1.kernel.numpy()), "db": t(d1.bias.numpy()), "O": t(d2.kernel.numpy()), "ob": t(d2.bias.numpy())}
    if quant_weights:
        p["c"], p["D"], p["O"] = q_weight(p["c"], 0), q_weight(p["D"], -1), q_weight(p["O"], -1)
    return p


def cnn_forward(p, x, f, q, pad=(5, 6)):
    h = q("in", x.permute(0, 2, 1))                                   # (n, 3, 200) channels first
    h = torch.relu(F.conv1d(F.pad(h, pad), p["c"]) + p["cb"][None, :, None])
    h = q("conv", F.max_pool1d(h, 4))                                  # (n, 196, 50)
    h = h.permute(0, 2, 1).reshape(len(h), -1)                         # Keras flattens (time, channel)
    h = q("dense", torch.relu(torch.cat([h, q("stats", f)], 1) @ p["D"] + p["db"]))
    return h @ p["O"] + p["ob"]


def evaluate(name, model, forward, params_of, x_in, calib, tests, ftests):
    """Accuracy per test set: Keras, re-implementation in float32, int8 weights, int8 weights + activations."""
    out = {}
    q = Quant()
    rows = {"Keras": None, "float32": (False, "float"), "int8 weights": (True, "float"),
            "int8 weights + activations": (True, "int8")}
    p32, p8 = params_of(model, False), params_of(model, True)
    q.mode = "calibrate"
    with torch.no_grad():
        forward(p32, t(x_in(calib[0])), t(calib[1]), q)
    for t_name, (x, y) in tests.items():
        xi, fi = x_in(x), ftests[t_name]
        keras_pred = model.predict([xi, fi], batch_size=1024, verbose=0).argmax(1)
        for row, spec in rows.items():
            if spec is None:
                pred = keras_pred
            else:
                q.mode = spec[1]
                with torch.no_grad():
                    pred = forward(p8 if spec[0] else p32, t(xi), t(fi), q).argmax(1).cpu().numpy()
            out[(row, t_name)] = float(np.mean(pred == y))
            if row == "float32":
                out[("agree", t_name)] = float(np.mean(pred == keras_pred))
    print(name, {k: round(v, 4) for k, v in out.items()}, flush=True)
    return out


# ---------------------------------------------------------------- streaming FIN

def streaming_embed(z, a, b):
    """z: (200,) z-scored axis; a, b: (64, 61) kernels. Accumulates mean_t a_t * b_t without the 200 x 64 maps."""
    zp = np.concatenate([np.zeros(30), z, np.zeros(30)])
    acc = np.zeros(a.shape[0])
    for t_ in range(len(z)):
        seg = zp[t_:t_ + a.shape[1]]
        acc += (a @ seg) * (b @ seg)
    return acc / len(z)


def main():
    lines = ["# int8 quantization and streaming FIN", "",
             "Written by `efficiency/int8_and_streaming.py`. Models trained exactly as in "
             "`cross-dataset/cross_dataset.py`, seed 42. int8 = symmetric post-training quantization (weights per output "
             "channel, activations per tensor, scales from 2000 training windows), simulated in float arithmetic. "
             "`float32` is the PyTorch re-implementation; `agrees with Keras` is the share of identical predictions.", ""]
    for direction in ("w2m", "m2w"):
        fin40, cnn, (xtr, ftr), tests, ftests = train(direction)
        idx = np.random.default_rng(0).choice(len(xtr), 2000, replace=False)
        calib = (xtr[idx], ftr[idx])
        res = {"FIN + 40": evaluate("FIN + 40", fin40, fin_forward, fin_params, lambda x: x, calib, tests, ftests),
               "CNN (+40 stat.)": evaluate("CNN", cnn, cnn_forward, cnn_params, centred, calib, tests, ftests)}
        lines += [f"## {cd.DIRECTIONS[direction]}", "",
                  "| model | test set | Keras | float32 (agrees with Keras) | int8 weights | int8 weights + activations |",
                  "|---|---|---:|---:|---:|---:|"]
        for m, r in res.items():
            for tn in tests:
                lines.append(f"| {m} | {tn} | {r[('Keras', tn)]:.4f} | {r[('float32', tn)]:.4f} "
                             f"({r[('agree', tn)]:.1%}) | {r[('int8 weights', tn)]:.4f} | "
                             f"{r[('int8 weights + activations', tn)]:.4f} |")
        lines.append("")
        if direction == "w2m":
            fin = fin40.fin
            a = fin.a.kernel.numpy()[:, 0, :].T.astype(np.float64)
            b = fin.b.kernel.numpy()[:, 0, :].T.astype(np.float64)
            x = tests["WISDM 27-36"][0][:50]
            diffs = []
            for w in x:
                for axis in w:
                    z = (axis - axis.mean()) / (axis.std() + 1e-6)
                    batch = (np.stack([np.convolve(np.pad(z, 30), k[::-1], "valid") for k in a])
                             * np.stack([np.convolve(np.pad(z, 30), k[::-1], "valid") for k in b])).mean(1)
                    diffs.append(np.abs(streaming_embed(z, a, b) - batch).max())
            lines += ["## Streaming FIN", "",
                      f"Max |difference| between the streaming and the batch time average over 50 WISDM test windows x "
                      f"3 axes: {max(diffs):.2e} (float64).", "",
                      "| | batch (as trained) | streaming |", "|---|---:|---:|",
                      "| stored per axis | two 200 x 64 maps: 25,600 floats = 100 KB | input window with padding (260) "
                      "+ 64 accumulators + 2 x 64 current outputs = 452 floats = 1.8 KB |",
                      "| FLOPs | same | same |", ""]
    with open(os.path.join(HERE, "results_int8.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
