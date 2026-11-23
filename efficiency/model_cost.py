    """Inference cost of one 3 x 200 window for the paper's two neural models: FIN + 40 (frozen FIN shared by the axes,
dense head) and the Ignatov CNN + 40 (cross-dataset/cross_dataset.py). Parameters, FLOPs counted by PyTorch's
FlopCounterMode on the real graph (Keras runs on the torch backend), memory and CPU latency at batch 1.

    python efficiency/model_cost.py        # writes efficiency/results.md

Runs on the CPU only, so it can share the machine with GPU training.
"""
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KERAS_BACKEND"] = "torch"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "data-processing")]

import keras  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from keras import layers  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

import config as cfg  # noqa: E402
from bilin_fin_temporal_plus_original import RhythmEnsemble, RhythmFIN, ignatov_features  # noqa: E402

N_FEATURES = 40


def fin40():
    model = RhythmEnsemble(RhythmFIN())  # the paper's head adds dropout, which costs nothing at inference
    model([np.zeros((1, 3, cfg.SEGMENT), np.float32), np.zeros((1, N_FEATURES), np.float32)])
    return model, lambda x, f: [x, f]


def cnn40():
    """Copy of ignatov_cnn in cross-dataset/cross_dataset.py (initialisers and regularisers do not change the cost)."""
    signal = keras.Input((cfg.SEGMENT, len(cfg.AXES)))
    features = keras.Input((N_FEATURES,))
    h = layers.Conv1D(196, 12, padding="same", activation="relu")(signal)
    h = layers.MaxPooling1D(4, padding="same")(h)
    h = layers.Concatenate()([layers.Flatten()(h), features])
    h = layers.Dense(1024, activation="relu")(h)
    h = layers.Dropout(0.85)(h)
    model = keras.Model([signal, features], layers.Dense(cfg.N_CLASSES, activation="softmax")(h))
    return model, lambda x, f: [x.reshape(len(x), cfg.SEGMENT, len(cfg.AXES)), f]


def flops(model, inputs):
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(inputs, training=False)
    return counter.get_total_flops()


def latency_ms(fn, runs=300):
    for _ in range(20):
        fn()
    t = []
    for _ in range(runs):
        s = time.perf_counter()
        fn()
        t.append(time.perf_counter() - s)
    return 1e3 * float(np.median(t))


def main():
    torch.set_num_threads(1)  # one core, closest to a phone or watch running it in the background
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, 3, cfg.SEGMENT)).astype(np.float32)
    f = rng.normal(size=(1, N_FEATURES)).astype(np.float32)

    rows = []
    for name, build in (("FIN + 40", fin40), ("CNN (+40 stat.)", cnn40)):
        model, prep = build()
        inputs = [np.ascontiguousarray(a) for a in prep(x, f)]
        params = model.count_params()
        fl = flops(model, inputs)
        with torch.no_grad():
            ms = latency_ms(lambda: model(inputs, training=False))
        rows.append((name, params, fl, ms))
        print(f"{name}: {params:,} parameters, {fl / 1e6:.2f} MFLOPs (matmul/conv), {ms:.2f} ms", flush=True)

    feat_ms = latency_ms(lambda: ignatov_features(x), runs=200)

    # biggest intermediate tensor kept at once (float32), from the architectures:
    # FIN: the two 200 x 64 convolution outputs of one axis (axes run one after the other)
    # CNN: the 200 x 196 convolution output
    act = {"FIN + 40": 2 * cfg.SEGMENT * 64 * 4, "CNN (+40 stat.)": cfg.SEGMENT * 196 * 4}

    lines = ["# Inference cost of one 10 s window (3 x 200 samples)", "",
             "Written by `efficiency/model_cost.py`. FLOPs counted by `torch.utils.flop_counter` (matrix "
             "multiplications and convolutions; element-wise products, means and activations add < 1 %). "
             f"Latency: median of 300 calls at batch 1, one CPU thread, {torch.__version__} on this laptop.", "",
             "| model | parameters | weights fp32 | weights int8 | largest activation fp32 | MFLOPs | CPU latency |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, params, fl, ms in rows:
        lines.append(f"| {name} | {params:,} | {params * 4 / 1024:,.0f} KB | {params / 1024:,.0f} KB | "
                     f"{act[name] / 1024:,.0f} KB | {fl / 1e6:.2f} | {ms:.2f} ms |")
    lines += ["", f"The 40 statistical features (NumPy, same window): {feat_ms:.2f} ms.", ""]
    with open(os.path.join(HERE, "results.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
