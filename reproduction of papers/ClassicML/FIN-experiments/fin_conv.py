import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import keras
import numpy as np
import pandas as pd
from keras import layers, ops

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data_processing", "wisdm_data")
MODEL_DIR = os.path.join(BASE_DIR, "trained_models")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "fin_mean_std_mad.weights.h5")
SCALE_PATH = os.path.join(MODEL_DIR, "fin_mean_std_mad_scale.npz")

SEGMENT = 200
AXES = ("x", "y", "z")
TARGET_NAMES = ("mean", "std", "mad")

SEED = 42
EPOCHS = 40
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

AUG_COPIES = 2
AUG_GAIN = (0.5, 2.0)
AUG_OFFSET = (-15.0, 15.0)



def load_axes(split):
    suffix = "" if split == "train" else "_test"
    return [
        pd.read_csv(
            os.path.join(DATA_DIR, f"data_{a}{suffix}_{SEGMENT}.csv"), header=None
        ).to_numpy(dtype=np.float64)
        for a in AXES
    ]


def compute_targets(windows):
    mean = windows.mean(axis=1)
    std = windows.std(axis=1, ddof=1)
    mad = np.abs(windows - mean[:, None]).mean(axis=1)
    return np.stack([mean, std, mad], axis=1)


def affine_augment(windows, rng, copies=AUG_COPIES):
    parts = [windows]
    for _ in range(copies):
        gain = rng.uniform(*AUG_GAIN, size=(len(windows), 1))
        offset = rng.uniform(*AUG_OFFSET, size=(len(windows), 1))
        parts.append(windows * gain + offset)
    return np.concatenate(parts, axis=0)




class FeatureImitationNetwork(keras.Model):
    def __init__(self, width=64, **kwargs):
        super().__init__(**kwargs)
        self.c1 = layers.Conv1D(width, 1, activation="relu")
        self.c2 = layers.Conv1D(width, 1, activation="relu")
        self.proj = layers.Dense(64, activation="relu")
        self.head = layers.Dense(len(TARGET_NAMES))  # linear: this is regression

    def embed(self, x):
        h = self.c2(self.c1(ops.expand_dims(x, axis=-1)))
        h = ops.mean(h, axis=1)

        return self.proj(h)

    def call(self, x):
        return self.head(self.embed(x))




def r2(true, pred):
    resid = ((true - pred) ** 2).sum(axis=0)
    total = ((true - true.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - resid / total


def mad_std_ratio(true, pred, min_std=1.0):

    keep = true[:, 1] > min_std
    tr = true[keep, 2] / true[keep, 1]
    pr = pred[keep, 2] / pred[keep, 1]
    return int(keep.sum()), tr.std(), pr.std(), np.corrcoef(tr, pr)[0, 1]




def train():
    keras.utils.set_random_seed(SEED)
    rng = np.random.default_rng(SEED)

    pooled = np.concatenate(load_axes("train"), axis=0)
    x_train = affine_augment(pooled, rng)
    y_train = compute_targets(x_train)

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)

    model = FeatureImitationNetwork()
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="mse")
    model.fit(
        x_train.astype(np.float32),
        ((y_train - y_mean) / y_std).astype(np.float32),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=2,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_weights(WEIGHTS_PATH)
    np.savez(SCALE_PATH, target_mean=y_mean, target_std=y_std)
    return model, y_mean, y_std, len(pooled), len(x_train)


def predict(model, y_mean, y_std, windows):
    raw = model.predict(windows.astype(np.float32), batch_size=1024, verbose=0)
    return raw.astype(np.float64) * y_std + y_mean


def load_fin():
    model = FeatureImitationNetwork()
    model(np.zeros((1, SEGMENT), np.float32))
    model.load_weights(WEIGHTS_PATH)
    scale = np.load(SCALE_PATH)
    return model, scale["target_mean"], scale["target_std"]


def main():
    model, y_mean, y_std, n_pooled, n_aug = train()
    print(f"\ntrained on {n_pooled} pooled windows -> {n_aug} after affine augmentation")
    print(f"parameters: {model.count_params():,}")

    test = load_axes("test")
    print("\n### Imitation quality on held-out WISDM users 27-36\n")
    print("| axis | R² mean | R² std | R² mad | MAE mean | MAE std | MAE mad |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for axis, windows in zip(AXES, test):
        true = compute_targets(windows)
        pred = predict(model, y_mean, y_std, windows)
        row = list(r2(true, pred)) + list(np.abs(true - pred).mean(axis=0))
        print(f"| {axis} | " + " | ".join(f"{v:.4f}" for v in row) + " |")

    print("\n### Reference: MAE of predicting the test-split mean of each target\n")
    print("| axis | mean | std | mad |")
    print("|---|---:|---:|---:|")
    for axis, windows in zip(AXES, test):
        true = compute_targets(windows)
        base = np.abs(true - true.mean(axis=0)).mean(axis=0)
        print(f"| {axis} | " + " | ".join(f"{v:.4f}" for v in base) + " |")

    true = compute_targets(test[0])
    n, tr_spread, pr_spread, corr = mad_std_ratio(true, predict(model, y_mean, y_std, test[0]))
    print(f"\n### mad/std ratio, X axis, {n} windows with std > 1.0 m/s²\n")
    print(f"true spread {tr_spread:.4f} | predicted spread {pr_spread:.4f} | correlation {corr:.4f}")
    print("(a collapsed predicted spread or a low correlation would mean mad was faked from std)")

    print(f"\nweights -> {WEIGHTS_PATH}")
    print(f"scaling -> {SCALE_PATH}")


if __name__ == "__main__":
    main()
