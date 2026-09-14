import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
from keras import layers

import wisdm_windowing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEGMENT = 200
AXES = ("x", "y", "z")
TARGETS = ("mean", "std", "mad")
ACTIVITIES = ("Jogging", "Walking", "Upstairs", "Downstairs", "Sitting", "Standing")

SPLITS = {"train": range(1, 21), "val": range(21, 27), "test": range(27, 37)}
BEST_PATH = os.path.join(BASE_DIR, "best_topologies.json")

N_TOPOLOGIES = 25
SEARCH_EPOCHS = 6
FINAL_EPOCHS = 60
BATCH_SIZE = 256
SEED = 42


def generate_topologies(count=100):
    rng = np.random.RandomState(1337)
    widths = [32, 64, 128, 256, 512, 1024, 1280]
    lrs = [0.01, 1e-02, 1e-03, 1e-04, 1e-05, 1e-06, 1e-07, 1e-08, 1e-09]
    momenta = [1e-05, 1e-06, 1e-07, 1e-08, 1e-09, 0]
    b1s = [0.94, 0.92, 0.9, 0.88, 0.86, 0.84, 0.82]
    b2s = [0.995, 0.99, 0.985, 0.98, 0.975, 0.97, 0.96, 0.95, 0.94]
    out = []
    for _ in range(count):
        n = rng.randint(3, 8)
        w, bn = [], []
        for _ in range(n):
            w.append(widths[rng.randint(len(widths))])
            bn.append(bool(rng.choice([False, True])))
        out.append(
            {
                "widths": w,
                "bn": bn,
                "lr": float(lrs[rng.randint(len(lrs))]),
                # float() matters: their grid ends in an int 0, and Keras 3's SGD rejects it
                "momentum": float(momenta[rng.randint(len(momenta))]),
                "nesterov": bool(rng.choice([False, True])),
                "b1": float(b1s[rng.randint(len(b1s))]),
                "b2": float(b2s[rng.randint(len(b2s))]),
                "opt": str(rng.choice(["SGD", "ADAM"])),
            }
        )
    return out


def load_splits():
    return {k: wisdm_windowing.per_axis(w) for k, (w, _) in wisdm_windowing.load_splits().items()}


def compute_targets(w):
    return wisdm_windowing.compute_targets(w, TARGETS)


def build_model(topology):
    model = keras.Sequential([keras.Input(shape=(SEGMENT,))])
    for width, bn in zip(topology["widths"], topology["bn"]):
        model.add(layers.Dense(width, activation="relu"))
        if bn:
            model.add(layers.BatchNormalization())
    model.add(layers.Dense(1))

    if topology["opt"] == "SGD":
        opt = keras.optimizers.SGD(
            learning_rate=topology["lr"],
            momentum=topology["momentum"],
            nesterov=topology["nesterov"] and topology["momentum"] > 0,
        )
    else:
        opt = keras.optimizers.Adam(
            learning_rate=topology["lr"], beta_1=topology["b1"], beta_2=topology["b2"]
        )
    model.compile(optimizer=opt, loss="mse")
    return model


def r2(true, pred):
    return 1.0 - ((true - pred) ** 2).sum() / ((true - true.mean()) ** 2).sum()


def run_feature(x, y, j, topologies, known_best=None):
    y_mean, y_std = y["train"][:, j].mean(), y["train"][:, j].std()
    y_fit = ((y["train"][:, j] - y_mean) / y_std).astype(np.float32)

    def score(model, split):
        pred = model.predict(x[split], batch_size=1024, verbose=0).ravel()
        return r2(y[split][:, j], pred.astype(np.float64) * y_std + y_mean)

    best_val, best_i = float("nan"), known_best
    if best_i is None:
        best_val = -np.inf
        for i, topo in enumerate(topologies):
            keras.utils.set_random_seed(SEED)
            model = build_model(topo)
            hist = model.fit(
                x["train"], y_fit, epochs=SEARCH_EPOCHS, batch_size=BATCH_SIZE, verbose=0
            )
            if not np.isfinite(hist.history["loss"][-1]):
                continue
            val = score(model, "val")
            if np.isfinite(val) and val > best_val:
                best_val, best_i = val, i

    keras.utils.set_random_seed(SEED)
    model = build_model(topologies[best_i])
    model.fit(x["train"], y_fit, epochs=FINAL_EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    topo = topologies[best_i]
    label = f"#{best_i} {len(topo['widths'])}L {topo['opt']} lr={topo['lr']:g}"
    return score(model, "test"), best_val, label, best_i


def load_cached_best():
    """Winners from an earlier search, reused unless --search forces a fresh one."""
    if not os.path.exists(BEST_PATH):
        return None
    cached = json.load(open(BEST_PATH, encoding="utf-8"))
    context = {"n_topologies": N_TOPOLOGIES, "segment": SEGMENT,
               "splits": {k: [v.start, v.stop] for k, v in SPLITS.items()}}
    if cached.get("context") != context:
        print("cached winners were selected under different settings - searching again\n")
        return None
    return cached["best"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", action="store_true", help="re-run the topology search")
    args = parser.parse_args()

    x = load_splits()
    y = {k: compute_targets(v) for k, v in x.items()}
    topologies = generate_topologies()[:N_TOPOLOGIES]
    cached = None if args.search else load_cached_best()

    print(f"train {len(x['train'])} / val {len(x['val'])} / test {len(x['test'])} windows")
    print(f"users 1-20 / 21-26 / 27-36, {N_TOPOLOGIES} topologies")
    print("reusing cached winners\n" if cached else "searching topologies\n")

    rows, best = [], {}
    for j, name in enumerate(TARGETS):
        test_r2, val_r2, label, idx = run_feature(
            x, y, j, topologies, None if cached is None else cached[name]
        )
        rows.append((name, test_r2, val_r2, label))
        best[name] = idx
        val = "cached" if np.isnan(val_r2) else f"{val_r2:.4f}"
        print(f"  {name:<5} test R²={test_r2:.4f}  val R²={val}  [{label}]", flush=True)

    if cached is None:
        json.dump(
            {"best": best,
             "context": {"n_topologies": N_TOPOLOGIES, "segment": SEGMENT,
                         "splits": {k: [v.start, v.stop] for k, v in SPLITS.items()}}},
            open(BEST_PATH, "w", encoding="utf-8"), indent=2,
        )
        print(f"\nwinners saved to {os.path.basename(BEST_PATH)}")

    print("\n| feature | test R² | val R² | topology |")
    print("|---|---:|---:|---|")
    for name, test_r2, val_r2, label in rows:
        val = "cached" if np.isnan(val_r2) else f"{val_r2:.4f}"
        print(f"| {name} | {test_r2:.4f} | {val} | {label} |")


if __name__ == "__main__":
    main()
