
import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
from keras import layers, ops

import wisdm_windowing
from sklearn.metrics import classification_report, f1_score

import fin_flat

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "ensemble_results")

SEGMENT = 200
AXES = ("x", "y", "z")
ACTIVITIES = ("Jogging", "Walking", "Upstairs", "Downstairs", "Sitting", "Standing")
N_CLASSES = len(ACTIVITIES)
SPLITS = {"train": range(1, 21), "val": range(21, 27), "test": range(27, 37)}
FEATURES = ("mean", "std", "mad")

FIN_EPOCHS = 60
HEAD_EPOCHS = 60
FREEZE_EPOCHS = 8
HEAD_WIDTH = 128
BATCH_SIZE = 256
FIN_LR = 1e-3
HEAD_LR = 1e-3
SEED = 42


compute_feature = wisdm_windowing.compute_feature
load_splits = wisdm_windowing.load_splits


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


def winning_topologies():
    topos = fin_flat.generate_topologies()[: fin_flat.N_TOPOLOGIES]
    best = fin_flat.load_cached_best()
    if best is None:
        raise SystemExit("run fin_flat.py first so best_topologies.json exists")
    return {f: topos[best[f]] for f in FEATURES}, best


def build_bank(topos, truncate, init, x_axes_train, x_axes_test, seed):
    fins, imitation = {}, {}
    for feature in FEATURES:
        keras.utils.set_random_seed(seed)
        model = FlatFIN(topos[feature], truncate)
        if init == "random":
            model(np.zeros((1, SEGMENT), np.float32))
            imitation[feature] = float("nan")
            fins[feature] = model
            continue

        y = compute_feature(x_axes_train, feature)
        y_mean, y_std = y.mean(), y.std()
        model.compile(optimizer=keras.optimizers.Adam(FIN_LR), loss="mse")
        model.fit(
            x_axes_train,
            ((y - y_mean) / y_std).astype(np.float32),
            epochs=FIN_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
        )
        true = compute_feature(x_axes_test, feature)
        pred = model.predict(x_axes_test, batch_size=1024, verbose=0).ravel()
        pred = pred.astype(np.float64) * y_std + y_mean
        imitation[feature] = 1.0 - ((true - pred) ** 2).sum() / ((true - true.mean()) ** 2).sum()
        fins[feature] = model
    return fins, imitation



class Ensemble(keras.Model):
    def __init__(self, fins, head_layers, **kw):
        super().__init__(**kw)
        self.fins = list(fins.values())
        self.hidden = [layers.Dense(HEAD_WIDTH, activation="relu") for _ in range(head_layers - 1)]
        self.out = layers.Dense(N_CLASSES, activation="softmax")

    def set_fin_trainable(self, flag):
        for f in self.fins:
            f.trainable = flag

    def call(self, x):
        parts = [fin.embed(x[:, i, :]) for i in range(len(AXES)) for fin in self.fins]
        h = ops.concatenate(parts, axis=1)
        for layer in self.hidden:
            h = layer(h)
        return self.out(h)


def run_regime(fins, head_layers, regime, data, seed):
    keras.utils.set_random_seed(seed)
    model = Ensemble(fins, head_layers)
    (xtr, ytr), (xva, yva), (xte, yte) = data["train"], data["val"], data["test"]

    def compile_with(trainable, lr):
        model.set_fin_trainable(trainable)
        model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    if regime == "frozen":
        compile_with(False, HEAD_LR)
        model.fit(xtr, ytr, epochs=HEAD_EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    elif regime == "gradual":
        compile_with(False, HEAD_LR)
        model.fit(xtr, ytr, epochs=FREEZE_EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        compile_with(True, HEAD_LR)
        model.fit(xtr, ytr, epochs=HEAD_EPOCHS - FREEZE_EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    else:
        compile_with(True, HEAD_LR)
        model.fit(xtr, ytr, epochs=HEAD_EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    def score(x, y):
        pred = model.predict(x, batch_size=1024, verbose=0).argmax(axis=1)
        return float((pred == y).mean()), f1_score(y, pred, average="macro"), pred

    val_acc, _, _ = score(xva, yva)
    test_acc, test_f1, pred = score(xte, yte)
    return {"val_acc": val_acc, "test_acc": test_acc, "test_f1": test_f1, "pred": pred}




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", choices=["fin", "random"], default="fin")
    parser.add_argument("--truncate", type=int, choices=[1, 2], default=1)
    parser.add_argument("--regimes", nargs="+", default=["frozen", "gradual", "unfrozen"])
    parser.add_argument("--head-layers", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[SEED])
    args = parser.parse_args()

    data = load_splits()
    x_axes_train = data["train"][0].reshape(-1, SEGMENT)
    x_axes_test = data["test"][0].reshape(-1, SEGMENT)
    topos, best = winning_topologies()

    probe = {f: FlatFIN(topos[f], args.truncate).embed_width for f in FEATURES}
    total = sum(probe.values()) * len(AXES)
    print(f"init={args.init}  truncate={args.truncate}")
    print(f"train {len(data['train'][0])} / val {len(data['val'][0])} / test {len(data['test'][0])}")
    print("topologies: " + ", ".join(f"{f}=#{best[f]}" for f in FEATURES))
    print("embedding widths: " + ", ".join(f"{f}={w}" for f, w in probe.items()))
    print(f"concat width: ({' + '.join(str(w) for w in probe.values())}) x 3 axes = {total}\n",
          flush=True)

    rows = []
    for seed in args.seeds:
        fins, imitation = build_bank(topos, args.truncate, args.init, x_axes_train, x_axes_test, seed)
        if args.init == "fin":
            print("  imitation R²: "
                  + "  ".join(f"{k}={v:.4f}" for k, v in imitation.items()), flush=True)
        for regime in args.regimes:
            for hl in args.head_layers:
                r = run_regime(fins, hl, regime, data, seed)
                rows.append((seed, regime, hl, r))
                print(
                    f"  seed={seed} {regime:<8} head={hl}L  val={r['val_acc']:.4f}  "
                    f"test={r['test_acc']:.4f}  macroF1={r['test_f1']:.4f}",
                    flush=True,
                )

    print(f"\n\n### FIN ensemble, init={args.init}, truncate={args.truncate}\n")
    print("| regime | head layers | val acc | test acc | macro F1 |")
    print("|---|---:|---:|---:|---:|")
    for regime in args.regimes:
        for hl in args.head_layers:
            sel = [r for _, rg, h, r in rows if rg == regime and h == hl]
            print(f"| {regime} | {hl} | {np.mean([r['val_acc'] for r in sel]):.4f} "
                  f"| {np.mean([r['test_acc'] for r in sel]):.4f} "
                  f"| {np.mean([r['test_f1'] for r in sel]):.4f} |")

    best_row = max(rows, key=lambda r: r[3]["val_acc"])
    print(f"\nbest by val: {best_row[1]}, {best_row[2]} head layers "
          f"-> test {best_row[3]['test_acc']:.4f}\n")
    print("```")
    print(classification_report(data["test"][1], best_row[3]["pred"],
                                target_names=ACTIVITIES, digits=4))
    print("```")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{args.init}_truncate{args.truncate}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([{"seed": s, "regime": rg, "head_layers": h, "val_acc": r["val_acc"],
                    "test_acc": r["test_acc"], "test_f1": r["test_f1"]}
                   for s, rg, h, r in rows], fh, indent=2)


if __name__ == "__main__":
    main()
