
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


def extra_features(windows):
    """The part of Ignatov's 40 handcrafted features the FINs do not already give: 31 columns.

    His block is 3 means, 3 stds, 3 mads, the average resultant, and a 10-bin histogram per axis.
    The FINs produce the first nine, so what is left is the resultant and the 30 histogram bins.
    Same role the handcrafted block plays in his CNN, with the FIN embedding standing in for the
    conv branch.
    """
    w = windows.astype(np.float64)
    out = [np.sqrt((w**2).sum(axis=1)).mean(axis=1)]

    for a in range(len(AXES)):
        x = w[:, a, :]
        lo = x.min(axis=1, keepdims=True)
        width = (x.max(axis=1, keepdims=True) - lo) / 10
        width = np.where(width == 0, 1.0, width)
        idx = np.clip(((x - (lo - width / 2)) / width).astype(int), 0, 9)
        counts = np.zeros((len(x), 10))
        np.add.at(counts, (np.arange(len(x))[:, None], idx), 1)
        out += list((counts / x.shape[1]).T)

    return np.stack(out, axis=1).astype(np.float32)


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


class ConvFIN(keras.Model):
    """Per-timestep map then a fixed mean pool. One net per feature, single output.

    Kernel 1 means each sample is mapped independently and the pool is a fixed mean, which matches
    the algebra of mean/std/mad and makes the net invariant to sample order. That invariance is why
    it imitates these three so cheaply, and also why its embedding carries nothing about timing.
    """

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


def make_fin(bank, topos, feature, truncate):
    return ConvFIN(truncate=truncate) if bank == "conv" else FlatFIN(topos[feature], truncate)


def winning_topologies():
    topos = fin_flat.generate_topologies()[: fin_flat.N_TOPOLOGIES]
    best = fin_flat.load_cached_best()
    if best is None:
        raise SystemExit("run fin_flat.py first so best_topologies.json exists")
    return {f: topos[best[f]] for f in FEATURES}, best


def build_bank(bank, topos, truncate, init, x_axes_train, x_axes_test, seed):
    fins, imitation = {}, {}
    for feature in FEATURES:
        keras.utils.set_random_seed(seed)
        model = make_fin(bank, topos, feature, truncate)
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


class TestTracker(keras.callbacks.Callback):
    def __init__(self, x, y):
        super().__init__()
        self.x, self.y, self.history = x, y, []

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.x, batch_size=1024, verbose=0).argmax(axis=1)
        self.history.append(float((pred == self.y).mean()))


def run_regime(fins, head_layers, regime, data, seed, extras=None, track_test=False):
    keras.utils.set_random_seed(seed)
    model = Ensemble(fins, head_layers, use_extra=extras is not None)
    (xtr, ytr), (xva, yva), (xte, yte) = data["train"], data["val"], data["test"]

    def inp(split, x):
        return [x, extras[split]] if extras is not None else x

    cbs = [TestTracker(inp("test", xte), yte)] if track_test else []

    def compile_with(trainable, lr):
        model.set_fin_trainable(trainable)
        model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    if regime == "frozen":
        compile_with(False, HEAD_LR)
        model.fit(inp("train", xtr), ytr, epochs=HEAD_EPOCHS, batch_size=BATCH_SIZE,
                  callbacks=cbs, verbose=0)
    elif regime == "gradual":
        compile_with(False, HEAD_LR)
        model.fit(inp("train", xtr), ytr, epochs=FREEZE_EPOCHS, batch_size=BATCH_SIZE,
                  callbacks=cbs, verbose=0)
        compile_with(True, HEAD_LR)
        model.fit(inp("train", xtr), ytr, epochs=HEAD_EPOCHS - FREEZE_EPOCHS,
                  batch_size=BATCH_SIZE, callbacks=cbs, verbose=0)
    else:
        compile_with(True, HEAD_LR)
        model.fit(inp("train", xtr), ytr, epochs=HEAD_EPOCHS, batch_size=BATCH_SIZE,
                  callbacks=cbs, verbose=0)

    def score(split, x, y):
        pred = model.predict(inp(split, x), batch_size=1024, verbose=0).argmax(axis=1)
        return float((pred == y).mean()), f1_score(y, pred, average="macro"), pred

    val_acc, _, _ = score("val", xva, yva)
    test_acc, test_f1, pred = score("test", xte, yte)
    out = {"val_acc": val_acc, "test_acc": test_acc, "test_f1": test_f1, "pred": pred}
    if track_test:
        out["max_test"] = max(cbs[0].history)
    return out




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", choices=["flat", "conv"], default="flat")
    parser.add_argument("--protocol", choices=["clean", "ignatov"], default="clean",
                        help="ignatov: train on users 1-26 (no val group) and report the best "
                             "test reading seen during training, as cnn_wisdm.py does")
    parser.add_argument("--extras", action="store_true",
                        help="concatenate the 31 handcrafted features the FINs do not supply")
    parser.add_argument("--init", choices=["fin", "random"], default="fin")
    parser.add_argument("--truncate", type=int, choices=[1, 2], default=1)
    parser.add_argument("--regimes", nargs="+", default=["frozen", "gradual", "unfrozen"])
    parser.add_argument("--head-layers", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[SEED])
    args = parser.parse_args()

    data = load_splits()
    if args.protocol == "ignatov":
        data["train"] = (np.concatenate([data["train"][0], data["val"][0]]),
                         np.concatenate([data["train"][1], data["val"][1]]))
    x_axes_train = data["train"][0].reshape(-1, SEGMENT)
    x_axes_test = data["test"][0].reshape(-1, SEGMENT)
    topos, best = winning_topologies() if args.bank == "flat" else ({}, {})

    probe = {f: make_fin(args.bank, topos, f, args.truncate).embed_width for f in FEATURES}
    total = sum(probe.values()) * len(AXES)
    print(f"bank={args.bank}  init={args.init}  truncate={args.truncate}")
    print(f"train {len(data['train'][0])} / val {len(data['val'][0])} / test {len(data['test'][0])}")
    if best:
        print("topologies: " + ", ".join(f"{f}=#{best[f]}" for f in FEATURES))
    print("embedding widths: " + ", ".join(f"{f}={w}" for f, w in probe.items()))
    print(f"concat width: ({' + '.join(str(w) for w in probe.values())}) x 3 axes = {total}\n",
          flush=True)

    extras = None
    if args.extras:
        raw = {k: extra_features(v[0]) for k, v in data.items()}
        mu, sd = raw["train"].mean(0), raw["train"].std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        extras = {k: ((v - mu) / sd).astype(np.float32) for k, v in raw.items()}
        n_extra = extras["train"].shape[1]
        print(f"extra handcrafted features: {n_extra} -> head input {total + n_extra}\n", flush=True)

    rows = []
    for seed in args.seeds:
        fins, imitation = build_bank(
            args.bank, topos, args.truncate, args.init, x_axes_train, x_axes_test, seed
        )
        if args.init == "fin":
            print("  imitation R²: "
                  + "  ".join(f"{k}={v:.4f}" for k, v in imitation.items()), flush=True)
        for regime in args.regimes:
            for hl in args.head_layers:
                r = run_regime(fins, hl, regime, data, seed, extras,
                               track_test=args.protocol == "ignatov")
                rows.append((seed, regime, hl, r))
                extra_col = f"  max={r['max_test']:.4f}" if "max_test" in r else ""
                print(
                    f"  seed={seed} {regime:<8} head={hl}L  val={r['val_acc']:.4f}  "
                    f"test={r['test_acc']:.4f}  macroF1={r['test_f1']:.4f}{extra_col}",
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
    tag = f"{args.bank}_{args.init}_truncate{args.truncate}" + ("_extras" if args.extras else "")
    path = os.path.join(OUT_DIR, f"{tag}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([{"seed": s, "regime": rg, "head_layers": h, "val_acc": r["val_acc"],
                    "test_acc": r["test_acc"], "test_f1": r["test_f1"]}
                   for s, rg, h, r in rows], fh, indent=2)


if __name__ == "__main__":
    main()
