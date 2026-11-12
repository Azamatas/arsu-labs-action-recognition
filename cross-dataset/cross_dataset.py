import os
import sys

os.environ["KERAS_BACKEND"] = "torch"
HERE = os.path.dirname(os.path.abspath(__file__))
FIN = os.path.dirname(HERE)
sys.path[:0] = [HERE, FIN, os.path.join(FIN, "data-processing")]

import argparse

import numpy as np

import config as cfg

FORMATS = ("eps",)

OUT = os.path.join(HERE, "results")
FLIP = np.array([-1, 1, -1], np.float32)[:, None]  # 180 degrees about y
MODELS = {"fin_frozen_dropout": ("FIN + 40, pretrained", "#2a78d6"),
          "fin_random_dropout": ("FIN + 40, random init", "#eb6834"),
          "ignatov_cnn": ("CNN + 40 statistical", "#7a4fc9")}
HEAD_DROPOUT = 0.5
DIRECTIONS = {"w2m": "WISDM -> MotionSense", "m2w": "MotionSense -> WISDM"}
CNN_LR = 5e-4


def load(direction):
    import motionsense_windowing as ms
    import wisdm_windowing as ww

    data = ww.load_splits()
    wisdm_train = (np.concatenate([data["train"][0], data["val"][0]]),  # users 1-26
                   np.concatenate([data["train"][1], data["val"][1]]))
    ms_windows, ms_users, ms_labels = ms.window_raw()
    if direction == "w2m":
        (x, y), tests = wisdm_train, {"MotionSense": (ms_windows, ms_labels), "WISDM 27-36": data["test"]}
    else:
        held_out = ms_users >= 19  # MotionSense users 19-24
        (x, y) = ms_windows[~held_out], ms_labels[~held_out]
        w_windows, _, w_labels = ww.window_raw()
        tests = {"WISDM, all users": (w_windows, w_labels),
                 "MotionSense 19-24": (ms_windows[held_out], ms_labels[held_out])}
    return np.concatenate([x, x * FLIP]), np.concatenate([y, y]), tests


def train(direction, seeds, models=tuple(MODELS)):
    import keras
    import torch
    from keras import layers, ops

    from bilin_fin_temporal_plus_original import RhythmEnsemble, RhythmFIN, ignatov_features, targets

    class TestAccuracy(keras.callbacks.Callback):
        """Accuracy on every test set after every epoch"""

        def __init__(self, tests):
            super().__init__()
            self.tests, self.curves = tests, {name: [] for name in tests}

        def on_epoch_end(self, epoch, logs=None):
            rng = torch.random.get_rng_state()
            for name, (x, y) in self.tests.items():
                pred = self.model.predict(x, batch_size=1024, verbose=0).argmax(axis=1)
                self.curves[name].append(float(np.mean(pred == y)))
            torch.random.set_rng_state(rng)

    class DropoutEnsemble(RhythmEnsemble):
        """RhythmEnsemble with dropout after the hidden layer."""

        def __init__(self, fin, **kw):
            super().__init__(fin, **kw)
            self.drop = layers.Dropout(HEAD_DROPOUT)

        def call(self, inputs, training=None):
            x, stats = inputs
            h = ops.concatenate([self.fin.embed(x[:, i, :]) for i in range(len(cfg.AXES))] + [stats], axis=1)
            for layer in self.hidden:
                h = self.drop(layer(h), training=training)
            return self.out(h)

    def ignatov_cnn(n_features):
        init = dict(kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.01),
                    bias_initializer=keras.initializers.Constant(0.01))
        l2 = keras.regularizers.L2(5e-4 / 2 / 200)  # his summed loss over a batch of 200, per window
        signal = keras.Input((cfg.SEGMENT, len(cfg.AXES)))
        features = keras.Input((n_features,))
        h = layers.Conv1D(196, 12, padding="same", activation="relu", **init)(signal)
        h = layers.MaxPooling1D(4, padding="same")(h)
        h = layers.Concatenate()([layers.Flatten()(h), features])
        h = layers.Dense(1024, activation="relu", kernel_regularizer=l2, bias_regularizer=l2, **init)(h)
        h = layers.Dropout(0.85)(h)  # his keep_prob = 0.15
        return keras.Model([signal, features], layers.Dense(cfg.N_CLASSES, activation="softmax", **init)(h))

    def centred(x):
        x = x - x.mean(axis=2, keepdims=True)
        return x.reshape(len(x), cfg.SEGMENT, len(cfg.AXES)).astype(np.float32)

    xtr, ytr, tests = load(direction)
    ftr = ignatov_features(xtr)
    mu, sd = ftr.mean(0), ftr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    standardise = lambda f: ((f - mu) / sd).astype(np.float32)
    ftr = standardise(ftr)
    ftests = {name: standardise(ignatov_features(x)) for name, (x, _) in tests.items()}
    print(f"{DIRECTIONS[direction]}: training on {len(xtr)} windows (with the flipped copies); test "
          + ", ".join(f"{name} {len(y)} windows" for name, (_, y) in tests.items()), flush=True)

    def fit(model, x, phases, lr, inputs):
        """Returns the test curves and final predictions."""
        track = TestAccuracy({name: (inputs(name), y) for name, (_, y) in tests.items()})
        for epochs, before in phases:
            before()
            model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy")
            model.fit(x, ytr, epochs=epochs, batch_size=cfg.BATCH_SIZE, verbose=0, callbacks=[track])
        preds = {name: model.predict(inputs(name), batch_size=1024, verbose=0).argmax(axis=1) for name in tests}
        return track.curves, preds

    def run(name, seed):
        fin_inputs = lambda t: [tests[t][0], ftests[t]]
        if name == "ignatov_cnn":
            keras.utils.set_random_seed(seed)
            return fit(ignatov_cnn(ftr.shape[1]), [centred(xtr), ftr], [(cfg.HEAD_EPOCHS, lambda: None)], CNN_LR,
                       lambda t: [centred(tests[t][0]), ftests[t]])
        keras.utils.set_random_seed(seed)
        fin = RhythmFIN()
        pretrained = name == "fin_frozen_dropout"
        if pretrained:
            axes = xtr.reshape(-1, cfg.SEGMENT)
            y = targets(axes)
            fin.compile(optimizer=keras.optimizers.Adam(cfg.FIN_LR), loss="mse")
            fin.fit(axes, ((y - y.mean(0)) / y.std(0)).astype(np.float32),
                    epochs=cfg.FIN_EPOCHS, batch_size=cfg.BATCH_SIZE, verbose=0)
        keras.utils.set_random_seed(seed)
        # pretrained: frozen for all epochs; random init: trained with the head from the first epoch
        return fit(DropoutEnsemble(fin), [xtr, ftr], [(cfg.HEAD_EPOCHS, lambda: setattr(fin, "trainable", not pretrained))],
                   cfg.HEAD_LR, fin_inputs)

    curves = {(m, t): [] for m in models for t in tests}
    preds = {(m, t): [] for m in models for t in tests}
    for seed in seeds:
        for name in models:
            c, p = run(name, seed)
            for t in tests:
                curves[name, t].append(c[t])
                preds[name, t].append(p[t])
            print(f"seed {seed}, {name}: " + ", ".join(f"{t} {np.mean(p[t] == tests[t][1]):.4f}" for t in tests),
                  flush=True)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{direction}.npz")
    old = dict(np.load(path)) if set(models) != set(MODELS) else {}  # retrained models replace theirs only
    if old and old["seeds"].tolist() != list(seeds):
        raise SystemExit(f"{path} has seeds {old['seeds'].tolist()}, retrain with the same seeds")
    np.savez(path, **{**old, "seeds": np.array(seeds), "tests": np.array(list(tests)),
                      **{f"truth|{t}": y for t, (_, y) in tests.items()},
                      **{f"curve|{m}|{t}": np.array(v) for (m, t), v in curves.items()},
                      **{f"pred|{m}|{t}": np.array(v) for (m, t), v in preds.items()}})
    print("saved", path)
    return path


def report(direction, path):
    """Mean +- s.d. over the seeds in results/<direction>.md."""
    from sklearn.metrics import f1_score

    data = np.load(path)
    seeds, tests = data["seeds"].tolist(), data["tests"].tolist()
    ms = lambda v: f"{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}"
    lines = [f"# {DIRECTIONS[direction]}", "",
             f"Written by `cross_dataset.py {direction}`; seeds {seeds}, scored after the last epoch. Activity "
             "columns are recall, mean over seeds.", ""]
    for t in tests:
        truth = data[f"truth|{t}"]
        lines += [f"## Test: {t}", "", "| model | accuracy | macro F1 | " + " | ".join(cfg.ACTIVITIES) + " |",
                  "|---|---:|---:|" + "---:|" * len(cfg.ACTIVITIES)]
        for m, (label, _) in MODELS.items():
            p = data[f"pred|{m}|{t}"]
            acc = [np.mean(q == truth) for q in p]
            f1 = [f1_score(truth, q, average="macro") for q in p]
            recall = [np.mean([np.mean(q[truth == i] == i) for q in p]) for i in range(cfg.N_CLASSES)]
            lines.append(f"| {label} | {ms(acc)} | {ms(f1)} | " + " | ".join(f"{r:.3f}" for r in recall) + " |")
        lines.append("")
    out = os.path.join(OUT, f"{direction}.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print("saved", out)


def std_bands(ax, x, bands, alpha=0.18):
    """Mean +- s.d. regions"""
    from itertools import combinations

    from matplotlib.colors import to_rgb

    for k in range(1, len(bands) + 1):
        for subset in combinations(bands, k):
            low = np.max([b[0] for b in subset], axis=0)
            high = np.min([b[1] for b in subset], axis=0)
            colour = np.ones(3)
            for b in subset:
                colour = colour * (1 - alpha) + np.array(to_rgb(b[2])) * alpha
            ax.fill_between(x, low, high, where=high > low, interpolate=True, color=colour, linewidth=0, zorder=0.5)


def plot(direction, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.3, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })
    data = np.load(path)
    target = data["tests"].tolist()[0]  # the other dataset
    truth = data[f"truth|{target}"]

    def save(fig, name):
        for ext in FORMATS:
            out = os.path.join(OUT, f"{direction}_{name}.{ext}")
            fig.savefig(out, bbox_inches="tight", dpi=600 if ext == "png" else None)
            print("saved", out)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.15, 1.95))  # 8 cm wide
    curves = {m: data[f"curve|{m}|{target}"] for m in MODELS}
    x = np.arange(1, next(iter(curves.values())).shape[1] + 1)
    std_bands(ax, x, [(c.mean(0) - c.std(0, ddof=1), c.mean(0) + c.std(0, ddof=1), MODELS[m][1])
                      for m, c in curves.items()])
    for m, c in curves.items():
        ax.plot(x, c.mean(0), color=MODELS[m][1], label=MODELS[m][0])
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"test accuracy ({target})")
    ax.set_xlim(1, x[-1])
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.4, zorder=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(pad=0.3)
    save(fig, "curves")

    short = ["Jog", "Walk", "Up", "Down", "Sit", "Stand"]
    shown = [m for m in MODELS if f"pred|{m}|{target}" in data]
    cols = 2 if len(shown) > 3 else len(shown)
    rows = -(-len(shown) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.72, 1.75 * rows if cols < 3 else 1.75),
                             squeeze=False)
    axes = axes.ravel()
    for ax, m in zip(axes, shown):
        label = MODELS[m][0]
        counts = np.zeros((cfg.N_CLASSES, cfg.N_CLASSES))
        for p in data[f"pred|{m}|{target}"]:
            np.add.at(counts, (truth, p), 1)
        share = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(cfg.N_CLASSES):
            for j in range(cfg.N_CLASSES):
                ax.text(j, i, f"{100 * share[i, j]:.0f}", ha="center", va="center", fontsize=5.5,
                        color="white" if share[i, j] > 0.6 else "black")
        ax.set_title(label)
        ax.set_xticks(range(cfg.N_CLASSES), short, rotation=90)
        ax.set_yticks(range(cfg.N_CLASSES), short)
        ax.set_xlabel("predicted")
        ax.tick_params(length=0)
    for ax in axes[::cols]:
        ax.set_ylabel("true")
    fig.tight_layout(pad=0.3, w_pad=0.6)
    save(fig, "confusion")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("direction", choices=list(DIRECTIONS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(cfg.SEEDS))
    parser.add_argument("--plot", action="store_true", help="only redraw the figures from results/<direction>.npz")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS),
                        help="train only these, merged into the existing results/<direction>.npz")
    args = parser.parse_args()
    path = os.path.join(OUT, f"{args.direction}.npz")
    if not args.plot:
        path = train(args.direction, args.seeds, args.models)
        report(args.direction, path)
    plot(args.direction, path)


if __name__ == "__main__":
    main()
