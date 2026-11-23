"""The smallest BiTFIN (circular, 32 filters, kernel 31, embedding 64, hidden 64; run as FFT + one matrix on the device)
in the within-dataset WISDM protocols of the TinyHAR and TinierHAR papers (their 10 user groups), trained with each
protocol's own recipe, so that it can be compared with our re-implementations of both models on equal terms.

  --protocol tinyhar    TinyHAR code settings (baseline/tinyhar/tinyhar_reproduce_wisdm_code.py): 5 s windows over the
                        whole recording (majority label), train step 50 %, test step 10 %, Adam 1e-3, batch 32,
                        <= 80 epochs, lr / 10 after 7 epochs without validation improvement, early stop after 15
  --protocol tinierhar  TinierHAR paper settings (baseline/tinierhar/tinierhar_reproduce_wisdm.py): 4 s windows inside
                        single-activity stretches, train step 50 %, test step 1, AdamW 1e-3, batch 256, <= 150 epochs,
                        same schedule
Model: 3 accelerometer axes through one circular FIN (pretrained to imitate spectral entropy, autocorrelation-peak height
and spectral peak on the training windows, lags 5 .. window / 2; then frozen) + the 40 statistical features -> Dense 64
-> dropout 0.5 -> softmax. Validation: a random 10 % of the training windows; best-validation-loss weights.

    python wisdm-groups/bitfin_wisdm_groups.py --protocol tinyhar --groups 0 1 2 3 4
    python wisdm-groups/bitfin_wisdm_groups.py --report
"""
import glob
import json
import os
import sys

os.environ["KERAS_BACKEND"] = "torch"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "data-processing"), os.path.join(ROOT, "baseline", "tinyhar")]

import keras  # noqa: E402
import numpy as np  # noqa: E402
from keras import layers, ops  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

import bilin_fin_temporal_plus_original as bfin  # noqa: E402
import config as cfg  # noqa: E402
import wisdm_windowing as ww  # noqa: E402
from bilin_fin_circular import circular  # noqa: E402

OUT = os.path.join(HERE, "results")
WIDTH, KERNEL, EMB, HIDDEN = 32, 31, 64, 64
RECIPES = {"tinyhar": dict(win=100, opt="adam", batch=32, epochs=80),
           "tinierhar": dict(win=80, opt="adamw", batch=256, epochs=150)}


def spectral_mflops(t, axes=3, n_classes=6):
    f = t // 2 + 1
    fin = axes * (2.5 * t * np.log2(t) + 3 * f + 2 * f * EMB)
    head = 2 * ((axes * EMB + 40) * HIDDEN + HIDDEN * n_classes)
    return (fin + head) / 1e6


def data(protocol, test_users):
    win = RECIPES[protocol]["win"]
    if protocol == "tinyhar":
        import tinyhar_reproduce_wisdm_code as c

        streams = c.streams()
        train_users = [u for u in streams if u not in test_users]
        xtr, ytr = c.windows(streams, train_users, win // 2)
        xte, yte = c.windows(streams, [u for u in test_users if u in streams], win // 10)
    else:
        import tinyhar_reproduce_wisdm as trw

        trw.WIN = win
        segs = trw.segments()
        xtr, ytr = trw.windows([s for s in segs if s[0] not in test_users], win // 2)
        xte, yte = trw.windows([s for s in segs if s[0] in test_users], 1)
    return xtr.astype(np.float32), ytr, xte.astype(np.float32), yte


class Head(keras.Model):
    def __init__(self, fin, **kw):
        super().__init__(**kw)
        self.fin = fin
        self.hidden = layers.Dense(HIDDEN, activation="relu")
        self.drop = layers.Dropout(0.5)
        self.out = layers.Dense(len(cfg.ACTIVITIES), activation="softmax")

    def call(self, inputs, training=None):
        x, stats = inputs
        h = ops.concatenate([self.fin.embed(x[:, i, :]) for i in range(3)] + [stats], axis=1)
        return self.out(self.drop(self.hidden(h), training=training))


def fold(protocol, test_users, seed):
    r = RECIPES[protocol]
    xtr, ytr, xte, yte = data(protocol, test_users)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ytr))
    va, tr = perm[:len(ytr) // 10], perm[len(ytr) // 10:]

    f_tr = bfin.ignatov_features(xtr)
    mu, sd = f_tr[tr].mean(0), f_tr[tr].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    feats = lambda f: ((f - mu) / sd).astype(np.float32)
    f_tr, f_te = feats(f_tr), feats(bfin.ignatov_features(xte))

    bfin.WIDTH, bfin.KERNEL, ww.MAX_LAG = WIDTH, KERNEL, r["win"] // 2
    keras.utils.set_random_seed(seed)
    fin = circular(bfin.RhythmFIN)()
    axes = xtr[tr].reshape(-1, r["win"])
    y = bfin.targets(axes)
    fin.compile(optimizer=keras.optimizers.Adam(cfg.FIN_LR), loss="mse")
    hist = fin.fit(axes, ((y - y.mean(0)) / y.std(0)).astype(np.float32), epochs=cfg.FIN_EPOCHS,
                   batch_size=cfg.BATCH_SIZE, verbose=0)
    fin.trainable = False

    keras.utils.set_random_seed(seed)
    model = Head(fin)
    opt = keras.optimizers.AdamW(1e-3) if r["opt"] == "adamw" else keras.optimizers.Adam(1e-3)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy")
    cb = [keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=7),
          keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)]
    h = model.fit([xtr[tr], f_tr[tr]], ytr[tr], validation_data=([xtr[va], f_tr[va]], ytr[va]),
                  epochs=r["epochs"], batch_size=r["batch"], callbacks=cb, verbose=0, shuffle=True)
    pred = model.predict([xte, f_te], batch_size=4096, verbose=0).argmax(1)
    return {"protocol": protocol, "users": list(test_users), "seed": seed, "epochs": len(h.history["loss"]),
            "imitation_mse": float(hist.history["loss"][-1]), "test_windows": int(len(yte)),
            "macro_f1": float(f1_score(yte, pred, average="macro")), "accuracy": float(np.mean(pred == yte))}


def report():
    refs = {"tinyhar": ("TinyHAR, ours (authors' code settings)", "baseline/tinyhar/results_reproduce_code", 83.47),
            "tinierhar": ("TinierHAR, ours", "baseline/tinierhar/results_reproduce", None)}
    lines = ["# Smallest BiTFIN in the TinyHAR / TinierHAR WISDM protocols (10 user groups, macro F1 %)", "",
             "Written by `wisdm-groups/bitfin_wisdm_groups.py`. BiTFIN: circular, 32 filters, kernel 31, embedding 64, "
             "hidden 64, + 40 statistical features; cost of its FFT + matrix form.", "",
             "| protocol | model | MFLOPs | folds | macro F1 | accuracy |", "|---|---|---:|---:|---:|---:|"]
    for p, (label, ref_dir, paper) in refs.items():
        rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, f"{p}_group*_seed*.json")))]
        if rows:
            lines.append(f"| {p} | **BiTFIN** | {spectral_mflops(RECIPES[p]['win']):.3f} | {len(rows)} | "
                         f"**{100 * np.mean([r['macro_f1'] for r in rows]):.2f}** | "
                         f"{100 * np.mean([r['accuracy'] for r in rows]):.2f} |")
        refs_rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ROOT, ref_dir, "group*_seed1.json")))]
        if refs_rows:
            done = {tuple(r["users"]) for r in rows} or {tuple(r["users"]) for r in refs_rows}
            same = [r for r in refs_rows if tuple(r["users"]) in done]
            lines.append(f"| {p} | {label} | | {len(same)} | {100 * np.mean([r['macro_f1'] for r in same]):.2f} | "
                         f"{100 * np.mean([r['accuracy'] for r in same]):.2f} |")
    with open(os.path.join(OUT, "bitfin_wisdm_groups.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    import argparse

    import tinyhar_reproduce_wisdm as trw

    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=list(RECIPES))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--groups", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not args.report:
        for gi in args.groups:
            res = fold(args.protocol, trw.GROUPS[gi], args.seed)
            with open(os.path.join(OUT, f"{args.protocol}_group{gi}_seed{args.seed}.json"), "w") as fh:
                json.dump(res, fh)
            print(f"{args.protocol} group {gi}: macro F1 {res['macro_f1']:.4f}, {res['epochs']} epochs", flush=True)
    report()
