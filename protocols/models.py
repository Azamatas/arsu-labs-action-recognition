"""How each model is prepared for a protocol: input preprocessing, construction, FIN pretraining, learning rate.

    spec = make("bitfin", features="acc7")         # or bifin, tinyhar, tinierhar, cnn_ignatov, knn, logreg, ...
    spec.setup(xtr, ctx)                            # statistics of the training windows
    model = spec.build(channels, length, n_classes, ctx, seed)   # pretrains and freezes the FIN if configured
    spec.inputs(x)                                  # tuple of arrays the model takes

Neural specs return torch modules trained by bifin.train; classical specs (spec.classical = True) are fitted by the
protocol with baseline.classical.common.fit_predict.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from bifin import config, cost, features, model as fin_model, spectral, targets, train
from baseline import tinierhar, tinyhar
from baseline.classical import MODELS as CLASSICAL, common as classical_common
from baseline import cnn_ignatov

NEURAL = ("bifin", "bitfin", "tinyhar", "tinierhar", "cnn_ignatov")


@dataclass
class Context:
    """What the protocol tells the model: the autocorrelation lag range of the imitation targets and which single
    axes the FIN is pretrained on (default: every channel of every training window)."""
    max_lag: int
    pretrain_axes: Callable = field(default=lambda xtr, seed, spec: xtr.reshape(-1, xtr.shape[-1]))


class FINSpec:
    classical = False

    def __init__(self, cfg, family):
        self.cfg, self.family = cfg, family
        self.lr = None
        self.info = {}

    def id(self):
        return f"bifin-{self.cfg.name()}"  # by configuration, whichever preset it started from

    def setup(self, xtr, ctx):
        self.std = features.Standardiser(features.compute(xtr, self.cfg.features))

    def inputs(self, x):
        return x.astype(np.float32), self.std(features.compute(x, self.cfg.features))

    def build(self, channels, length, n_classes, ctx, seed, xtr=None):
        train.seed_all(seed)
        m = fin_model.build(self.cfg, channels, n_classes).to(train.DEV)
        self.frozen = ()
        if self.cfg.pretrained:
            axes = ctx.pretrain_axes(xtr, seed, self)
            self.info["imitation_mse"] = train.pretrain_fin(m.fin, axes, ctx.max_lag, seed)
            train.freeze(m.fin)
            self.frozen = (m.fin,)
            if self.cfg.circular:
                sample = axes[np.random.default_rng(seed).choice(len(axes), min(5000, len(axes)), replace=False)]
                self.info["spectral_error"] = spectral.max_relative_error(m.fin, sample)
        train.seed_all(seed)  # the head starts from the same initial weights whatever the pretraining did
        fin_model.glorot_(m.hidden), fin_model.glorot_(m.out)
        p_conv, f_conv = cost.conv_form(self.cfg, channels, length, n_classes)
        self.info.update(params=p_conv, mflops=f_conv / 1e6)
        if self.cfg.circular:
            p_spec, f_spec = cost.spectral_form(self.cfg, channels, length, n_classes)
            self.info.update(params_spectral=p_spec, mflops_spectral=f_spec / 1e6)
        return m

    def config(self):
        return self.cfg.as_dict()


class TinySpec:
    """TinyHAR (unpadded convolutions, as the authors' code) or TinierHAR on the per-channel standardised signal."""
    classical = False

    def __init__(self, family):
        self.family, self.lr, self.info, self.frozen = family, None, {}, ()

    def id(self):
        return self.family

    def setup(self, xtr, ctx):
        self.mu = xtr.mean(axis=(0, 2), keepdims=True)
        self.sd = xtr.std(axis=(0, 2), keepdims=True) + 1e-8

    def inputs(self, x):
        return (((x - self.mu) / self.sd).astype(np.float32),)

    def build(self, channels, length, n_classes, ctx, seed, xtr=None):
        train.seed_all(seed)
        m = (tinyhar.TinyHAR(channels, length, n_classes, pad=0) if self.family == "tinyhar"
             else tinierhar.TinierHAR(channels, length, n_classes))
        self.info["params"] = cost.count(m)
        return m.to(train.DEV)

    def config(self):
        return {"pad": 0} if self.family == "tinyhar" else {"m": 4, "n": 16}


class CNNSpec:
    """Ignatov's CNN with the 40 statistical features ('acc40') or the 9 imitation targets of the FIN ('imitated9')."""
    classical = False

    def __init__(self, feats="acc40"):
        self.feats, self.lr, self.info, self.frozen = feats, cnn_ignatov.LR, {}, ()

    def id(self):
        return f"cnn_ignatov-{self.feats}"

    def _features(self, x):
        if self.feats == "imitated9":
            return np.concatenate([targets.imitated(x[:, a], self.max_lag) for a in range(3)], axis=1)
        return features.compute(x, self.feats)

    def setup(self, xtr, ctx):
        self.max_lag = ctx.max_lag
        self.std = features.Standardiser(self._features(xtr))

    def inputs(self, x):
        return cnn_ignatov.centred(x).astype(np.float32), self.std(self._features(x))

    def build(self, channels, length, n_classes, ctx, seed, xtr=None):
        train.seed_all(seed)
        n_feat = 9 if self.feats == "imitated9" else features.size(self.feats)
        m = cnn_ignatov.IgnatovCNN(channels, length, n_feat, n_classes)
        self.info["params"] = cost.count(m)
        return m.to(train.DEV)

    def config(self):
        return {"features": self.feats}


class ClassicalSpec:
    classical = True

    def __init__(self, family, feats):
        self.family, self.feats, self.module, self.info = family, feats, CLASSICAL[family], {}

    def id(self):
        return f"{self.family}-{self.feats}"

    def features(self, x, ctx):
        return classical_common.compute(x, self.feats, ctx.max_lag)

    def fit_predict(self, xtr, ytr, tests, seed, ctx):
        """tests: {name: windows}. Returns {name: predictions}."""
        f = {k: self.features(v, ctx) for k, v in tests.items()}
        return classical_common.fit_predict(self.module.make, self.module.scale(self.feats), self.features(xtr, ctx),
                                            ytr, f, seed, shuffle=self.module.SHUFFLE)

    def config(self):
        return {"features": self.feats}


def make(name, **opts):
    """opts: ModelConfig fields for bifin / bitfin (None = preset value); features for cnn_ignatov and classical."""
    if name in config.PRESETS:
        return FINSpec(config.preset(name, **opts), name)
    if name in ("tinyhar", "tinierhar"):
        return TinySpec(name)
    if name == "cnn_ignatov":
        return CNNSpec(opts.get("features") or "acc40")
    if name in CLASSICAL:
        return ClassicalSpec(name, opts.get("features") or "statistical")
    raise ValueError(name)
