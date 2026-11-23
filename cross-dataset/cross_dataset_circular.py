"""The paper's cross-dataset experiment (cross_dataset.py, Table 4) for the pretrained FIN + 40 only, with CIRCULAR
convolutions in the FIN (bilin_fin_circular.py), so that the trained FIN is exactly a linear map of the power
spectrum. Everything else (windows, splits, flip augmentation, pretraining, epochs, seeds) as in cross_dataset.py.

    python cross-dataset/cross_dataset_circular.py w2m      # results_circular/w2m.npz and .md
    python cross-dataset/cross_dataset_circular.py m2w
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

import cross_dataset as cd  # noqa: E402  (sets the keras backend)

import bilin_fin_temporal_plus_original as bfin  # noqa: E402
from bilin_fin_circular import circular  # noqa: E402

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("direction", choices=list(cd.DIRECTIONS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(cd.cfg.SEEDS))
    args = parser.parse_args()
    bfin.RhythmFIN = circular(bfin.RhythmFIN)
    cd.OUT = os.path.join(HERE, "results_circular")
    cd.MODELS = {"fin_frozen_dropout": ("FIN + 40, pretrained, circular", cd.MODELS["fin_frozen_dropout"][1])}
    path = cd.train(args.direction, args.seeds, ["fin_frozen_dropout"])
    cd.report(args.direction, path)
