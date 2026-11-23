"""TinierHAR (baseline/tinierhar/tinierhar.py, M = 4, N = 16) in the DAGHAR leave-one-dataset-out protocol, exactly as
tinyhar_daghar.py runs TinyHAR: accelerometer + gyroscope 6 x 60, train + validation splits of the five sources,
batches of 60 (12 per source), 5000 steps, Adam lr 1e-3 and 1e-4 chosen on the sources' test splits, best checkpoint,
evaluation on all splits of the target. 7,285 parameters, 150.6k multiply-adds (0.30 MFLOPs) per window.

    python daghar_benchmark/tinierhar_daghar.py --seeds 0      # results_tinierhar/<fold>_seed<s>.json, tinierhar_daghar.md
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "baseline", "tinierhar")]

import tinyhar_daghar as td  # noqa: E402
from tinierhar import TinierHAR  # noqa: E402

OUT = os.path.join(HERE, "results_tinierhar")


def report():
    td.report()
    src = os.path.join(OUT, "tinyhar_daghar.md")
    text = open(src, encoding="utf-8").read()
    os.remove(src)
    text = text.replace("# TinyHAR,", "# TinierHAR,").replace("`daghar_benchmark/tinyhar_daghar.py`",
                                                               "`daghar_benchmark/tinierhar_daghar.py`")
    with open(os.path.join(OUT, "tinierhar_daghar.md"), "w", encoding="utf-8") as fh:
        fh.write(text)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--folds", nargs="+", choices=td.DOMAINS, default=td.DOMAINS)
    args = parser.parse_args()
    td.TinyHAR = lambda channels, length, n_classes, pad=0: TinierHAR(channels, length, n_classes)
    td.OUT = OUT
    os.makedirs(OUT, exist_ok=True)
    for seed in args.seeds:
        for target in args.folds:
            r = td.fold(target, seed)
            with open(os.path.join(OUT, f"{target}_seed{seed}.json"), "w", encoding="utf-8") as fh:
                json.dump(r, fh, indent=1)
            print(f"seed {seed}, target {target}: {100 * r['target_acc']:.1f} %", flush=True)
            report()
