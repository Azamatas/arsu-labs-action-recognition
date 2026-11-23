"""Run one model in one protocol for some seeds; results go to results/<protocol>/<variant>/<model id>/seed<s>.json.

    python run.py cross_dataset --model bifin --seeds 42 0 1 2 3
    python run.py cross_dataset --model bifin --random-init --seeds 42           # FIN trained with the head
    python run.py cross_dataset --model cnn_ignatov --direction m2w --rates --seeds 42
    python run.py daghar_lodo --model bitfin --features acc7 --seeds 0 1 2
    python run.py daghar_lodo --model bitfin --pooling mean --targets UCI WISDM
    python run.py scarcity --model bifin --features none --users 7 --seeds 0 1 2 3 4
    python run.py within_dataset --model knn --features raw_y --dataset wisdm

Models: bifin (the paper's: 64 filters, kernel 61, head 128), bitfin (circular, 32 filters, kernel 31, head 64),
tinyhar, tinierhar, cnn_ignatov, and the classical knn, logreg, random_forest, xgb. Size options (--width ...)
override the bifin / bitfin preset. --features: a statistical set for bifin / bitfin (acc40, acc10, acc7, acc4, none,
acc7_gyro7, acc10_gyro10), acc40 or imitated9 for cnn_ignatov, raw_y / statistical / temporal for the classical
models. Existing results are skipped unless --force.
"""
import argparse
import os

os.environ.setdefault("PYTHONHASHSEED", "0")

from protocols import PROTOCOLS  # noqa: E402
from protocols.models import make  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("protocol", choices=list(PROTOCOLS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--force", action="store_true", help="rerun even if the result exists")
    g = parser.add_argument_group("model options")
    g.add_argument("--features")
    g.add_argument("--width", type=int)
    g.add_argument("--kernel", type=int)
    g.add_argument("--emb", type=int)
    g.add_argument("--head", type=int)
    g.add_argument("--pooling", choices=["concat", "mean", "meanmax"])
    g.add_argument("--circular", action=argparse.BooleanOptionalAction, default=None)
    g.add_argument("--random-init", action="store_true", help="no FIN pretraining; the FIN is trained with the head")
    args, rest = parser.parse_known_args()
    protocol = PROTOCOLS[args.protocol]
    sub = argparse.ArgumentParser(prog=f"run.py {args.protocol}")
    protocol.add_args(sub)
    args = argparse.Namespace(**vars(args), **vars(sub.parse_args(rest)))

    opts = dict(features=args.features, width=args.width, kernel=args.kernel, emb=args.emb, head=args.head,
                pooling=args.pooling, circular=args.circular, pretrained=False if args.random_init else None)
    for seed in args.seeds:
        protocol.run(make(args.model, **opts), args, seed)


if __name__ == "__main__":
    main()
