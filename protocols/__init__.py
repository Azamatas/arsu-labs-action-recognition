"""Evaluation protocols: which windows train and test, and the training recipe. Each module has NAME, add_args(parser)
and run(spec, args, seed); the model comes in as a spec from protocols.models."""
from . import cross_dataset, daghar_lodo, scarcity, within_dataset

PROTOCOLS = {m.NAME: m for m in (cross_dataset, within_dataset, daghar_lodo, scarcity)}
