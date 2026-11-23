"""Classical baselines, one module per model: knn, logreg, random_forest, xgb (each: make(seed), scale(features),
SHUFFLE). Shared code in common.py."""
from . import knn, logreg, random_forest, xgb

MODELS = {"knn": knn, "logreg": logreg, "random_forest": random_forest, "xgb": xgb}
