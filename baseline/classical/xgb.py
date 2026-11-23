"""XGBoost, 100 trees, unscaled features."""
from xgboost import XGBClassifier

SHUFFLE = True


def make(seed):
    return XGBClassifier(n_estimators=100, random_state=seed, eval_metric="mlogloss")


def scale(features):
    return False
