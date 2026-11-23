"""Random forest, 100 trees, unscaled features."""
from sklearn.ensemble import RandomForestClassifier

SHUFFLE = False  # the seed already drives the forest


def make(seed):
    return RandomForestClassifier(n_estimators=100, random_state=seed)


def scale(features):
    return False
