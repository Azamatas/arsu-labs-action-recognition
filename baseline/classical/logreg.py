"""Logistic regression on standardised features."""
from sklearn.linear_model import LogisticRegression

SHUFFLE = True


def make(seed):
    return LogisticRegression(max_iter=1000, random_state=seed)


def scale(features):
    return True
