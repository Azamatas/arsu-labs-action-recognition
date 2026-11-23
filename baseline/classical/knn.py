"""1-nearest neighbour, Manhattan distance; features standardised except the raw signal (as Ignatov's 1-NN)."""
from sklearn.neighbors import KNeighborsClassifier

SHUFFLE = True


def make(seed):
    return KNeighborsClassifier(n_neighbors=1, p=1)


def scale(features):
    return features != "raw_y"
