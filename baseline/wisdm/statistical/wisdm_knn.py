import os
import sys
import numpy as np
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
sys.path[:0] = [FIN_DIR, os.path.join(FIN_DIR, "data-processing")]

import wisdm_windowing as ww
from config import ACTIVITIES

# Reading the data

print('loading the dataset')

data = ww.load_splits()
windows_train = np.concatenate([data["train"][0], data["val"][0]])  # WISDM users 1-26
labels_train = np.concatenate([data["train"][1], data["val"][1]])
windows_test, labels_test = data["test"]                             # WISDM users 27-36

# Features: the raw y axis of each window, as Ignatov's 1-NN

def features(windows):
    return windows[:, 1].astype(np.float64)


data_train = features(windows_train)
data_test = features(windows_test)
print(str(len(data_train)) + " training windows, loading done, " + str(data_train.shape[1]) + " features")

# Classification

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42  # optional: python script.py <seed>

# the model is deterministic, so the seed shuffles the order of the training windows
order = np.random.default_rng(seed).permutation(len(labels_train))
data_train, labels_train = data_train[order], labels_train[order]

knn = KNeighborsClassifier(n_neighbors=1, p=1)
knn.fit(data_train, labels_train)

predictions = knn.predict(data_test)

print('accuracy:' + str(np.sum(predictions == labels_test)/predictions.shape[0]))
print(classification_report(labels_test, predictions, target_names=ACTIVITIES, digits = 4))
