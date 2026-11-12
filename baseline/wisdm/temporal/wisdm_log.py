import os
import sys
import numpy as np
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
sys.path[:0] = [FIN_DIR, os.path.join(FIN_DIR, "data-processing")]

import wisdm_windowing as ww
from config import ACTIVITIES, TEMPORAL
from wisdm_windowing import temporal_axis

# Reading the data

print('loading the dataset')

data = ww.load_splits()
windows_train = np.concatenate([data["train"][0], data["val"][0]])  # WISDM users 1-26
labels_train = np.concatenate([data["train"][1], data["val"][1]])
windows_test, labels_test = data["test"]                             # WISDM users 27-36

# Features: the 33 temporal features (11 per axis)

def features(windows):
    axes = [temporal_axis(windows[:, a].astype(np.float64)) for a in range(3)]
    return np.stack([axes[a][f] for f in TEMPORAL for a in range(3)], axis=1)


data_train = features(windows_train)
data_test = features(windows_test)
print(str(len(data_train)) + " training windows, loading done, " + str(data_train.shape[1]) + " features")

# Classification

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42  # optional: python script.py <seed>

# the model is deterministic, so the seed shuffles the order of the training windows
order = np.random.default_rng(seed).permutation(len(labels_train))
data_train, labels_train = data_train[order], labels_train[order]

scaler = StandardScaler()
data_train = scaler.fit_transform(data_train)
data_test = scaler.transform(data_test)

clf = LogisticRegression(max_iter=1000, random_state=seed)
clf.fit(data_train, labels_train)

predictions = clf.predict(data_test)

print('accuracy:' + str(np.sum(predictions == labels_test)/predictions.shape[0]))
print(classification_report(labels_test, predictions, target_names=ACTIVITIES, digits = 4))
