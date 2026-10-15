import os
import sys
import numpy as np
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))))
sys.path[:0] = [FIN_DIR, os.path.join(FIN_DIR, "data-processing")]

import motionsense_windowing as ms
import wisdm_windowing as ww
from config import ACTIVITIES, TEMPORAL
from wisdm_windowing import temporal_axis

# Reading the data

print('loading the dataset')

data = ww.load_splits()
windows_train = np.concatenate([data["train"][0], data["val"][0]])  # WISDM users 1-26
labels_train = np.concatenate([data["train"][1], data["val"][1]])
windows_test, _, labels_test = ms.window_raw()                       # all 24 MotionSense users

# Flip: every training window once more, rotated 180 degrees about y (x and z change sign)

windows_train = np.concatenate([windows_train, windows_train * np.array([-1, 1, -1])[:, None]])
labels_train = np.concatenate([labels_train, labels_train])

# Features: the 33 temporal features (11 per axis)

def features(windows):
    axes = [temporal_axis(windows[:, a].astype(np.float64)) for a in range(3)]
    return np.stack([axes[a][f] for f in TEMPORAL for a in range(3)], axis=1)


data_train = features(windows_train)
data_test = features(windows_test)
print(str(len(data_train)) + " training windows, loading done, " + str(data_train.shape[1]) + " features")

# Classification

scaler = StandardScaler()
data_train = scaler.fit_transform(data_train)
data_test = scaler.transform(data_test)

clf = KNeighborsClassifier(n_neighbors=1, p=1)
clf.fit(data_train, labels_train)

predictions = clf.predict(data_test)

print('MotionSense accuracy:' + str(np.sum(predictions == labels_test)/predictions.shape[0]))
print(classification_report(labels_test, predictions, target_names=ACTIVITIES, digits = 4))
