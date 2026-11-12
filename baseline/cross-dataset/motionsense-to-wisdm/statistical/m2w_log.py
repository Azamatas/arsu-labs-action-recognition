import os
import sys
import numpy as np
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))))
sys.path[:0] = [FIN_DIR, os.path.join(FIN_DIR, "data-processing")]

import motionsense_windowing as ms
import wisdm_windowing as ww
import statistical_features as ignatov
from config import ACTIVITIES

# Reading the data

print('loading the dataset')

windows_ms, users_ms, labels_ms = ms.window_raw()
held_out = users_ms >= 19                                            # MotionSense users 19-24, the in-dataset test
windows_train, labels_train = windows_ms[~held_out], labels_ms[~held_out]   # MotionSense users 1-18
windows_ms_test, labels_ms_test = windows_ms[held_out], labels_ms[held_out]
windows_test, _, labels_test = ww.window_raw()                       # all 36 WISDM users (none used for training)

# Flip: every training window once more, rotated 180 degrees about y (x and z change sign)

windows_train = np.concatenate([windows_train, windows_train * np.array([-1, 1, -1])[:, None]])
labels_train = np.concatenate([labels_train, labels_train])

# Features: Ignatov's 40 statistical features

def features(windows):
    return np.stack([ignatov.extract_basic_features(*w) for w in windows.astype(np.float64)])


data_train = features(windows_train)
data_test = features(windows_test)
data_ms_test = features(windows_ms_test)
print(str(len(data_train)) + " training windows, loading done, " + str(data_train.shape[1]) + " features")

# Classification

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42  # optional: python script.py <seed>

# the model is deterministic, so the seed shuffles the order of the training windows
order = np.random.default_rng(seed).permutation(len(labels_train))
data_train, labels_train = data_train[order], labels_train[order]

scaler = StandardScaler()
data_train = scaler.fit_transform(data_train)
data_test = scaler.transform(data_test)
data_ms_test = scaler.transform(data_ms_test)

clf = LogisticRegression(max_iter=1000, random_state=seed)
clf.fit(data_train, labels_train)

predictions = clf.predict(data_test)

print('WISDM, all users accuracy:' + str(np.sum(predictions == labels_test)/predictions.shape[0]))
print(classification_report(labels_test, predictions, target_names=ACTIVITIES, digits = 4))

predictions = clf.predict(data_ms_test)

print('MotionSense users 19-24 accuracy:' + str(np.sum(predictions == labels_ms_test)/predictions.shape[0]))
print(classification_report(labels_ms_test, predictions, target_names=ACTIVITIES, digits = 4))
