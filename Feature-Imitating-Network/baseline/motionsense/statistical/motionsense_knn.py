import os
import sys
import numpy as np
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
sys.path[:0] = [FIN_DIR, os.path.join(FIN_DIR, "data-processing")]

import motionsense_windowing as ms
import statistical_features as ignatov
from config import ACTIVITIES

# Reading the data

print('loading the dataset')

windows, users, labels = ms.window_raw()
train = users <= 17
windows_train, labels_train = windows[train], labels[train]   # MotionSense users 1-17
windows_test, labels_test = windows[~train], labels[~train]   # MotionSense users 18-24

# Features: Ignatov's 40 statistical features

def features(windows):
    return np.stack([ignatov.extract_basic_features(*w) for w in windows.astype(np.float64)])


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

print('MotionSense users 18-24 accuracy:' + str(np.sum(predictions == labels_test)/predictions.shape[0]))
print(classification_report(labels_test, predictions, target_names=ACTIVITIES, digits = 4))
