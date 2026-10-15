import os
import sys
import numpy as np
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))))
sys.path[:0] = [FIN_DIR, os.path.join(FIN_DIR, "data-processing")]

import motionsense_windowing as ms
import wisdm_windowing as ww
import statistical_features as ignatov
from config import ACTIVITIES

# Reading the data

print('loading the dataset')

windows_train, _, labels_train = ms.window_raw()                     # all 24 MotionSense users
windows_test, labels_test = ww.load_splits()["test"]                 # WISDM users 27-36

# Flip: every training window once more, rotated 180 degrees about y (x and z change sign)

windows_train = np.concatenate([windows_train, windows_train * np.array([-1, 1, -1])[:, None]])
labels_train = np.concatenate([labels_train, labels_train])

# Features: Ignatov's 40 statistical features

def features(windows):
    return np.stack([ignatov.extract_basic_features(*w) for w in windows.astype(np.float64)])


data_train = features(windows_train)
data_test = features(windows_test)
print(str(len(data_train)) + " training windows, loading done, " + str(data_train.shape[1]) + " features")

# Classification

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42  # optional: python script.py <seed>

clf = RandomForestClassifier(n_estimators=100, random_state=seed)
clf.fit(data_train, labels_train)

predictions = clf.predict(data_test)

print('WISDM users 27-36 accuracy:' + str(np.sum(predictions == labels_test)/predictions.shape[0]))
print(classification_report(labels_test, predictions, target_names=ACTIVITIES, digits = 4))
