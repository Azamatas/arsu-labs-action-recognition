import os
import sys
import numpy as np
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))))
sys.path[:0] = [FIN_DIR, os.path.join(FIN_DIR, "data-processing")]

import motionsense_windowing as ms
import wisdm_windowing as ww
from config import ACTIVITIES, TEMPORAL
from wisdm_windowing import temporal_axis

# Reading the data

print('loading the dataset')

windows_train, _, labels_train = ms.window_raw()                     # all 24 MotionSense users
windows_test, labels_test = ww.load_splits()["test"]                 # WISDM users 27-36

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

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42  # optional: python script.py <seed>

clf = XGBClassifier(n_estimators=100, random_state=seed, eval_metric='mlogloss')
clf.fit(data_train, labels_train)

predictions = clf.predict(data_test)

print('WISDM users 27-36 accuracy:' + str(np.sum(predictions == labels_test)/predictions.shape[0]))
print(classification_report(labels_test, predictions, target_names=ACTIVITIES, digits = 4))
