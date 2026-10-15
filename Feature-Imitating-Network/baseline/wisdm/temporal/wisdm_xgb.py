import os
import sys
import numpy as np
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# Reading the data

print('loading the dataset')

length = 200

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
DATA_DIR = os.path.join(os.path.dirname(FIN_DIR), "reproduction of papers", "ClassicML", "data_processing", "wisdm_data")
sys.path[:0] = [FIN_DIR, os.path.join(FIN_DIR, "data-processing")]

from config import TEMPORAL
from wisdm_windowing import temporal_axis


def load(name):
    return np.loadtxt(os.path.join(DATA_DIR, name + "_" + str(length) + ".csv"), delimiter = ',')


def features(axes):
    """The 11 temporal features of config.TEMPORAL on each of the three axes: 33 per window."""
    per_axis = [temporal_axis(a) for a in axes]
    return np.stack([per_axis[a][f] for f in TEMPORAL for a in range(3)], axis=1)


data_train = features([load("data_x"), load("data_y"), load("data_z")])
data_test = features([load("data_x_test"), load("data_y_test"), load("data_z_test")])
labels_train = load("answers")
labels_test = load("answers_test")

print(str(length) + ", loading done, " + str(data_train.shape[1]) + " features")

# Classification

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42  # optional: python script.py <seed>

clf = XGBClassifier(n_estimators=100, random_state=seed, eval_metric='mlogloss')
clf.fit(data_train, labels_train - 1)

predictions = clf.predict(data_test) + 1

print('accuracy:' + str(np.sum(predictions == labels_test)/predictions.shape[0]))
print(classification_report(labels_test, predictions, digits = 4))
