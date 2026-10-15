import os
import sys
import numpy as np
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# Reading the data

print('loading the dataset')

length = 200

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))))
DATA_DIR = os.path.join(REPO_DIR, "reproduction of papers", "ClassicML", "data_processing", "wisdm_data")

f1 = open(os.path.join(DATA_DIR, "basic_features_" + str(length) + ".csv"))
f3 = open(os.path.join(DATA_DIR, "basic_features_test_" + str(length) + ".csv"))
f2 = open(os.path.join(DATA_DIR, "answers_" + str(length) + ".csv"))
f4 = open(os.path.join(DATA_DIR, "answers_test_" + str(length) + ".csv"))

data_train = np.loadtxt(fname = f1, delimiter = ',')
labels_train = np.loadtxt(fname = f2, delimiter = ',')
data_test = np.loadtxt(fname = f3, delimiter = ',')
labels_test = np.loadtxt(fname = f4, delimiter = ',')

f1.close(); f2.close(); f3.close(); f4.close()
print(str(length) + ", loading done")

# Classification

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42  # optional: python script.py <seed>

rf = RandomForestClassifier(n_estimators=100, random_state=seed)
rf.fit(data_train, labels_train)

predictions = rf.predict(data_test)

print('accuracy:' + str(np.sum(predictions == labels_test)/predictions.shape[0]))
print(classification_report(labels_test, predictions, digits = 4))

