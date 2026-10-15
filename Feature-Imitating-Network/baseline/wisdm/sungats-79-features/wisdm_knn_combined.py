import os
import numpy as np
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Reading the data

print('loading the dataset')

length = 200

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))))
DATA_DIR = os.path.join(REPO_DIR, "reproduction of papers", "ClassicML", "data_processing", "wisdm_data")

f1 = open(os.path.join(DATA_DIR, "combined_features_" + str(length) + ".csv"))
f3 = open(os.path.join(DATA_DIR, "combined_features_test_" + str(length) + ".csv"))
f2 = open(os.path.join(DATA_DIR, "answers_" + str(length) + ".csv"))
f4 = open(os.path.join(DATA_DIR, "answers_test_" + str(length) + ".csv"))

data_train = np.loadtxt(fname = f1, delimiter = ',')
labels_train = np.loadtxt(fname = f2, delimiter = ',')
data_test = np.loadtxt(fname = f3, delimiter = ',')
labels_test = np.loadtxt(fname = f4, delimiter = ',')

f1.close(); f2.close(); f3.close(); f4.close()
print(str(length) + ", loading done, " + str(data_train.shape[1]) + " features")

# Classification

scaler = StandardScaler()
data_train = scaler.fit_transform(data_train)
data_test = scaler.transform(data_test)

knn = KNeighborsClassifier(n_neighbors=1, p=1)
knn.fit(data_train, labels_train)

predictions = knn.predict(data_test)

print('accuracy:' + str(np.sum(predictions == labels_test)/predictions.shape[0]))
print(classification_report(labels_test, predictions, digits = 4))
