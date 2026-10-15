import os
import numpy as np
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
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

with open(os.path.join(DATA_DIR, "combined_feature_names.txt")) as f:
    feature_names = f.read().splitlines()

# Classification

scaler = StandardScaler()
data_train = scaler.fit_transform(data_train)
data_test = scaler.transform(data_test)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(data_train, labels_train)

predictions = clf.predict(data_test)

print('accuracy:' + str(np.sum(predictions == labels_test)/predictions.shape[0]))
print(classification_report(labels_test, predictions, digits = 4))

importance = np.abs(clf.coef_).mean(axis=0)
top_idx = np.argsort(importance)[::-1][:15]
print('top 15 features by mean |coefficient| across classes:')
for i in top_idx:
    print(f'  {feature_names[i]}: {importance[i]:.4f}')
