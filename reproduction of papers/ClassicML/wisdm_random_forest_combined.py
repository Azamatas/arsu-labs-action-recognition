import os
import numpy as np
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# Reading the data

print('loading the dataset')

length = 200

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_processing", "wisdm_data")

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

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(data_train, labels_train)

predictions = rf.predict(data_test)

print('accuracy:' + str(np.sum(predictions == labels_test)/predictions.shape[0]))
print(classification_report(labels_test, predictions, digits = 4))

top_idx = np.argsort(rf.feature_importances_)[::-1][:15]
print('top 15 features by importance:')
for i in top_idx:
    print(f'  {feature_names[i]}: {rf.feature_importances_[i]:.4f}')
