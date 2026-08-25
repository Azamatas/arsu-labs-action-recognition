# Report on An improved human activity recognition technique based on convolutional neural network paper


Overall, we achieved **98.13%** accuracy on the same WISDM dataset, which is higher than papers stated **97.20%** accuracy.

Only three mistakes appeared on confusion matrix in between these classes:

    Downstairs → Upstairs,
    Upstairs → Walking,
    Walking → Upstairs.


---
**Notes:**
- Model successfully reproduced.
- Minor bugs in code due to depreciated functions and older versions of libraries.

---
# Results for UniMiB 50Hz:

| Metric           | Value                                                                 |
|------------------|-----------------------------------------------------------------------|
| **Precision**    | [0.75, 0.71, 0.55, 0.61, 0.61, 0.70, 0.83, 1.00, 0.80, 0.91, 0.72, 1.00, 0.84, 0.83, 0.84, 0.54, 0.97] |
| **Recall**       | [0.68, 0.90, 0.77, 0.55, 0.57, 0.61, 1.00, 0.84, 0.53, 0.97, 0.84, 1.00, 0.84, 0.80, 0.87, 0.45, 0.90] |
| **Accuracy**     | 0.7716                                                                |
| **Hamming Loss** | 0.2284                                                                |

#### Per-class Precision and Recall

| Class | Precision | Recall |
|-------|-----------|--------|
| 1     | 0.75      | 0.68   |
| 2     | 0.71      | 0.90   |
| 3     | 0.55      | 0.77   |
| 4     | 0.61      | 0.55   |
| 5     | 0.61      | 0.57   |
| 6     | 0.70      | 0.61   |
| 7     | 0.83      | 1.00   |
| 8     | 1.00      | 0.84   |
| 9     | 0.80      | 0.53   |
| 10    | 0.91      | 0.97   |
| 11    | 0.72      | 0.84   |
| 12    | 1.00      | 1.00   |
| 13    | 0.84      | 0.84   |
| 14    | 0.83      | 0.80   |
| 15    | 0.84      | 0.87   |
| 16    | 0.54      | 0.45   |
| 17    | 0.97      | 0.90   |

**Overall Accuracy:** `0.7716`  
**Hamming Loss:** `0.2284`

---
## UniMiB testing

Accuracy: 0.1524798927613941

Macro F1: 0.13127900559404534
              precision    recall  f1-score   support

  Downstairs       0.22      0.10      0.13      1324
     Jogging       0.00      0.00      0.00      1985
    Upstairs       0.15      0.56      0.23       921
     Walking       0.17      0.15      0.16      1738

   micro avg       0.16      0.15      0.16      5968
   macro avg       0.13      0.20      0.13      5968
weighted avg       0.12      0.15      0.11      5968