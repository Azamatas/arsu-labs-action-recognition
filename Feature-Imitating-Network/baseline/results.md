# Classic baselines

Every script trains on one dataset and tests on one dataset. Windows are 10 s (200 samples at 20 Hz) with 50 % overlap. The flip in the cross-dataset runs adds every training window once more rotated 180 degrees about y (x and z change sign), the phone facing the other way in the pocket.

Features: **40 statistical** = Ignatov's (mean, std and mean absolute deviation per axis, average resultant acceleration, 10-bin histogram per axis); **79 combined** = the 40 plus 13 more statistics per axis; **33 temporal** = the 11 features of `config.TEMPORAL` on each axis (mean-crossing rate, autocorrelation peak and lag, dominant frequency, spectral peak and entropy, jerk std and mad, power in 0-3, 3-6 and 6-10 Hz).

Models: random forest and XGBoost with 100 trees, mean ± standard deviation over seeds 42, 0, 1, 2, 3 (`python <script> <seed>`); 1-NN (Manhattan) and logistic regression have no randomness and run once. 1-NN and logistic regression are standardised, except `wisdm/statistical/wisdm_knn.py`, which (as in the original ClassicML script) runs 1-NN on the raw y-axis signal instead of the features. Activity columns are per-class F1, mean over seeds. `wisdm/temporal/wisdm_random_forest_temporal.py` is a separate experiment (several feature sets, LOSO) and is not in these tables.

## WISDM -> WISDM

Ignatov's own split and windows (`ClassicML/data_processing/wisdm_data/*_200.csv`): trained on WISDM users 1-26 (7,367 windows), tested on users 27-36 (3,026 windows).

| features | model | script | accuracy | macro F1 | Jogging | Walking | Upstairs | Downstairs | Sitting | Standing |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 statistical | 1-NN | `statistical/wisdm_knn.py` | 0.6844 | 0.4819 | 0.8606 | 0.8667 | 0.1915 | 0.0528 | 0.4982 | 0.4213 |
| 40 statistical | Logistic regression | `statistical/wisdm_log.py` | 0.8063 | 0.7584 | 0.8653 | 0.8502 | 0.4990 | 0.6200 | 0.8659 | 0.8499 |
| 40 statistical | Random forest | `statistical/wisdm_random_forest.py` | 0.8273 ± 0.0018 | 0.7948 ± 0.0020 | 0.8807 | 0.8584 | 0.6395 | 0.6049 | 0.8934 | 0.8917 |
| 40 statistical | XGBoost | `statistical/wisdm_xgb.py` | 0.8410 ± 0.0000 | 0.8070 ± 0.0000 | 0.8803 | 0.8824 | 0.6172 | 0.6986 | 0.8771 | 0.8864 |
| 79 combined | 1-NN | `sungats-79-features/wisdm_knn_combined.py` | 0.8110 | 0.7823 | 0.8731 | 0.8623 | 0.5424 | 0.6281 | 0.8966 | 0.8914 |
| 79 combined | Logistic regression | `sungats-79-features/wisdm_log_combined.py` | 0.8328 | 0.8056 | 0.8884 | 0.8596 | 0.6032 | 0.6679 | 0.9096 | 0.9048 |
| 79 combined | Random forest | `sungats-79-features/wisdm_random_forest_combined.py` | 0.8371 ± 0.0041 | 0.8048 ± 0.0045 | 0.8931 | 0.8610 | 0.6645 | 0.6496 | 0.8753 | 0.8854 |
| 79 combined | XGBoost | `sungats-79-features/wisdm_xgb_combined.py` | 0.8328 ± 0.0000 | 0.7899 ± 0.0000 | 0.8898 | 0.8638 | 0.6160 | 0.7090 | 0.7935 | 0.8672 |
| 33 temporal | 1-NN | `temporal/wisdm_knn.py` | 0.8549 | 0.7815 | 0.9191 | 0.9158 | 0.6523 | 0.7313 | 0.7546 | 0.7160 |
| 33 temporal | Logistic regression | `temporal/wisdm_log.py` | 0.8698 | 0.8185 | 0.9127 | 0.9247 | 0.7147 | 0.7400 | 0.8299 | 0.7890 |
| 33 temporal | Random forest | `temporal/wisdm_random_forest.py` | 0.8922 ± 0.0025 | 0.8382 ± 0.0049 | 0.9372 | 0.9383 | 0.7348 | 0.8302 | 0.8156 | 0.7731 |
| 33 temporal | XGBoost | `temporal/wisdm_xgb.py` | **0.8929 ± 0.0000** | 0.8396 ± 0.0000 | 0.9381 | 0.9385 | 0.7310 | 0.8383 | 0.8168 | 0.7750 |

## MotionSense -> MotionSense

MotionSense windows of `data-processing/motionsense_windowing.py`: trained on users 1-17, tested on users 18-24 (1,551 windows). No flip.

| features | model | script | accuracy | macro F1 | Jogging | Walking | Upstairs | Downstairs | Sitting | Standing |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 statistical | 1-NN | `statistical/motionsense_knn.py` | 0.8266 | 0.7949 | 0.9606 | 0.7983 | 0.5977 | 0.5462 | 0.9304 | 0.9360 |
| 40 statistical | Logistic regression | `statistical/motionsense_log.py` | 0.8549 | 0.8128 | 0.9718 | 0.8712 | 0.5799 | 0.5328 | 0.9580 | 0.9630 |
| 40 statistical | Random forest | `statistical/motionsense_random_forest.py` | 0.8576 ± 0.0096 | 0.8348 ± 0.0107 | 0.9772 | 0.7883 | 0.6188 | 0.6713 | 0.9754 | 0.9779 |
| 40 statistical | XGBoost | `statistical/motionsense_xgb.py` | 0.8407 ± 0.0000 | 0.8066 ± 0.0000 | 0.9793 | 0.8259 | 0.5813 | 0.5517 | 0.9467 | 0.9544 |
| 33 temporal | 1-NN | `temporal/motionsense_knn.py` | 0.8846 | 0.8645 | 0.9004 | 0.9323 | 0.8122 | 0.7478 | 0.8906 | 0.9036 |
| 33 temporal | Logistic regression | `temporal/motionsense_log.py` | **0.9323** | 0.9113 | 0.9790 | 0.9770 | 0.8232 | 0.7897 | 0.9582 | 0.9410 |
| 33 temporal | Random forest | `temporal/motionsense_random_forest.py` | 0.9250 ± 0.0048 | 0.9028 ± 0.0056 | 0.9642 | 0.9632 | 0.8190 | 0.7652 | 0.9521 | 0.9530 |
| 33 temporal | XGBoost | `temporal/motionsense_xgb.py` | 0.9207 ± 0.0000 | 0.9028 ± 0.0000 | 0.9756 | 0.9726 | 0.8372 | 0.7745 | 0.9255 | 0.9313 |

## WISDM -> MotionSense

Trained on WISDM users 1-26 (7,350 windows of `data-processing/wisdm_windowing.py`) plus each of them flipped (14,700), tested on all 24 MotionSense users (5,193 windows).

| features | model | script | accuracy | macro F1 | Jogging | Walking | Upstairs | Downstairs | Sitting | Standing |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 statistical | 1-NN | `statistical/w2m_knn.py` | 0.7400 | 0.7087 | 0.9774 | 0.7740 | 0.4484 | 0.4332 | 0.7993 | 0.8198 |
| 40 statistical | Logistic regression | `statistical/w2m_log.py` | 0.8484 | 0.7851 | 0.9707 | 0.8747 | 0.5331 | 0.4587 | 0.9359 | 0.9371 |
| 40 statistical | Random forest | `statistical/w2m_random_forest.py` | 0.8647 ± 0.0016 | 0.8108 ± 0.0011 | 0.9583 | 0.8784 | 0.5916 | 0.5185 | 0.9596 | 0.9583 |
| 40 statistical | XGBoost | `statistical/w2m_xgb.py` | **0.8704 ± 0.0000** | 0.8266 ± 0.0000 | 0.9738 | 0.8969 | 0.6168 | 0.5910 | 0.9400 | 0.9411 |
| 33 temporal | 1-NN | `temporal/w2m_knn.py` | 0.8265 | 0.8201 | 0.9733 | 0.9489 | 0.7602 | 0.7032 | 0.7671 | 0.7677 |
| 33 temporal | Logistic regression | `temporal/w2m_log.py` | 0.8683 | 0.8694 | 0.9358 | 0.9483 | 0.8492 | 0.8411 | 0.8202 | 0.8222 |
| 33 temporal | Random forest | `temporal/w2m_random_forest.py` | 0.8449 ± 0.0042 | 0.8403 ± 0.0048 | 0.8058 | 0.8703 | 0.8452 | 0.8457 | 0.8636 | 0.8111 |
| 33 temporal | XGBoost | `temporal/w2m_xgb.py` | 0.8535 ± 0.0000 | 0.8480 ± 0.0000 | 0.8330 | 0.8971 | 0.8436 | 0.8375 | 0.8592 | 0.8178 |

## MotionSense -> WISDM

Trained on all 24 MotionSense users (5,193 windows) plus each of them flipped (10,386), tested on WISDM users 27-36 (3,017 windows of `data-processing/wisdm_windowing.py`).

| features | model | script | accuracy | macro F1 | Jogging | Walking | Upstairs | Downstairs | Sitting | Standing |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 statistical | 1-NN | `statistical/m2w_knn.py` | 0.7149 | 0.6985 | 0.8361 | 0.7500 | 0.4587 | 0.4676 | 0.8323 | 0.8462 |
| 40 statistical | Logistic regression | `statistical/m2w_log.py` | 0.8127 | 0.7655 | 0.8617 | 0.8959 | 0.4992 | 0.5167 | 0.9153 | 0.9043 |
| 40 statistical | Random forest | `statistical/m2w_random_forest.py` | 0.8059 ± 0.0045 | 0.7774 ± 0.0040 | 0.8740 | 0.8491 | 0.5782 | 0.5698 | 0.9036 | 0.8896 |
| 40 statistical | XGBoost | `statistical/m2w_xgb.py` | 0.7971 ± 0.0000 | 0.7593 ± 0.0000 | 0.8922 | 0.8242 | 0.5825 | 0.5714 | 0.7912 | 0.8943 |
| 33 temporal | 1-NN | `temporal/m2w_knn.py` | 0.7696 | 0.7237 | 0.8717 | 0.8527 | 0.5044 | 0.5665 | 0.7927 | 0.7539 |
| 33 temporal | Logistic regression | `temporal/m2w_log.py` | 0.8250 | 0.7890 | 0.8666 | 0.8963 | 0.6176 | 0.6631 | 0.8705 | 0.8202 |
| 33 temporal | Random forest | `temporal/m2w_random_forest.py` | **0.8335 ± 0.0082** | 0.7933 ± 0.0078 | 0.8540 | 0.9477 | 0.6255 | 0.5910 | 0.8822 | 0.8590 |
| 33 temporal | XGBoost | `temporal/m2w_xgb.py` | 0.8333 ± 0.0000 | 0.7925 ± 0.0000 | 0.8918 | 0.9297 | 0.5879 | 0.5961 | 0.8913 | 0.8580 |
