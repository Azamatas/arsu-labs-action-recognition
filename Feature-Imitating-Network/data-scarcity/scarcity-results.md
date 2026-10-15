# Data-scarcity results

The classifier is the same everywhere unless noted: FIN embedding of each axis -> Dense 128 -> softmax, no
hand-made features, 60 epochs, batch 256. "Frozen" = the pretrained FIN stays fixed and only the head learns;
the random-init FIN trains together with the head. Test accuracy after the last epoch.

## 7 training users, 5 seeds — `data_scarcity_real_warped.ipynb`

Training users 12, 18, 19, 20, 31, 34, 36; test on the other 29 WISDM users (7,710 windows). Seeds 42, 0, 1, 2, 3:
the seed sets the warped windows, the FIN pretraining and the head training. No flip.

| FIN pretraining | accuracy | macro F1 | Jogging | Walking | Upstairs | Downstairs | Sitting | Standing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| real + warped windows of the 7 users, frozen | **0.8291 ± 0.0131** | **0.7148 ± 0.0071** | 0.840 | 0.951 | 0.540 | 0.584 | 0.845 | 0.679 |
| MotionSense + real + warped, frozen | 0.8273 ± 0.0149 | 0.7090 ± 0.0135 | 0.842 | 0.954 | 0.524 | 0.556 | 0.860 | 0.673 |
| none (random-init, trainable) | 0.7993 ± 0.0132 | 0.6935 ± 0.0135 | 0.842 | 0.860 | 0.557 | 0.672 | 0.779 | 0.723 |

Activity columns are recall, mean over seeds.

Accuracy per seed:

| seed | real + warped | MotionSense + real + warped | random-init |
|---:|---:|---:|---:|
| 42 | 0.8450 | 0.8460 | 0.8058 |
| 0 | 0.8243 | 0.8271 | 0.7820 |
| 1 | 0.8257 | 0.8125 | 0.8062 |
| 2 | 0.8388 | 0.8130 | 0.8134 |
| 3 | 0.8115 | 0.8377 | 0.7891 |

Imitation R² of the pretrained FIN on the 29 unseen users' real windows:

| FIN pretraining | spectral entropy | AC peak | spectral peak |
|---|---:|---:|---:|
| real + warped | 0.9096 ± 0.0078 | 0.9249 ± 0.0042 | 0.8328 ± 0.0097 |
| MotionSense + real + warped | 0.9342 ± 0.0050 | 0.9411 ± 0.0017 | 0.8799 ± 0.0110 |

Warped windows: a real single-axis window at a random tempo (0.75-1.35x), randomly flipped around its mean and/or
reversed in time, with a little Gaussian noise; 12,006 per seed (`data-processing/augmentation.py`).

## Earlier results

Different users, seeds or notebook versions than above, so compare only within a row.

| experiment | training users | seeds | FIN pretraining | pretrained FIN | random-init | notes |
|---|---|---:|---|---:|---:|---|
| `data_scarcity.ipynb` | 7 random (5 draws) | 5 | warped, frozen | 0.820 | 0.802 | pretrained and trainable: 0.814 |
| `data_scarcity.ipynb` | 2 random (5 draws) | 5 | warped, frozen | 0.673 | 0.630 | pretrained and trainable: 0.643 |
| 150 epochs (notebook removed) | 7 random (2 draws) | 2 | warped, frozen | 0.838 at epoch 11 | 0.833 best (epoch 27), 0.824 at 150 | how fast each model gets there |
| MotionSense only (earlier version of `data_scarcity_motionsense.ipynb`) | 2, 3, 9, 11, 16, 20, 25 | 1 | MotionSense, frozen | 0.843 | 0.797 | |
| MotionSense + 2 users (notebook removed) | 23, 29 | 1 | MotionSense + the 2 users' windows, frozen | 0.625 | 0.529 | random-init reaches 0.71 at epoch 7, then overfits |
| plain CNN (earlier version of `fin_vs_cnn.ipynb`) | 7 random (5 draws) | 5 | warped, frozen | 0.820 | — | plain CNN (Conv2D 32 -> Conv2D 64 -> Dense 128): 0.737 |
