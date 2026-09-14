# ARSU Labs Action Recognition — Results

## 1. WISDM — Classical ML (ClassicML)

6 classes: Jogging, Walking, Upstairs, Downstairs, Sitting, Standing. 200-sample windows (10s @ 20Hz),
50% overlap, split by subject (test users 27-36). 7,367 train / 3,026 test windows.

### Single seed (`random_state=42`)

| Model | 40 features | 79 features |
|---|---:|---:|
| Random Forest | 82.72% | 83.34% |
| Logistic Regression | 80.63% | 83.28% |
| XGBoost | 84.10% | 83.28% |
| KNN (1-NN, Manhattan) | 77.99% | 81.10% |

### 10-seed sweep (seeds 0-9)

| Feature set | Model | Mean | Std | Min | Max |
|---|---|---:|---:|---:|---:|
| 40 | RF | 82.87% | ±0.31% | 82.52% | 83.48% |
| 40 | LogReg | 80.63% | ±0.00% | 80.63% | 80.63% |
| 40 | XGBoost | 84.10% | ±0.00% | 84.10% | 84.10% |
| 40 | KNN | 77.99% | ±0.00% | 77.99% | 77.99% |
| 79 | RF | 83.90% | ±0.28% | 83.44% | 84.34% |
| 79 | LogReg | 83.28% | ±0.00% | 83.28% | 83.28% |
| 79 | XGBoost | 83.28% | ±0.00% | 83.28% | 83.28% |
| 79 | KNN | 81.10% | ±0.00% | 81.10% | 81.10% |

### Random Forest, 79 features — per-class F1

| Class | 40 features | 79 features |
|---|---:|---:|
| Jogging | 0.8814 | 0.8918 |
| Walking | 0.8623 | 0.8542 |
| Upstairs | 0.6540 | 0.6690 |
| Downstairs | 0.6017 | 0.6339 |
| Sitting | 0.8920 | 0.8771 |
| Standing | 0.8908 | 0.8876 |
| Weighted F1 | 0.8233 | 0.8296 |

---

## 2. WISDM — CNN (`cnn_wisdm.py`, ClassicML)

Ignatov's hybrid: a shallow conv branch over the raw window, concatenated with the handcrafted
features, then dense and softmax. Reproduced in PyTorch (TensorFlow has no GPU support on native
Windows, and 50,000 iterations of the 9840x1024 dense layer takes ~9 h on CPU against ~6 min on
a GPU). **The original protocol was preserved exactly, bugs included** — see below.

| Metric | Value |
|---|---:|
| Claimed accuracy (README, segment=200) | ~93% |
| Claimed accuracy (README, segment=50) | ~90% |
| **Reproduced, 40 features, 5 seeds** | **0.9344 ± 0.0011** |
| Reproduced, 79 features, 2 seeds | **0.9408 ± 0.0000** |

The reproduced figures use the paper's own reporting metric: the best of ~50 test-set readings
taken every 1000 iterations. Accuracy at the final iteration instead, which is what an
independent evaluation would report, is 0.9303 ± 0.0013 (40 features) and 0.9356 ± 0.0003 (79) —
so that metric inflates the result by about 0.4 points.

### Feature set — 40 against 79

Only the feature matrix changes; architecture, protocol and seeds are identical.

| Feature set | max accuracy | final accuracy | Upstairs F1 | Downstairs F1 |
|---|---:|---:|---:|---:|
| 40 (Kwapisz) | 0.9337 | 0.9299 | 0.7887 | 0.8522 |
| 79 (+ 13 stats per axis) | **0.9408** | **0.9356** | **0.7978** | **0.8674** |

Paired on seeds 42 and 0. **+0.71 points** — six times the seed-to-seed spread of ±0.11, and the
largest single improvement measured anywhere in this project. The gain concentrates in the static
classes: Sitting F1 0.8870 → 0.9272, Standing 0.8876 → 0.9204.

Upstairs recall stays at ~0.71-0.73 throughout. That failure is untouched by the extra features,
by FIN substitutions, or by anything else tried.

### Per-class, 40 features, seed 42, final iteration

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Jogging | 0.9531 | 0.9809 | 0.9668 |
| Walking | 0.9424 | 0.9825 | 0.9620 |
| Upstairs | 0.8550 | 0.7320 | 0.7887 |
| Downstairs | 0.9098 | 0.8014 | 0.8522 |
| Sitting | 0.9573 | 0.8263 | 0.8870 |
| Standing | 0.8272 | 0.9576 | 0.8876 |
| **Accuracy** | | | **0.9289** |
| Macro avg | 0.9075 | 0.8801 | 0.8907 |

### Bugs found in `cnn_wisdm.py`

The split itself is sound — `test-user-start=27` makes it subject-disjoint, which is what the
`HAR using CNN.ipynb` reproduction got wrong. These are the defects that remain.

| Bug | Line | Description |
|---|---|---|
| Model selection on the test set | 196-197 | The test set is evaluated every 1000 iterations and the best of ~50 readings is reported. Measured inflation over the final-iteration figure: +0.40 points. |
| Per-split feature standardization | 96-100 | `features_test` is centred and scaled by the **test** split's own statistics, `features` by the train split's, so the two arms sit in different coordinate frames. Measured shift between the splits: up to 0.28 train-std units. |
| `keep_prob`, not `dropout_rate` | 151, 172 | TF1's `tf.nn.dropout` takes a KEEP probability, and the code passes it the variable named `dropout_rate` = 0.15. So **85% of units are dropped**, survivors scaled by 6.67. Almost certainly a naming mix-up. |
| Axis interleaving | 126 | `hstack((data_x, data_y, data_z))` lays a row out as `[x_0..x_199, y_0..y_199, z_0..z_199]`, then `reshape(-1, 1, 200, 3)` reads element (t, c) from flat index `3t+c`. A "channel" is therefore three **consecutive samples of one axis**, not x/y/z at one instant, and the 200 "timesteps" run across all three axes end to end. Verified numerically over all 600 positions. Same defect as the one catalogued for `HAR using CNN.ipynb`. |

The last one means the conv branch trains on a scrambled signal, which is part of why the
handcrafted features carry so much of this architecture's performance.

---

## 3. WISDM — CNN (Keras, `CNN/` reproduction)

| Version | Accuracy | Macro F1 | Train users | Test windows |
|---|---:|---:|---:|---:|
| Original notebook (`HAR using CNN.ipynb`) | 98.13% | — | 5 | 107 |
| Original weights, evaluated on held-out WISDM subjects | 58.8% | 0.465 | — | — |
| Subject-split rebuild (`har_cnn_subject_split.py` / `.ipynb`) | 86.47% | 0.8137 | 36 | 7,817 |

### Subject-split rebuild — per-class F1

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Jogging | 0.9174 | 0.9567 | 0.9366 | 2264 |
| Walking | 0.9303 | 0.9230 | 0.9266 | 3037 |
| Upstairs | 0.5633 | 0.6024 | 0.5822 | 835 |
| Downstairs | 0.7418 | 0.7165 | 0.7290 | 762 |
| Sitting | 0.9852 | 0.8193 | 0.8946 | 487 |
| Standing | 0.8362 | 0.7917 | 0.8133 | 432 |
| **Accuracy** | | | **0.8647** | 7817 |
| Macro avg | 0.8290 | 0.8016 | 0.8137 | |
| Weighted avg | 0.8672 | 0.8647 | 0.8652 | |

### Bugs found in `HAR using CNN.ipynb`

| Bug | Cell(s) | Description |
|---|---|---|
| Class-balancing leakage | 12 | `.head(3555)` per activity draws from the first N rows of an unshuffled, user-ordered file — only 5 of 36 users (17, 20, 27, 33, 36) end up in the dataset. |
| Scaler fit before split | 17 | `StandardScaler().fit_transform(X)` runs on the full dataset before `train_test_split`, so test-set statistics leak into training normalization. |
| Axis interleaving | 18 | `frames.append([x, y, z])` builds shape `(N, 3, 80)`; `.reshape(-1, 80, 3)` reinterprets memory instead of transposing, so each of the "3 channels" at a timestep is 3 consecutive samples of one axis, not x/y/z at one instant. |
| Subject leakage | 10, 19 | The `user` column is dropped before splitting, then `train_test_split` is applied to overlapping windows — the same person's data can appear in both train and test. |
| Overlapping-window leakage | 18, 19 | `hop_size = frame_size / 2` makes adjacent windows share 50% of their samples; the random split can place near-duplicate windows on both sides. |
| Test set used as validation | 23 | `model.fit(..., validation_data=(X_test, y_test))` for 250 epochs — the reported accuracy is the best value seen on the test set itself, not an independent evaluation. |

---

## 4. UniMiB-SHAR — CNN reproduction

### In-domain, UniMiB 50Hz, 17 classes

| Metric | Value |
|---|---:|
| Accuracy | 0.7716 |
| Hamming Loss | 0.2284 |

| Class | Precision | Recall |
|---|---:|---:|
| 1 | 0.75 | 0.68 |
| 2 | 0.71 | 0.90 |
| 3 | 0.55 | 0.77 |
| 4 | 0.61 | 0.55 |
| 5 | 0.61 | 0.57 |
| 6 | 0.70 | 0.61 |
| 7 | 0.83 | 1.00 |
| 8 | 1.00 | 0.84 |
| 9 | 0.80 | 0.53 |
| 10 | 0.91 | 0.97 |
| 11 | 0.72 | 0.84 |
| 12 | 1.00 | 1.00 |
| 13 | 0.84 | 0.84 |
| 14 | 0.83 | 0.80 |
| 15 | 0.84 | 0.87 |
| 16 | 0.54 | 0.45 |
| 17 | 0.97 | 0.90 |

### Axis-interleaving check (verified on `datasets/UniMiB-SHAR/data/acc_data.mat`)

| Reshape method | Per-axis sigma |
|---|---|
| `reshape(-1, 151, 3)` (as used in repo) | [6.266, 6.312, 6.289] |
| `reshape(-1, 3, 151).transpose(0, 2, 1)` (axis-correct) | [5.058, 8.457, 4.616] |

### Cross-dataset — WISDM-trained CNN tested on UniMiB (4 shared classes)

| Metric | Value |
|---|---:|
| Accuracy (as reported) | 0.1525 |
| Macro F1 (as reported) | 0.1313 |
| Accuracy, 6-way argmax (notebook's own saved output) | 0.1029 |
| Accuracy, restricted to 4 shared classes | 0.1548 |
| + WISDM scaler instead of target-fitted scaler | 0.2002 |
| + orientation sign-fix only | 0.1925 |
| + correct 50→20Hz resample only | 0.1614 |
| + WISDM scaler AND orientation fix | 0.2187 |
| Properly trained subject-disjoint model, global z-score | 0.2661 |
| + per-window z-score + orientation fix + correct resample | 0.4596 |
| Best label-permutation (24-way sweep) | 0.3686 |
| Identity label mapping (worst) | 0.1548 |

| Model / condition | WISDM accuracy |
|---|---:|
| Reported (leaked) | 0.9813 |
| Same weights, held-out WISDM subjects | 0.5876 |
| Properly trained, subject-disjoint | 0.8602 |

---

## 5. HAPT / Opportunity — IF-ConvTransformer reproduction

| Dataset | Paper Macro F1 | Reproduced Macro F1 | Reproduced Accuracy |
|---|---:|---:|---:|
| HAPT | 85.58% | 86.76% ± 8.90 | 95.25% ± 4.02 |
| Opportunity | 47.41% | 46.35% ± 5.93 | 88.16% ± 1.38 |

---

## 6. Cross-dataset generalization candidates

### UCI HAR — WISDM-trained RF surrogate tested on UCI

| Configuration | Accuracy |
|---|---:|
| In-domain WISDM, held-out subjects | 0.8301 |
| Cross-dataset WISDM → UCI, 5 classes | 0.5051 |
| Without axis swap (X↔Y) | 0.4581 |
| Left in g units (no ×9.80665) | 0.2452 |
| Locomotion-only subset (Walk/Up/Down) | 0.4373 |
| Static-only subset (Sit/Stand) | 0.5678 |
| Static-vs-locomotion binary | 0.9970 |
| Sitting recall | 0.08 |
| Walking recall | 0.15 |
| Downstairs recall | 0.80 |
| Upstairs recall | 0.57 |

### Units verification (measured on repo's own raw files)

| Dataset | mean\|a\| | Units | Gravity axis |
|---|---:|---|---|
| UCI HAR (raw) | 1.039 | g | X (0.881) |
| WISDM (raw) | 12.060 | m/s² | Y (6.86) |

### Other candidates — assessed, not run

| Dataset | Overlapping classes | Sampling rate | Placement | Seamlessness (1-10) | Expected accuracy |
|---|---:|---:|---|---:|---|
| PAMAP2 | 6/6 | 100Hz | wrist/chest/ankle | 7 | 30-45% |
| USC-HAD | 6/6 | 100Hz | front right hip | 7 | 45-65% |
| MHEALTH | 4/6 | 50Hz | chest/wrist/ankle | 5-6 | not estimated |
| HHAR | 5/6 | variable | phone/watch, varied | 4 | not estimated |

---

---

## 7. WISDM — Feature Imitating Network (FIN), mean/std/mad

Saba-Sadiya, Alhanai & Ghassemi, *Feature Imitating Networks*, ICASSP 2022
([arXiv:2110.04831](https://arxiv.org/abs/2110.04831),
[code](https://github.com/sari-saba-sadiya/Feature-Imitating-Networks)).

Not a reproduction of the paper's experiments (those are ECG/EEG); this applies the method to
WISDM. A single network takes one raw 200-sample axis and outputs `mean`, `std` (ddof=1) and
`mad` — the first 9 of ClassicML's 40 features when applied to x, y and z with shared weights.

Script: `ClassicML/FIN-experiments/fin_mean_std_mad.py`. 8,643 parameters. Trained on users 1-26
(all three axes pooled, 22,101 windows → 66,303 after exact affine augmentation), 40 epochs.

Architecture: a per-timestep map (two 1x1 convolutions, weights shared across all 200 timesteps),
a fixed mean pool over time, then `Dense(64) → Dense(3)`. The pool is not learned.

### Imitation quality, held-out users 27-36

| axis | R² mean | R² std | R² mad | MAE mean | MAE std | MAE mad |
|---|---:|---:|---:|---:|---:|---:|
| x | 0.9990 | 0.9944 | 0.9904 | 0.1098 | 0.1409 | 0.1770 |
| y | 0.9984 | 0.9940 | 0.9949 | 0.1404 | 0.1440 | 0.1283 |
| z | 0.9974 | 0.9792 | 0.9827 | 0.1151 | 0.1817 | 0.1268 |

MAE is in m/s². The same weights serve all three axes, which is what makes the model reusable.

### Reference — MAE of predicting the test-split mean of each target

| axis | mean | std | mad |
|---|---:|---:|---:|
| x | 3.1457 | 2.2111 | 1.9148 |
| y | 2.7380 | 2.1569 | 1.8615 |
| z | 1.6765 | 1.2584 | 1.0071 |

The FIN's error is 8-25× smaller than this constant-predictor floor.

### mad/std ratio check — X axis, 2,677 windows with std > 1 m/s²

| quantity | value |
|---|---:|
| true spread of mad/std | 0.0438 |
| predicted spread | 0.0473 |
| correlation with true ratio | 0.6405 |

`std` and `mad` are strongly correlated, so a network can score well by emitting `mad ≈ c·std`
for a fixed `c` without computing `mad` at all. The correct `c` is shape-dependent (√(2/π) ≈ 0.798
for Gaussian noise, 2√2/π ≈ 0.900 for a sine), so a faked `mad` shows a collapsed spread and poor
correlation. The spread here is honest but the correlation is only moderate: this network leans
partly on the relationship between the two rather than computing each independently.

### Notes on the method as published

| Deviation from the paper | Why |
|---|---|
| Raw windows, never z-scored | The reference repo standardizes each window before computing the target. That forces `mean` ≡ 0, `std` ≡ 1, `mad` ≈ 0.8, making all three targets constants. Harmless for the skew/kurtosis the repo demonstrates (affine-invariant), fatal for these three. |
| All three axes pooled in training | Training on X alone drops imitation R² to 0.68-0.92 on Y and Z. Not a range problem but a density one: X's window means centre near 0, Y's near +8.5 (gravity), and only ~8% of X windows fall in Y's core (mean, std) region. |
| Per-timestep shape, not the paper's flat MLP | All three features are a time-average of a per-timestep function, so an architecture built that way needs far fewer weights to express them. |
| No BatchNorm | These targets *are* the input's scale and offset; normalizing activations across the batch deletes the answer and makes predictions batch-dependent. |

### Architecture against capacity

Same targets, same data, same number of gradient steps — only the network's shape differs.
Measured with a PyTorch implementation, single seed 42.

| architecture | params | R² mean | R² std | R² mad |
|---|---:|---:|---:|---:|
| flat MLP (the paper's shape) | 92,803 | 0.9974 | 0.9914 | 0.9769 |
| flat MLP, widened | 862,723 | 0.9986 | **0.9959** | 0.9817 |
| per-timestep map + pool | **8,643** | **0.9997** | 0.9905 | **0.9901** |

The widened control exists because the paper's topology search spanned 3-15M parameters, so a 92k
flat net is small by its standards. The per-timestep shape wins on `mean` and `mad` with **100×
fewer parameters**, but loses on `std` — there the widened flat net is better. So for these three
features the shape argument holds for two of them and not the third.

It holds much more clearly for the average resultant acceleration, where the same comparison gives
R² 0.9969 at 4,481 parameters against 0.8601 at 1,556,737 — **347× fewer weights and a far better
fit**.

A fourth variant was tried and removed: it added an explicit centring stage and passed the
arithmetically computed window mean straight to the head. It scored better on every target
(R² 1.0000 / 0.9998 / 1.0000, ratio correlation 0.983), but the `mean` output was then a
pass-through rather than an imitation, so the figure measured nothing about the network.

---

## 8. FIN initialization vs random initialization on the HAR task

`ClassicML/FIN-experiments/fin-features-estimation.py`. Backbone pretrained on mean/std/mad, its
output layer discarded, then fine-tuned with an extra Dense layer on 6-class WISDM. 5 seeds
(42, 0, 1, 2, 3), paired. The backbone here is the flat `200 → 256 → 128 → 64` shape.

| init | test accuracy | macro F1 |
|---|---:|---:|
| random | 0.7571 ± 0.0060 | 0.6819 ± 0.0112 |
| FIN | 0.7623 ± 0.0133 | 0.6869 ± 0.0205 |

| | test accuracy | macro F1 |
|---|---:|---:|
| mean paired difference | +0.0052 | +0.0050 |
| paired t (df=4) | 0.668 | 0.600 |
| p | **0.54** | **0.58** |
| seeds favouring FIN | 3/5 | 3/5 |

**No significant difference.** Per-seed differences run from −0.0185 to +0.0238 — the sign flips.
The effect is about a third of the seed-to-seed noise; establishing it at 80% power would need
~87 seeds.

FIN's variance is 4.8× *higher* than random's on test accuracy (F-test p = 0.16, not established
at n=5), which runs opposite to the paper's headline claim of reduced variance.

Best-val accuracy (~0.87) sits ~11 points above test accuracy: `validation_split=0.1` draws a
random 10% of training windows, which share subjects and 50% sample overlap with the training
set. Test accuracy is unaffected, but `best_val_accuracy` should not be quoted as a result.

### Context

Both arms sit well below the rest of the project — the architecture is three Dense layers on raw
windows, no convolution and no handcrafted features.

| model | test accuracy |
|---|---:|
| this MLP, random init | 0.7571 |
| this MLP, FIN init | 0.7623 |
| Random Forest, 79 features | 0.8390 |
| plain CNN, subject-split | 0.8647 |
| Ignatov CNN + 40 features | 0.9303 |

0.76 is roughly what mean/std/mad alone support, so both arms are limited by the architecture
rather than by the initialization.
