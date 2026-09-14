# FIN architectures in this folder

Three networks, all imitating `mean`, `std` (ddof=1) and `mad` of a raw 200-sample accelerometer
window. RMSE is in m/s², measured on held-out users 27-36.

## Summary

| script | architecture | params | RMSE mean | RMSE std | RMSE mad |
|---|---|---:|---:|---:|---:|
| `fin_conv.py` | per-timestep map → pool over time | **8,643** | **0.1310** | **0.2102** | **0.1843** |
| `fin_flat.py` | flat MLP, one network per feature | 202,657 / 1,476,481 | 0.2242 | 0.4172 | 0.4095 |
| `fin-features-estimation.py` | flat MLP, used as an initialization | 92,803 | 0.5458 | 0.4387 | 0.3954 |

As a fraction of each target's own spread on the test split:

| script                      | mean | std | mad |
|-----------------------------|---:|---:|---:|
| `fin_conv.py`               | **3.9%** | **9.0%** | **9.3%** |
| `fin_flat.py`               | 4.8% | 17.3% | 19.8% |
| `fin-features-estimation.py` | 16.1% | 18.9% | 20.0% |

**Rows are not strictly comparable.** They differ in training split, augmentation and epoch budget
— each row is what running that script produces, not a controlled comparison. The per-network
sections below state the protocol for each.

---

## 1. `fin_conv.py`

One network, one raw axis in, three numbers out. The same weights are applied to x, y and z, which
is what makes it reusable.

```
input: 200 samples (one axis)
  Conv1D(64, kernel 1) + ReLU        1 number  -> 64, applied to each sample separately
  Conv1D(64, kernel 1) + ReLU        64 -> 64, same
  mean over time                     200 vectors of 64 -> one vector of 64
  Dense(64) + ReLU
  Dense(3)                           mean, std, mad
```

| | |
|---|---|
| weight layers | 4 |
| parameters | 8,643 |
| training | users 1-26, three axes pooled (22,101 → 66,303 windows after affine augmentation), 40 epochs |
| test | users 27-36, all three axes |

Kernel size 1 means the convolution sees exactly one sample at a time, so the two convolutions are
a small MLP (1 → 64 → 64) run independently at each of the 200 timesteps with shared weights. The
pool is a fixed arithmetic mean, not learned.

The shape mirrors the features themselves: each is a time-average of a per-sample function. A
consequence is that the network is invariant to the order of the samples — correct for these three
targets, but it could not represent anything that depends on timing, such as a fundamental
frequency.

Target spread on this test split: mean 3.381, std 2.326, mad 1.977.

---

## 2. `fin_flat.py`

The published recipe: a flat stack, **one network per feature**, single scalar output, with depth,
width, optimizer and learning rate chosen by their random topology search (seed 1337, first 25 of
100 configs).

```
input: 200 samples
  Flatten
  Dense(w) + ReLU (+ BatchNorm)      x 4, widths from the searched grid
  Dense(1)                           one number
```

| feature | winning topology | weight layers | parameters |
|---|---|---:|---:|
| mean | #13, widths [256, 512, 32, 64], ADAM lr=0.01 | 4 + output = 5 | 202,657 |
| std | #14, widths [1024, 512, 1280, 64], SGD lr=0.01 | 4 + output = 5 | 1,476,481 |
| mad | #14, same | 4 + output = 5 | 1,476,481 |

| | |
|---|---|
| training | users 1-20, three axes pooled, no augmentation, 60 epochs |
| topology selection | users 21-26, never fitted on; winners cached in `best_topologies.json` |
| test | users 27-36, scored once |

Every layer sees all 200 samples from the start, so the network has to discover for itself that
they should be processed one at a time and then averaged. It reaches a worse fit than network 1
while using **171× more parameters** on `std`.

All three winners drew lr=0.01, the largest value in their grid. Six of that grid's nine learning
rates are 1e-06 or below, where nothing trains within the search budget.

Target spread on this test split: mean 4.631, std 2.409, mad 2.071. The windows here are cut from
the raw file rather than read from ClassicML's CSVs, so this split is not identical to the one
above.

---

## 3. `fin-features-estimation.py`

The backbone used as an **initialization** for a HAR classifier rather than as a feature imitator
in its own right. The RMSE below is how well it imitates before that output layer is discarded.

```
input: 200 samples
  Dense(256) + ReLU
  Dense(128) + ReLU
  Dense(64) + ReLU                   <- backbone kept
  Dense(3)                           <- discarded before the classifier
```

| | |
|---|---|
| weight layers | 4 (3 kept) |
| parameters | 92,803 |
| training | users 1-26, three axes pooled, no augmentation, 40 epochs, `validation_split=0.1` |
| test | users 27-36 |

Downstream the backbone is applied to x, y and z with shared weights, the three 64-vectors are
concatenated, and a `Dense(128) → Dense(6)` head classifies the activity. Its imitation quality is
the weakest of the three, partly because it trains without augmentation and for fewer effective
steps; imitation is not what it is optimized for.

---

## Reference: predicting each target's mean

| split | mean | std | mad |
|---|---:|---:|---:|
| networks 1 and 3 (users 27-36, ClassicML windows) | 3.381 | 2.326 | 1.977 |
| network 2 (users 27-36, raw-cut windows) | 4.631 | 2.409 | 2.071 |

A constant predictor scores roughly these values as RMSE, so every network above is well clear of
the floor. Network 1 is 8-25× below it; networks 2 and 3 are 5-20× below.
