`fin_ensemble.py`. Three FINs (mean, std, mad), each truncated to its penultimate layer, applied
to x, y and z with shared weights, concatenated to 576 dims, then a classification head.

| regime | acc FIN | acc random | Δ acc | F1 FIN | F1 random | Δ F1 |
|---|---:|---:|---:|---:|---:|---:|
| frozen | 0.6730 | 0.6555 | +0.0175 | 0.6544 | 0.6185 | +0.0359 |
| gradual | **0.7812** | 0.7453 | **+0.0358** | **0.7185** | 0.6723 | **+0.0462** |
| unfrozen | 0.7590 | 0.7416 | +0.0174 | 0.7052 | 0.6694 | +0.0358 |

## Freeze everything but the head

`frozen` is the weakest regime — 0.6730

Head depth matters here. With the FINs frozen, accuracy falls m with
depth: 0.7328 → 0.6845 → 0.6377 → 0.6371 for 1, 2, 3, 4 layers.

## Freeze first, then unfreeze

`gradual` (frozen 8 epochs, then everything trains) is the **best regime in both arms** and beats
`unfrozen` at all four head depths.

The informative part is how unevenly it helps:

| arm | gradual − unfrozen |
|---|---:|
| FIN | **+0.0222** |
| random | +0.0038 |


After thawing, the FIN branch trains at the same rate as the head (1e-3).

