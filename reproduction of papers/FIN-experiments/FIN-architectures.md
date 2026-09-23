# FIN architectures

RMSE in m/s² on held-out users 27-36, imitating `mean`, `std` (ddof=1) and `mad` of a raw
200-sample axis.

| script | architecture | params | RMSE mean | RMSE std | RMSE mad |
|---|---|---:|---:|---:|---:|
| `fin_conv.py` | `Conv1D(64, k=1) ×2 → mean over time → Dense(64) → Dense(3)` | **8,643** | **0.1310** | **0.2102** | **0.1843** |
| `fin_flat.py` | `Flatten → Dense ×4 (+BatchNorm) → Dense(1)`, one net per feature | 202,657 / 1,476,481 | 0.2242 | 0.4172 | 0.4095 |
| `fin_ensemble.py` | three `fin_flat` nets truncated → concat 576 → head of 1-4 Dense → softmax(6) | 3,159,081 – 3,263,273 | — | — | — |

`fin_conv.py` is one network with three outputs; kernel 1 means each sample is mapped
independently and the pool is a fixed mean. `fin_flat.py` is the published shape with the authors'
topology search (seed 1337): #13 for `mean`, #14 for `std` and `mad`.

`fin_ensemble.py` classifies rather than imitates, so it has no RMSE — its accuracy is in
`FIN-results.md`. Its range covers head depths 1 to 4; the three truncated trunks are 3,147,939 of
that, so the head is at most 3.5% of the model.


