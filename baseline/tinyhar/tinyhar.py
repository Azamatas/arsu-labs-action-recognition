"""TinyHAR (Zhou et al., "TinyHAR: A Lightweight Deep Learning Model Designed for Human Activity Recognition",
ISWC 2022, doi:10.1145/3544794.3558467) in the paper's cross-dataset protocol.

Re-implemented in PyTorch from the paper and the authors' code (github.com/teco-kit/ISWC22-HAR, models/TinyHAR.py,
configs/model.yaml; the repository has no licence, so nothing is copied). Configuration of the paper:
  * filter_num F = 20; 4 per-channel convolutions, kernel 5 along time, stride 2 on the 2nd and 4th, ReLU + BatchNorm
  * cross-channel interaction: one transformer block over the sensor channels at every time step
    (4 heads x 16, pre-norm, feed-forward 16 with GELU), dropout 0.1
  * cross-channel fusion: FC from C * F to 2F (ReLU); temporal interaction: one LSTM layer (2F)
  * temporal aggregation "tnaive": tanh attention over time, context added to the last step with a learnable gamma
  * linear classifier
Training as the neural models of cross-dataset/cross_dataset.py: same windows and splits, flipped copies, Adam 1e-3,
60 epochs, batch 256, scored after the last epoch. Input standardised per channel with the training statistics (as
the authors' pipeline does); no statistical features (TinyHAR uses the raw signal only).

    python baseline/tinyhar/tinyhar.py --seeds 42       # writes baseline/tinyhar/results/tinyhar.md and .npz
"""
import os
import sys

os.environ["KERAS_BACKEND"] = "torch"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # baseline/tinyhar -> repository root
sys.path[:0] = [ROOT, os.path.join(ROOT, "data-processing"), os.path.join(ROOT, "cross-dataset")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import config as cfg  # noqa: E402
import cross_dataset as cd  # noqa: E402

OUT = os.path.join(HERE, "results")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
F = 20


class ChannelTransformer(nn.Module):
    """One pre-norm transformer block over the sensor channels (tokens = channels, features = F)."""

    def __init__(self, dim, heads=4, dim_head=16, mlp_dim=16):
        super().__init__()
        inner = heads * dim_head
        self.heads, self.scale = heads, dim_head ** -0.5
        self.norm1, self.qkv, self.proj = nn.LayerNorm(dim), nn.Linear(dim, 3 * inner, bias=False), nn.Linear(inner, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))

    def forward(self, x):  # (n, C, F)
        n, c, _ = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = (t.view(n, c, self.heads, -1).transpose(1, 2) for t in (q, k, v))
        a = torch.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1) @ v
        x = x + self.proj(a.transpose(1, 2).reshape(n, c, -1))
        return x + self.ff(self.norm2(x))


class TinyHAR(nn.Module):
    def __init__(self, channels, length, n_classes, f=F, dropout=0.1, pad=2):  # pad=0 as in the authors' code
        super().__init__()
        convs, cin = [], 1
        for i in range(4):
            stride = 2 if i % 2 == 1 else 1
            convs += [nn.Conv2d(cin, f, (5, 1), stride=(stride, 1), padding=(pad, 0)), nn.ReLU(), nn.BatchNorm2d(f)]
            cin = f
        self.conv = nn.Sequential(*convs)
        self.interaction = ChannelTransformer(f)
        self.drop = nn.Dropout(dropout)
        self.fusion = nn.Linear(channels * f, 2 * f)
        self.lstm = nn.LSTM(2 * f, 2 * f, batch_first=True)
        self.att1, self.att2 = nn.Linear(2 * f, 2 * f), nn.Linear(2 * f, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.out = nn.Linear(2 * f, n_classes)

    def forward(self, x):  # x: (n, C, L)
        h = self.conv(x.permute(0, 2, 1).unsqueeze(1))           # (n, F, L*, C)
        n, f, t, c = h.shape
        h = h.permute(0, 2, 3, 1).reshape(n * t, c, f)           # every time step: C tokens of size F
        h = self.drop(self.interaction(h)).reshape(n, t, c * f)
        h = torch.relu(self.fusion(h))                           # (n, L*, 2F)
        h, _ = self.lstm(h)
        w = torch.softmax(self.att2(torch.tanh(self.att1(h))), dim=1)
        context = (w * h).sum(1)
        return self.out(h[:, -1] + self.gamma * context)


def run(direction, seed):
    xtr, ytr, tests = cd.load(direction)  # paper windows, flipped copies included
    mu, sd = xtr.mean(axis=(0, 2), keepdims=True), xtr.std(axis=(0, 2), keepdims=True) + 1e-8
    norm = lambda x: torch.as_tensor(((x - mu) / sd).astype(np.float32))
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TinyHAR(len(cfg.AXES), cfg.SEGMENT, cfg.N_CLASSES).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    x, y = norm(xtr), torch.as_tensor(ytr, dtype=torch.long)
    rng = np.random.default_rng(seed)
    for _ in range(cfg.HEAD_EPOCHS):
        model.train()
        for idx in np.array_split(rng.permutation(len(y)), int(np.ceil(len(y) / cfg.BATCH_SIZE))):
            opt.zero_grad()
            loss_fn(model(x[idx].to(DEV)), y[idx].to(DEV)).backward()
            opt.step()
    model.eval()
    preds = {}
    with torch.no_grad():
        for name, (xt, _) in tests.items():
            xt = norm(xt)
            preds[name] = torch.cat([model(xt[i:i + 1024].to(DEV)).argmax(1).cpu()
                                     for i in range(0, len(xt), 1024)]).numpy()
    return preds, tests, sum(p.numel() for p in model.parameters())


def main():
    import argparse

    from sklearn.metrics import f1_score

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    seeds = parser.parse_args().seeds
    os.makedirs(OUT, exist_ok=True)
    lines = ["# TinyHAR in the paper's cross-dataset protocol", "",
             f"Written by `baseline/tinyhar/tinyhar.py`; seeds {seeds}, scored after the last epoch. "
             "Same windows, splits, flip augmentation, epochs and batch size as the neural models of Table 4.", "",
             "| direction | test set | accuracy | macro F1 |", "|---|---|---:|---:|"]
    save, n_params = {}, None
    for direction in ("w2m", "m2w"):
        res = {}
        for seed in seeds:
            preds, tests, n_params = run(direction, seed)
            for t_name, p in preds.items():
                y = tests[t_name][1]
                res.setdefault(t_name, []).append((np.mean(p == y), f1_score(y, p, average="macro")))
                save[f"{direction}|{seed}|{t_name}"] = p
                save[f"{direction}|truth|{t_name}"] = y
            print(f"{direction} seed {seed}: " + ", ".join(f"{t} {np.mean(p == tests[t][1]):.4f}"
                                                          for t, p in preds.items()), flush=True)
        for t_name, v in res.items():
            a, f = np.array(v).T
            ms = lambda z: f"{z.mean():.4f}" + (f" ± {z.std(ddof=1):.4f}" if len(z) > 1 else "")
            lines.append(f"| {cd.DIRECTIONS[direction]} | {t_name} | {ms(a)} | {ms(f)} |")
    lines += ["", f"TinyHAR parameters (3 accelerometer channels, F = {F}): {n_params:,}"]
    np.savez(os.path.join(OUT, "tinyhar.npz"), **save)
    with open(os.path.join(OUT, "tinyhar.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
