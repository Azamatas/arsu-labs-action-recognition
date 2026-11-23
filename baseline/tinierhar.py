"""TinierHAR (Bian, Liu, Fortes Rey, Geissler, Lukowicz, "TinierHAR: Towards Ultra-Lightweight Deep Learning Models for
Efficient Human Activity Recognition on Edge Devices", arXiv 2507.07949, 2025).

Re-implemented in PyTorch from the paper and from reading the authors' code (github.com/zhaxidele/TinierHAR,
models/TinierHAR.py, configs/model.yaml); the repository has no licence, so nothing is copied. Configuration of the
paper (M = 4 filters, N = 16 GRU units):
  * input (batch, channels C, time T), treated as a 1 x T x C image
  * 2 + 4 residual blocks; each: depthwise convolution 5 x 1 along time (groups = input maps), pointwise 1 x 1,
    BatchNorm, ReLU, and in the first two blocks max pooling 2 x 1 along time; maps 1 -> M -> 2M -> 2M (x 4);
    shortcut: identity, or 1 x 1 convolution + BatchNorm when the maps change, pooled like the block
  * per time step the 2M maps of all C channels are flattened, dropout 0.3, bidirectional GRU with N units
  * temporal aggregation: softmax attention over time (one linear score per step), weighted sum
  * linear classifier
"""
import torch
import torch.nn as nn



class Block(nn.Module):
    def __init__(self, cin, cout, pool):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(cin, cin, (5, 1), padding=(2, 0), groups=cin), nn.Conv2d(cin, cout, 1),
                                  nn.BatchNorm2d(cout), nn.ReLU())
        self.pool = nn.MaxPool2d((2, 1)) if pool else nn.Identity()
        self.short = nn.Identity() if cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1), nn.BatchNorm2d(cout))

    def forward(self, x):
        return self.pool(self.conv(x)) + self.pool(self.short(x))


class TinierHAR(nn.Module):
    def __init__(self, channels, length, n_classes, m=4, n=16, extra_blocks=4, dropout=0.3):
        super().__init__()
        blocks = [Block(1, m, True), Block(m, 2 * m, True)] + [Block(2 * m, 2 * m, False) for _ in range(extra_blocks)]
        self.blocks = nn.Sequential(*blocks)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(2 * m * channels, n, batch_first=True, bidirectional=True)
        self.score = nn.Linear(2 * n, 1)
        self.out = nn.Linear(2 * n, n_classes)

    def forward(self, x):  # x: (batch, C, T)
        h = self.blocks(x.transpose(1, 2).unsqueeze(1))  # (batch, maps, T', C)
        h = h.permute(0, 2, 1, 3).flatten(2)  # (batch, T', maps * C)
        h, _ = self.gru(self.drop(h))
        w = torch.softmax(self.score(h), dim=1)
        return self.out((w * h).sum(1))


def cost(channels, length, n_classes, m=4, n=16, extra_blocks=4):
    """Parameters and multiply-adds of one window (convolutions counted by hand; GRU: 3 gates, input + recurrent)."""
    model = TinierHAR(channels, length, n_classes, m, n, extra_blocks)
    params = sum(p.numel() for p in model.parameters())
    macs, t = 0, length
    for cin, cout, pool in [(1, m, True), (m, 2 * m, True)] + [(2 * m, 2 * m, False)] * extra_blocks:
        macs += t * channels * (5 * cin + cin * cout + (cin * cout if cin != cout else 0))
        t = t // 2 if pool else t
    macs += 2 * t * 3 * (2 * m * channels * n + n * n) + t * 2 * n * 2 + 2 * n * n_classes
    return params, macs
