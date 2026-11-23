"""TinyHAR (Zhou et al., "TinyHAR: A Lightweight Deep Learning Model Designed for Human Activity Recognition",
ISWC 2022, doi:10.1145/3544794.3558467).

Re-implemented in PyTorch from the paper and the authors' code (github.com/teco-kit/ISWC22-HAR, models/TinyHAR.py,
configs/model.yaml; the repository has no licence, so nothing is copied). Configuration of the paper:
  * filter_num F = 20; 4 per-channel convolutions, kernel 5 along time, stride 2 on the 2nd and 4th, ReLU + BatchNorm
  * cross-channel interaction: one transformer block over the sensor channels at every time step
    (4 heads x 16, pre-norm, feed-forward 16 with GELU), dropout 0.1
  * cross-channel fusion: FC from C * F to 2F (ReLU); temporal interaction: one LSTM layer (2F)
  * temporal aggregation "tnaive": tanh attention over time, context added to the last step with a learnable gamma
  * linear classifier
Input: the raw signal standardised per channel with the training statistics (as the authors' pipeline does).
"""
import torch
import torch.nn as nn

F = 20


class ChannelTransformer(nn.Module):
    """One pre-norm transformer block over the sensor channels (tokens = channels, features = F)."""

    def __init__(self, dim, heads=4, dim_head=16, mlp_dim=16):
        super().__init__()
        inner = heads * dim_head
        self.heads, self.scale = heads, dim_head ** -0.5
        self.norm1, self.qkv = nn.LayerNorm(dim), nn.Linear(dim, 3 * inner, bias=False)
        self.proj = nn.Linear(inner, dim)
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
