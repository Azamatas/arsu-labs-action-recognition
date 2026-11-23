"""FIN and the classifier built on it (PyTorch).

FIN, for one axis x of length T:
    z = (x - mean x) / (std x + 1e-6)                       per-window z-score
    m_k = mean_t (a_k * z)_t (b_k * z)_t,  k = 1..width     two linear convolutions, product, time mean
    emb = ReLU(W m + c)                                     dense layer
and, during pretraining only, a linear layer emb -> the 3 imitation targets.

Convolutions are cross-correlations aligned as Keras 'same' (left padding (K - 1) // 2), zero-padded or circular.
Initialisation is Glorot-uniform with zero biases, as in the Keras implementation of the paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import features


def glorot_(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def zscore(x):
    return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, unbiased=False, keepdim=True) + 1e-6)


class FIN(nn.Module):
    def __init__(self, width, kernel, emb, circular=False, n_targets=3):
        super().__init__()
        self.kernel, self.circular = kernel, circular
        self.a = nn.Conv1d(1, width, kernel, bias=False)
        self.b = nn.Conv1d(1, width, kernel, bias=False)
        self.proj = nn.Linear(width, emb)
        self.head = nn.Linear(emb, n_targets)
        glorot_(self)

    def embed(self, x):
        """x: (n, T) single axes -> (n, emb)."""
        h = zscore(x).unsqueeze(1)
        left = (self.kernel - 1) // 2
        h = F.pad(h, (left, self.kernel - 1 - left), mode="circular" if self.circular else "constant")
        return torch.relu(self.proj((self.a(h) * self.b(h)).mean(dim=-1)))

    def forward(self, x):
        return self.head(self.embed(x))


class Classifier(nn.Module):
    """Frozen (or trainable) FIN on every channel + statistical features -> Dense ReLU -> Dropout -> logits.

    pooling: 'concat' keeps one embedding per channel; 'mean' / 'meanmax' pool the embeddings over the three axes of
    each sensor (accelerometer = channels 0-2, gyroscope = 3-5)."""

    def __init__(self, fin, emb, channels, n_features, head, n_classes, dropout=0.5, pooling="concat"):
        super().__init__()
        self.fin, self.pooling, self.channels = fin, pooling, channels
        per_sensor = {"concat": 3, "mean": 1, "meanmax": 2}[pooling]
        n_in = (channels // 3) * per_sensor * emb + n_features
        self.hidden = nn.Linear(n_in, head)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(head, n_classes)
        glorot_(self.hidden), glorot_(self.out)

    def forward(self, x, stats):
        n, c, t = x.shape
        e = self.fin.embed(x.reshape(n * c, t)).reshape(n, c, -1)
        if self.pooling == "concat":
            parts = [e.flatten(1)]
        else:
            parts = []
            for s in range(0, c, 3):
                g = e[:, s:s + 3]
                parts.append(g.mean(1))
                if self.pooling == "meanmax":
                    parts.append(g.amax(1))
        h = torch.relu(self.hidden(torch.cat(parts + [stats], dim=1)))
        return self.out(self.drop(h))


def build(cfg, channels, n_classes):
    fin = FIN(cfg.width, cfg.kernel, cfg.emb, cfg.circular)
    return Classifier(fin, cfg.emb, channels, features.size(cfg.features), cfg.head, n_classes, cfg.dropout,
                      cfg.pooling)
