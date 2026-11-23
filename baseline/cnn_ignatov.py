"""CNN + statistical features of Ignatov (2018), the neural baseline of the paper, in PyTorch.

Conv1D 196 filters, kernel 12, 'same', ReLU -> max pooling 4 -> flatten, concatenated with the statistical
features -> Dense 1024 ReLU (L2 5e-4 / 2 / 200 per window, as his summed loss over batches of 200) -> dropout 0.85
(his keep_prob 0.15) -> Dense. Weights truncated normal (s.d. 0.01), biases 0.01. Trained with Adam 5e-4.

Input: each window centred per axis and then reshaped (n, 3, T) -> (n, T, 3) WITHOUT transposing, exactly as the
paper's Keras code did (it mixes axes and time steps; kept so that the baseline is the one we reported).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

LR = 5e-4
L2 = 5e-4 / 2 / 200


def centred(x):
    """(n, C, T) -> (n, C, T) model input: per-axis mean removed, then the paper's reshape to (n, T, C) read back as
    channels-first for Conv1d."""
    n, c, t = x.shape
    x = x - x.mean(axis=2, keepdims=True)
    return x.reshape(n, t, c).transpose(0, 2, 1).copy()


class IgnatovCNN(nn.Module):
    def __init__(self, channels, length, n_features, n_classes):
        super().__init__()
        self.conv = nn.Conv1d(channels, 196, 12)
        self.pool = nn.MaxPool1d(4, ceil_mode=True)
        self.dense = nn.Linear(196 * -(-length // 4) + n_features, 1024)
        self.drop = nn.Dropout(0.85)
        self.out = nn.Linear(1024, n_classes)
        for m in (self.conv, self.dense, self.out):
            nn.init.trunc_normal_(m.weight, std=0.01, a=-0.02, b=0.02)
            nn.init.constant_(m.bias, 0.01)

    def forward(self, x, stats):
        h = torch.relu(self.conv(F.pad(x, (5, 6))))  # Keras 'same' for kernel 12: 5 left, 6 right
        h = self.pool(h).transpose(1, 2).flatten(1)  # time-major, as Keras flattens (T', 196)
        h = torch.relu(self.dense(torch.cat([h, stats], dim=1)))
        return self.out(self.drop(h))

    def penalty(self):
        return L2 * (self.dense.weight.pow(2).sum() + self.dense.bias.pow(2).sum())
