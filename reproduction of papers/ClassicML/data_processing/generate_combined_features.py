import os

import numpy as np
from scipy.stats import kurtosis, skew

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "wisdm_data")

SEGMENT_LENGTH = 200

EXTRA_STAT_NAMES = [
    "min", "max", "median", "iqr", "range", "skew", "kurtosis",
    "energy", "rms", "third_moment", "fourth_moment",
    "mean_abs_first_diff", "mean_abs_second_diff",
]

BASIC_FEATURE_NAMES = (
    [f"{stat}_{axis}" for stat in ["mean", "std"] for axis in "xyz"]
    + [f"mad_{axis}" for axis in "xyz"]
    + ["avg_resultant_accel"]
    + [f"hist{i}_{axis}" for axis in "xyz" for i in range(10)]
)

COMBINED_FEATURE_NAMES = BASIC_FEATURE_NAMES + [
    f"{stat}_{axis}" for axis in "xyz" for stat in EXTRA_STAT_NAMES
]


def extract_extra_features(x):
    """x: (n_windows, window_length) raw samples for one axis."""
    mean = x.mean(axis=1, keepdims=True)
    diff1 = np.diff(x, axis=1)
    diff2 = np.diff(diff1, axis=1)

    feats = [
        x.min(axis=1),
        x.max(axis=1),
        np.median(x, axis=1),
        np.percentile(x, 75, axis=1) - np.percentile(x, 25, axis=1),
        x.max(axis=1) - x.min(axis=1),
        skew(x, axis=1, bias=False),
        kurtosis(x, axis=1, fisher=True, bias=False),
        (x ** 2).sum(axis=1),
        np.sqrt((x ** 2).mean(axis=1)),
        ((x - mean) ** 3).mean(axis=1),
        ((x - mean) ** 4).mean(axis=1),
        np.abs(diff1).mean(axis=1),
        np.abs(diff2).mean(axis=1),
    ]
    return np.stack(feats, axis=1)


def build_combined(suffix):
    basic = np.loadtxt(os.path.join(DATA_DIR, f"basic_features_{suffix}.csv"), delimiter=",")
    x = np.loadtxt(os.path.join(DATA_DIR, f"data_x_{suffix}.csv"), delimiter=",")
    y = np.loadtxt(os.path.join(DATA_DIR, f"data_y_{suffix}.csv"), delimiter=",")
    z = np.loadtxt(os.path.join(DATA_DIR, f"data_z_{suffix}.csv"), delimiter=",")

    extra = np.concatenate(
        [extract_extra_features(axis_data) for axis_data in (x, y, z)], axis=1
    )
    combined = np.concatenate([basic, extra], axis=1)

    out_path = os.path.join(DATA_DIR, f"combined_features_{suffix}.csv")
    np.savetxt(out_path, combined, delimiter=",", fmt="%.6f")
    return combined.shape


if __name__ == "__main__":
    train_shape = build_combined(str(SEGMENT_LENGTH))
    test_shape = build_combined(f"test_{SEGMENT_LENGTH}")

    print(f"train combined features: {train_shape}")
    print(f"test combined features: {test_shape}")
    assert len(COMBINED_FEATURE_NAMES) == train_shape[1]

    with open(os.path.join(DATA_DIR, "combined_feature_names.txt"), "w") as f:
        f.write("\n".join(COMBINED_FEATURE_NAMES))
    print(f"total features: {len(COMBINED_FEATURE_NAMES)}")
