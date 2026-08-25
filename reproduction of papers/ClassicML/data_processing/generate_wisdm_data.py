# a rewrite of the original MATLAB/OCTAVE code to Python

import argparse
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "datasets", "wisdm_raw_data.txt")
OUT_DIR = os.path.join(BASE_DIR, "wisdm_data")

ACTIVITY_TO_LABEL = {
    "Jogging": 1,
    "Walking": 2,
    "Upstairs": 3,
    "Downstairs": 4,
    "Sitting": 5,
    "Standing": 6,
}
N_CLASSES = 6


def extract_basic_features(x, y, z):
    n = len(x)
    features = [x.mean(), y.mean(), z.mean(), x.std(ddof=1), y.std(ddof=1), z.std(ddof=1)]
    features += [
        np.mean(np.abs(x - x.mean())),
        np.mean(np.abs(y - y.mean())),
        np.mean(np.abs(z - z.mean())),
    ]
    features.append(np.mean(np.sqrt(x**2 + y**2 + z**2)))
    for axis in (x, y, z):
        # matches MATLAB hist(axis, 10): bin edges padded by half a bin width
        # beyond min/max, unlike numpy's default edges-at-min/max histogram.
        lo, hi = axis.min(), axis.max()
        binwidth = (hi - lo) / 10 or 1.0
        edges = np.linspace(lo - binwidth / 2, hi + binwidth / 2, 11)
        counts, _ = np.histogram(axis, bins=edges)
        features.extend(counts / n)
    return np.array(features)


def find_block_starts(activity):
    n = len(activity)
    changes = np.flatnonzero(activity[:-1] != activity[1:]) + 1
    return np.concatenate(([0], changes, [n]))


def segment(user_id, activity, x, y, z, segment_length):
    block_starts = find_block_starts(activity)

    window_starts = []
    for i in range(len(block_starts) - 1):
        k = block_starts[i]
        block_end = block_starts[i + 1]
        while k + segment_length < block_end:
            window_starts.append(k)
            k += segment_length // 2

    n_windows = len(window_starts)
    data_x = np.empty((n_windows, segment_length))
    data_y = np.empty((n_windows, segment_length))
    data_z = np.empty((n_windows, segment_length))
    features = np.empty((n_windows, 40))
    labels = np.empty(n_windows, dtype=int)
    users = np.empty(n_windows, dtype=int)

    for i, k in enumerate(window_starts):
        xs = x[k : k + segment_length]
        ys = y[k : k + segment_length]
        zs = z[k : k + segment_length]
        data_x[i] = xs
        data_y[i] = ys
        data_z[i] = zs
        features[i] = extract_basic_features(xs, ys, zs)
        labels[i] = ACTIVITY_TO_LABEL[activity[k]]
        users[i] = user_id[k]

    return data_x, data_y, data_z, features, labels, users


def one_hot(labels):
    vectors = np.zeros((len(labels), N_CLASSES))
    vectors[np.arange(len(labels)), labels - 1] = 1
    return vectors


def write_csv(path, array, fmt):
    np.savetxt(path, array, delimiter=",", fmt=fmt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-length", type=int, default=200)
    parser.add_argument("--test-user-start", type=int, default=27)
    parser.add_argument("--test-user-end", type=int, default=36)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    print("loading the data")
    df = pd.read_csv(
        RAW_DATA_PATH,
        header=None,
        names=["user", "activity", "time", "x", "y", "z"],
        usecols=[0, 1, 2, 3, 4, 5],
        engine="python",
        on_bad_lines="skip",
    )
    df = df[df["activity"].isin(ACTIVITY_TO_LABEL)]

    user_id = df["user"].to_numpy()
    activity = df["activity"].to_numpy()
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    z = df["z"].to_numpy()

    print("generating new dataset")
    data_x, data_y, data_z, features, labels, users = segment(
        user_id, activity, x, y, z, args.segment_length
    )
    labels_vec = one_hot(labels)

    test_mask = (users >= args.test_user_start) & (users <= args.test_user_end)

    os.makedirs(OUT_DIR, exist_ok=True)
    L = args.segment_length

    print("writing test data to file")
    write_csv(os.path.join(OUT_DIR, f"answers_test_{L}.csv"), labels[test_mask], "%d")
    write_csv(os.path.join(OUT_DIR, f"answers_vectors_test_{L}.csv"), labels_vec[test_mask], "%d")
    write_csv(os.path.join(OUT_DIR, f"data_x_test_{L}.csv"), data_x[test_mask], "%.4f")
    write_csv(os.path.join(OUT_DIR, f"data_y_test_{L}.csv"), data_y[test_mask], "%.4f")
    write_csv(os.path.join(OUT_DIR, f"data_z_test_{L}.csv"), data_z[test_mask], "%.4f")
    write_csv(os.path.join(OUT_DIR, f"basic_features_test_{L}.csv"), features[test_mask], "%.4f")

    train_mask = ~test_mask
    shuffle = np.random.permutation(train_mask.sum())

    print("writing training data to file")
    write_csv(os.path.join(OUT_DIR, f"answers_{L}.csv"), labels[train_mask][shuffle], "%d")
    write_csv(os.path.join(OUT_DIR, f"answers_vectors_{L}.csv"), labels_vec[train_mask][shuffle], "%d")
    write_csv(os.path.join(OUT_DIR, f"data_x_{L}.csv"), data_x[train_mask][shuffle], "%.4f")
    write_csv(os.path.join(OUT_DIR, f"data_y_{L}.csv"), data_y[train_mask][shuffle], "%.4f")
    write_csv(os.path.join(OUT_DIR, f"data_z_{L}.csv"), data_z[train_mask][shuffle], "%.4f")
    write_csv(os.path.join(OUT_DIR, f"basic_features_{L}.csv"), features[train_mask][shuffle], "%.4f")

    print(f"training and test data was generated ({train_mask.sum()} train / {test_mask.sum()} test windows)")


if __name__ == "__main__":
    main()
