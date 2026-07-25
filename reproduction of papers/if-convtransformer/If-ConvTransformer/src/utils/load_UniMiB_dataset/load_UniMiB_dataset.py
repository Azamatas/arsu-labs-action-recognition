import scipy.io
import numpy as np
from scipy.signal import resample


def load_UniMiB_data(DATA_DIR, target_hz=None):

    print("[UniMiB Loader] Loading .mat files...")

    mat_data = scipy.io.loadmat(f"{DATA_DIR}/acc_data.mat")
    mat_labels = scipy.io.loadmat(f"{DATA_DIR}/acc_labels.mat")

    X = mat_data["acc_data"]  # (11771, 453)
    y = mat_labels["acc_labels"][:, 0].squeeze()  # (11771,) labels 1-17
    subjects = mat_labels["acc_labels"][:, 1].squeeze()  # (11771,) subject ids

    print(f"[UniMiB Loader] Raw X.shape        = {X.shape}")
    print(f"[UniMiB Loader] Raw y.shape        = {y.shape}")
    print(f"[UniMiB Loader] Raw subjects.shape = {subjects.shape}")

    if len(X.shape) == 2:
        X = X.reshape(-1, 151, 3)
    print(f"[UniMiB Loader] After reshape      = {X.shape}")  # (11771, 151, 3)

    if target_hz is not None:
        n_timesteps_new = int(X.shape[1] * target_hz / 50)
        X_ds = np.zeros((X.shape[0], n_timesteps_new, 3), dtype=np.float32)
        for i in range(X.shape[0]):
            X_ds[i] = resample(X[i], n_timesteps_new, axis=0)
        X = X_ds
        print(f"[UniMiB Loader] After downsample   = {X.shape}")

    X = X.transpose(0, 2, 1).astype(np.float32)
    print(f"[UniMiB Loader] After transpose    = {X.shape}")  # (11771, 3, 151)

    n_samples = X.shape[0]
    window_size = X.shape[2]

    X_padded = np.zeros((n_samples, 9, window_size), dtype=np.float32)
    X_padded[:, 0:3, :] = X  # acc data -> channels 0,1,2

    print(f"[UniMiB Loader] After padding      = {X_padded.shape}")  # (11771, 9, 151)

    X_padded = np.expand_dims(X_padded, axis=1)
    print(f"[UniMiB Loader] After expand_dims  = {X_padded.shape}")  # (11771, 1, 9, 151)

    y = (y - 1).astype(np.int64)
    print(f"[UniMiB Loader] y unique labels    = {np.unique(y)}")
    print("[UniMiB Loader] Done! Final shapes:")
    print(f"  X_padded : {X_padded.shape}")
    print(f"  y        : {y.shape}")
    print(f"  subjects : {subjects.shape}")

    return X_padded, y, subjects
