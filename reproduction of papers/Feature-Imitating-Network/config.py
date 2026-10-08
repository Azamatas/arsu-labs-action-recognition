SEGMENT = 200
AXES = ("x", "y", "z")
ACTIVITIES = ("Jogging", "Walking", "Upstairs", "Downstairs", "Sitting", "Standing")
N_CLASSES = len(ACTIVITIES)
SPLITS = {"train": range(1, 21), "val": range(21, 27), "test": range(27, 37)}
FEATURES = ("mean", "std", "mad")

# windowing and temporal features
FS = 20.0
MIN_LAG, MAX_LAG = 5, 60
BANDS = {"band_lo": (0.0, 3.0), "band_mid": (3.0, 6.0), "band_hi": (6.0, 10.0)}
TEMPORAL = ("mcr", "ac_peak", "ac_lag", "dom_freq", "spec_peak", "spec_entropy",
            "jerk_std", "jerk_mad", "band_lo", "band_mid", "band_hi")
TOP3 = ("spec_entropy", "ac_peak", "spec_peak")


# unfrozen/gradual/frozen regimes
SEED = 42
SEEDS = (42, 0, 1, 2, 3)

FIN_EPOCHS = 60
FIN_LR = 1e-3
HEAD_EPOCHS = 60
FREEZE_EPOCHS = 8
HEAD_LAYERS = 2
HEAD_WIDTH = 128
HEAD_LR = 1e-3
BATCH_SIZE = 256
