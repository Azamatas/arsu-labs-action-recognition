"""BiFIN: bilinear feature-imitating network for activity recognition.

    config.py    ModelConfig and the presets BIFIN (paper) and BITFIN (smallest, circular)
    targets.py   the temporal features the FIN learns to imitate
    features.py  statistical features fed to the head next to the FIN embeddings
    model.py     FIN and Classifier (PyTorch)
    spectral.py  the circular FIN as one linear map of the power spectrum
    cost.py      parameters and FLOPs
    train.py     training loops
"""
