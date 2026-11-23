"""Raw datasets (datasets/) -> windows of shape (n, channels, time) at 20 Hz.

    wisdm.windows()          WISDM v1.1, accelerometer, 10 s windows
    motionsense.windows()    MotionSense, accelerometer, 10 s windows
    daghar.read(domain, split)   DAGHAR standardized view, accelerometer + gyroscope, 3 s windows
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = os.path.join(ROOT, "datasets")
