"""Activity names and label codes.

WISDM / MotionSense protocols (cross-dataset, within-dataset, data scarcity) use the six WISDM v1.1 activities in the
order of the paper. DAGHAR uses its own "standard activity code" (0 sit, 1 stand, 2 walk, 3 upstairs, 4 downstairs,
5 run), kept as it is so that our numbers stay comparable with Napoli & Borin.
"""
PAPER = ("Jogging", "Walking", "Upstairs", "Downstairs", "Sitting", "Standing")
DAGHAR = ("Sitting", "Standing", "Walking", "Upstairs", "Downstairs", "Running")
SHORT = {"Jogging": "Jog", "Running": "Run", "Walking": "Walk", "Upstairs": "Up", "Downstairs": "Down",
         "Sitting": "Sit", "Standing": "Stand"}
