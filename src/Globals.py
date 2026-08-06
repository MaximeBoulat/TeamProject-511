import os
import random

import numpy as np
import tensorflow as tf

SEED = 42

TARGET_COMPOSERS = ["Bach", "Beethoven", "Chopin", "Mozart"]
LABEL2IDX = {c: i for i, c in enumerate(TARGET_COMPOSERS)}
IDX2LABEL = {i: c for c, i in LABEL2IDX.items()}

# Consumed by DataPreprocessor (minimum-notes filter), FeatureExtractor
# (LSTM window length), and ModelTrainer (LSTM input shape).
SEQ_LEN = 100

# Consumed by PipelineBuilder (augmentation clipping) and ModelTrainer
# (embedding input_dim).
PITCH_VOCAB = 128       # MIDI pitches 0-127
PAD_TOKEN = 128          # reserved padding index -> embedding input_dim = PITCH_VOCAB + 1

# Consumed by FeatureExtractor (piano-roll window length) and ModelTrainer
# (CNN input shape).
PR_TIME = 128

# Consumed by PipelineBuilder (roll pitch-shift bound) and ModelTrainer
# (CNN input shape).
PR_PITCH = 128           # piano-roll pitch rows (fixed by MIDI)


def set_seeds(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
