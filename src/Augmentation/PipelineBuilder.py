import numpy as np
import tensorflow as tf

from Globals import PITCH_VOCAB, PR_PITCH, SEED


class PipelineBuilder:
    BATCH_SIZE = 64
    AUG_TRANSPOSITIONS = [-2, -1, 1, 2]   # semitone shifts; [] disables transposition

    def __init__(self, batch_size=BATCH_SIZE, aug_transpositions=AUG_TRANSPOSITIONS, seed=SEED):
        self.batch_size = batch_size
        self.aug_transpositions = aug_transpositions
        self.seed = seed
        self.shifts = tf.constant([0] + aug_transpositions, dtype=tf.int32)

    def _aug_seq(self, x, y):
        s = self.shifts[tf.random.uniform([], 0, len(self.aug_transpositions) + 1, tf.int32)]
        x = tf.clip_by_value(tf.cast(x, tf.int32) + s, 0, PITCH_VOCAB - 1)
        return x, y

    @staticmethod
    def _shift_roll(x, s):
        # x: (PR_PITCH, PR_TIME, 1); row index = pitch, so shifting rows transposes.
        def up():   return tf.pad(x, [[s, 0], [0, 0], [0, 0]])[:PR_PITCH]
        def down(): return tf.pad(x, [[0, -s], [0, 0], [0, 0]])[-PR_PITCH:]
        return tf.cond(s > 0, up, down)

    def _aug_roll(self, x, y):
        s = self.shifts[tf.random.uniform([], 0, len(self.aug_transpositions) + 1, tf.int32)]
        x = tf.cond(tf.equal(s, 0), lambda: x, lambda: self._shift_roll(x, s))
        return x, y

    def make_ds(self, X, y, kind, training):
        if kind == "seq":
            ds = tf.data.Dataset.from_tensor_slices((X.astype(np.int32), y))
            aug = self._aug_seq
        else:
            ds = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y))
            aug = self._aug_roll
        if training:
            ds = ds.shuffle(min(len(y), 20000), seed=self.seed).map(
                aug, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def build_all(self, train_arr, val_arr, test_arr):
        Xseq_train, yseq_train, _, _ = train_arr["seq"]
        Xseq_val,   yseq_val,   _, _ = val_arr["seq"]
        Xseq_test,  yseq_test,  _, _ = test_arr["seq"]
        Xcnn_train, ycnn_train, _, _ = train_arr["cnn"]
        Xcnn_val,   ycnn_val,   _, _ = val_arr["cnn"]
        Xcnn_test,  ycnn_test,  _, _ = test_arr["cnn"]

        return {
            "seq_train": self.make_ds(Xseq_train, yseq_train, "seq", training=True),
            "seq_val":   self.make_ds(Xseq_val,   yseq_val,   "seq", training=False),
            "seq_test":  self.make_ds(Xseq_test,  yseq_test,  "seq", training=False),
            "cnn_train": self.make_ds(Xcnn_train, ycnn_train, "cnn", training=True),
            "cnn_val":   self.make_ds(Xcnn_val,   ycnn_val,   "cnn", training=False),
            "cnn_test":  self.make_ds(Xcnn_test,  ycnn_test,  "cnn", training=False),
        }
