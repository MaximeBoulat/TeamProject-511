from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

from Globals import IDX2LABEL, PR_TIME, SEQ_LEN, TARGET_COMPOSERS


class FeatureExtractor:
    SEQ_STRIDE = 50           # hop between windows (controls overlap / augmentation)
    PR_FS = 8                 # piano-roll sampling frequency (frames per second)
    PR_STRIDE = 64             # hop between piano-roll windows
    MAX_WINDOWS_PER_FILE = 20  # keeps very long pieces from dominating a split
    CACHE_DIR = Path("build/cache")
    VISUALIZATIONS_DIR = Path("build/visualizations/feature_extraction")

    def __init__(self, seq_len=SEQ_LEN, seq_stride=SEQ_STRIDE, pr_time=PR_TIME,
                 pr_fs=PR_FS, pr_stride=PR_STRIDE,
                 max_windows_per_file=MAX_WINDOWS_PER_FILE,
                 cache_dir=CACHE_DIR, visualizations_dir=VISUALIZATIONS_DIR):
        self.seq_len = seq_len
        self.seq_stride = seq_stride
        self.pr_time = pr_time
        self.pr_fs = pr_fs
        self.pr_stride = pr_stride
        self.max_windows_per_file = max_windows_per_file
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations_dir = Path(visualizations_dir)
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)

    def extract_both(self, path):
        try:
            pm = pretty_midi.PrettyMIDI(path)
        except Exception:
            return None, None
        notes = [(n.start, n.pitch) for inst in pm.instruments
                 if not inst.is_drum for n in inst.notes]
        seq = None
        if len(notes) >= self.seq_len:
            notes.sort(key=lambda x: x[0])
            seq = np.array([p for _, p in notes], dtype=np.int16)
        roll = None
        if len(pm.instruments) > 0:
            r = pm.get_piano_roll(fs=self.pr_fs)            # (128, T) velocity
            if r.shape[1] >= self.pr_time:
                roll = (r > 0).astype(np.uint8)              # binarize
        return seq, roll

    def sequence_windows(self, seq):
        out = []
        for start in range(0, len(seq) - self.seq_len + 1, self.seq_stride):
            out.append(seq[start:start + self.seq_len])
            if len(out) >= self.max_windows_per_file:
                break
        return out

    def roll_windows(self, roll):
        out = []
        for start in range(0, roll.shape[1] - self.pr_time + 1, self.pr_stride):
            out.append(roll[:, start:start + self.pr_time])
            if len(out) >= self.max_windows_per_file:
                break
        return out

    def build_split_arrays(self, files, labels):
        seq_X, seq_y, seq_fid, skipped_seq = [], [], [], 0
        cnn_X, cnn_y, cnn_fid, skipped_cnn = [], [], [], 0
        for fid, (path, lab) in enumerate(zip(files, labels)):
            seq, roll = self.extract_both(path)
            if seq is None:
                skipped_seq += 1
            else:
                for w in self.sequence_windows(seq):
                    seq_X.append(w); seq_y.append(lab); seq_fid.append(fid)
            if roll is None:
                skipped_cnn += 1
            else:
                for w in self.roll_windows(roll):
                    cnn_X.append(w); cnn_y.append(lab); cnn_fid.append(fid)
        Xseq = np.array(seq_X, dtype=np.int16)
        Xcnn = np.array(cnn_X, dtype=np.uint8)[..., np.newaxis]   # add channel dim
        return {
            "seq": (Xseq, np.array(seq_y, dtype=np.int32),
                    np.array(seq_fid, dtype=np.int64), skipped_seq),
            "cnn": (Xcnn, np.array(cnn_y, dtype=np.int32),
                    np.array(cnn_fid, dtype=np.int64), skipped_cnn),
        }

    def cached_split_arrays(self, split, files, labels):
        p = self.cache_dir / f"{split}.npz"
        if p.exists():
            z = np.load(p)
            return {"seq": (z["sX"], z["sy"], z["sfid"], int(z["sskip"])),
                    "cnn": (z["cX"], z["cy"], z["cfid"], int(z["cskip"]))}
        out = self.build_split_arrays(files, labels)
        (sX, sy, sfid, sskip), (cX, cy, cfid, cskip) = out["seq"], out["cnn"]
        np.savez_compressed(p, sX=sX, sy=sy, sfid=sfid, sskip=sskip,
                             cX=cX, cy=cy, cfid=cfid, cskip=cskip)
        return out

    @staticmethod
    def class_distribution(y):
        return {IDX2LABEL[i]: int((y == i).sum()) for i in range(len(TARGET_COMPOSERS))}

    @staticmethod
    def class_weights(y):
        classes = np.arange(len(TARGET_COMPOSERS))
        w = compute_class_weight("balanced", classes=classes, y=y)
        return {int(c): float(wi) for c, wi in zip(classes, w)}

    def visualize(self, yseq_train, ycnn_train):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for a, (y, t) in zip(ax, [(yseq_train, "LSTM train windows"),
                                  (ycnn_train, "CNN train windows")]):
            vals = [(y == i).sum() for i in range(len(TARGET_COMPOSERS))]
            a.bar(TARGET_COMPOSERS, vals, color=sns.color_palette("deep", len(TARGET_COMPOSERS)))
            a.set_title(t)
            a.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        out_path = self.visualizations_dir / "window_class_distribution.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
