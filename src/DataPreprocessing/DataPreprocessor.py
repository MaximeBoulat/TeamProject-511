import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pretty_midi
import seaborn as sns

from Globals import SEQ_LEN, TARGET_COMPOSERS


class DataPreprocessor:
    VISUALIZATIONS_DIR = Path("build/visualizations/preprocessing")

    def __init__(self, min_notes=SEQ_LEN, visualizations_dir=VISUALIZATIONS_DIR):
        self.min_notes = min_notes
        self.visualizations_dir = Path(visualizations_dir)
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def md5_of(path, chunk=1 << 20):
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(chunk), b""):
                h.update(block)
        return h.hexdigest()

    @staticmethod
    def probe_midi(path):
        try:
            pm = pretty_midi.PrettyMIDI(path)
            return sum(len(inst.notes) for inst in pm.instruments)
        except Exception:
            return -1

    def deduplicate(self, raw_df):
        df = raw_df.copy()
        df["md5"] = df["path"].map(self.md5_of)
        before = len(df)
        dedup_df = df.drop_duplicates(subset="md5").reset_index(drop=True)
        return dedup_df, before - len(dedup_df)

    def filter_usable(self, dedup_df):
        dedup_df = dedup_df.copy()
        dedup_df["n_notes"] = dedup_df["path"].map(self.probe_midi)
        unreadable = int((dedup_df["n_notes"] < 0).sum())
        empty = int((dedup_df["n_notes"] == 0).sum())
        clean_df = dedup_df[dedup_df["n_notes"] >= self.min_notes].reset_index(drop=True)
        stats = {"unreadable": unreadable, "empty": empty, "usable": len(clean_df)}
        return clean_df, stats

    def process(self, raw_df):
        dedup_df, n_dupes = self.deduplicate(raw_df)
        clean_df, stats = self.filter_usable(dedup_df)
        stats["duplicates_removed"] = n_dupes
        return clean_df, dedup_df, stats

    @staticmethod
    def summarize(raw_df, dedup_df, clean_df):
        summary = (pd.DataFrame({
                "raw_matched": raw_df["composer"].value_counts(),
                "after_dedup": dedup_df["composer"].value_counts(),
                "usable": clean_df["composer"].value_counts(),
            })
            .reindex(TARGET_COMPOSERS)
            .fillna(0).astype(int))
        summary.loc["TOTAL"] = summary.sum()
        return summary

    def visualize(self, clean_df):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        clean_df["composer"].value_counts().reindex(TARGET_COMPOSERS).plot(
            kind="bar", ax=ax[0], color=sns.color_palette("deep", len(TARGET_COMPOSERS)))
        ax[0].set_title("Usable files per composer")
        ax[0].set_ylabel("file count")
        ax[0].tick_params(axis="x", rotation=45)
        clean_df["n_notes"].plot(kind="hist", bins=40, ax=ax[1], color="steelblue")
        ax[1].set_title("Distribution of notes per file")
        ax[1].set_xlabel("notes")
        plt.tight_layout()
        out_path = self.visualizations_dir / "class_distribution.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
