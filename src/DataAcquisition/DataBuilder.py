from pathlib import Path

import pandas as pd

from Globals import TARGET_COMPOSERS


class DataBuilder:
    DATASET_PATH = "build/datasets/dataset_4composers"
    MIDI_GLOBS = ("*.mid", "*.midi", "*.MID", "*.MIDI")

    def __init__(self, dataset_path=DATASET_PATH, composers=TARGET_COMPOSERS):
        self.dataset_path = Path(dataset_path)
        self.composers = composers

    def verify_layout(self):
        if not self.dataset_path.is_dir():
            raise FileNotFoundError(
                f"DATASET_PATH does not exist: {self.dataset_path}\n"
                "Set DataBuilder.DATASET_PATH to the folder that contains the "
                "composer sub-folders (Bach/, Beethoven/, Chopin/, Mozart/)."
            )
        counts = {}
        for comp in self.composers:
            comp_dir = self.dataset_path / comp
            if comp_dir.is_dir():
                counts[comp] = sum(1 for ext in self.MIDI_GLOBS for _ in comp_dir.rglob(ext))
            else:
                counts[comp] = None
        return counts

    def infer_composer(self, path):
        matches = [c for c in self.composers if c.lower() in str(path).lower()]
        return matches[0] if len(matches) == 1 else None

    def build_index(self):
        records = []
        for ext in self.MIDI_GLOBS:
            for f in self.dataset_path.rglob(ext):
                comp = self.infer_composer(f)
                if comp is not None:
                    records.append({"path": str(f), "composer": comp})
        df = (pd.DataFrame(records)
                .drop_duplicates(subset="path")
                .sort_values("path")
                .reset_index(drop=True))
        if len(df) == 0:
            raise RuntimeError(
                "No MIDI files found. Check dataset_path and its contents.")
        return df
