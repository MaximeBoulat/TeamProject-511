from sklearn.model_selection import train_test_split

from Globals import IDX2LABEL, LABEL2IDX, SEED, TARGET_COMPOSERS


class DataSplitter:
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15  # fraction of the WHOLE dataset (taken from train remainder)

    def __init__(self, test_size=TEST_SIZE, val_size=VAL_SIZE, seed=SEED):
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed

    def split(self, clean_df):
        files_all = clean_df["path"].values
        labels_all = clean_df["composer"].map(LABEL2IDX).values

        f_trainval, f_test, y_trainval, y_test = train_test_split(
            files_all, labels_all, test_size=self.test_size,
            stratify=labels_all, random_state=self.seed)
        val_relative = self.val_size / (1.0 - self.test_size)
        f_train, f_val, y_train, y_val = train_test_split(
            f_trainval, y_trainval, test_size=val_relative,
            stratify=y_trainval, random_state=self.seed)

        assert set(f_train) & set(f_test) == set()
        assert set(f_train) & set(f_val) == set()
        assert set(f_val) & set(f_test) == set()

        return {
            "train": (f_train, y_train),
            "val": (f_val, y_val),
            "test": (f_test, y_test),
        }

    @staticmethod
    def describe(splits):
        report = {}
        for name, (files, y) in splits.items():
            dist = {IDX2LABEL[i]: int((y == i).sum()) for i in range(len(TARGET_COMPOSERS))}
            report[name] = {"n_files": len(files), "class_counts": dist}
        return report
