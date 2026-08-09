from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

from Globals import TARGET_COMPOSERS


class ModelEvaluator:
    VISUALIZATIONS_DIR = Path("build/visualizations/evaluation")

    def __init__(self, visualizations_dir=VISUALIZATIONS_DIR):
        self.visualizations_dir = Path(visualizations_dir)
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, model, ds, y, file_ids, title):
        proba = model.predict(ds, verbose=0)

        # ---- window level ----
        y_pred_w = proba.argmax(axis=1)
        acc_w = accuracy_score(y, y_pred_w)
        pw_macro, rw_macro, fw_macro, _ = precision_recall_fscore_support(
            y, y_pred_w, average="macro", zero_division=0)
        pw_w, rw_w, fw_w, _ = precision_recall_fscore_support(
            y, y_pred_w, average="weighted", zero_division=0)

        # ---- piece level: average window probabilities per file, then argmax ----
        pieces = pd.DataFrame(proba)
        pieces["fid"] = file_ids
        pieces["true"] = y
        agg = pieces.groupby("fid").agg({**{i: "mean" for i in range(len(TARGET_COMPOSERS))},
                                         "true": "first"})
        y_true_p = agg["true"].values
        y_pred_p = agg[list(range(len(TARGET_COMPOSERS)))].values.argmax(axis=1)
        acc_p = accuracy_score(y_true_p, y_pred_p)
        pp_macro, rp_macro, fp_macro, _ = precision_recall_fscore_support(
            y_true_p, y_pred_p, average="macro", zero_division=0)

        print(f"===== {title} =====")
        print(f"Window accuracy       : {acc_w:.4f}  (n={len(y)})")
        print(f"Window macro    P/R/F1: {pw_macro:.4f} / {rw_macro:.4f} / {fw_macro:.4f}")
        print(f"Window weighted P/R/F1: {pw_w:.4f} / {rw_w:.4f} / {fw_w:.4f}")
        print(f"Piece  accuracy       : {acc_p:.4f}  (n={len(y_true_p)})")
        print(f"Piece  macro    P/R/F1: {pp_macro:.4f} / {rp_macro:.4f} / {fp_macro:.4f}\n")
        print("Per-class classification report (piece level):")
        print(classification_report(y_true_p, y_pred_p, target_names=TARGET_COMPOSERS,
                                    zero_division=0))

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        sns.heatmap(confusion_matrix(y, y_pred_w), annot=True, fmt="d", cmap="Blues",
                    xticklabels=TARGET_COMPOSERS, yticklabels=TARGET_COMPOSERS, ax=ax[0])
        ax[0].set_title(f"{title} - window-level confusion")
        ax[0].set_xlabel("predicted"); ax[0].set_ylabel("true")
        sns.heatmap(confusion_matrix(y_true_p, y_pred_p), annot=True, fmt="d", cmap="Blues",
                    xticklabels=TARGET_COMPOSERS, yticklabels=TARGET_COMPOSERS, ax=ax[1])
        ax[1].set_title(f"{title} - piece-level confusion")
        ax[1].set_xlabel("predicted"); ax[1].set_ylabel("true")
        plt.tight_layout()
        fig.savefig(self.visualizations_dir / f"{title.lower()}_confusion.png")
        plt.close(fig)

        return {"model": title,
                "accuracy_window": acc_w, "f1_macro_window": fw_macro, "f1_weighted_window": fw_w,
                "accuracy_piece": acc_p, "precision_macro_piece": pp_macro,
                "recall_macro_piece": rp_macro, "f1_macro_piece": fp_macro}

    def compare(self, lstm_metrics, cnn_metrics):
        comparison = pd.DataFrame([lstm_metrics, cnn_metrics]).set_index("model").round(4)

        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(comparison)); w = 0.35
        ax.bar(x - w/2, comparison["accuracy_window"], w, label="window-level accuracy")
        ax.bar(x + w/2, comparison["accuracy_piece"],  w, label="piece-level accuracy")
        ax.set_xticks(x); ax.set_xticklabels(comparison.index)
        ax.axhline(0.25, ls="--", c="gray", lw=1, label="chance (25%)")
        ax.set_ylim(0, 1); ax.legend(loc="lower right")
        ax.set_title("LSTM vs CNN - window- vs piece-level test accuracy")
        plt.tight_layout()
        fig.savefig(self.visualizations_dir / "model_comparison.png")
        plt.close(fig)

        comparison.to_csv(self.visualizations_dir / "model_comparison.csv")
        return comparison
