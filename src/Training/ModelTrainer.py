import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers

from Globals import PITCH_VOCAB, PR_PITCH, PR_TIME, SEQ_LEN, TARGET_COMPOSERS, set_seeds


class _HistoryLike:
    def __init__(self, data):
        self.history = data


class ModelTrainer:
    MAX_EPOCHS = 60
    PATIENCE = 8
    ARTIFACTS_DIR = Path("build/artifacts")            # model checkpoints + training histories
    VISUALIZATIONS_DIR = Path("build/visualizations/training")

    LSTM_GRID = [
        {"lr": 1e-3, "dropout": 0.3},
        {"lr": 5e-4, "dropout": 0.3},
        {"lr": 1e-3, "dropout": 0.5},
    ]
    CNN_GRID = [
        {"lr": 1e-3, "dropout": 0.3},
        {"lr": 5e-4, "dropout": 0.3},
        {"lr": 1e-3, "dropout": 0.5},
    ]

    def __init__(self, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                 artifacts_dir=ARTIFACTS_DIR, visualizations_dir=VISUALIZATIONS_DIR,
                 lstm_grid=LSTM_GRID, cnn_grid=CNN_GRID):
        self.max_epochs = max_epochs
        self.patience = patience
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations_dir = Path(visualizations_dir)
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)
        self.lstm_grid = lstm_grid
        self.cnn_grid = cnn_grid

        # ~2x faster LSTM/CNN training on NVIDIA GPUs; harmless elsewhere.
        # Models keep their softmax in float32, as Keras recommends.
        if tf.config.list_physical_devices("GPU"):
            tf.keras.mixed_precision.set_global_policy("mixed_float16")

    def build_lstm(self, units1=128, units2=64, emb_dim=64, dropout=0.3, lr=1e-3):
        set_seeds()
        model = models.Sequential([
            layers.Input(shape=(SEQ_LEN,)),
            layers.Embedding(input_dim=PITCH_VOCAB + 1, output_dim=emb_dim, mask_zero=False),
            layers.LSTM(units1, return_sequences=True, dropout=dropout),
            layers.LSTM(units2, dropout=dropout),
            layers.Dense(64, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(len(TARGET_COMPOSERS), activation="softmax", dtype="float32"),
        ], name="LSTM_composer_classifier")
        model.compile(optimizer=optimizers.Adam(lr),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def build_cnn(self, f1=32, f2=64, f3=128, dropout=0.3, lr=1e-3):
        set_seeds()
        model = models.Sequential([
            layers.Input(shape=(PR_PITCH, PR_TIME, 1)),
            layers.Conv2D(f1, 3, padding="same", activation="relu"),
            layers.BatchNormalization(), layers.MaxPooling2D(2),
            layers.Conv2D(f2, 3, padding="same", activation="relu"),
            layers.BatchNormalization(), layers.MaxPooling2D(2),
            layers.Conv2D(f3, 3, padding="same", activation="relu"),
            layers.BatchNormalization(), layers.MaxPooling2D(2),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(len(TARGET_COMPOSERS), activation="softmax", dtype="float32"),
        ], name="CNN_composer_classifier")
        model.compile(optimizer=optimizers.Adam(lr),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def train_model(self, build_fn, tr_ds, va_ds, class_weight, tag, **kw):
        set_seeds()
        model = build_fn(**kw)
        ckpt_path = str(self.artifacts_dir / f"best_{tag}.keras")
        cbs = [
            callbacks.EarlyStopping(monitor="val_loss", patience=self.patience,
                                     restore_best_weights=True),
            callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
        ]
        history = model.fit(
            tr_ds, validation_data=va_ds, epochs=self.max_epochs,
            class_weight=class_weight, callbacks=cbs, verbose=2)
        with open(self.artifacts_dir / f"history_{tag}.json", "w") as fh:
            json.dump(history.history, fh)
        return model, history

    def grid_search(self, build_fn, grid, tr_ds, va_ds, cw, tag):
        rows, best = [], None
        for i, params in enumerate(grid):
            cfg_tag = f"{tag}_cfg{i}"
            model_path = self.artifacts_dir / f"best_{cfg_tag}.keras"
            history_path = self.artifacts_dir / f"history_{cfg_tag}.json"
            if model_path.exists():
                model = tf.keras.models.load_model(str(model_path))
                if history_path.exists():
                    with open(history_path) as fh:
                        hist_data = json.load(fh)
                else:
                    hist_data = {"val_accuracy": [0], "accuracy": [0],
                                 "val_loss": [], "loss": []}
                hist = _HistoryLike(hist_data)
            else:
                model, hist = self.train_model(build_fn, tr_ds, va_ds, cw,
                                               tag=cfg_tag, **params)
            val_acc = max(hist.history["val_accuracy"])
            rows.append({**params, "best_val_acc": round(val_acc, 4)})
            if best is None or val_acc > best["val_acc"]:
                best = {"val_acc": val_acc, "params": params,
                        "model": model, "history": hist}
        table = pd.DataFrame(rows)
        return best, table

    def load_best_from_disk(self, tag, grid):
        best_val_acc, best_idx, best_h = -1, None, None
        for i in range(len(grid)):
            hpath = self.artifacts_dir / f"history_{tag}_cfg{i}.json"
            mpath = self.artifacts_dir / f"best_{tag}_cfg{i}.keras"
            if not (hpath.exists() and mpath.exists()):
                continue
            with open(hpath) as fh:
                h = json.load(fh)
            v = max(h.get("val_accuracy", [0]))
            if v > best_val_acc:
                best_val_acc, best_idx, best_h = v, i, h
        if best_idx is None:
            raise FileNotFoundError(
                f"No complete checkpoints found for '{tag}'. Run training first.")
        model = tf.keras.models.load_model(
            str(self.artifacts_dir / f"best_{tag}_cfg{best_idx}.keras"))
        return model, _HistoryLike(best_h)

    def train_lstm(self, pipelines, class_weight):
        best, table = self.grid_search(self.build_lstm, self.lstm_grid,
                                        pipelines["seq_train"], pipelines["seq_val"],
                                        class_weight, tag="lstm")
        return best["model"], best["history"], table

    def train_cnn(self, pipelines, class_weight):
        best, table = self.grid_search(self.build_cnn, self.cnn_grid,
                                        pipelines["cnn_train"], pipelines["cnn_val"],
                                        class_weight, tag="cnn")
        return best["model"], best["history"], table

    def visualize(self, history, title):
        h = history.history
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(h["loss"], label="train"); ax[0].plot(h["val_loss"], label="val")
        ax[0].set_title(f"{title} - loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
        ax[1].plot(h["accuracy"], label="train"); ax[1].plot(h["val_accuracy"], label="val")
        ax[1].set_title(f"{title} - accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
        plt.tight_layout()
        out_path = self.visualizations_dir / f"{title.lower()}_curves.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
