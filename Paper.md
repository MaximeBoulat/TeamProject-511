# Paper

# Abstract

Identifying the composer of a musical piece is difficult even for trained listeners. This project builds deep learning classifiers that predict the composer of a musical score directly from its MIDI file, comparing the two architectures required by the project brief: a Long Short-Term Memory (LSTM) network and a Convolutional Neural Network (CNN). Working from the Kaggle *midi-classic-music* corpus filtered to Bach, Beethoven, Chopin, and Mozart (1,637 files in the full corpus; 200 files used after a 50-file-per-composer cap for computational feasibility), we clean the data (duplicate removal by content hash, corrupt-file exclusion), split at the file level to prevent leakage, and convert each piece into many fixed-length training windows. The two models consume complementary views of the same music: the LSTM reads ordered pitch-token sequences through an embedding layer, while the CNN reads binarized piano-roll windows as single-channel images. Training uses class weights to counter composer imbalance, transposition-based data augmentation, early stopping, and a small grid search over learning rate and dropout. Models are evaluated with accuracy, precision, recall, and F1 (macro and weighted) plus per-composer confusion matrices.

# Data Collection

- **Source.** Kaggle dataset `[blanderbuss/midi-classic-music](https://www.kaggle.com/datasets/blanderbuss/midi-classic-music)` (Fedorak, 2019) — a corpus of classical MIDI files spanning 175 composers (3,929 files). Per the project instructions, we filter to four target composers: **Bach, Beethoven, Chopin, and Mozart**, yielding **1,637 MIDI files** in the full corpus. Due to WSL memory constraints (fully materializing piano-roll windows as NumPy arrays at full scale would exceed 10 GB), we apply a **50-file-per-composer cap**, using **200 files** for all experiments.
- **Format.** MIDI is symbolic, not audio: each file is a list of timed note events (note-on/note-off, pitch 0–127, velocity) per instrument. This removes the need for audio signal processing and lets us derive features directly from note events.
- **Labels.** The composer label is inferred from the file path (each composer has a dedicated folder; the parser applies an exactly-one-match rule so ambiguous paths are excluded).
- **Per-composer profile** (from the team EDA):


Full corpus profile (from EDA on all 1,637 files):

| Composer  | Full corpus files | Used (capped) | Median duration (s) | Mean notes/s | Mean instruments |
| --------- | ----------------- | ------------- | ------------------- | ------------ | ---------------- |
| Bach      | 1,024             | 50            | 77                  | 9.4          | 4.8              |
| Beethoven | 219               | 50            | 410                 | 14.2         | 6.9              |
| Chopin    | 136               | 50            | 158                 | 11.2         | 2.5              |
| Mozart    | 256               | 50            | 348                 | 13.8         | 7.3              |

After the cap, the 200 files are split 70/15/15 (train 139, val 31, test 30), stratified by composer. Despite equal file counts per composer, the *window* distribution is unequal because piece lengths differ greatly. Actual CNN training-window shares: **Beethoven 28.8%, Mozart 29.3%, Chopin 25.3%, Bach 16.6%**. Bach's short median duration (77 s vs Beethoven's 410 s) means 50 Bach files yield far fewer windows than 50 Beethoven files — reversing the file-level imbalance. This directly motivates the class-weighting strategy in training.

# Data Pre-processing

1. **Parse validation.** Every file is opened with `pretty_midi`; unreadable files are dropped. EDA on the full 1,637-file corpus found 2 corrupt files (both `KeySignatureError`), a 99.9% parse success rate.
2. **De-duplication.** Files are hashed (MD5) and exact byte-duplicates removed — EDA found 21 duplicate groups (42 files), all within-composer (e.g., the same Chopin étude under two filenames). Removing them prevents the same piece from appearing on both sides of a split.
3. **File cap.** Before splitting, a random 50-file-per-composer cap is applied (seed 42), reducing the working set to 200 files. This keeps peak RAM under ~1.5 GB when the full window arrays are materialized.
4. **Leakage-safe splitting.** The train/validation/test split (70/15/15) is performed at the *file* level, stratified by composer, with an assertion that no file appears in two splits. Actual counts: train 139, val 31, test 30. Windowing happens *after* splitting, so windows from one piece can never leak across splits.
5. **Windowing.** Each piece is cut into fixed-length, overlapping windows that inherit the piece's composer label. Window parameters: LSTM — 100 notes, stride 50; CNN — 128 frames (16 s at 8 fps), stride 64 frames. A per-file cap of 20 windows prevents very long Beethoven and Mozart pieces from dominating. Actual training-window counts: LSTM — Bach 2,000 / Beethoven 3,190 / Chopin 3,270 / Mozart 3,170 (total 11,630); CNN — Bach 1,795 / Beethoven 3,115 / Chopin 2,740 / Mozart 3,165 (total 10,815).
6. **Data augmentation.** Training windows (only) are additionally transposed by ±1 and ±2 semitones — pitch shifts preserve compositional style while diversifying the input distribution.
7. **Class weights.** Inverse-frequency class weights are applied during training to counter the window-share imbalance (Bach underrepresented at 16.6% of CNN windows due to short piece duration).

# Feature Extraction

The same cleaned files feed two complementary representations, one per model:

- **Note-token sequences (LSTM view).** All non-drum notes are sorted by onset time and reduced to their MIDI pitch (0–127). Each training example is a window of 100 consecutive pitch tokens (stride 50, i.e. 50% overlap), passed through a learned embedding layer. This preserves the *order* in which notes arrive — melodic contour and voice-leading — which is the signal an LSTM is built to exploit.
- **Binary piano rolls (CNN view).** Each piece is rendered as a piano roll at 8 frames per second — a 128 × T matrix whose entry (p, t) indicates whether pitch p sounds at frame t — then binarized and cut into 128-frame (16 s) windows with 50% overlap. Composer style shows up as visual texture: chord shapes, voice spacing, rhythmic density. The window becomes a 128 × 128 single-channel image.

Features beyond pitch and timing (tempo estimates, instrument counts, note density, pitch range) were computed during EDA to characterize the composers; they confirm the classes are separable in aggregate (e.g., Bach's low note density and Beethoven's high polyphony) but are not fed to the models directly.

# Model Building

Both models end in a softmax over the 4 composers and are trained with sparse categorical cross-entropy and the Adam optimizer.

- **LSTM.** `Embedding(129 → 64)` → `LSTM(128, return_sequences=True)` → `LSTM(64)` → `Dense(64, relu)` → `Dropout` → `Dense(4, softmax)`. Recurrent dropout is applied inside both LSTM layers.
- **CNN.** Three convolutional blocks — `Conv2D(32/64/128, 3×3)` each followed by `BatchNormalization` and `MaxPooling2D` — then `GlobalAveragePooling2D` → `Dense(128, relu)` → `Dropout` → `Dense(4, softmax)`.


# Model Training

- **Optimizer/loss.** Adam, sparse categorical cross-entropy, batch size 64, up to 60 epochs.
- **Callbacks.** Early stopping on validation loss (patience 8) with best-weights restore; `ModelCheckpoint` saving the best model to disk; `ReduceLROnPlateau` (halve the learning rate after 3 stalled epochs, floor 1e-5) to stabilize training when the loss plateaus.
- **Imbalance handling.** Class weights computed from the training-window distribution, plus the per-file window cap, address the two imbalance mechanisms found in EDA (file count and piece duration).
- **Reproducibility.** All random seeds (Python, NumPy, TensorFlow) fixed at 42; models are rebuilt with re-seeded initializers for every grid-search configuration.
- **Environment.** TensorFlow/Keras with `pretty_midi` for parsing and scikit-learn for metrics; runs on Google Colab (GPU runtime) or locally. The team also validated a local WSL2 + CUDA GPU setup (TensorFlow 2.21), which required pre-loading the pip-installed NVIDIA libraries and disabling XLA convolution autotuning (`--xla_gpu_autotune_level=0`) to work around a known WSL failure mode.

# Model Evaluation

Evaluation is on the held-out test split, never used during training or model selection.

- **Metrics.** Accuracy, precision, recall (per the project instructions) and F1, reported both macro-averaged (treats all composers equally — sensitive to minority-class failure) and weighted (reflects the actual class mix). Full per-composer classification reports and confusion matrices are produced for both models, along with training/validation curves for over/underfitting diagnosis.
- **Final results — model comparison (test set):**

| Model | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 |
| ----- | -------- | ------- | ------- | -------- | ----------- |
| LSTM  | 0.4671   | 0.4322  | 0.4636  | 0.4323   | 0.4417      |
| CNN   | **0.6895** | **0.6846** | **0.6699** | **0.6486** | **0.6650** |

Chance baseline (4 classes): 25%. The CNN reaches 69% accuracy — 2.76× above chance — while the LSTM reaches 47% — 1.87× above chance. The CNN is the clear winner on every metric.

- **Per-composer breakdown:**

LSTM (501 test windows):

| Composer  | Precision | Recall | F1   | Support |
| --------- | --------- | ------ | ---- | ------- |
| Bach      | 0.51      | 0.82   | 0.63 | 103     |
| Beethoven | 0.28      | 0.13   | 0.18 | 98      |
| Chopin    | 0.50      | 0.51   | 0.50 | 160     |
| Mozart    | 0.43      | 0.40   | 0.42 | 140     |

CNN (467 test windows):

| Composer  | Precision | Recall | F1   | Support |
| --------- | --------- | ------ | ---- | ------- |
| Bach      | 0.71      | **0.94** | **0.81** | 103 |
| Beethoven | 0.48      | 0.28   | 0.35 | 93      |
| Chopin    | 0.65      | **0.93** | 0.77 | 139     |
| Mozart    | **0.90**  | 0.53   | 0.67 | 132     |


# Model Optimization

- **Grid search.** Both models are tuned over a grid of learning rate {1e-3, 5e-4} × dropout {0.3, 0.5}, each configuration trained with the full callback stack and selected by best validation accuracy.

LSTM grid-search results:

| lr     | dropout | best val acc |
| ------ | ------- | ------------ |
| 0.001  | 0.3     | 0.4377       |
| 0.0005 | 0.3     | 0.4170       |
| 0.001  | 0.5     | **0.5491**   |

CNN grid-search results:

| lr     | dropout | best val acc |
| ------ | ------- | ------------ |
| 0.001  | 0.3     | 0.5760       |
| 0.0005 | 0.3     | 0.5580       |
| 0.001  | 0.5     | **0.5820**   |

Both models selected `lr=0.001, dropout=0.5`. Higher dropout (0.5 vs 0.3) consistently improved validation accuracy for both architectures, suggesting that with only ~11,000 training windows the models are prone to overfitting and benefit from stronger regularization. Halving the learning rate (`lr=5e-4`) hurt both models — the plateau scheduler already handles rate decay dynamically, so a lower starting rate appears to simply slow convergence without benefit.

- **Tuning levers identified but deferred** (documented for future work): window length and stride, piano-roll frame rate, number of LSTM units / CNN filters, richer note tokens combining pitch with duration and velocity, event-based encodings, multi-channel piano rolls separating instruments, and hybrid CRNN architectures (convolutions feeding an LSTM).
- **Future improvements.** More composers, k-fold cross-validation for tighter confidence intervals, cross-dataset validation, and explicit handling of polyphony/voice separation.

# Analysis

*CNN vs. LSTM gap (69% vs. 47%).* The 22-point accuracy gap favors the CNN across every metric. The piano-roll representation exposes spatial structure — chord voicing, rhythmic density, register — that the CNN's convolutional filters exploit directly. The LSTM receives only an ordered sequence of pitch integers, discarding duration, rhythm, and harmonic texture; recovering composer style purely from pitch order is a harder inductive problem. This result confirms the intuition that *how* music looks on a grid is more discriminative than *what notes come in sequence*.

*Beethoven is the hardest composer for both models.* The LSTM achieves only F1=0.18 on Beethoven (recall 0.13); the CNN reaches F1=0.35 (recall 0.28). Beethoven's style bridges Classical and Romantic periods — his harmonic language and textures overlap with Mozart's (Classical) and Chopin's (Romantic), giving the models the least separable signal. Beethoven is also the only composer for whom the CNN's high precision (0.48) fails to translate into high recall, meaning a substantial fraction of Beethoven windows are being assigned to other classes.

*Bach is the easiest for the CNN (recall 0.94).* Baroque counterpoint — multiple independent melodic voices moving simultaneously — produces a visually distinctive piano-roll texture that the convolutional filters identify reliably. The same distinction is visible in the EDA: Bach's mean note density (9.4 notes/s) is the lowest of the four composers, and the polyphonic layering creates a characteristic vertical pattern in the roll.

*Mozart precision vs. recall asymmetry in the CNN (precision 0.90, recall 0.53).* The CNN is very confident when it predicts Mozart but misses roughly half of Mozart's test windows. This suggests that Mozart windows have a clean, recognizable prototype (high precision) but that style varies enough across pieces that many windows do not match that prototype closely (low recall), leading to misclassification as Beethoven or Chopin.

*CNN overfitting pattern.* The CNN training logs show training accuracy reaching 100% within 6 epochs while validation accuracy plateaued at 55–58% — a BatchNormalization-on-sparse-binary-input pathology where running batch statistics are poorly estimated on sparse binary piano rolls. The best-weights restore via `ModelCheckpoint` recovered a useful model despite the diverging validation loss, but the large train/val gap indicates the model is memorizing window-level patterns rather than fully generalizing. Removing BatchNorm or switching to LayerNorm remains the primary recommended next step.

*LSTM training instability.* The LSTM training logs show oscillating accuracy across epochs at lr=1e-3, which the `ReduceLROnPlateau` scheduler partially corrected. The best configuration (lr=0.001, dropout=0.5) converged slowly over ~18 epochs before early stopping, suggesting the LSTM is extracting a genuine but weak signal from pitch sequences.

