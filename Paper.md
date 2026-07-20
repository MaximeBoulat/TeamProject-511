# Paper

# Abstract

Identifying the composer of a musical piece is difficult even for trained listeners. This project builds deep learning classifiers that predict the composer of a musical score directly from its MIDI file, comparing the two architectures required by the project brief: a Long Short-Term Memory (LSTM) network and a Convolutional Neural Network (CNN). Working from the Kaggle *midi-classic-music* corpus filtered to Bach, Beethoven, Chopin, and Mozart (1,637 files in the full corpus; 300 files used after a 75-file-per-composer cap for computational feasibility), we clean the data (duplicate removal by content hash, corrupt-file exclusion), split at the file level to prevent leakage, and convert each piece into many fixed-length training windows. The two models consume complementary views of the same music: the LSTM reads ordered pitch-token sequences through an embedding layer, while the CNN reads binarized piano-roll windows as single-channel images. Training uses class weights to counter composer imbalance, transposition-based data augmentation, early stopping, and a small grid search over learning rate and dropout. Models are evaluated with accuracy, precision, recall, and F1 (macro and weighted) plus per-composer confusion matrices.

# Data Collection

- **Source.** Kaggle dataset `[blanderbuss/midi-classic-music](https://www.kaggle.com/datasets/blanderbuss/midi-classic-music)` (Fedorak, 2019) — a corpus of classical MIDI files spanning 175 composers (3,929 files). Per the project instructions, we filter to four target composers: **Bach, Beethoven, Chopin, and Mozart**, yielding **1,637 MIDI files** in the full corpus. Due to WSL memory constraints (fully materializing piano-roll windows as NumPy arrays at full scale would exceed 10 GB), we apply a **75-file-per-composer cap**, using **300 files** for all experiments.
- **Format.** MIDI is symbolic, not audio: each file is a list of timed note events (note-on/note-off, pitch 0–127, velocity) per instrument. This removes the need for audio signal processing and lets us derive features directly from note events.
- **Labels.** The composer label is inferred from the file path (each composer has a dedicated folder; the parser applies an exactly-one-match rule so ambiguous paths are excluded).
- **Per-composer profile** (from the team EDA):


Full corpus profile (from EDA on all 1,637 files):

| Composer  | Full corpus files | Used (capped) | Median duration (s) | Mean notes/s | Mean instruments |
| --------- | ----------------- | ------------- | ------------------- | ------------ | ---------------- |
| Bach      | 1,024             | 75            | 77                  | 9.4          | 4.8              |
| Beethoven | 220               | 75            | 410                 | 14.2         | 6.9              |
| Chopin    | 136               | 75            | 158                 | 11.2         | 2.5              |
| Mozart    | 257               | 75            | 348                 | 13.8         | 7.3              |

After the cap, the 300 files are split 70/15/15 (train 210, val 45, test 45), stratified by composer. Despite equal file counts per composer, the *window* distribution is unequal because piece lengths differ greatly. Actual CNN training-window shares: **Beethoven 30.1%, Mozart 28.9%, Chopin 24.1%, Bach 16.9%**. Bach's short median duration (77 s vs Beethoven's 410 s) means 75 Bach files yield far fewer windows than 75 Beethoven files — reversing the file-level imbalance. This directly motivates the class-weighting strategy in training.

# Data Pre-processing

1. **Parse validation.** Every file is opened with `pretty_midi`; unreadable files are dropped. EDA on the full 1,637-file corpus found 2 corrupt files (both `KeySignatureError`), a 99.9% parse success rate.
2. **De-duplication.** Files are hashed (MD5) and exact byte-duplicates removed — EDA found 21 duplicate groups (42 files), all within-composer (e.g., the same Chopin étude under two filenames). Removing them prevents the same piece from appearing on both sides of a split.
3. **File cap.** Before splitting, a random 75-file-per-composer cap is applied (seed 42), reducing the working set to 300 files. This keeps peak RAM under ~1.5 GB when the full window arrays are materialized.
4. **Leakage-safe splitting.** The train/validation/test split (70/15/15) is performed at the *file* level, stratified by composer, with an assertion that no file appears in two splits. Actual counts: train 210, val 45, test 45. Windowing happens *after* splitting, so windows from one piece can never leak across splits.
5. **Windowing.** Each piece is cut into fixed-length, overlapping windows that inherit the piece's composer label. Window parameters: LSTM — 100 notes, stride 50; CNN — 128 frames (16 s at 8 fps), stride 64 frames. A per-file cap of 20 windows prevents very long Beethoven and Mozart pieces from dominating. Actual training-window counts (with augmentation): LSTM — Bach 3,185 / Beethoven 4,970 / Chopin 4,720 / Mozart 4,745 (total 17,620); CNN — Bach 2,810 / Beethoven 4,985 / Chopin 3,995 / Mozart 4,795 (total 16,585).
6. **Data augmentation.** Training windows (only) are additionally transposed by ±1 and ±2 semitones — pitch shifts preserve compositional style while diversifying the input distribution.
7. **Class weights.** Inverse-frequency class weights are applied during training to counter the window-share imbalance (Bach underrepresented at ~17% of CNN windows due to short piece duration).

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
- **Environment.** TensorFlow/Keras with `pretty_midi` for parsing and scikit-learn for metrics; runs on Google Colab (GPU runtime) or locally. The team also validated a local WSL2 + CUDA GPU setup (TensorFlow 2.16), which required pre-loading the pip-installed NVIDIA libraries and disabling XLA convolution autotuning (`--xla_gpu_autotune_level=0`) to work around a known WSL failure mode.

# Model Evaluation

Evaluation is on the held-out test split, never used during training or model selection.

- **Metrics.** Accuracy, precision, recall (per the project instructions) and F1, reported both macro-averaged (treats all composers equally — sensitive to minority-class failure) and weighted (reflects the actual class mix). Full per-composer classification reports and confusion matrices are produced for both models, along with training/validation curves for over/underfitting diagnosis.
- **Final results — model comparison (test set):**

| Model | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 |
| ----- | -------- | ------- | ------- | -------- | ----------- |
| LSTM  | 0.4044   | 0.3863  | 0.4343  | 0.3826   | 0.3758      |
| CNN   | **0.5971** | **0.6341** | **0.6075** | **0.6167** | **0.5984** |

Chance baseline (4 classes): 25%. The CNN reaches 60% accuracy — 2.39× above chance — while the LSTM reaches 40% — 1.62× above chance. The CNN is the clear winner on every metric.

- **Per-composer breakdown:**

LSTM (732 test windows):

| Composer  | Precision | Recall | F1   | Support |
| --------- | --------- | ------ | ---- | ------- |
| Bach      | 0.34      | 0.68   | 0.46 | 104     |
| Beethoven | 0.37      | 0.23   | 0.28 | 205     |
| Chopin    | 0.48      | 0.64   | 0.55 | 219     |
| Mozart    | 0.35      | 0.18   | 0.24 | 204     |

CNN (680 test windows):

| Composer  | Precision | Recall | F1   | Support |
| --------- | --------- | ------ | ---- | ------- |
| Bach      | **0.85**  | 0.64   | 0.73 | 112     |
| Beethoven | 0.49      | 0.48   | 0.48 | 207     |
| Chopin    | 0.65      | **0.73** | 0.69 | 172   |
| Mozart    | 0.55      | 0.57   | 0.56 | 189     |


# Model Optimization

- **Grid search.** Both models are tuned over a grid of learning rate {1e-3, 5e-4} × dropout {0.3, 0.5}, each configuration trained with the full callback stack and selected by best validation accuracy.

LSTM grid-search results:

| lr     | dropout | best val acc |
| ------ | ------- | ------------ |
| 0.001  | 0.3     | **0.5303**   |
| 0.0005 | 0.3     | 0.3402       |
| 0.001  | 0.5     | 0.3099       |

CNN grid-search results:

| lr     | dropout | best val acc |
| ------ | ------- | ------------ |
| 0.001  | 0.3     | 0.6109       |
| 0.0005 | 0.3     | **0.6331**   |
| 0.001  | 0.5     | 0.6183       |

The LSTM selected `lr=0.001, dropout=0.3`; the CNN selected `lr=0.0005, dropout=0.3`. For the LSTM, higher dropout (0.5) badly hurt validation accuracy (0.31 vs 0.53), suggesting that with the pitch-sequence representation the model is already adequately regularized by the recurrent architecture and class weights — additional dropout at 0.5 merely disrupts learning. For the CNN, a lower starting learning rate (5e-4) outperformed 1e-3, likely because the spiky validation loss under the default rate was slowing the search for a flat optimum. In both cases the plateau scheduler handles dynamic rate decay, but the CNN benefited from a more cautious initial step size.

- **Tuning levers identified but deferred** (documented for future work): window length and stride, piano-roll frame rate, number of LSTM units / CNN filters, richer note tokens combining pitch with duration and velocity, event-based encodings, multi-channel piano rolls separating instruments, and hybrid CRNN architectures (convolutions feeding an LSTM).
- **Future improvements.** More composers, k-fold cross-validation for tighter confidence intervals, cross-dataset validation, and explicit handling of polyphony/voice separation.

# Analysis

*CNN vs. LSTM gap (60% vs. 40%).* The 20-point accuracy gap favors the CNN across every metric. The piano-roll representation exposes spatial structure — chord voicing, rhythmic density, register — that the CNN's convolutional filters exploit directly. The LSTM receives only an ordered sequence of pitch integers, discarding duration, rhythm, and harmonic texture; recovering composer style purely from pitch order is a harder inductive problem. This result confirms the intuition that *how* music looks on a grid is more discriminative than *what notes come in sequence*.

*Beethoven is the hardest composer for both models.* The LSTM achieves only F1=0.28 on Beethoven (recall 0.23); the CNN reaches F1=0.48 (recall 0.48). Beethoven's style bridges Classical and Romantic periods — his harmonic language and textures overlap with Mozart's (Classical) and Chopin's (Romantic), giving the models the least separable signal. Despite the CNN's overall advantage, Beethoven still ranks last for both architectures.

*Bach has the highest CNN precision (0.85).* Baroque counterpoint — multiple independent melodic voices moving simultaneously — produces a visually distinctive piano-roll texture that the convolutional filters identify reliably: when the CNN predicts Bach it is correct 85% of the time. Recall (0.64), however, is only moderate, meaning a meaningful fraction of Bach windows are being classified as other composers (likely those with similarly dense polyphony like Beethoven).

*Chopin achieves the highest CNN recall (0.73).* Romantic piano music features characteristic textures — arpeggiated left-hand accompaniment, sustained pedal, expressive phrasing — that create recognizable temporal patterns in the piano roll. The CNN catches most Chopin windows, though at somewhat lower precision (0.65), indicating some non-Chopin windows share surface similarity.

*Mozart is evenly balanced in the CNN (precision 0.55, recall 0.57).* Earlier experiments at a smaller file cap showed a strong precision/recall asymmetry for Mozart (very high precision, low recall). With 75 files per composer the asymmetry has smoothed out, suggesting that additional Mozart training examples expose the full stylistic range, reducing the false-negative rate while slightly diluting the prototype's purity.

*CNN overfitting pattern.* The CNN training logs show training accuracy reaching ~99% within the first few epochs while validation accuracy oscillated widely before settling — a BatchNormalization-on-sparse-binary-input pathology where running batch statistics are poorly estimated on sparse binary piano rolls. `ModelCheckpoint` recovered a useful model despite the volatile validation loss, but the large train/val gap indicates memorization at the window level. Removing BatchNorm or switching to LayerNorm remains the primary recommended next step.

*LSTM grid-search sensitivity.* Dropout 0.5 dramatically reduced LSTM validation accuracy (0.31 vs 0.53 at dropout 0.3), and halving the learning rate to 5e-4 also hurt (0.34 vs 0.53). The LSTM is sensitive to over-regularization in this setting: with only ~3,500 pre-augmentation training windows per run, strong dropout prevents the recurrent layers from learning composer-specific temporal patterns at all.
