# Paper

# Abstract

Identifying the composer of a musical piece is difficult even for trained listeners. This project builds deep learning classifiers that predict the composer of a musical score directly from its MIDI file, comparing the two architectures required by the project brief: a Long Short-Term Memory (LSTM) network and a Convolutional Neural Network (CNN). Working from the Kaggle *midi-classic-music* corpus filtered to Bach, Beethoven, Chopin, and Mozart (1,637 files), we clean the data (duplicate removal by content hash, corrupt-file exclusion), split at the file level to prevent leakage, and convert each piece into many fixed-length training windows. The two models consume complementary views of the same music: the LSTM reads ordered pitch-token sequences through an embedding layer, while the CNN reads binarized piano-roll windows as single-channel images. Training uses class weights to counter composer imbalance, transposition-based data augmentation, early stopping, and a small grid search over learning rate and dropout. Models are evaluated with accuracy, precision, recall, and F1 (macro and weighted) plus per-composer confusion matrices. Final test metrics to be inserted once the team notebook has been executed end to end. A preliminary pilot study on a 9-composer dataset showed an LSTM reaching 57.1% piece-level accuracy against an 11% chance baseline, and surfaced training pathologies (BatchNormalization on sparse binary input, duration-driven class imbalance) that directly shaped the final pipeline.

# Data Collection

- **Source.** Kaggle dataset `[blanderbuss/midi-classic-music](https://www.kaggle.com/datasets/blanderbuss/midi-classic-music)` (Fedorak, 2019) — a corpus of classical MIDI files spanning 175 composers (~~3,929 files). Per the project instructions, we filter to four target composers: **Bach, Beethoven, Chopin, and Mozart**, yielding **1,637 MIDI files (~~43 MB)**.
- **Format.** MIDI is symbolic, not audio: each file is a list of timed note events (note-on/note-off, pitch 0–127, velocity) per instrument. This removes the need for audio signal processing and lets us derive features directly from note events.
- **Labels.** The composer label is inferred from the file path (each composer has a dedicated folder; the parser applies an exactly-one-match rule so ambiguous paths are excluded).
- **Per-composer profile** (from the team EDA):


| Composer  | Files | Total hours | Median duration (s) | Mean notes/s | Mean instruments |
| --------- | ----- | ----------- | ------------------- | ------------ | ---------------- |
| Bach      | 1,024 | 44.4        | 77                  | 9.4          | 4.8              |
| Beethoven | 219   | 31.3        | 410                 | 14.2         | 6.9              |
| Chopin    | 136   | 8.3         | 158                 | 11.2         | 2.5              |
| Mozart    | 256   | 28.5        | 348                 | 13.8         | 7.3              |


The corpus is heavily imbalanced at the file level (Bach has 4.7× more files than Chopin), but Bach's pieces are short while Beethoven's and Mozart's are long, so the imbalance in *training windows* is milder though still substantial: projected window shares are Bach 37.3%, Beethoven 29.2%, Mozart 26.2%, **Chopin 7.3%**. This directly motivates the class-weighting strategy in training.

# Data Pre-processing

1. **Parse validation.** Every file is opened with `pretty_midi`; unreadable files are dropped. EDA found 2 corrupt files out of 1,637 (both `KeySignatureError`), a 99.9% parse success rate.
2. **De-duplication.** Files are hashed (MD5) and exact byte-duplicates removed — EDA found 21 duplicate groups (42 files), all within-composer (e.g., the same Chopin étude under two filenames). Removing them prevents the same piece from appearing on both sides of a split.
3. **Leakage-safe splitting.** The train/validation/test split (70/15/15) is performed at the *file* level, stratified by composer, with an assertion that no file appears in two splits. Windowing happens *after* splitting, so windows from one piece can never leak across splits.
4. **Windowing.** A whole piece is a single labeled example, but pieces vary enormously in length (Bach median 77 s vs Beethoven 410 s). Each piece is therefore cut into many fixed-length, overlapping windows that inherit the piece's composer label. This multiplies hundreds of files into thousands of training examples and acts as implicit augmentation. A per-file window cap limits very long pieces from dominating the training distribution.
5. **Data augmentation.** Training windows (only) are additionally transposed by ±1 and ±2 semitones — pitch shifts preserve compositional style while diversifying the input distribution.
6. **Class weights.** Inverse-frequency class weights are applied during training to counter the window-share imbalance documented in the EDA (Chopin at 7.3%).

# Feature Extraction

The same cleaned files feed two complementary representations, one per model:

- **Note-token sequences (LSTM view).** All non-drum notes are sorted by onset time and reduced to their MIDI pitch (0–127). Each training example is a window of 100 consecutive pitch tokens (stride 50, i.e. 50% overlap), passed through a learned embedding layer. This preserves the *order* in which notes arrive — melodic contour and voice-leading — which is the signal an LSTM is built to exploit.
- **Binary piano rolls (CNN view).** Each piece is rendered as a piano roll at 8 frames per second — a 128 × T matrix whose entry (p, t) indicates whether pitch p sounds at frame t — then binarized and cut into 128-frame (16 s) windows with 50% overlap. Composer style shows up as visual texture: chord shapes, voice spacing, rhythmic density. The window becomes a 128 × 128 single-channel image.

Features beyond pitch and timing (tempo estimates, instrument counts, note density, pitch range) were computed during EDA to characterize the composers; they confirm the classes are separable in aggregate (e.g., Bach's low note density and Beethoven's high polyphony) but are not fed to the models directly.

# Model Building

Both models end in a softmax over the 4 composers and are trained with sparse categorical cross-entropy and the Adam optimizer.

- **LSTM.** `Embedding(129 → 64)` → `LSTM(128, return_sequences=True)` → `LSTM(64)` → `Dense(64, relu)` → `Dropout` → `Dense(4, softmax)`. Recurrent dropout is applied inside both LSTM layers.
- **CNN.** Three convolutional blocks — `Conv2D(32/64/128, 3×3)` each followed by `BatchNormalization` and `MaxPooling2D` — then `GlobalAveragePooling2D` → `Dense(128, relu)` → `Dropout` → `Dense(4, softmax)`.

**Design lesson from the pilot study.** In a first pass on a different, 9-composer dataset (NN_midi_files_extended), a CNN with BatchNormalization trained on sparse binary piano rolls diverged catastrophically at inference time: training accuracy climbed to 79% while validation accuracy stayed at chance, because the running batch statistics were badly estimated on sparse binary input. Combined with aggressive early stopping (patience 5 on validation loss), best-weights restore returned an effectively untrained network that predicted a single class. The final pipeline mitigates this with longer patience, learning-rate reduction on plateau, and checkpoint-based model selection; if the pathology reappears on the 4-composer data, the documented fallback is removing BatchNorm from the convolutional blocks.

# Model Training

- **Optimizer/loss.** Adam, sparse categorical cross-entropy, batch size 64, up to 60 epochs.
- **Callbacks.** Early stopping on validation loss (patience 8) with best-weights restore; `ModelCheckpoint` saving the best model to disk; `ReduceLROnPlateau` (halve the learning rate after 3 stalled epochs, floor 1e-5) — the pilot study showed LSTM training oscillating between 31% and 51% accuracy across epochs at a fixed 1e-3 learning rate, which the plateau scheduler addresses.
- **Imbalance handling.** Class weights computed from the training-window distribution, plus the per-file window cap, address the two imbalance mechanisms found in EDA (file count and piece duration).
- **Reproducibility.** All random seeds (Python, NumPy, TensorFlow) fixed at 42; models are rebuilt with re-seeded initializers for every grid-search configuration.
- **Environment.** TensorFlow/Keras with `pretty_midi` for parsing and scikit-learn for metrics; runs on Google Colab (GPU runtime) or locally. The team also validated a local WSL2 + CUDA GPU setup (TensorFlow 2.21), which required pre-loading the pip-installed NVIDIA libraries and disabling XLA convolution autotuning (`--xla_gpu_autotune_level=0`) to work around a known WSL failure mode.

# Model Evaluation

Evaluation is on the held-out test split, never used during training or model selection.

- **Metrics.** Accuracy, precision, recall (per the project instructions) and F1, reported both macro-averaged (treats all composers equally — sensitive to minority-class failure) and weighted (reflects the actual class mix). Full per-composer classification reports and confusion matrices are produced for both models, along with training/validation curves for over/underfitting diagnosis.
- **Final results.** To be inserted after the team notebook is executed end to end: the model-comparison table (test accuracy, macro-F1, weighted-F1 for LSTM vs CNN), the two confusion matrices, and the easiest/most-confused composer pairs.
- **Pilot-study results (9-composer dataset, piece-level).** The LSTM reached **57.1% accuracy** (macro precision 0.583, macro recall 0.556) against an 11% chance baseline. The error pattern was musically coherent: Bach and Byrd — the most distinctive textures (Baroque counterpoint, Renaissance polyphony) — were classified perfectly, while the classical/romantic cluster (Chopin, Handel, Hummel, Mendelssohn, Mozart) smeared together. Aggregating window predictions to piece level by averaging window probabilities lifted accuracy from 50.4% to 57.1%, confirming aggregation as a cheap, reliable gain. Composers with short pieces (Bartok) were starved of training windows and scored zero — the failure that motivated class weights and window caps in the final pipeline.

# Model Optimization

- **Grid search.** Both models are tuned over a small grid of learning rate {1e-3, 5e-4} × dropout {0.3, 0.5}, each configuration trained with the full callback stack and selected by best validation accuracy. Winning configurations and the grid-search tables to be inserted from the executed notebook.
- **Tuning levers identified but deferred** (documented for future work): window length and stride, piano-roll frame rate, number of LSTM units / CNN filters, richer note tokens combining pitch with duration and velocity, event-based encodings, multi-channel piano rolls separating instruments, and hybrid CRNN architectures (convolutions feeding an LSTM).
- **Future improvements.** More composers, k-fold cross-validation for tighter confidence intervals, cross-dataset validation, and explicit handling of polyphony/voice separation.

