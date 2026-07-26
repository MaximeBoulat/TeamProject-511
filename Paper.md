# Paper

# Abstract

Identifying the composer of a musical piece is difficult even for trained listeners. This project builds deep learning classifiers that predict the composer of a musical score directly from its MIDI file, comparing the two architectures required by the project brief: a Long Short-Term Memory (LSTM) network and a Convolutional Neural Network (CNN). Working from the Kaggle *midi-classic-music* corpus filtered to Bach, Beethoven, Chopin, and Mozart (1,637 matching files; 1,611 usable after cleaning), we clean the data (duplicate removal by content hash, corrupt-file exclusion), split at the file level to prevent leakage, and convert each piece into many fixed-length training windows. Windows are stored unaugmented (`int16` pitch tokens / `uint8` piano-roll frames) with pitch-transposition augmentation applied on the fly inside a `tf.data` pipeline rather than pre-computed as extra float32 copies — this is what lets the **full corpus** train directly, with no per-composer file cap. The two models consume complementary views of the same music: the LSTM reads ordered pitch-token sequences through an embeddin*g layer, while the CNN reads binarized piano-roll windows as single-channel images. Training uses class weights to counter compo*ser imbalance, early stopping, and a small grid search over learning rate and dropout. Models are evaluated at both **window level** (every window scored independently) and **piece level** (a file's window probabilities averaged, then argmax — closer to how the model would actually be used), with accuracy, precision, recall, and F1 (macro and weighted), plus per-composer confusion matrices. On the held-out test set, the CNN reaches 76.1% window-level / 90.1% piece-level accuracy, and the LSTM reaches 65.6% / 80.6% — both far above the 25% chance baseline, with the CNN ahead on every metric.

# Data Collection

- **Source.** Kaggle dataset `[blanderbuss/midi-classic-music](https://www.kaggle.com/datasets/blanderbuss/midi-classic-music)` (Fedorak, 2019) — a corpus of classical MIDI files spanning 175 composers (3,929 files). Per the project instructions, we filter to four target composers: **Bach, Beethoven, Chopin, and Mozart**, yielding **1,637 MIDI files**.
- **No file cap.** An earlier version of this pipeline capped the working set to 75 files per composer (300 files total) because fully materializing augmented piano-roll windows as float32 arrays would exceed 10 GB of RAM. This version stores windows unaugmented and in compact dtypes (`int16` pitch tokens, `uint8` binary piano-roll frames), and applies pitch-transposition augmentation on the fly inside a `tf.data` pipeline instead of pre-computing four transposed copies per window. That removes the RAM bottleneck, so the **full corpus is used directly**.
- **Format.** MIDI is symbolic, not audio: each file is a list of timed note events (note-on/note-off, pitch 0–127, velocity) per instrument. This removes the need for audio signal processing and lets us derive features directly from note events.
- **Labels.** The composer label is inferred from the file path (each composer has a dedicated folder; the parser applies an exactly-one-match rule so ambiguous paths are excluded).
- **Per-composer profile:**


| Composer  | Full corpus files | Usable (after cleaning) | Median duration (s) | Mean notes/s | Mean instruments |
| --------- | ----------------- | ----------------------- | ------------------- | ------------ | ---------------- |
| Bach      | 1,024             | 1,012                   | 77                  | 9.4          | 4.8              |
| Beethoven | 220               | 213                     | 410                 | 14.2         | 6.9              |
| Chopin    | 136               | 132                     | 158                 | 11.2         | 2.5              |
| Mozart    | 257               | 254                     | 348                 | 13.8         | 7.3              |


Duration/rate/instrument statistics are from the team's exploratory data analysis on the full corpus. Bach's file count (1,024) dwarfs the other three composers combined (613), while its median piece duration (77 s) is the shortest by a wide margin. This tension — many short Bach files vs. fewer, longer files for the other three composers — is the central driver of the class-imbalance story below.

# Data Pre-processing

1. **Parse validation.** Every file is opened with `pretty_midi`; unreadable files are dropped. Of the 1,637 matched files, 2 were unreadable/invalid.
2. **De-duplication.** Files are hashed (MD5) and exact byte-duplicates removed — 21 duplicates were found and dropped, all within-composer. Removing them prevents the same piece from appearing on both sides of a split.
3. **Minimum-length filter.** Files with fewer than 100 notes (the LSTM window length) can't yield a single training window and are dropped. After de-duplication and this filter, **1,611 usable files** remain (Bach 1,012, Beethoven 213, Chopin 132, Mozart 254).
4. **Leakage-safe splitting.** The train/validation/test split (70/15/15) is performed at the *file* level, stratified by composer, with an assertion that no file appears in two splits. Actual counts: train 1,127 (Bach 708, Beethoven 149, Chopin 92, Mozart 178), val 242 (Bach 152, Beethoven 32, Chopin 20, Mozart 38), test 242 (identical class counts to val). Windowing happens *after* splitting, so windows from one piece can never leak across splits.
5. **Windowing.** Each piece is cut into fixed-length, overlapping windows that inherit the piece's composer label. Window parameters: LSTM — 100 notes, stride 50; CNN — 128 frames (16 s at 8 fps), stride 64 frames. A per-file cap of 20 windows prevents very long Beethoven and Mozart pieces from dominating. Actual window counts: LSTM — train 16,002 / val 3,363 / test 3,326; CNN — train 14,738 / val 3,146 / test 3,011. No files were skipped at the windowing stage.
6. **Data augmentation.** Training windows (only) are randomly transposed by one of {−2, −1, 0, +1, +2} semitones, resampled every epoch inside the `tf.data` pipeline (rather than pre-computed as five stored copies). Pitch shifts preserve compositional style while diversifying the input distribution; validation and test windows are never augmented.
7. **Class weights.** Inverse-frequency class weights are computed from the training-window counts. LSTM training windows: Bach 8,299 (51.9%), Mozart 3,305 (20.7%), Beethoven 2,732 (17.1%), Chopin 1,666 (10.4%) — class weights {Bach 0.482, Beethoven 1.464, Chopin 2.401, Mozart 1.21}. CNN training windows: Bach 7,371 (50.0%), Mozart 3,260 (22.1%), Beethoven 2,679 (18.2%), Chopin 1,428 (9.7%) — class weights {Bach 0.5, Beethoven 1.375, Chopin 2.58, Mozart 1.13}.

**Imbalance direction reverses without the file cap.** Under the earlier 75-files-per-composer design, equal file counts let piece *duration* dominate the window-level imbalance — Bach's short pieces made it the minority window class despite equal file representation. At full-corpus scale, the corpus's natural per-composer file-count skew dominates instead: Bach alone supplies 708 of 1,127 training files, so it becomes the *majority* window class again (52% of LSTM windows) even though its pieces are still short. Chopin, the composer with the fewest files, is now the smallest window class. Class weights correct for this in both directions.

# Feature Extraction

The same cleaned files feed two complementary representations, one per model, both derived from a **single parse** of each file (parsing dominates extraction time, so parsing twice — as an earlier version of this pipeline did — would roughly double the time needed to process the full corpus):

- **Note-token sequences (LSTM view).** All non-drum notes are sorted by onset time and reduced to their MIDI pitch (0–127), stored as `int16`. Each training example is a window of 100 consecutive pitch tokens (stride 50, i.e. 50% overlap), passed through a learned embedding layer. This preserves the *order* in which notes arrive — melodic contour and voice-leading — which is the signal an LSTM is built to exploit.
- **Binary piano rolls (CNN view).** Each piece is rendered as a piano roll at 8 frames per second — a 128 × T matrix whose entry (p, t) indicates whether pitch p sounds at frame t — then binarized, stored as `uint8`, and cut into 128-frame (16 s) windows with 50% overlap. Composer style shows up as visual texture: chord shapes, voice spacing, rhythmic density. The window becomes a 128 × 128 single-channel image.

Each window also carries the id of its source file, which is what makes piece-level evaluation (Model Evaluation, below) possible. Extracted windows are cached to disk per split so re-running the notebook doesn't re-parse the full corpus.

# Model Building

Both models end in a softmax over the 4 composers and are trained with sparse categorical cross-entropy and the Adam optimizer. The output `Dense` layer is pinned to `float32` (Keras's recommended practice under mixed-precision training; see Model Training).

- **LSTM.** `Embedding(129 → 64)` → `LSTM(128, return_sequences=True, dropout)` → `LSTM(64, dropout)` → `Dense(64, relu)` → `Dropout` → `Dense(4, softmax)`. 160,900 parameters, all trainable (628.52 KB).
- **CNN.** Three convolutional blocks — `Conv2D(32/64/128, 3×3)` each followed by `BatchNormalization` and `MaxPooling2D` — then `GlobalAveragePooling2D` → `Dense(128, relu)` → `Dropout` → `Dense(4, softmax)`. 110,596 parameters: 110,148 trainable, 448 non-trainable (BatchNorm statistics); 432.02 KB.



# Model Training

- **Optimizer/loss.** Adam, sparse categorical cross-entropy, batch size 64, up to 60 epochs.
- **Callbacks.** Early stopping on validation loss (patience 8) with best-weights restore; `ModelCheckpoint` saving the best model to disk; `ReduceLROnPlateau` (halve the learning rate after 3 stalled epochs, floor 1e-5) to stabilize training when the loss plateaus.
- **Data pipeline.** Training, validation, and test windows are wrapped in `tf.data.Dataset` pipelines (shuffle → augment → batch → prefetch for training; batch → prefetch for validation/test) and passed directly to `model.fit`, rather than materializing augmented NumPy arrays.
- **Mixed precision.** When a GPU is available, TensorFlow's `mixed_float16` policy is enabled (float16 compute, float32 softmax output) for faster training; this run used an NVIDIA GeForce GTX 1080 (compute capability 6.1) under TensorFlow 2.16.1.
- **Imbalance handling.** Class weights computed from the training-window distribution, plus the per-file window cap, address the two imbalance mechanisms described above (raw file count and piece duration).
- **Reproducibility.** All random seeds (Python, NumPy, TensorFlow) fixed at 42; models are rebuilt with re-seeded initializers for every grid-search configuration.
- **Environment.** TensorFlow/Keras with `pretty_midi` for parsing and scikit-learn for metrics; the team validated a local WSL2 + CUDA GPU setup, which required pre-loading the pip-installed NVIDIA libraries before importing TensorFlow.



# Model Evaluation

Evaluation is on the held-out test split, never used during training or model selection, at two levels:

- **Window level.** Every window is scored independently, as in earlier versions of this pipeline.
- **Piece level.** A file's window probability vectors are averaged, then argmax is taken — one prediction per piece. This is closer to how the model would actually be used, and is only meaningful now that file ids are tracked through the pipeline.

**Final results — model comparison (test set):**


| Model | Window acc | Window macro F1 | Window weighted F1 | Piece acc  | Piece macro P | Piece macro R | Piece macro F1 |
| ----- | ---------- | --------------- | ------------------ | ---------- | ------------- | ------------- | -------------- |
| LSTM  | 0.6563     | 0.5655          | 0.6446             | 0.8058     | 0.7442        | 0.6819        | 0.6875         |
| CNN   | **0.7612** | **0.7001**      | **0.7551**         | **0.9008** | **0.8591**    | **0.8219**    | **0.8393**     |


Chance baseline (4 classes): 25%. The CNN reaches 90.1% piece-level accuracy — 3.6× above chance — while the LSTM reaches 80.6% — 3.2× above chance. Piece-level aggregation lifts both models well above their window-level scores (LSTM +14.9 points, CNN +14.0 points), consistent with averaging out independent per-window errors. The CNN is the clear winner on every metric at both levels.

**Per-composer breakdown (piece level, 242 test pieces):**

LSTM (accuracy 0.81):


| Composer  | Precision | Recall | F1   | Support |
| --------- | --------- | ------ | ---- | ------- |
| Bach      | 0.88      | 0.95   | 0.91 | 152     |
| Beethoven | **0.86**  | 0.38   | 0.52 | 32      |
| Chopin    | 0.67      | 0.80   | 0.73 | 20      |
| Mozart    | 0.57      | 0.61   | 0.59 | 38      |


CNN (accuracy 0.90):


| Composer  | Precision | Recall   | F1   | Support |
| --------- | --------- | -------- | ---- | ------- |
| Bach      | 0.95      | **0.99** | 0.97 | 152     |
| Beethoven | 0.76      | 0.69     | 0.72 | 32      |
| Chopin    | **0.94**  | 0.85     | 0.89 | 20      |
| Mozart    | 0.78      | 0.76     | 0.77 | 38      |




# Model Optimization

- **Grid search.** Both models are tuned over a grid of learning rate {1e-3, 5e-4} × dropout {0.3, 0.5} (three configurations each — the third pairs the higher learning rate with the higher dropout), each configuration trained fresh with the full callback stack and selected by best validation accuracy.

LSTM grid-search results:


| lr     | dropout | best val acc |
| ------ | ------- | ------------ |
| 0.0010 | 0.3     | **0.6958**   |
| 0.0005 | 0.3     | 0.6812       |
| 0.0010 | 0.5     | 0.6215       |


CNN grid-search results:


| lr     | dropout | best val acc |
| ------ | ------- | ------------ |
| 0.0010 | 0.3     | 0.7524       |
| 0.0005 | 0.3     | **0.8026**   |
| 0.0010 | 0.5     | 0.7962       |


The LSTM selected `lr=0.001, dropout=0.3`; the CNN selected `lr=0.0005, dropout=0.3` — the same configurations an earlier, smaller-data run of this grid search had selected, but at substantially higher validation accuracy (LSTM 0.696 vs 0.530; CNN 0.803 vs 0.633) now that the full corpus is available. Dropout 0.5 measurably hurt both models relative to their best 0.3 configuration, suggesting the class-weighted, tf.data-augmented training signal is already well regularized without the extra dropout.

- **Tuning levers identified but deferred** (documented for future work): window length and stride, piano-roll frame rate, number of LSTM units / CNN filters, richer note tokens combining pitch with duration and velocity, event-based encodings, multi-channel piano rolls separating instruments, and hybrid CRNN architectures (convolutions feeding an LSTM).
- **Future improvements.** More composers, k-fold cross-validation for tighter confidence intervals, cross-dataset validation, and explicit handling of polyphony/voice separation.



# Analysis

*CNN vs. LSTM gap.* The CNN leads the LSTM by 10.5 points window-level and 9.5 points piece-level accuracy, and on every other metric. The piano-roll representation exposes spatial structure — chord voicing, rhythmic density, register — that the CNN's convolutional filters exploit directly. The LSTM receives only an ordered sequence of pitch integers, discarding duration, rhythm, and harmonic texture; recovering composer style purely from pitch order is a harder inductive problem.

*Beethoven is the hardest composer for both models — not the rarest one.* Beethoven has the lowest F1 of the four composers for both LSTM (0.52) and CNN (0.72), even though Chopin has fewer training files and windows. The confusion matrices show why: for both models, most of Beethoven's misclassified pieces are predicted as Mozart (10 of 20 LSTM errors, 6 of 10 CNN errors), not Chopin. Beethoven's style bridges Classical and Romantic periods, and empirically the overlap with Mozart's Classical vocabulary is the dominant source of confusion, more than the Romantic overlap with Chopin. This confirms that stylistic ambiguity, not window count, is what makes a class hard once class weighting has equalized the training signal.

*Bach is the easiest composer for both models, despite being the majority class.* Bach reaches F1 0.91 (LSTM) and 0.97 (CNN) — the class weighting (Bach weight ≈ 0.48–0.5, the lowest of the four) successfully prevents the model from simply defaulting to the majority class, while Bach's genuinely distinctive Baroque counterpoint texture (dense, regular polyphony) still makes it the easiest to recognize. Chopin, the rarest class by file count, is *not* the hardest to classify (F1 0.73 LSTM, 0.89 CNN) — further evidence that raw class frequency is not what drives difficulty here.

*CNN overfitting pattern.* The CNN's training curves show a volatile first ~~8 epochs: training accuracy climbs smoothly past 80%, but validation loss spikes as high as 5.1 and validation accuracy briefly collapses to near-chance (~~0.18–0.20) more than once, before both curves stabilize from epoch ~10 onward and converge to train accuracy ≈0.91 / validation accuracy ≈0.79 by the final epoch. This is consistent with a BatchNormalization-on-sparse-binary-input pathology — batch statistics are poorly estimated on sparse piano rolls early in training — but on the full corpus the pathology resolves rather than persisting, and `ModelCheckpoint(monitor="val_loss")` recovers a stable, low-loss model regardless. The LSTM shows no comparable instability: its training and validation loss/accuracy curves track closely together for all ~43 epochs, indicating a well-regularized fit with little overfitting.

*Piece-level evaluation matters.* Averaging window probabilities per file before taking the argmax lifted accuracy by 14–15 points for both models. Individual 100-note or 16-second windows are a noisy, partial view of a piece; a full composition gives the model many independent "votes," and errors that are uncorrelated across windows of the same piece cancel out under averaging. Since a deployed system would classify whole pieces, piece-level accuracy (80.6% LSTM, 90.1% CNN) is the more representative number for this task, not the window-level number.