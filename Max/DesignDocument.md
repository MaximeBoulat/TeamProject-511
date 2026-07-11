# Technical document

## Objective

Build a deep learning classifier that predicts the composer of a musical score. The instructions require two architectures — an LSTM and a CNN — so the plan trains both on a shared preprocessing pipeline and compares them.

## Technical specifications

### Dataset

The provided dataset (NN_midi_files_extended) contains MIDI files for 9 composers: Bach, Bartok, Byrd, Chopin, Handel, Hummel, Mendelssohn, Mozart, and Schumann. It comes pre-split: roughly 369 training files (38–42 per composer), 35 dev files, and 35 test files. The dev split serves as the validation set during training; the test split is touched only for final evaluation.

### Representation: from MIDI to sequences of numbers

MIDI is not audio — it is already symbolic: a list of timed note events (note-on, note-off, pitch 0–127, velocity) per instrument. This removes the need for any audio signal processing.

Each file is converted to a **piano roll**: time is discretized into fixed steps (8 frames per second), and each frame is a 128-dimensional binary vector indicating which pitches are sounding. A piece thus becomes a matrix of shape time × 128.

Because pieces have different lengths and one piece is a single labeled example, each piano roll is cut into fixed-length **windows of 240 frames (30 seconds) with 50% overlap**. This yields thousands of training examples from hundreds of files and acts as data augmentation. Every window inherits the composer label of its piece, and the piece id is kept alongside so that predictions can be aggregated back to piece level.

The same window feeds both models, viewed two ways:

- **LSTM view:** a sequence of 240 time steps, each a 128-dimensional frame vector.
- **CNN view:** a 240 × 128 single-channel image, where composer style shows up as visual texture (chord shapes, voice spacing, rhythmic density).

### Model architectures

**CNN:** three Conv2D blocks (32, 64, 128 filters, batch normalization, max pooling), global average pooling, dropout, dense softmax over 9 classes.

**LSTM:** two stacked LSTM layers (128 units each), dropout, dense softmax over 9 classes.

### Training

- Adam optimizer, sparse categorical cross-entropy, batch size 64.
- Early stopping on dev-set loss with best-weights restore.
- Parsed piano rolls cached to disk (.npz) so the MIDI parsing cost is paid once.

### Evaluation

Accuracy, precision, and recall (per instructions) reported at two levels: per window, and per piece by averaging the window probabilities of each piece — the piece-level number is the one that matches the actual use case. A confusion matrix identifies which composers get mistaken for each other.

### Optimization

Hyperparameters to tune after the baselines run: window length and overlap, frame rate, number of units/filters, dropout rate, and learning rate.

### Environment

Conda environment `511-team-project` (Python 3.11): TensorFlow/Keras for the models, pretty_midi for MIDI parsing, scikit-learn for metrics.

### Development plan

1. Preprocessing pipeline: scan splits, parse MIDI, build windowed arrays, cache.
2. Baseline CNN trained and evaluated end to end.
3. Baseline LSTM on the same arrays.
4. Piece-level aggregation and full metrics (accuracy, precision, recall, confusion matrix).
5. Hyperparameter tuning.
6. EDA and report material (deferred from the first pass).

## Results

### First attempt

Preprocessing produced 7,371 training windows from 369 pieces (672 dev / 742 test windows). Headline: the LSTM genuinely learned (57.1% piece-level test accuracy against an 11% chance baseline); the CNN result is a training pathology, not a modeling verdict — it predicts a single class for every input and its numbers are meaningless.

**CNN.** Train accuracy climbed steadily (48% → 79%) while dev accuracy sat at 3–20% with dev loss exploding to 11. This divergence is the classic BatchNormalization failure signature on sparse binary input: training normalizes with batch statistics (train metrics look fine), inference uses running averages, which are badly estimated on piano rolls, so the network produces garbage in inference mode. Early stopping compounded it — best dev loss was epoch 1, so best-weights restore handed back an essentially untrained network that degenerately outputs "byrd" (recall 1.000 for Byrd, 0.000 for the rest; 11.4% piece-level accuracy = Byrd's share of pieces).

**LSTM.** Piece-level test: 57.1% accuracy, macro precision 0.583, macro recall 0.556. The per-composer pattern is musically coherent: Bach and Byrd perfect (the most distinctive textures — Baroque counterpoint, Renaissance polyphony); the classical/romantic cluster (Chopin, Handel, Hummel, Mendelssohn, Mozart) smears together, with Hummel absorbing traffic (precision 0.273, recall 0.750) and Mendelssohn recognized precisely but rarely (precision 1.000, recall 0.250). Bartok scores zero everywhere — a data cause: window counts are proportional to piece length, not file count, and Bartok's pieces are short (21–39 windows per split vs Hummel's 146–201), so the window-level training distribution under-represents him. Schumann suffers the same plus having only 3 test pieces. Training was unstable (accuracy oscillating 0.31 ↔ 0.51 across epochs), pointing at a too-aggressive learning rate. Window-to-piece aggregation helped as designed: 50.4% → 57.1% on test.

**Fixes for the second pass, by expected payoff:**

1. Remove BatchNorm from the CNN and retrain (until then there is no CNN result at all); relax early stopping (patience ~10 or monitor accuracy).
2. Add class weights, or cap windows per piece, to correct the duration-driven imbalance that zeroes out Bartok.
3. Lower the LSTM learning rate (~1e-4) or add ReduceLROnPlateau.
4. Consider shorter windows (e.g. 15 s) so short pieces contribute more examples.
