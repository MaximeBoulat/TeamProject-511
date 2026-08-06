# Paper

# Abstract



# Introduction and literature review

Music as a medium a offers a unique and fertile playground for data science and deep learning experimentation. The reason why is that it is naturally well suited to be broken down into bite sized chunks for analysis, it comes pre-labeled, there exists a large quantity of it in the wild, its underlying data representation is continuous and messy yet predictable and uniquely identifying patterns are repeated for each musician across their musicography.

The science of analyzing music to extract meaningful information from it (Musical Information Retrieval, MIR) finds its origins in the science of speech recognition. Before the advent of Deep Learning, and as is the case for other fields of study such as computer vision and Natural Language Processing, researchers relied heavily on handcrafted features which were either used directly to compare against known thresholds, or used to train classical classifiers such as Logistic Regression, Decision Trees or Support Vector Machines. Foote (1997) proposed to use MFCC coefficients to construct a learning tree vector quantizer, which could then be used to produce a unique signature for each track. Similarity search could then be performed on the resulting signatures. Tzanetakis and Cook (2002) outlined a series of mathematical formulas to extract uniquely identifying features from musical tracks for genre classification. 

The deep learning revolution kicked off by Krizhevsky et al. (2012) with AlexNet extended to MIR and CNNs became the defacto gold standard for music classification (with musical track data converted to "image" data). However it was widely believed that the raw audio data needed to be converted to a spectrogram before ingestion. Dieleman and Schrauwen (2014), were the first to demonstrate that CNNs were able to process raw music audio directly and learn how to extract the prominent features directly from it.

In this paper, we propose to implement a musical classification system which can correctly predict the composer for a given track out of 4 possible composers. We propose to use two different deep neural network architectures to that effect: a CNN architecture and a LSTM architecture. As a model able to identify hierarchical patterns in image-like data, the CNN is uniquely positioned to recognize recurring complex patterns such as chord voicing, voice spacing, rhythmic density, and register specific to each composer. The LSTM, in turn, is specially designed to recognize patterns in time-series or sequential data thanks to the recursiveness of its processing loop, and is therefore particularly well suited for analyzing music.

First, we discuss data collection and pre-processing, then the feature extraction step that converts cleaned MIDI files into the pitch-token and piano-roll representations each model consumes. We then describe the CNN and LSTM architectures, the training procedure, and the window- and piece-level evaluation used to compare them, before optimizing each model's hyperparameters and analyzing the results.

# Data Collection

The data used in the experiment was obtained from a public dataset hosted on Kaggle consisting of 3,929 midi files organized by composer (Fedorak, 2019). This original batch was filtered down to contain only the four main composers: Back, Beethoven, Chopin and Mozart, which left 1637 files total.

Some basic data hygiene was performed on the raw data, with duplicates (identified with md5 hashing) and unuseable files removed.

Then the entire folder was traversed and everty track inventoried into a tracking table where every track was mapped to its physical file and assigned the correct composer class label according to its position in the file hierarchy.

This allowed us to perform some exploratory analysis, as illustrated in Table 1.

**Table 1**
Per composer profile


| Composer  | Full corpus files | Usable (after cleaning) | Median duration (s) | Mean notes/s | Mean instruments |
| --------- | ----------------- | ----------------------- | ------------------- | ------------ | ---------------- |
| Bach      | 1,024             | 1,012                   | 77                  | 9.4          | 4.8              |
| Beethoven | 220               | 213                     | 410                 | 14.2         | 6.9              |
| Chopin    | 136               | 132                     | 158                 | 11.2         | 2.5              |
| Mozart    | 257               | 254                     | 348                 | 13.8         | 7.3              |


Bach's file count (1,024) dwarfs the other three composers combined (613), while its median piece duration (77 s) is the shortest by a wide margin. This tension — many short Bach files vs. fewer, longer files for the other three composers — is the central driver of the class-imbalance story below.

# Data Pre-processing


In order to maximize the standardizedness and the quantity of samples that could be extracted from the midi tracks, it was decided to adopt a windowing strategy where each track was broken down in chunks of fixed length with overlap between them and where made to inherit their track's class label. These chunks, called windows, formed the basis of the sample space that was used to train the models in this experiment.

Because each model type have different requirements in terms of input, the way the windows were created from the raw files differed according to which model they were intended for.  
  
For the LSTM, the windows were measured by quantity of notes, since the LSTM would ingest flat vectors. Each window was made to be 100 notes long with a stride of 50. For the CNN, since the input format would be a 2D image representation of the data, a fixed interval served as the boundary. Each window was made to be 16 seconds (128  frames at 8fps) and the stride 64 frames.

It is important to note that the conversion of midi tracks into windows was performed before the train-test split to avoid leakage of data from the same track across training and testing. In addition the train-test split was perfomred using stratified split to preserve the class representation of the original dataset.

In addition to windowing, the data was augmented using random pitch transposition during training with each window getting one random draw from `{-2, -1, 0, +1, +2}` per epoch.

Finaly, the remaining class imbalance was mitigated by using reverse frequency weights based on the frequencies observed in the training windows as displayed in Table 2. 


**Table 2**
Training window class frequencies and inverse-frequency weights

| Composer  | LSTM windows | LSTM % | LSTM weight | CNN windows | CNN % | CNN weight |
| --------- | ------------ | ------ | ----------- | ----------- | ----- | ---------- |
| Bach      | 8,299        | 51.9%  | 0.482       | 7,371       | 50.0% | 0.500      |
| Beethoven | 2,732        | 17.1%  | 1.464       | 2,679       | 18.2% | 1.375      |
| Chopin    | 1,666        | 10.4%  | 2.401       | 1,428       | 9.7%  | 2.580      |
| Mozart    | 3,305        | 20.7%  | 1.210       | 3,260       | 22.1% | 1.130      |


# Feature Extraction

The conversion of midi tracks into windows explained above already hints at the kind of feature extraction that was performed for each model. For the LSTM, all the notes except for the percussions were extracted from the midi file and were arranged as a vector of pitch values sorted by time and everything else was discarded. This was assumed to be enough to preserve the underlying structure of the music and its composer's specific style.

For the CNN, all the notes (except the drums) were arranged on a 2D canvas where the x axis was time and the y axis was the range of possible pitch values. All the notes were extracted from the midi files and displayed as boolean flags on the canvas, effectively turning the midi tracks into images.

Each window also carried the id of its source file, which is what made piece-level evaluation (Model Evaluation, below) possible. 

# Model Building

Both models end in a softmax over the 4 composers and are trained with sparse categorical cross-entropy and the Adam optimizer. 

- **LSTM.** `Embedding(129 → 64)` → `LSTM(128, return_sequences=True, dropout)` → `LSTM(64, dropout)` → `Dense(64, relu)` → `Dropout` → `Dense(4, softmax)`. 160,900 parameters, all trainable (628.52 KB).
- **CNN.** Three convolutional blocks — `Conv2D(32/64/128, 3×3)` each followed by `BatchNormalization` and `MaxPooling2D` — then `GlobalAveragePooling2D` → `Dense(128, relu)` → `Dropout` → `Dense(4, softmax)`. 110,596 parameters: 110,148 trainable, 448 non-trainable (BatchNorm statistics); 432.02 KB.


- **Optimizer/loss.** Adam, sparse categorical cross-entropy, batch size 64, up to 60 epochs.
- **Callbacks.** Early stopping on validation loss (patience 8) with best-weights restore; `ModelCheckpoint` saving the best model to disk; `ReduceLROnPlateau` (halve the learning rate after 3 stalled epochs, floor 1e-5) to stabilize training when the loss plateaus.


# Model Training


As shown in Figure 1, the training histories for both models show a steady convergence toward an optimum, with the best checkpoint reached at epoch 35 with validation loss 0.82 and validation accuracy 0.69 for the LSTM, and at epoch 17 with validation loss 0.55 and validation accuracy 0.80 for the CNN. Besides early wild fluctuations on the validation set for the CNN which stabilized around epoch 10, both training histories show a healthy trend indicating that both models were able to learn from the data without overfitting.

![Figure 1 - Training curves](Resources/TrainingGraphs.png)

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

![](Resources/Results.png)


**Per-composer breakdown (piece level, 242 test pieces):**

LSTM (accuracy 0.81):


| Composer  | Precision | Recall | F1   | Support |
| --------- | --------- | ------ | ---- | ------- |
| Bach      | 0.88      | 0.95   | 0.91 | 152     |
| Beethoven | **0.86**  | 0.38   | 0.52 | 32      |
| Chopin    | 0.67      | 0.80   | 0.73 | 20      |
| Mozart    | 0.57      | 0.61   | 0.59 | 38      |

![](Resources/ConfusionMatrixLSTM.png)


CNN (accuracy 0.90):


| Composer  | Precision | Recall   | F1   | Support |
| --------- | --------- | -------- | ---- | ------- |
| Bach      | 0.95      | **0.99** | 0.97 | 152     |
| Beethoven | 0.76      | 0.69     | 0.72 | 32      |
| Chopin    | **0.94**  | 0.85     | 0.89 | 20      |
| Mozart    | 0.78      | 0.76     | 0.77 | 38      |

![](Resources/ConfusionMatrixCNN.png)


# Model Optimization

- **Grid search.** Both models were tuned over a grid of learning rate {1e-3, 5e-4} × dropout {0.3, 0.5} (three configurations each — the third pairs the higher learning rate with the higher dropout), each configuration trained fresh with the full callback stack and selected by best validation accuracy.

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


The LSTM selected `lr=0.001, dropout=0.3`; the CNN selected `lr=0.0005, dropout=0.3`. Dropout 0.5 measurably hurt both models relative to their best 0.3 configuration, suggesting the class-weighted, tf.data-augmented training signal is already well regularized without the extra dropout.



# Conclusion

Both models were able to learn from the data and make predictions with a certain degree of confidence, pointing to a strong validation of the design of the pipeline at every step (data preprocessing, feature extraction, model architecture and hyperparameters).

*CNN vs. LSTM gap.* The CNN leads the LSTM by 10.5 points window-level and 9.5 points piece-level accuracy, and on every other metric. The piano-roll representation exposes spatial structure — chord voicing, rhythmic density, register — that the CNN's convolutional filters exploit directly. The LSTM receives only an ordered sequence of pitch integers, discarding duration, rhythm, and harmonic texture; recovering composer style purely from pitch order is a harder inductive problem.

*Beethoven is the hardest composer for both models — not the rarest one.* Beethoven has the lowest F1 of the four composers for both LSTM (0.52) and CNN (0.72), even though Chopin has fewer training files and windows. The confusion matrices show why: for both models, most of Beethoven's misclassified pieces are predicted as Mozart (10 of 20 LSTM errors, 6 of 10 CNN errors), not Chopin. Beethoven's style bridges Classical and Romantic periods, and empirically the overlap with Mozart's Classical vocabulary is the dominant source of confusion, more than the Romantic overlap with Chopin. This confirms that stylistic ambiguity, not window count, is what makes a class hard once class weighting has equalized the training signal.

*Bach is the easiest composer for both models, despite being the majority class.* Bach reaches F1 0.91 (LSTM) and 0.97 (CNN) — the class weighting (Bach weight ≈ 0.48–0.5, the lowest of the four) successfully prevents the model from simply defaulting to the majority class, while Bach's genuinely distinctive Baroque counterpoint texture (dense, regular polyphony) still makes it the easiest to recognize. Chopin, the rarest class by file count, is *not* the hardest to classify (F1 0.73 LSTM, 0.89 CNN) — further evidence that raw class frequency is not what drives difficulty here.

*CNN overfitting pattern.* The CNN's training curves show a volatile first 8 epochs: training accuracy climbs smoothly past 80%, but validation loss spikes as high as 5.1 and validation accuracy briefly collapses to near-chance (0.18–0.20) more than once, before both curves stabilize from epoch ~10 onward and converge to train accuracy ≈0.91 / validation accuracy ≈0.79 by the final epoch. This is consistent with a BatchNormalization-on-sparse-binary-input pathology — batch statistics are poorly estimated on sparse piano rolls early in training — but on the full corpus the pathology resolves rather than persisting, and `ModelCheckpoint(monitor="val_loss")` recovers a stable, low-loss model regardless. The LSTM shows no comparable instability: its training and validation loss/accuracy curves track closely together for all ~43 epochs, indicating a well-regularized fit with little overfitting.

*Piece-level evaluation matters.* Averaging window probabilities per file before taking the argmax lifted accuracy by 14–15 points for both models. Individual 100-note or 16-second windows are a noisy, partial view of a piece; a full composition gives the model many independent "votes," and errors that are uncorrelated across windows of the same piece cancel out under averaging. Since a deployed system would classify whole pieces, piece-level accuracy (80.6% LSTM, 90.1% CNN) is the more representative number for this task, not the window-level number.

- **Tuning levers identified but deferred** (documented for future work): window length and stride, piano-roll frame rate, number of LSTM units / CNN filters, richer note tokens combining pitch with duration and velocity, event-based encodings, multi-channel piano rolls separating instruments, and hybrid CRNN architectures (convolutions feeding an LSTM).
- **Future improvements.** More composers, k-fold cross-validation for tighter confidence intervals, cross-dataset validation, and explicit handling of polyphony/voice separation.


# References

Dieleman, S., & Schrauwen, B. (2014). End-to-end learning for music audio. In *2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)* (pp. 6964–6968). IEEE.

Fedorak, B. (2019). *midi_classic_music* [Data set]. Kaggle. https://www.kaggle.com/datasets/blanderbuss/midi-classic-music

Foote, J. (1997). Content-based retrieval of music and audio. *Multimedia Storage and Archiving Systems II*, 138–147.

Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. *IEEE Transactions on Speech and Audio Processing*, *10*(5), 293–302. https://doi.org/10.1109/TSA.2002.800560

