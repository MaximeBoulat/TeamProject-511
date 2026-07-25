# Draft section for Paper.md (Model Optimization)

Note for the team: keeping this here until PR #3 merges so we don't get a merge
conflict in Paper.md. Figures are in Francisco/results/, the executed notebook
is Francisco/full_corpus_experiment.ipynb.

---

## Data scaling

The 75-file-per-composer cap in the main experiments exists only because of
memory limits, not for any methodological reason. To measure what it costs, we
removed it using a more compact pipeline: base windows stored as uint8/int16
(instead of float32 with the augmented copies pre-computed) and semitone
transposition applied randomly in the input pipeline. This cuts memory by about
25x, enough to train on the full cleaned corpus (1,605 files). Both
architectures were rerun unchanged (same seed, splits, window parameters, and
best grid-search configurations) on the capped subset and on the full corpus in
the same session. We also added piece-level evaluation, where the window
probability vectors of each file are averaged before taking the argmax, which
matches the real use case of classifying a whole piece rather than a 16-second
excerpt.

| Training set | Model | Window acc. | Piece acc. | Macro F1 (piece) |
| ------------ | ----- | ----------- | ---------- | ---------------- |
| 300 files (cap 75) | LSTM | 0.381 | 0.511 | 0.503 |
| 300 files (cap 75) | CNN  | 0.282 | 0.244* | 0.098* |
| 1,605 files (full) | LSTM | 0.709 | 0.863 | 0.784 |
| 1,605 files (full) | CNN  | 0.751 | 0.896 | 0.822 |

\* Degenerate run: on the capped data the CNN reproduced the batch-normalization
failure observed in the first pass (validation loss diverging while training
accuracy climbs), and early stopping restored a near-untrained model. The same
architecture trained stably on the full corpus, which suggests the pathology is
a small-data problem.

Data scale turned out to matter far more than tuning: removing the cap improved
piece-level accuracy by roughly 35 points, while the grid search moved results
by only a few. Aggregating windows to pieces added about 15 further points
(LSTM 0.709 to 0.863, CNN 0.751 to 0.896), confirming the design-document
expectation that windows are the right training unit but pieces are the right
evaluation unit. Per-composer results at full scale line up with the stylistic
picture from the EDA: Bach is easiest (piece F1 0.97), Beethoven hardest (0.67),
with Chopin (0.88) and Mozart (0.77) in between.

Since the corpus is 63% Bach, the stratified test set is Bach-heavy too, so we
report macro-averaged F1 alongside accuracy; at 0.82 for the CNN it confirms the
gain is not just the majority class. One caveat: the capped CNN row here is not
directly comparable to the main experiment's capped CNN (59.7% window accuracy)
because augmentation is sampled per epoch rather than pre-materialized, which
changes the number of gradient steps per epoch and therefore the early-stopping
behavior. The controlled comparison is between the rows of this table, which
share an identical pipeline.
