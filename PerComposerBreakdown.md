# Per-composer breakdown

---

### Cap-75 (75 files per composer, 300 total) — window-level evaluation

LSTM (732 test windows):

| Composer  | Precision | Recall | F1   | Support |
| --------- | --------- | ------ | ---- | ------- |
| Bach      | 0.34      | 0.68   | 0.46 | 104     |
| Beethoven | 0.37      | 0.23   | 0.28 | 205     |
| Chopin    | 0.48      | 0.64   | 0.55 | 219     |
| Mozart    | 0.35      | 0.18   | 0.24 | 204     |

CNN (680 test windows):

| Composer  | Precision | Recall   | F1   | Support |
| --------- | --------- | -------- | ---- | ------- |
| Bach      | **0.85**  | 0.64     | 0.73 | 112     |
| Beethoven | 0.49      | 0.48     | 0.48 | 207     |
| Chopin    | 0.65      | **0.73** | 0.69 | 172     |
| Mozart    | 0.55      | 0.57     | 0.56 | 189     |

---

### Full corpus (1,637 files, no cap) — piece-level evaluation (241 test pieces)

Piece-level: window probability vectors are averaged per file, then argmax is taken.

LSTM (241 test pieces, acc = 0.87):

| Composer  | Precision | Recall   | F1       | Support |
| --------- | --------- | -------- | -------- | ------- |
| Bach      | **0.94**  | **0.95** | **0.94** | 152     |
| Beethoven | 0.78      | 0.58     | 0.67     | 31      |
| Chopin    | 0.82      | 0.90     | 0.86     | 20      |
| Mozart    | 0.68      | 0.74     | 0.71     | 38      |

CNN (241 test pieces, acc = 0.89):

| Composer  | Precision | Recall   | F1       | Support |
| --------- | --------- | -------- | -------- | ------- |
| Bach      | **0.96**  | **0.97** | **0.97** | 152     |
| Beethoven | 0.80      | 0.52     | 0.63     | 31      |
| Chopin    | 0.79      | **0.95** | 0.86     | 20      |
| Mozart    | 0.72      | 0.82     | 0.77     | 38      |