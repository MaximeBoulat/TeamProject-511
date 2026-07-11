# Composer Classification (LSTM vs CNN) — Run Guide & Submission Checklist

## Deliverables in this folder
- `Project_Notebook-Team.ipynb` — the complete, executable notebook.
- `Project_Notebook-Team.html` — HTML export of the notebook (regenerate after you run it so it shows your outputs).
- `Project_Report-Team.docx` — APA 7 technical report (editable).
- `Project_Report-Team.pdf` — APA 7 technical report (PDF).
- `requirements.txt` — Python packages.
- `README-and-Checklist.md` — this file.

> Rename `-Team` to your team/author name in all six filenames if required by your course.

## How each uploaded file was used
- **`midi-classic-music-metadata.json`** — a Croissant *metadata descriptor* for the Kaggle dataset
  *midi_classic_music* (Fedorak, 2019). It contains the dataset's source URL, size, creator, and dates, but
  **no MIDI files**. It was used to (a) identify and cite the dataset source and (b) confirm the dataset's scale
  (3,929 files, 175 composers). It is **not** the dataset itself, so the actual MIDI files must be downloaded
  before running the notebook. No other files were uploaded.

## Required packages
See `requirements.txt`. On Google Colab, only `pretty_midi`, `music21`, and `kaggle` typically need installing;
the notebook's first cell handles this automatically.

## Running in Google Colab (recommended)
1. Open `Project_Notebook-Team.ipynb` in Colab.
2. (Optional) `Runtime → Change runtime type → GPU`.
3. Run the **Setup** cells (Section 1) to install libraries.
4. **Provide the dataset** (Section 3) using one of the three options:
   - **Option A — Kaggle API (auto-download):** upload your `kaggle.json` token, then uncomment the Option A cell.
     It downloads and unzips `blanderbuss/midi-classic-music`.
   - **Option B — Google Drive:** put the dataset in Drive, mount it, and point `DATASET_PATH` to it.
   - **Option C — Manual upload:** upload a zip and unzip it in the notebook.
5. Set `DATASET_PATH` (Section 2) to the folder that contains the composer sub-folders. The parser searches
   recursively, so any nested layout works as long as composer names appear in the file paths.
6. `Runtime → Run all`.
7. Read the printed metrics/figures and **paste the real numbers into the highlighted placeholders** in the report.
8. Re-export the notebook to HTML (`File → Download → .html`) so the HTML contains your outputs.

## Running locally (Jupyter)
1. `pip install -r requirements.txt`
2. Download the dataset from Kaggle and unzip it somewhere.
3. Open the notebook, set `DATASET_PATH` to that folder, and run all cells.

## Notes on runtime
- `MAX_WINDOWS_PER_FILE` and `MAX_FILES_PER_COMPOSER` (Section 2) cap the amount of data for speed. Set them to
  `None` to use everything. Start with the defaults if you are on CPU.
- The grid search trains each model three times. Reduce `LSTM_GRID`/`CNN_GRID` to one entry for a fast first pass.

---

## Submission checklist

### Data handling
- [ ] Dataset downloaded and `DATASET_PATH` set correctly
- [ ] Filtered to Bach, Beethoven, Chopin, Mozart only
- [ ] Composer labels verified via path-based inference (exactly-one-match rule)
- [ ] Duplicates removed (MD5), unreadable/empty files dropped
- [ ] Per-composer counts recorded **before and after** preprocessing (notebook Section 4)
- [ ] Class imbalance analyzed (charts + counts)
- [ ] File-level, stratified train/val/test split with leakage assertion passing

### Modeling
- [ ] LSTM built (Embedding → LSTM×2 → Dropout → Dense softmax)
- [ ] CNN built (Conv×3 + BN + Pool → GAP → Dense softmax)
- [ ] Random seeds set; sparse categorical cross-entropy; Adam
- [ ] Early stopping + checkpoint + LR reduction callbacks
- [ ] Class weights applied
- [ ] Grid search over learning rate and dropout run; best config selected

### Evaluation
- [ ] Test accuracy, precision, recall, F1 (macro **and** weighted) reported
- [ ] Classification report + confusion matrix for both models
- [ ] Training/validation loss and accuracy curves plotted
- [ ] Model comparison table generated from real metrics
- [ ] Easiest/most-confused composers and over/underfitting discussed

### Report (APA 7)
- [ ] Title page, abstract, all required sections present
- [ ] Placeholders (highlighted) replaced with **real** results — no invented numbers
- [ ] In-text citations match the reference list
- [ ] Report matches what the notebook actually does

### Files
- [ ] `Project_Notebook-Team.ipynb`
- [ ] `Project_Notebook-Team.html` (re-exported with outputs)
- [ ] `Project_Report-Team.pdf`
- [ ] (optional) `Project_Report-Team.docx`, `requirements.txt`
