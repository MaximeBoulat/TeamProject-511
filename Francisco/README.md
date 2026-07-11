# Francisco — Dataset Exploration

## Dataset

Kaggle [`blanderbuss/midi-classic-music`](https://www.kaggle.com/datasets/blanderbuss/midi-classic-music),
filtered to the 4 composers required by the project instructions:
**Bach, Beethoven, Chopin, Mozart** (~1,637 MIDI files, 43 MB).

`data/` is gitignored. To recreate it:

```bash
curl -sL -o midi-classic-music.zip \
  "https://www.kaggle.com/api/v1/datasets/download/blanderbuss/midi-classic-music"
python - <<'EOF'
import zipfile, os
z = zipfile.ZipFile("midi-classic-music.zip")
targets = {"Bach", "Beethoven", "Chopin", "Mozart"}
for info in z.infolist():
    parts = info.filename.split('/')
    if len(parts) >= 3 and parts[0] == 'midiclassics' and parts[1] in targets and not info.is_dir():
        dest = os.path.join("data", parts[1], *parts[2:])
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with z.open(info) as src, open(dest, 'wb') as dst:
            dst.write(src.read())
EOF
# Beethoven ships 6 nested zips — unpack them too:
cd data/Beethoven && for z in *.zip; do unzip -q -o "$z" && rm "$z"; done
```

## EDA

```bash
pip install pretty_midi numpy pandas matplotlib tabulate
python eda_exploration.py          # writes eda/
```

Outputs in `eda/`:
- `file_metadata.csv` — one row per file: duration, note count, pitch stats,
  note density, instrument count, tempo estimate, md5 (for duplicate detection),
  parse status, and projected 30-second window count.
- `eda_report.md` — summary tables (per-composer stats, class balance,
  corrupt files, duplicates) + figures.
- `fig_*.png` — class balance, duration, note density, pitch range, instruments.
