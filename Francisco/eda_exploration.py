"""
Exploratory Data Analysis — Composer Classification Dataset (MSAAI-511 Team Project)

Dataset: Kaggle `blanderbuss/midi-classic-music` (midiclassics), filtered to the
4 composers required by the project instructions: Bach, Beethoven, Chopin, Mozart.

Parses every MIDI file with pretty_midi and produces:
  eda/file_metadata.csv   — one row per file (duration, notes, pitch stats, ...)
  eda/eda_report.md       — summary tables + findings
  eda/fig_*.png           — charts

Usage:
  python eda_exploration.py [--data-dir data] [--out-dir eda]
"""

import argparse
import hashlib
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # pretty_midi emits many benign RuntimeWarnings

import pretty_midi  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COMPOSERS = ["Bach", "Beethoven", "Chopin", "Mozart"]
COLORS = {"Bach": "#4C72B0", "Beethoven": "#DD8452", "Chopin": "#55A868", "Mozart": "#C44E52"}

# Window parameters used to project how many training examples the
# preprocessing pipeline will produce (30 s windows, 50% overlap, 8 fps)
WINDOW_SECONDS = 30
WINDOW_OVERLAP = 0.5


def md5(path, chunk=1 << 16):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def n_windows(duration_s):
    """Number of fixed-length windows a piece yields (matches planned pipeline)."""
    if duration_s < WINDOW_SECONDS:
        return 1 if duration_s > 0 else 0
    hop = WINDOW_SECONDS * (1 - WINDOW_OVERLAP)
    return int((duration_s - WINDOW_SECONDS) // hop) + 1


def parse_file(path, composer):
    row = {
        "composer": composer,
        "path": str(path),
        "filename": path.name,
        "file_kb": path.stat().st_size / 1024,
        "md5": md5(path),
        "parse_ok": False,
        "error": "",
    }
    try:
        pm = pretty_midi.PrettyMIDI(str(path))
        notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
        row["parse_ok"] = True
        row["duration_s"] = pm.get_end_time()
        row["n_instruments"] = len(pm.instruments)
        row["n_notes"] = len(notes)
        row["has_drums"] = any(i.is_drum for i in pm.instruments)
        if notes:
            pitches = np.array([n.pitch for n in notes])
            vels = np.array([n.velocity for n in notes])
            row["pitch_min"] = int(pitches.min())
            row["pitch_max"] = int(pitches.max())
            row["pitch_mean"] = float(pitches.mean())
            row["pitch_std"] = float(pitches.std())
            row["velocity_mean"] = float(vels.mean())
            row["notes_per_sec"] = len(notes) / row["duration_s"] if row["duration_s"] > 0 else 0
        try:
            row["tempo_est"] = float(pm.estimate_tempo())
        except Exception:
            row["tempo_est"] = np.nan
        row["n_windows"] = n_windows(row["duration_s"])
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"[:200]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="eda")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    rows = []
    for composer in COMPOSERS:
        files = sorted((data_dir / composer).rglob("*.[mM][iI][dD]"))
        print(f"{composer}: {len(files)} files")
        for i, f in enumerate(files):
            rows.append(parse_file(f, composer))
            if (i + 1) % 200 == 0:
                print(f"  ... {i + 1}/{len(files)}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "file_metadata.csv", index=False)

    ok = df[df.parse_ok].copy()
    bad = df[~df.parse_ok]

    # Exact-duplicate detection (same bytes, possibly across composers)
    dup_groups = df[df.duplicated("md5", keep=False)].sort_values("md5")
    cross_dups = (
        dup_groups.groupby("md5")["composer"].nunique().pipe(lambda s: s[s > 1])
    )

    # ---------- Figures ----------
    # 1. Files + projected windows per composer
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = df.composer.value_counts().reindex(COMPOSERS)
    axes[0].bar(counts.index, counts.values, color=[COLORS[c] for c in counts.index])
    axes[0].set_title("MIDI files per composer")
    axes[0].bar_label(axes[0].containers[0])
    wins = ok.groupby("composer").n_windows.sum().reindex(COMPOSERS)
    axes[1].bar(wins.index, wins.values, color=[COLORS[c] for c in wins.index])
    axes[1].set_title(f"Projected {WINDOW_SECONDS}s windows (50% overlap)")
    axes[1].bar_label(axes[1].containers[0], fmt="%d")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_class_balance.png", dpi=150)

    # 2. Duration distribution (log scale)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [ok[ok.composer == c].duration_s.clip(lower=1) for c in COMPOSERS]
    bp = ax.boxplot(data, labels=COMPOSERS, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], COMPOSERS):
        patch.set_facecolor(COLORS[c])
    ax.set_yscale("log")
    ax.set_ylabel("Duration (s, log scale)")
    ax.set_title("Piece duration by composer")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_duration.png", dpi=150)

    # 3. Note density
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c in COMPOSERS:
        vals = ok[ok.composer == c].notes_per_sec.dropna()
        ax.hist(vals, bins=60, range=(0, 60), alpha=0.55, label=c, color=COLORS[c])
    ax.set_xlabel("Notes per second")
    ax.set_ylabel("Files")
    ax.set_title("Note density by composer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_note_density.png", dpi=150)

    # 4. Pitch range
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, c in enumerate(COMPOSERS):
        sub = ok[ok.composer == c]
        ax.errorbar(
            [i] * 2,
            [sub.pitch_min.mean(), sub.pitch_max.mean()],
            fmt="none",
        )
        ax.scatter([i], [sub.pitch_mean.mean()], color=COLORS[c], s=80, zorder=3)
        ax.vlines(i, sub.pitch_min.mean(), sub.pitch_max.mean(), color=COLORS[c], lw=6, alpha=0.4)
    ax.set_xticks(range(len(COMPOSERS)))
    ax.set_xticklabels(COMPOSERS)
    ax.set_ylabel("MIDI pitch")
    ax.set_title("Average pitch range (bar) and mean pitch (dot)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_pitch_range.png", dpi=150)

    # 5. Instruments per file
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c in COMPOSERS:
        vals = ok[ok.composer == c].n_instruments.clip(upper=20)
        ax.hist(vals, bins=range(1, 22), alpha=0.55, label=c, color=COLORS[c])
    ax.set_xlabel("Instrument tracks (capped at 20)")
    ax.set_ylabel("Files")
    ax.set_title("Instrument tracks per file")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_instruments.png", dpi=150)

    # ---------- Report ----------
    def table(d):
        return d.to_markdown()

    summary = (
        ok.groupby("composer")
        .agg(
            files=("path", "count"),
            total_hours=("duration_s", lambda s: s.sum() / 3600),
            median_dur_s=("duration_s", "median"),
            mean_notes=("n_notes", "mean"),
            mean_notes_per_sec=("notes_per_sec", "mean"),
            mean_instruments=("n_instruments", "mean"),
            windows=("n_windows", "sum"),
        )
        .round(2)
        .reindex(COMPOSERS)
    )
    win_share = (summary.windows / summary.windows.sum() * 100).round(1)

    lines = []
    lines.append("# EDA Report — Composer Classification Dataset\n")
    lines.append(f"Source: Kaggle `blanderbuss/midi-classic-music`, filtered to {', '.join(COMPOSERS)}.\n")
    lines.append(f"- Total files: **{len(df)}**")
    lines.append(f"- Parsed OK: **{len(ok)}** | Corrupt/unparseable: **{len(bad)}**")
    lines.append(f"- Exact byte-duplicates: **{dup_groups.md5.nunique()}** groups "
                 f"({len(dup_groups)} files), cross-composer: **{len(cross_dups)}**\n")
    lines.append("## Per-composer summary\n")
    lines.append(table(summary))
    lines.append("\n\n## Projected window share (class balance for training)\n")
    lines.append(table(win_share.rename("window_%").to_frame()))
    lines.append("\n\n## Corrupt files\n")
    if len(bad):
        lines.append(table(bad[["composer", "filename", "error"]].reset_index(drop=True)))
    else:
        lines.append("None.")
    lines.append("\n\n## Duplicate groups (first 30)\n")
    if len(dup_groups):
        lines.append(table(dup_groups[["md5", "composer", "filename"]].head(30).reset_index(drop=True)))
    else:
        lines.append("None.")
    lines.append("\n\n## Figures\n")
    for f in ["fig_class_balance", "fig_duration", "fig_note_density", "fig_pitch_range", "fig_instruments"]:
        lines.append(f"![{f}]({f}.png)\n")
    (out_dir / "eda_report.md").write_text("\n".join(lines))

    print("\nDone. Outputs in", out_dir)
    print(summary)


if __name__ == "__main__":
    main()
