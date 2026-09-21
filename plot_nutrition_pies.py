#!/usr/bin/env python3
"""
plot_nutrition_pies.py

Usage:
  - Provide absolute paths as args:
      python /full/path/plot_nutrition_pies.py /full/path/cleaned_nutrition.csv /full/path/output_folder
  - Or run without args and the script will prompt you to paste absolute paths.

Creates three donut PNGs for columns:
  energy_source, carbon_source, electron_source
Saves files into the provided output folder and prints saved file paths.

Requires: pandas, matplotlib
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# --- Helpers ---
def prompt_if_missing(arg_index, prompt_text):
    try:
        return Path(sys.argv[arg_index]).expanduser().resolve()
    except Exception:
        p = input(prompt_text).strip()
        return Path(p).expanduser().resolve()

def make_donut(series, top_n=8, unknown_label='unknown'):
    # count and ensure unknown presence
    counts = series.fillna(unknown_label).astype(str).str.strip().replace('', unknown_label).value_counts()
    if unknown_label not in counts:
        counts[unknown_label] = 0
    counts = counts.sort_values(ascending=False)
    if len(counts) > top_n:
        top = counts.iloc[:top_n].copy()
        other = counts.iloc[top_n:].sum()
        top['Other'] = other
        counts = top
    return counts

def plot_and_save(counts, title, out_path, cmap='tab20'):
    fig, ax = plt.subplots(figsize=(6,6))
    colors = plt.get_cmap(cmap).colors
    wedges, texts = ax.pie(counts, labels=None, colors=colors[:len(counts)], startangle=90, counterclock=False, wedgeprops=dict(width=0.34, edgecolor='w'))
    # center circle for donut
    centre_circle = plt.Circle((0,0),0.66,fc='white')
    fig.gca().add_artist(centre_circle)
    ax.axis('equal')
    # legend with counts and percent
    total = counts.sum()
    labels = [f"{k} — {v} ({v/total:.1%})" for k,v in counts.items()]
    ax.legend(wedges, labels, title=title, loc='center left', bbox_to_anchor=(1,0.5))
    plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")

# --- Main ---
def main():
    # get paths (either args or prompt)
    if len(sys.argv) >= 3:
        csv_path = Path(sys.argv[1]).expanduser().resolve()
        out_dir = Path(sys.argv[2]).expanduser().resolve()
    else:
        csv_path = Path(input("Enter absolute path to cleaned_nutrition.csv: ").strip()).expanduser().resolve()
        out_dir = Path(input("Enter absolute path to output folder (will be created if needed): ").strip()).expanduser().resolve()

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    # Column names expected (adjust below if your cleaned CSV uses different names)
    cols = ['energy_source', 'carbon_source', 'electron_source']
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print("ERROR: CSV missing expected columns:", missing)
        print("Available columns:", list(df.columns))
        sys.exit(1)

    # Parameters: change top_n if you want fewer/more categories before grouping into Other
    top_n = 8

    for col in cols:
        counts = make_donut(df[col], top_n=top_n, unknown_label='unknown')
        outfile = out_dir / f"{col}_donut.png"
        plot_and_save(counts, title=col.replace('_',' ').title(), out_path=outfile)

if __name__ == "__main__":
    main()