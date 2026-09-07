#!/usr/bin/env python3
"""
clean_bacdive_nutrition.py
Usage:
    python clean_bacdive_nutrition.py input.csv output_cleaned.csv
Reads column 'nutritionType' (adjust INPUT_COL if different) and writes a cleaned CSV.
"""

import sys
import re
from pathlib import Path
import pandas as pd

# ---------------- Config ----------------
INPUT_COL = "nutritionType"   # adjust if your file uses a different column name
ORIGINAL_COL = "original_nutrition"
OUT_EXTRA_COLS = [
    "energy_source",     # phototroph / chemotroph / mixotroph / unknown
    "electron_source",   # organotroph / lithotroph / unknown
    "carbon_source",     # autotroph / heterotroph / mixotroph / unknown
    "parsed_tokens",     # for auditing
    "mapping_confidence" # high / medium / low / manual_review
]

# Known prefixes & special tokens
PREFIXES = ["photo", "phototroph", "photolitho", "photolithoautotroph", "chemo", "chemotroph", "chemoauto", 
            "mixo", "mixotroph", "organo", "organotroph", "litho", "lithotroph", "auto", "autotroph",
            "hetero", "heterotroph", "diazo", "methano", "methyl", "methanotroph", "copiotroph",
            "oligotroph", "photoheterotroph", "photolithoautotroph", "photolithotroph"]

# ---------------- Helpers ----------------
def normalize_text(s):
    if pd.isna(s):
        return ""
    s = str(s).strip()
    # remove HTML/LaTeX wrapper fragments that sometimes appear in scraped data
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"&nbsp;", " ", s)
    # unify separators to pipe
    s = re.sub(r'[\|/;,+]', '|', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def split_labels(s):
    # split on pipe or whitespace; preserve order and unique
    parts = []
    for piece in re.split(r'\|', s):
        piece = piece.strip()
        if not piece:
            continue
        # commonly entries are concatenated (e.g. chemoorganoheterotroph). attempt to split known prefixes
        pieces = decompose_concatenated(piece.lower())
        for p in pieces:
            if p and p not in parts:
                parts.append(p)
    return parts

def decompose_concatenated(token):
    t = token.lower()
    # quick mapping for some full-form known compound tokens
    explicit_map = {
        "chemoorganoheterotroph": ["chemo","organo","hetero"],
        "chemoorganoheterotrophs": ["chemo","organo","hetero"],
        "chemoorganotroph": ["chemo","organo"],
        "chemolithoautotroph": ["chemo","litho","auto"],
        "chemolithotroph": ["chemo","litho"],
        "photolithoautotroph": ["photo","litho","auto"],
        "photoautotroph": ["photo","auto"],
        "photoheterotroph": ["photo","hetero"],
        "organoheterotroph": ["organo","hetero"],
        "lithoheterotroph": ["litho","hetero"],
        "chemoautolithotroph": ["chemo","auto","litho"],  # catch spelling variants
        "photolithotroph": ["photo","litho"],
        "chemoautotroph": ["chemo","auto"],
        "methylotroph": ["methyl"],
        "methanotroph": ["methano"],
        "diazotroph": ["diazo"],
    }
    if t in explicit_map:
        return explicit_map[t]
    # fallback: try to iteratively strip known prefixes
    tokens = []
    remaining = t
    # check also for hyphenated words already removed earlier
    found_any = True
    while remaining and found_any:
        found_any = False
        for p in sorted(PREFIXES, key=len, reverse=True):
            if remaining.startswith(p):
                tokens.append(p if p.isalpha() else p)
                remaining = remaining[len(p):]
                found_any = True
                break
    # if some leftover and not matched, append leftover as token
    if remaining:
        tokens.append(remaining)
    # final normalization to simple canonical tokens (short forms)
    canon = []
    for tk in tokens:
        tk = tk.strip()
        if not tk: continue
        # normalize longer forms to canonical short prefixes
        if tk.startswith("photo"):
            canon.append("photo")
        elif tk.startswith("chemo"):
            canon.append("chemo")
        elif tk.startswith("mixo"):
            canon.append("mixo")
        elif tk.startswith("organo"):
            canon.append("organo")
        elif tk.startswith("litho"):
            canon.append("litho")
        elif tk.startswith("auto"):
            canon.append("auto")
        elif tk.startswith("hetero"):
            canon.append("hetero")
        elif tk.startswith("methan") or tk.startswith("methyl"):
            canon.append("methyl")
        elif tk.startswith("diaz"):
            canon.append("diazo")
        elif tk in ("organotroph","organotrophs","organotrophy"):
            canon.append("organo")
        elif tk in ("lithotroph","lithotrophs"):
            canon.append("litho")
        else:
            canon.append(tk)
    # dedupe keeping order
    seen = set(); out=[]
    for c in canon:
        if c not in seen:
            out.append(c); seen.add(c)
    return out

def map_tokens(tokens):
    energy = "unknown"
    electron = "unknown"
    carbon = "unknown"
    tset = set(tokens)

    # energy_source
    if "mixo" in tset or ("photo" in tset and "chemo" in tset):
        energy = "mixotroph"
    elif "photo" in tset:
        energy = "phototroph"
    elif "chemo" in tset:
        energy = "chemotroph"
    elif "photolitho" in tset:
        energy = "phototroph"

    # electron_source
    if "organo" in tset:
        electron = "organotroph"
    elif "litho" in tset:
        electron = "lithotroph"

    # carbon_source
    if "mixo" in tset:
        carbon = "mixotroph"
    elif "auto" in tset:
        carbon = "autotroph"
    elif "hetero" in tset:
        carbon = "heterotroph"
    # heuristics: chemo + organo often implies heterotroph if carbon still unknown
    if carbon == "unknown" and energy == "chemotroph" and electron == "organotroph":
        carbon = "heterotroph"
    return energy, electron, carbon

def confidence_flag(original, tokens, mapped):
    energy, electron, carbon = mapped
    axes_known = sum(1 for x in (energy,electron,carbon) if x!="unknown")
    if axes_known >= 2:
        return "high"
    if tokens:
        return "medium"
    if pd.isna(original) or str(original).strip()=="":
        return "low"
    return "manual_review"

# ---------------- Processing ----------------
def process_df(df, input_col=INPUT_COL):
    if input_col not in df.columns:
        raise KeyError(f"Column '{input_col}' not found. Columns: {list(df.columns)}")
    out_rows = []
    for _, row in df.iterrows():
        raw = row.get(input_col, "")
        norm = normalize_text(raw)
        tokens = split_labels(norm)
        mapped = map_tokens(tokens)
        conf = confidence_flag(raw, tokens, mapped)
        parsed = ";".join(tokens)
        # build output row: keep original columns + new audit fields
        out_row = row.to_dict()
        out_row[ORIGINAL_COL] = raw
        out_row["energy_source"] = mapped[0]
        out_row["electron_source"] = mapped[1]
        out_row["carbon_source"] = mapped[2]
        out_row["parsed_tokens"] = parsed
        out_row["mapping_confidence"] = conf
        out_rows.append(out_row)
    return pd.DataFrame(out_rows)

def main():
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    default_in = project_dir / "bacdive_HmE6dj.csv"
    default_out = project_dir / "cleaned_nutrition.csv"

    if len(sys.argv) == 1:
        in_path = default_in
        out_path = default_out
    elif len(sys.argv) == 3:
        in_path = Path(sys.argv[1])
        out_path = Path(sys.argv[2])
    else:
        print("Usage: python clean_bacdive_nutrition.py [input.csv output_cleaned.csv]")
        print(f"Default input: {default_in}")
        print(f"Default output: {default_out}")
        sys.exit(1)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path, dtype=str, keep_default_na=False)
    cleaned = process_df(df, input_col=INPUT_COL)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False)
    print(f"Saved cleaned CSV to {out_path} ({len(cleaned)} rows).")

if __name__ == "__main__":
    main()