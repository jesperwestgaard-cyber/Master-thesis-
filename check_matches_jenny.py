import pandas as pd
import re
import argparse
import os

def normalize(name):
    if pd.isna(name):
        return ""
    name = str(name).strip().lower()
    name = re.sub(r'\s+', ' ', name)
    return name

def extract_species_from_bacdive_label(label):
    """
    BacDive strainLabel looks like: 'Succinispira mobilis 105'
    or 'Bacillus safensis subsp. safensis 1264'
    We want just the species-level binomial (or trinomial for subsp.),
    stripping the trailing strain ID.
    """
    if pd.isna(label):
        return ""
    label = str(label).strip()

    # Handle "Genus species subsp. subspecies STRAINID"
    m = re.match(r'^([A-Z][a-z]+ [a-z0-9\-]+ subsp\. [a-z0-9\-]+)\s+.*$', label)
    if m:
        return normalize(m.group(1))

    # Handle normal "Genus species STRAINID..." (strain id = everything after 2nd word)
    parts = label.split()
    if len(parts) >= 2:
        genus, species = parts[0], parts[1]
        return normalize(f"{genus} {species}")

    return normalize(label)


def load_bacdive(path):
    df = pd.read_csv(path, sep=',')
    df.columns = [c.strip() for c in df.columns]
    df['species_clean'] = df['strainLabel'].apply(extract_species_from_bacdive_label)
    return df


def load_jenny(path):
    df = pd.read_csv(path, sep=';')
    df.columns = [c.strip() for c in df.columns]
    df['species_clean'] = df['species'].apply(normalize)
    return df


def main():
    parser = argparse.ArgumentParser(description="Compare species names between BacDive and Jenny's dataset (ignoring strain IDs).")
    parser.add_argument('--bacdive', required=True, help="Path to BacDive CSV (tab-separated)")
    parser.add_argument('--jenny', required=True, help="Path to Jenny's reducedDataset.csv (semicolon-separated)")
    parser.add_argument('--outdir', default='.', help="Directory to write output reports")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    bacdive_df = load_bacdive(args.bacdive)
    jenny_df = load_jenny(args.jenny)

    bacdive_species = set(bacdive_df['species_clean']) - {""}
    jenny_species = set(jenny_df['species_clean']) - {""}

    missing_from_jenny = sorted(bacdive_species - jenny_species)
    extra_in_jenny = sorted(jenny_species - bacdive_species)
    matched = sorted(bacdive_species & jenny_species)

    total_bacdive = len(bacdive_species)
    n_matched = len(matched)
    n_missing = len(missing_from_jenny)
    pct_matched = (n_matched / total_bacdive * 100) if total_bacdive else 0

    print("=" * 60)
    print("SPECIES COMPARISON SUMMARY (strain IDs ignored)")
    print("=" * 60)
    print(f"Unique species in BacDive file:      {total_bacdive}")
    print(f"Unique species in Jenny's file:       {len(jenny_species)}")
    print(f"Matched (present in both):            {n_matched} ({pct_matched:.1f}%)")
    print(f"Missing from Jenny's dataset:         {n_missing}")
    print(f"Extra in Jenny's (not in BacDive):    {len(extra_in_jenny)}")
    print("=" * 60)

    # Full report: one row per unique BacDive species with match status
    report_rows = []
    for sp in sorted(bacdive_species):
        report_rows.append({
            "species_normalized": sp,
            "in_jenny_dataset": sp in jenny_species
        })
    report_df = pd.DataFrame(report_rows)
    report_path = os.path.join(args.outdir, "species_comparison_report.csv")
    report_df.to_csv(report_path, index=False)

    missing_path = os.path.join(args.outdir, "bacdive_species_missing_from_jenny.csv")
    pd.DataFrame({"species_normalized": missing_from_jenny}).to_csv(missing_path, index=False)

    extra_path = os.path.join(args.outdir, "jenny_species_not_in_bacdive.csv")
    pd.DataFrame({"species_normalized": extra_in_jenny}).to_csv(extra_path, index=False)

    print(f"\nReports written to: {args.outdir}")
    print(f"  - {report_path}")
    print(f"  - {missing_path}")
    print(f"  - {extra_path}")

    if n_missing > 0:
        print(f"\n⚠️  {n_missing} species from BacDive are NOT in Jenny's dataset.")
        print("   Per Daniel's instructions, these would need genome download + annotation.")
    else:
        print("\n✅ All BacDive species are already present in Jenny's dataset. You can use her file directly.")


if __name__ == "__main__":
    main()