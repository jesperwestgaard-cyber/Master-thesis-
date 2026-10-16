import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

# Build the ML feature/label table from completed mapper results and BacDive data.
# Run directly, or let annotate_genomes.sh call this after all annotations finish.
# The output is ml_feature_matrix.csv in the project root; strain URLs are excluded.
BASE = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE / "proteomes_downloads"
RESULTS_DIR = BASE / "kofam_results"
BACDIVE_CSV = BASE / "csv filer" / "bacdive_HmE6dj.csv"
OUTPUT = BASE / "ml_feature_matrix.csv"
KO_PATTERN = re.compile(r"K\d{5}\Z")


def species_key(name: str) -> str:
    words = re.sub(r"[_\s]+", " ", name.strip()).casefold().split()
    return " ".join(words[:2])


def parse_mapper_result(path: Path) -> set[str]:
    # Mapper output contains protein ID in column 1 and assigned KO in column 2.
    kos = set()
    with path.open("r", encoding="utf-8", errors="replace") as result_file:
        for line in result_file:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) >= 2 and KO_PATTERN.fullmatch(fields[1].strip()):
                kos.add(fields[1].strip())
    return kos


def load_bacdive_records(species_to_assemblies: dict[str, list[str]]):
    # Match BacDive strain labels to downloaded species; combine multiple records
    # for the same BacDive strain so its nutrition types become multilabel targets.
    required_columns = {"nutritionType", "strain", "strainLabel"}
    records = {}
    unmatched_rows = 0
    matched_rows = 0

    with BACDIVE_CSV.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            sys.exit(f"BacDive CSV must contain columns: {sorted(required_columns)}")
        trophy_columns = [column for column in reader.fieldnames if column != "strain"]

        for row in reader:
            assemblies = species_to_assemblies.get(species_key(row["strainLabel"]), [])
            if not assemblies:
                unmatched_rows += 1
                continue
            matched_rows += 1

            strain_id = urlparse(row["strain"]).path.rstrip("/").split("/")[-1]
            if not strain_id:
                sys.exit(f"Missing BacDive strain ID for {row['strainLabel']}")

            record = records.setdefault(
                strain_id,
                {
                    "assemblies": assemblies,
                    "values": defaultdict(set),
                    "nutrition_types": set(),
                },
            )
            if record["assemblies"] != assemblies:
                sys.exit(f"Inconsistent proteome match for BacDive strain {strain_id}")

            for column in trophy_columns:
                value = (row.get(column) or "").strip()
                if value:
                    record["values"][column].add(value)
            record["nutrition_types"].update(
                value.strip()
                for value in (row.get("nutritionType") or "").split("|")
                if value.strip()
            )

    return records, trophy_columns, matched_rows, unmatched_rows


def main() -> None:
    # Require a finished mapper file for every downloaded assembly before writing.
    protein_files = sorted(INPUT_DIR.rglob("protein.faa"))
    if not protein_files:
        sys.exit(f"No protein.faa files found under {INPUT_DIR}")

    sample_to_result = {}
    species_to_assemblies = defaultdict(list)
    for protein_file in protein_files:
        sample = protein_file.parent.name
        if sample in sample_to_result:
            sys.exit(f"Duplicate assembly accession found: {sample}")
        sample_to_result[sample] = RESULTS_DIR / f"{sample}.mapper.tsv"
        species_to_assemblies[species_key(protein_file.parents[3].name)].append(sample)

    missing = [
        sample
        for sample, result in sample_to_result.items()
        if not result.is_file() or result.stat().st_size == 0
    ]
    if missing:
        preview = ", ".join(missing[:10])
        remaining = len(missing) - min(len(missing), 10)
        suffix = f", and {remaining} more" if remaining else ""
        sys.exit(
            f"Annotations are incomplete: {len(missing)} of "
            f"{len(sample_to_result)} genomes have no completed mapper result. "
            f"Missing: {preview}{suffix}. No matrix was written."
        )

    bacdive_records, trophy_columns, matched_rows, unmatched_rows = (
        load_bacdive_records(species_to_assemblies)
    )
    if not bacdive_records:
        sys.exit("No BacDive records matched the downloaded proteomes.")

    matched_assemblies = {
        assembly
        for record in bacdive_records.values()
        for assembly in record["assemblies"]
    }
    assembly_to_kos = {
        assembly: parse_mapper_result(sample_to_result[assembly])
        for assembly in matched_assemblies
    }
    all_kos = sorted(set().union(*assembly_to_kos.values()))
    if not all_kos:
        sys.exit("No threshold-passing KO assignments found; no matrix was written.")

    nutrition_types = sorted(
        {
            nutrition_type
            for record in bacdive_records.values()
            for nutrition_type in record["nutrition_types"]
        }
    )
    target_columns = [f"target__{nutrition_type}" for nutrition_type in nutrition_types]
    samples = []
    for strain_id, record in bacdive_records.items():
        for assembly in record["assemblies"]:
            sample = {
                "sample": f"bacdive_{strain_id}_{assembly}",
                "assembly": assembly,
                "nutrition_types": record["nutrition_types"],
                "trophy_values": record["values"],
                "kos": assembly_to_kos[assembly],
            }
            samples.append(sample)

    columns = ["sample", "assembly"] + trophy_columns + target_columns + all_kos
    temporary_output = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    with temporary_output.open("w", newline="", encoding="utf-8") as matrix_file:
        writer = csv.writer(matrix_file)
        writer.writerow(columns)
        for sample in sorted(samples, key=lambda row: row["sample"]):
            values = sample["trophy_values"]
            row = [sample["sample"], sample["assembly"]]
            row.extend("|".join(sorted(values[column])) for column in trophy_columns)
            row.extend(
                int(nutrition_type in sample["nutrition_types"])
                for nutrition_type in nutrition_types
            )
            row.extend(int(ko in sample["kos"]) for ko in all_kos)
            writer.writerow(row)
    os.replace(temporary_output, OUTPUT)

    empty_samples = sum(not sample["kos"] for sample in samples)
    assignments = sum(len(sample["kos"]) for sample in samples)
    print(f"Annotated genomes: {len(sample_to_result)}")
    print(f"Matched BacDive records: {matched_rows}")
    print(f"Samples: {len(samples)}")
    print(f"Unmatched BacDive records excluded: {unmatched_rows}")
    print(f"Nutrition targets: {len(target_columns)}")
    print(f"KO features: {len(all_kos)}")
    print(f"Binary assignments: {assignments}")
    print(f"Samples with no assigned KOs: {empty_samples}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()