#!/usr/bin/env python3
"""Evaluate KO prevalence filtering without changing the raw baseline."""

import csv
import sys
from pathlib import Path

import numpy as np

from compare_ml_baselines import (
    N_SPLITS,
    build_models,
    grouped_multilabel_folds,
    load_aligned_matrices,
    score_predictions,
)

BASE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE / "ml_feature_engineering_results"
# Zero is the unfiltered control; positive cutoffs keep KOs seen in at least
# that many training assemblies for the current fold.
MIN_PREVALENCE_ASSEMBLIES = (0, 1, 2, 5, 10)


def assembly_level_prevalence(x, groups):
    """Count each KO at most once per assembly, not once per BacDive record."""
    assemblies = np.unique(groups)
    assembly_matrix = np.zeros((len(assemblies), x.shape[1]), dtype=np.uint8)
    for index, assembly in enumerate(assemblies):
        assembly_rows = x[groups == assembly]
        if not np.all(assembly_rows == assembly_rows[0]):
            raise ValueError(f"KO profiles differ between samples for assembly {assembly}.")
        assembly_matrix[index] = assembly_rows[0]
    return assemblies, assembly_matrix


def main() -> None:
    x, y, groups, samples, ko_columns, target_columns = load_aligned_matrices()
    assemblies, assembly_x = assembly_level_prevalence(x, groups)
    folds = grouped_multilabel_folds(y, groups)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    prediction_rows = []

    for minimum_prevalence in MIN_PREVALENCE_ASSEMBLIES:
        for model_name in build_models():
            print(
                f"Minimum KO prevalence {minimum_prevalence} assemblies; "
                f"training {model_name}...",
                flush=True,
            )
            predictions = np.zeros_like(y)
            retained_counts = []

            for fold_number, (train_indices, test_indices) in enumerate(folds, start=1):
                train_assemblies = np.unique(groups[train_indices])
                train_assembly_rows = np.isin(assemblies, train_assemblies)
                prevalence = assembly_x[train_assembly_rows].sum(axis=0)
                retained_features = (
                    np.ones(prevalence.shape, dtype=bool)
                    if minimum_prevalence == 0
                    else prevalence >= minimum_prevalence
                )
                if not retained_features.any():
                    sys.exit(
                        f"No KO features remain at cutoff {minimum_prevalence} "
                        f"in fold {fold_number}."
                    )
                retained_counts.append(int(retained_features.sum()))

                # Learn the prevalence filter on training assemblies only.
                # The test fold cannot influence which KO columns are retained.
                model = build_models()[model_name]
                model.fit(x[train_indices][:, retained_features], y[train_indices])
                predictions[test_indices] = model.predict(
                    x[test_indices][:, retained_features]
                ).astype(np.uint8)
                print(f"  fold {fold_number}/{N_SPLITS}: {retained_counts[-1]} KOs", flush=True)

            model_metrics = [
                score_predictions(model_name, trait, y[:, index], predictions[:, index])
                for index, trait in enumerate(target_columns)
            ]
            metric_rows.extend(
                {
                    "model": row["model"],
                    "min_prevalence_assemblies": minimum_prevalence,
                    "trait": row["trait"],
                    "precision": row["precision"],
                    "accuracy": row["accuracy"],
                    "mcc": row["mcc"],
                    "mean_retained_kos": float(np.mean(retained_counts)),
                }
                for row in model_metrics
            )
            metric_rows.append(
                {
                    "model": model_name,
                    "min_prevalence_assemblies": minimum_prevalence,
                    "trait": "MACRO_AVERAGE",
                    "precision": float(np.mean([row["precision"] for row in model_metrics])),
                    "accuracy": float(np.mean([row["accuracy"] for row in model_metrics])),
                    "mcc": float(np.mean([row["mcc"] for row in model_metrics])),
                    "mean_retained_kos": float(np.mean(retained_counts)),
                }
            )

            for row_index, sample in enumerate(samples):
                prediction = {
                    "model": model_name,
                    "min_prevalence_assemblies": minimum_prevalence,
                    "sample": sample,
                    "assembly": groups[row_index],
                    "fold": next(
                        fold_number
                        for fold_number, (_, test_indices) in enumerate(folds, start=1)
                        if row_index in test_indices
                    ),
                }
                for target_index, trait in enumerate(target_columns):
                    prediction[f"true__{trait}"] = y[row_index, target_index]
                    prediction[f"pred__{trait}"] = predictions[row_index, target_index]
                prediction_rows.append(prediction)

    metrics_path = OUTPUT_DIR / "ko_prevalence_metrics.csv"
    predictions_path = OUTPUT_DIR / "ko_prevalence_oof_predictions.csv"
    metric_columns = [
        "model",
        "min_prevalence_assemblies",
        "trait",
        "precision",
        "accuracy",
        "mcc",
        "mean_retained_kos",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=metric_columns)
        writer.writeheader()
        writer.writerows(metric_rows)

    prediction_columns = [
        "model",
        "min_prevalence_assemblies",
        "sample",
        "assembly",
        "fold",
        *[
            column
            for trait in target_columns
            for column in (f"true__{trait}", f"pred__{trait}")
        ],
    ]
    with predictions_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=prediction_columns)
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(f"Samples: {len(samples)}; assemblies: {len(assemblies)}; raw KOs: {len(ko_columns)}")
    print(f"Metrics: {metrics_path}")
    print(f"Out-of-fold predictions: {predictions_path}")


if __name__ == "__main__":
    main()