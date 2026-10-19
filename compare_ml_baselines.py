#!/usr/bin/env python3
"""Compare multilabel classifiers on raw KOfam KO features.

Reads the project X and Y CSVs, aligns rows by sample ID, and evaluates four
classifiers with five-fold cross-validation grouped by genome assembly. Writes
per-trait and macro-average metrics plus out-of-fold predictions.
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

BASE = Path(__file__).resolve().parent.parent
# Keep these paths project-relative so the script works from a repository clone.
X_PATH = BASE / "ml_feature_matrix.csv"
Y_PATH = BASE / "ml_target_matrix.csv"
OUTPUT_DIR = BASE / "ml_baseline_results"
KO_PATTERN = re.compile(r"K\d{5}\Z")
# Fixed folds and model seeds make comparisons reproducible.
N_SPLITS = 5
RANDOM_STATE = 42


def load_aligned_matrices():
    """Load binary KO inputs and six binary targets, aligned by sample ID."""
    # X also contains nutrition labels and target columns; using them as inputs
    # would leak the answer into the model, so retain only K-number columns.
    x_frame = pd.read_csv(X_PATH)
    y_frame = pd.read_csv(Y_PATH)
    required = {"sample", "assembly"}
    if not required.issubset(x_frame.columns) or not required.issubset(y_frame.columns):
        sys.exit("Both matrices must contain sample and assembly columns.")
    if x_frame["sample"].duplicated().any() or y_frame["sample"].duplicated().any():
        sys.exit("Duplicate sample IDs found; cannot align matrices safely.")

    x_by_sample = x_frame.set_index("sample")
    y_by_sample = y_frame.set_index("sample")
    shared_samples = x_frame.loc[x_frame["sample"].isin(y_by_sample.index), "sample"]
    if set(shared_samples) != set(y_by_sample.index):
        sys.exit("Y contains samples absent from X; review the matrix join before training.")

    ko_columns = [column for column in x_frame.columns if KO_PATTERN.fullmatch(column)]
    target_columns = [column for column in y_frame.columns if column not in required]
    if not ko_columns or len(target_columns) != 6:
        sys.exit(f"Expected KO features and six Y targets; found {len(ko_columns)} KOs and {len(target_columns)} targets.")

    x = x_by_sample.loc[shared_samples, ko_columns].astype(np.uint8)
    y = y_by_sample.loc[shared_samples, target_columns].astype(np.uint8)
    groups = x_by_sample.loc[shared_samples, "assembly"].to_numpy()
    samples = shared_samples.to_numpy()

    if not x.isin([0, 1]).all().all() or not y.isin([0, 1]).all().all():
        sys.exit("Expected binary KO features and binary target values.")
    if len(np.unique(groups)) < N_SPLITS:
        sys.exit(f"Need at least {N_SPLITS} assemblies for grouped cross-validation.")
    for target in target_columns:
        positive_groups = np.unique(groups[y[target].to_numpy() == 1])
        if len(positive_groups) < N_SPLITS:
            sys.exit(
                f"Target {target} has positive labels in only {len(positive_groups)} assemblies; "
                f"cannot evaluate reliably with {N_SPLITS} grouped folds."
            )
    return x.to_numpy(), y.to_numpy(), groups, samples, ko_columns, target_columns


def build_models():
    """Return the four baseline estimators with imbalance-aware settings."""
    return {
        # One-vs-rest fits one binary classifier for each nutritional trait.
        "LogisticRegression": OneVsRestClassifier(
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            n_jobs=1,
        ),
        "LinearSVM": OneVsRestClassifier(
            LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=20000,
            ),
            n_jobs=1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=4,
        ),
        "XGBoost": OneVsRestClassifier(
            XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                eval_metric="logloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=4,
                verbosity=0,
            ),
            n_jobs=1,
        ),
    }


def grouped_multilabel_folds(y, groups):
    """Balance trait positives across folds without splitting assemblies.

    Rare-label assemblies seed separate folds first; remaining groups are
    greedily assigned to minimize differences in trait-positive and sample
    counts. The fixed random seed resolves ties reproducibly.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    group_to_indices = {}
    for index, group in enumerate(groups):
        group_to_indices.setdefault(group, []).append(index)

    group_items = list(group_to_indices.items())
    tie_breakers = rng.random(len(group_items))
    totals = np.maximum(y.sum(axis=0), 1)
    desired_positives = y.sum(axis=0) / N_SPLITS
    desired_samples = len(y) / N_SPLITS
    group_items = sorted(
        zip(group_items, tie_breakers),
        key=lambda item: (
            -np.max(y[item[0][1]].sum(axis=0) / totals),
            item[1],
        ),
    )

    fold_groups = [[] for _ in range(N_SPLITS)]
    fold_positive_counts = np.zeros((N_SPLITS, y.shape[1]), dtype=float)
    fold_sample_counts = np.zeros(N_SPLITS, dtype=float)

    def assign_group(fold, group, indices):
        group_positive_counts = y[indices].sum(axis=0)
        fold_groups[fold].append(group)
        fold_positive_counts[fold] += group_positive_counts
        fold_sample_counts[fold] += len(indices)

    # Seed every fold with a rare-label assembly first; this is especially
    # important for traits such as Photo that have few positive assemblies.
    for fold, ((group, indices), _) in enumerate(group_items[:N_SPLITS]):
        assign_group(fold, group, indices)

    for (group, indices), _ in group_items[N_SPLITS:]:
        group_positive_counts = y[indices].sum(axis=0)
        fold_costs = []
        for fold in range(N_SPLITS):
            candidate_counts = fold_positive_counts.copy()
            candidate_counts[fold] += group_positive_counts
            candidate_sizes = fold_sample_counts.copy()
            candidate_sizes[fold] += len(indices)
            positive_error = np.square(
                (candidate_counts - desired_positives)
                / np.maximum(desired_positives, 1)
            ).sum(axis=1).sum()
            sample_error = np.square(
                (candidate_sizes - desired_samples) / desired_samples
            ).sum()
            fold_costs.append(positive_error + 0.25 * sample_error)
        best_folds = np.flatnonzero(np.isclose(fold_costs, np.min(fold_costs)))
        selected_fold = int(rng.choice(best_folds))
        assign_group(selected_fold, group, indices)

    group_to_fold = {
        group: fold
        for fold, fold_group_list in enumerate(fold_groups)
        for group in fold_group_list
    }
    all_indices = np.arange(len(y))
    folds = []
    for fold in range(N_SPLITS):
        test_indices = np.array(
            [index for index, group in enumerate(groups) if group_to_fold[group] == fold],
            dtype=int,
        )
        train_indices = np.setdiff1d(all_indices, test_indices)
        folds.append((train_indices, test_indices))

    for trait_index in range(y.shape[1]):
        if any(y[test, trait_index].sum() == 0 for _, test in folds):
            raise ValueError("Grouped fold assignment left a trait absent from a test fold.")
    return folds


def score_predictions(model_name, trait, truth, prediction):
    """Calculate positive-class precision, accuracy, and Matthews correlation."""
    return {
        "model": model_name,
        "trait": trait,
        "precision": precision_score(truth, prediction, zero_division=0),
        "accuracy": accuracy_score(truth, prediction),
        "mcc": matthews_corrcoef(truth, prediction),
    }


def main() -> None:
    """Train each baseline on each fold and save out-of-fold evaluation data."""
    x, y, groups, samples, ko_columns, target_columns = load_aligned_matrices()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    folds = grouped_multilabel_folds(y, groups)
    models = build_models()
    metric_rows = []
    prediction_rows = []

    for model_name, model in models.items():
        print(f"Training {model_name} with {N_SPLITS}-fold assembly-grouped CV...", flush=True)
        predictions = np.zeros_like(y)
        for fold, (train_indices, test_indices) in enumerate(folds, start=1):
            # Groups are assigned to one test fold only, preventing assemblies
            # represented by multiple BacDive strains from crossing the split.
            if len(np.unique(groups[train_indices])) != len(set(groups[train_indices])):
                sys.exit("Assembly leakage detected in training fold.")
            model.fit(x[train_indices], y[train_indices])
            predictions[test_indices] = model.predict(x[test_indices]).astype(np.uint8)
            print(f"  fold {fold}/{N_SPLITS} complete", flush=True)

        for index, trait in enumerate(target_columns):
            metric_rows.append(
                score_predictions(model_name, trait, y[:, index], predictions[:, index])
            )
        metric_rows.append(
            {
                # Macro-average gives each trait equal weight despite imbalance.
                "model": model_name,
                "trait": "MACRO_AVERAGE",
                "precision": np.mean(
                    [row["precision"] for row in metric_rows if row["model"] == model_name]
                ),
                "accuracy": np.mean(
                    [row["accuracy"] for row in metric_rows if row["model"] == model_name]
                ),
                "mcc": np.mean(
                    [row["mcc"] for row in metric_rows if row["model"] == model_name]
                ),
            }
        )

        for row_index, sample in enumerate(samples):
            prediction = {
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
            prediction_rows.append({"model": model_name, **prediction})

    # Predictions are out-of-fold: every saved prediction comes from a model
    # that did not train on that sample or any sample from its assembly.
    metrics_path = OUTPUT_DIR / "raw_ko_baseline_metrics.csv"
    predictions_path = OUTPUT_DIR / "raw_ko_oof_predictions.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["model", "trait", "precision", "accuracy", "mcc"])
        writer.writeheader()
        writer.writerows(metric_rows)
    prediction_columns = ["model", "sample", "assembly", "fold"] + [
        column
        for trait in target_columns
        for column in (f"true__{trait}", f"pred__{trait}")
    ]
    with predictions_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=prediction_columns)
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(f"Samples: {len(samples)}; assemblies: {len(np.unique(groups))}")
    print(f"Raw KO features: {len(ko_columns)}; traits: {len(target_columns)}")
    print(f"Metrics: {metrics_path}")
    print(f"Out-of-fold predictions: {predictions_path}")


if __name__ == "__main__":
    main()