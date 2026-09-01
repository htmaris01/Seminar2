"""Command line entry point for local runs and reproducible experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from toxic_comments.config import RAW_DATA_DIR, RESULTS_DIR
from toxic_comments.experiment import run_experiment
from toxic_comments.repositories import CsvFileDatasetRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toxic comment classification experiment")
    parser.add_argument(
        "--data",
        type=Path,
        default=RAW_DATA_DIR / "train.csv",
        help="Path to the Kaggle train.csv file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR,
        help="Directory where result CSV files will be written.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of k-fold splits.")
    parser.add_argument("--max-features", type=int, default=50_000, help="TF-IDF vocabulary size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = CsvFileDatasetRepository(args.data)
    fold_results, summary = run_experiment(
        repository=repository,
        output_dir=args.output,
        n_splits=args.folds,
        max_features=args.max_features,
    )
    print(f"Saved fold metrics: {args.output / 'cross_validation_results.csv'}")
    print(f"Saved summary metrics: {args.output / 'summary_results.csv'}")
    print(summary)
