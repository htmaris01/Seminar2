"""Business workflow for loading data, training models, and saving results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from toxic_comments.config import HEAVY_TEXT_COLUMN
from toxic_comments.evaluation import cross_validate_model, summarize_results
from toxic_comments.folds import make_kfold_splits
from toxic_comments.cleaning import process_cleaning
from toxic_comments.repositories import DatasetRepository, validate_training_data
from toxic_comments.models.registry import build_models


def run_experiment(
    repository: DatasetRepository,
    output_dir: Path,
    n_splits: int = 5,
    max_features: int = 50_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run baseline and ML classifier evaluation with a repository abstraction."""

    data = process_cleaning(
        validate_training_data(repository.load()),
        is_train=True,
        verbose=False,
    )
    data = data[data["is_empty_heavy"] == 0].reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = build_models(max_features=max_features)
    splits = make_kfold_splits(data, n_splits=n_splits, text_column=HEAVY_TEXT_COLUMN)

    fold_results = pd.concat(
        [
            cross_validate_model(
                model,
                data,
                model_name=name,
                n_splits=n_splits,
                text_column=HEAVY_TEXT_COLUMN,
                splits=splits,
            )
            for name, model in models.items()
        ],
        ignore_index=True,
    )
    summary = summarize_results(fold_results)

    fold_results.to_csv(output_dir / "cross_validation_results.csv", index=False)
    summary.to_csv(output_dir / "summary_results.csv")
    return fold_results, summary
