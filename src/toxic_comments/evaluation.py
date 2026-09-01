"""Evaluation utilities for multi-label toxic comment classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold

from toxic_comments.config import LABEL_COLUMNS, TEXT_COLUMN


@dataclass(frozen=True)
class FoldResult:
    fold: int
    model_name: str
    subset_accuracy: float
    hamming_loss: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_f1: float
    micro_roc_auc: float | None


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Compute metrics suitable for multi-label classification."""

    metrics = {
        "subset_accuracy": accuracy_score(y_true, y_pred),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_roc_auc": None,
    }

    if y_score is not None:
        try:
            metrics["micro_roc_auc"] = roc_auc_score(y_true, y_score, average="micro")
        except ValueError:
            metrics["micro_roc_auc"] = None

    return metrics


def cross_validate_model(
    estimator,
    data: pd.DataFrame,
    model_name: str,
    n_splits: int = 5,
    random_state: int = 42,
    text_column: str = TEXT_COLUMN,
) -> pd.DataFrame:
    """Run k-fold cross validation and return fold-level metrics."""

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    x = data[text_column]
    y = data[LABEL_COLUMNS].to_numpy()
    results: list[FoldResult] = []

    for fold_index, (train_index, test_index) in enumerate(splitter.split(x), start=1):
        fold_estimator = clone(estimator)
        fold_estimator.fit(x.iloc[train_index], y[train_index])

        y_pred = fold_estimator.predict(x.iloc[test_index])
        y_score = _predict_scores(fold_estimator, x.iloc[test_index])
        metrics = evaluate_predictions(y[test_index], y_pred, y_score)

        results.append(
            FoldResult(
                fold=fold_index,
                model_name=model_name,
                subset_accuracy=float(metrics["subset_accuracy"]),
                hamming_loss=float(metrics["hamming_loss"]),
                micro_precision=float(metrics["micro_precision"]),
                micro_recall=float(metrics["micro_recall"]),
                micro_f1=float(metrics["micro_f1"]),
                macro_f1=float(metrics["macro_f1"]),
                micro_roc_auc=_optional_float(metrics["micro_roc_auc"]),
            )
        )

    return pd.DataFrame([result.__dict__ for result in results])


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold-level metrics by model using mean and standard deviation."""

    metric_columns = [
        "subset_accuracy",
        "hamming_loss",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_f1",
        "micro_roc_auc",
    ]
    return results.groupby("model_name")[metric_columns].agg(["mean", "std"]).round(4)


def _predict_scores(estimator, x_test: pd.Series) -> np.ndarray | None:
    if not hasattr(estimator, "predict_proba"):
        return None

    probabilities = estimator.predict_proba(x_test)
    if isinstance(probabilities, list):
        estimators = getattr(estimator, "estimators_", [])
        scores = []
        for label_index, class_probability in enumerate(probabilities):
            classes = getattr(estimators[label_index], "classes_", None)
            if classes is None or 1 not in classes:
                scores.append(np.zeros(class_probability.shape[0]))
                continue

            positive_class_index = int(np.where(classes == 1)[0][0])
            scores.append(class_probability[:, positive_class_index])

        return np.column_stack(scores)

    if isinstance(probabilities, np.ndarray) and probabilities.ndim == 3:
        return probabilities[:, :, 1].T

    return probabilities


def _optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)
