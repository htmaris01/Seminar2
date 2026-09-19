#!/usr/bin/env bash
# Chạy script này TỪ THƯ MỤC GỐC repo (nơi có pyproject.toml), vd: ~/Seminar2
# Ghi đè trực tiếp toàn bộ file đã sửa: Method 3 + fix tiến độ/incremental-save + save/load model + GPU check (CLI và notebook).
set -e

mkdir -p src/toxic_comments/models tests notebooks

cat > src/toxic_comments/cli.py << 'FILE_EOF'
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
    parser.add_argument(
        "--include-transformers",
        action="store_true",
        help=(
            "Also fine-tune the RoBERTa-based models (e.g. Method 3, label "
            "dependency). Off by default because it fine-tunes a transformer "
            "once per fold — consider a smaller --folds value (e.g. 3) when "
            "this is set."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help=(
            "Force the transformer models onto this device. Default (unset) "
            "auto-detects GPU vs CPU. Pass --device cuda to fail fast if no "
            "GPU is actually available, instead of silently falling back to "
            "a very slow CPU run."
        ),
    )
    return parser.parse_args()


def _print_device_status(requested_device: str | None) -> None:
    """Print which device the transformer models will actually run on.

    Mirrors the GPU-check cell in
    notebooks/train_and_save_roberta_label_dependency.ipynb, so this is
    visible in the terminal too instead of only happening silently inside
    ``_roberta_base.RobertaMultiLabelBase.fit``.
    """

    import torch

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested but torch.cuda.is_available() is "
            "False — no GPU visible in this environment. Fix the GPU setup "
            "(or drop --device to fall back to CPU) before running."
        )

    device = requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        print(f"✅ Dùng GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  KHÔNG CÓ GPU — đang chạy CPU, sẽ rất chậm cho model transformer.")


def main() -> None:
    args = parse_args()
    if args.include_transformers:
        _print_device_status(args.device)

    repository = CsvFileDatasetRepository(args.data)
    fold_results, summary = run_experiment(
        repository=repository,
        output_dir=args.output,
        n_splits=args.folds,
        max_features=args.max_features,
        include_transformer_models=args.include_transformers,
        device=args.device,
    )
    print(f"Saved fold metrics: {args.output / 'cross_validation_results.csv'}")
    print(f"Saved summary metrics: {args.output / 'summary_results.csv'}")
    print(summary)

FILE_EOF

cat > src/toxic_comments/experiment.py << 'FILE_EOF'
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
    include_transformer_models: bool = False,
    device: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run baseline and ML classifier evaluation with a repository abstraction.

    Fold results are appended to ``cross_validation_results.csv`` and the
    summary is refreshed after every fold — not just once at the very end.
    This matters most for the slow transformer models (see
    ``models/roberta_label_dependency.py``): a long-running fine-tune can
    now be inspected mid-run (e.g. with the ``view_experiment_results``
    notebook) and, if it crashes or is interrupted, the folds/models that
    already finished are not lost.
    """

    data = process_cleaning(
        validate_training_data(repository.load()),
        is_train=True,
        verbose=False,
    )
    data = data[data["is_empty_heavy"] == 0].reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = build_models(
        max_features=max_features,
        include_transformer_models=include_transformer_models,
        device=device,
    )
    splits = make_kfold_splits(data, n_splits=n_splits, text_column=HEAVY_TEXT_COLUMN)

    fold_results_path = output_dir / "cross_validation_results.csv"
    summary_path = output_dir / "summary_results.csv"
    # Start this run from a clean file rather than appending to a stale one
    # left over from a previous run.
    if fold_results_path.exists():
        fold_results_path.unlink()

    all_results: list[pd.DataFrame] = []

    def _persist_fold(result) -> None:
        row = pd.DataFrame([result.__dict__])
        row.to_csv(
            fold_results_path,
            mode="a",
            header=not fold_results_path.exists(),
            index=False,
        )
        # Cheap to recompute after every fold, and means summary_results.csv
        # always reflects everything finished so far, not just the final state.
        so_far = pd.concat(all_results + [row], ignore_index=True)
        summarize_results(so_far).to_csv(summary_path)

    for name, model in models.items():
        model_fold_results = cross_validate_model(
            model,
            data,
            model_name=name,
            n_splits=n_splits,
            text_column=HEAVY_TEXT_COLUMN,
            splits=splits,
            on_fold_complete=_persist_fold,
        )
        all_results.append(model_fold_results)

    fold_results = pd.concat(all_results, ignore_index=True)
    summary = summarize_results(fold_results)
    summary.to_csv(summary_path)
    return fold_results, summary

FILE_EOF

cat > src/toxic_comments/evaluation.py << 'FILE_EOF'
"""Evaluation utilities for multi-label toxic comment classification."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

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
    splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
    on_fold_complete: Callable[[FoldResult], None] | None = None,
) -> pd.DataFrame:
    """Run k-fold cross validation and return fold-level metrics.

    Prints per-fold progress (with elapsed time) so a slow model — e.g. a
    RoBERTa fine-tune — doesn't look frozen. If ``on_fold_complete`` is
    given, it's called with each ``FoldResult`` right after that fold
    finishes, so callers (see ``experiment.run_experiment``) can persist
    results incrementally instead of only at the very end.
    """

    if splits is None:
        from toxic_comments.folds import make_fold_splits

        splits = make_fold_splits(
            data,
            n_splits=n_splits,
            random_state=random_state,
            text_column=text_column,
            strategy="kfold",
        )

    x = data[text_column]
    y = data[LABEL_COLUMNS].to_numpy()
    results: list[FoldResult] = []

    for fold_index, (train_index, test_index) in enumerate(splits, start=1):
        fold_start = time.perf_counter()
        fold_estimator = clone(estimator)
        fold_estimator.fit(x.iloc[train_index], y[train_index])

        y_pred = fold_estimator.predict(x.iloc[test_index])
        y_score = _predict_scores(fold_estimator, x.iloc[test_index])
        metrics = evaluate_predictions(y[test_index], y_pred, y_score)

        result = FoldResult(
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
        elapsed = time.perf_counter() - fold_start
        print(
            f"[{model_name}] fold {fold_index}/{len(splits)} done in "
            f"{elapsed:.1f}s — macro_f1={result.macro_f1:.4f}, "
            f"micro_f1={result.micro_f1:.4f}"
        )

        if on_fold_complete is not None:
            on_fold_complete(result)

        results.append(result)

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

FILE_EOF

cat > src/toxic_comments/models/registry.py << 'FILE_EOF'
from toxic_comments.models.baseline import build_dummy_baseline
from toxic_comments.models.tfidf_logreg import build_tfidf_logistic_regression


def build_models(
    max_features: int = 50_000,
    include_transformer_models: bool = False,
    device: str | None = None,
):
    """Return the registry of models to evaluate.

    ``include_transformer_models`` is off by default: fine-tuning a RoBERTa
    model inside a 5-fold CV loop is far more expensive than the sklearn
    baselines above, so leaving it off keeps existing quick runs
    (``python -m toxic_comments``) and the test suite fast. Pass
    ``--include-transformers`` on the CLI, or the flag directly here, to add
    Method 3 (and, once implemented, Methods 1/2/4) to the comparison.

    ``device`` is forwarded to the transformer model(s) — ``None`` (default)
    auto-detects GPU vs CPU; pass ``"cuda"``/``"cpu"`` to force one.
    """

    models = {
        "dummy_most_frequent": build_dummy_baseline(),
        "tfidf_logistic_regression": build_tfidf_logistic_regression(
            max_features=max_features
        ),
    }

    if include_transformer_models:
        from toxic_comments.models.roberta_label_dependency import (
            build_roberta_label_dependency,
        )

        models["roberta_label_dependency"] = build_roberta_label_dependency(device=device)

    return models

FILE_EOF

cat > src/toxic_comments/models/_roberta_base.py << 'FILE_EOF'
"""Shared base class for RoBERTa-based multi-label classifiers.

This module gives Method 1 (RoBERTa+BCE), Method 2 (RoBERTa+ASL), and
Method 3 (RoBERTa + label-dependency layer, this integration) one
tokenization/training/inference loop to share, so each variant only needs to
override the model architecture (``_build_model``) and, if needed, the loss
function (``_compute_loss``). Everything else — including the
``get_params``/``set_params``/``clone`` contract that
``evaluation.cross_validate_model`` relies on via ``sklearn.base.clone`` — is
implemented once here.

Design notes
------------
- Every constructor parameter is stored verbatim as an attribute with the
  same name (no mutation, no derived state) so ``sklearn.base.clone`` can
  rebuild an untrained copy of the estimator from ``get_params()`` alone.
  Anything computed from data (tokenizer, model weights, label statistics)
  is only ever set inside ``fit`` and named with a trailing underscore
  (``self.model_``), which is the scikit-learn convention for "fitted"
  attributes that ``clone`` must NOT copy.
- Fine-tuning a transformer inside a 5-fold CV loop is expensive. See
  ``models/registry.py`` and ``cli.py`` for the ``--include-transformers``
  opt-in flag that keeps the existing fast baseline-only runs unaffected.
- ``cross_validate_model`` clones + fits + discards one model per fold — by
  design, since the point of that loop is comparing methods, not producing
  a deployable artifact. Use ``save``/``load`` below (see
  ``notebooks/train_and_save_roberta_label_dependency.ipynb``) to train once
  on a train/test split and persist that model to disk instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from toxic_comments.config import LABEL_COLUMNS


class _TextLabelDataset(Dataset):
    """Wraps tokenizer output (+ optional labels) for a torch DataLoader."""

    def __init__(self, encodings: dict[str, torch.Tensor], labels: np.ndarray | None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: tensor[index] for key, tensor in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.float32)
        return item


class RobertaMultiLabelBase(BaseEstimator, ClassifierMixin):
    """Shared fit/predict loop for RoBERTa-based multi-label classifiers.

    Subclasses must implement ``_build_model`` and may override
    ``_compute_loss``. See ``roberta_label_dependency.py`` for Method 3.
    """

    def __init__(
        self,
        pretrained_model_name: str = "roberta-base",
        max_length: int = 128,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 2,
        num_labels: int = len(LABEL_COLUMNS),
        random_state: int = 42,
        device: str | None = None,
    ) -> None:
        self.pretrained_model_name = pretrained_model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_labels = num_labels
        self.random_state = random_state
        self.device = device

    # ---- extension points for subclasses --------------------------------
    def _build_model(self, y: np.ndarray) -> nn.Module:
        """Build and return the torch module for this variant.

        Receives the *training fold's* labels so subclasses that need label
        statistics (e.g. Method 3's co-occurrence adjacency) can compute
        them here — this keeps the statistic fold-local and avoids leaking
        information from the held-out fold.
        """

        raise NotImplementedError

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Default loss is plain BCE. Method 2 (ASL) overrides this."""

        return nn.functional.binary_cross_entropy_with_logits(logits, targets)

    # ---- scikit-learn estimator contract ---------------------------------
    def fit(self, X: pd.Series, y: np.ndarray) -> "RobertaMultiLabelBase":
        y = np.asarray(y)
        torch.manual_seed(self.random_state)
        self.device_ = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer_ = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        self.model_ = self._build_model(y).to(self.device_)

        loader = self._make_loader(X, y, shuffle=True)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate)

        self.model_.train()
        for _ in range(self.num_epochs):
            for batch in loader:
                targets = batch.pop("labels").to(self.device_)
                batch = {key: value.to(self.device_) for key, value in batch.items()}

                optimizer.zero_grad()
                logits = self.model_(**batch)
                loss = self._compute_loss(logits, targets)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X: pd.Series) -> np.ndarray:
        loader = self._make_loader(X, y=None, shuffle=False)
        self.model_.eval()
        batches: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(self.device_) for key, value in batch.items()}
                logits = self.model_(**batch)
                batches.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(batches, axis=0)

    def predict(self, X: pd.Series, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # ---- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Persist the fitted model, tokenizer, and reconstruction params.

        Works for any subclass without extra code: ``self.model_.state_dict()``
        already includes registered buffers (e.g. Method 3's co-occurrence
        adjacency), so subclass-specific state is captured automatically.
        """

        if not hasattr(self, "model_"):
            raise RuntimeError("Cannot save an unfitted estimator — call fit() first.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model_.state_dict(), path / "model_state_dict.pt")
        self.tokenizer_.save_pretrained(path)
        (path / "params.json").write_text(json.dumps(self.get_params(), indent=2))

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "RobertaMultiLabelBase":
        """Reconstruct a fitted estimator saved with ``save``. Ready to
        ``predict``/``predict_proba`` immediately — no need to call ``fit``.
        """

        path = Path(path)
        params = json.loads((path / "params.json").read_text())
        estimator = cls(**params)
        estimator.device_ = device or ("cuda" if torch.cuda.is_available() else "cpu")
        estimator.tokenizer_ = AutoTokenizer.from_pretrained(path)

        # _build_model may need label statistics (e.g. Method 3's adjacency).
        # A zero-filled placeholder is fine here: the real values are part of
        # the state_dict loaded right below, which overwrites this anyway.
        placeholder_y = np.zeros((1, estimator.num_labels))
        estimator.model_ = estimator._build_model(placeholder_y).to(estimator.device_)

        state_dict = torch.load(path / "model_state_dict.pt", map_location=estimator.device_)
        estimator.model_.load_state_dict(state_dict)
        estimator.model_.eval()
        return estimator

    # ---- helpers ----------------------------------------------------------
    def _make_loader(self, X: pd.Series, y: np.ndarray | None, shuffle: bool) -> DataLoader:
        encodings = self.tokenizer_(
            list(X),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        dataset = _TextLabelDataset(dict(encodings), y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

FILE_EOF

cat > src/toxic_comments/models/roberta_label_dependency.py << 'FILE_EOF'
"""Method 3 — RoBERTa + explicit label-dependency graph layer + BCE.

Research role (see the 5-stage research design)
-------------------------------------------------
Representation (RoBERTa encoder) and loss (plain BCE) are held IDENTICAL to
Method 1. The only change relative to Method 1 is one new mechanism: a small
graph message-passing layer over the six toxicity labels, using the
*empirical label co-occurrence* from the training fold as a fixed adjacency
— e.g. ``severe_toxic`` almost always co-occurring with ``toxic``. This
isolates the contribution of explicit label-dependency modeling from
representation changes (Method 1) and imbalance-handling changes (Method 2),
so that Method 4's combined model can later be attributed correctly in the
ablations.

This is intentionally NOT the same mechanism as Method 4's label-attention
head (which attends over *tokens* to build a per-label representation).
Method 3 instead lets each label's *own* representation be refined by its
*correlated labels'* representations — a GraphSAGE-style aggregation step
over a 6-node label graph, using the co-occurrence matrix as a fixed,
data-driven adjacency instead of a learned one.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from transformers import AutoModel

from toxic_comments.config import LABEL_COLUMNS
from toxic_comments.models._roberta_base import RobertaMultiLabelBase


def compute_cooccurrence_adjacency(y: np.ndarray) -> torch.Tensor:
    """Row-normalized empirical label co-occurrence from the training fold.

    ``adjacency[i, j]`` approximates P(label_j = 1 | label_i = 1), estimated
    only from the labels passed in (the training fold), so no information
    leaks from the held-out fold. The diagonal is zeroed so a label does not
    "propagate to itself" in the message-passing step below.
    """

    y = np.asarray(y, dtype=float)
    co_occurrence = y.T @ y
    label_counts = np.clip(y.sum(axis=0), a_min=1.0, a_max=None)
    adjacency = co_occurrence / label_counts[:, None]
    np.fill_diagonal(adjacency, 0.0)
    return torch.tensor(adjacency, dtype=torch.float32)


class LabelDependencyGraphLayer(nn.Module):
    """One message-passing step over the six toxicity labels.

    Each label starts from its own hidden vector (a per-label linear
    projection of the pooled RoBERTa representation). It is then updated
    using a weighted sum of the *other* labels' vectors, weighted by the
    fixed co-occurrence adjacency, before a final per-label linear scorer
    produces the logit.
    """

    def __init__(self, hidden_size: int, dep_dim: int, num_labels: int, adjacency: torch.Tensor):
        super().__init__()
        self.num_labels = num_labels
        self.dep_dim = dep_dim
        self.register_buffer("adjacency", adjacency)
        self.label_projection = nn.Linear(hidden_size, num_labels * dep_dim)
        self.self_transform = nn.Linear(dep_dim, dep_dim)
        self.neighbor_transform = nn.Linear(dep_dim, dep_dim)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(dep_dim, 1)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        batch_size = pooled.shape[0]
        label_hidden = self.label_projection(pooled).view(batch_size, self.num_labels, self.dep_dim)

        # For each label l: sum_j adjacency[l, j] * label_hidden[:, j, :]
        neighbor_hidden = torch.einsum("lj,bjd->bld", self.adjacency, label_hidden)

        updated = self.activation(
            self.self_transform(label_hidden) + self.neighbor_transform(neighbor_hidden)
        )
        logits = self.classifier(updated).squeeze(-1)  # [batch, num_labels]
        return logits


class _RobertaWithLabelDependency(nn.Module):
    """RoBERTa encoder -> [CLS] pooling -> LabelDependencyGraphLayer."""

    def __init__(self, pretrained_model_name: str, num_labels: int, dep_dim: int, adjacency: torch.Tensor):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dependency_layer = LabelDependencyGraphLayer(hidden_size, dep_dim, num_labels, adjacency)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token representation
        return self.dependency_layer(pooled)


class RobertaLabelDependencyClassifier(RobertaMultiLabelBase):
    """Method 3: RoBERTa encoder + explicit label-dependency graph layer, BCE loss.

    All constructor parameters are re-declared explicitly (rather than via
    ``**kwargs``) because scikit-learn's ``get_params``/``clone`` machinery
    introspects each estimator's own ``__init__`` signature — a subclass
    that swallows parent params into ``**kwargs`` breaks that contract.
    """

    def __init__(
        self,
        dep_dim: int = 64,
        pretrained_model_name: str = "roberta-base",
        max_length: int = 128,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 2,
        num_labels: int = len(LABEL_COLUMNS),
        random_state: int = 42,
        device: str | None = None,
    ) -> None:
        super().__init__(
            pretrained_model_name=pretrained_model_name,
            max_length=max_length,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            num_labels=num_labels,
            random_state=random_state,
            device=device,
        )
        self.dep_dim = dep_dim

    def _build_model(self, y: np.ndarray) -> nn.Module:
        adjacency = compute_cooccurrence_adjacency(y)
        return _RobertaWithLabelDependency(
            pretrained_model_name=self.pretrained_model_name,
            num_labels=self.num_labels,
            dep_dim=self.dep_dim,
            adjacency=adjacency,
        )
    # _compute_loss is inherited unchanged from the base class (plain BCE) —
    # this is deliberate: Method 3 isolates the dependency-layer variable only.


def build_roberta_label_dependency(
    pretrained_model_name: str = "roberta-base",
    dep_dim: int = 64,
    max_length: int = 128,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 2,
    device: str | None = None,
) -> RobertaLabelDependencyClassifier:
    """Factory matching the project's existing ``build_*`` model convention.

    ``device`` defaults to ``None``, which auto-detects GPU vs CPU inside
    ``fit`` (see ``_roberta_base.RobertaMultiLabelBase.fit``). Pass
    ``device="cuda"`` explicitly if you want fit() to raise immediately when
    no GPU is available, instead of silently falling back to a very slow
    CPU run.
    """

    return RobertaLabelDependencyClassifier(
        dep_dim=dep_dim,
        pretrained_model_name=pretrained_model_name,
        max_length=max_length,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        device=device,
    )

FILE_EOF

cat > tests/test_label_dependency_layer.py << 'FILE_EOF'
"""Tests for Method 3 (RoBERTa + label-dependency graph layer).

These tests deliberately avoid downloading the pretrained ``roberta-base``
weights (no network access needed), so they can run in CI / on a laptop
without a Hugging Face Hub connection. They cover:

1. The co-occurrence adjacency computation.
2. The graph message-passing layer's forward pass (shape + gradient flow).
3. That the sklearn estimator contract (``get_params`` / ``clone``) holds,
   which is what ``evaluation.cross_validate_model`` relies on.

A full end-to-end fit/predict test against real ``roberta-base`` weights
should be run manually (e.g. on Colab, where network + GPU are available)
before trusting numbers from this model.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sklearn.base import clone  # noqa: E402

from toxic_comments.config import LABEL_COLUMNS  # noqa: E402
from toxic_comments.models.roberta_label_dependency import (  # noqa: E402
    LabelDependencyGraphLayer,
    RobertaLabelDependencyClassifier,
    compute_cooccurrence_adjacency,
)


def test_adjacency_captures_strong_cooccurrence():
    # severe_toxic (col 1) only ever appears alongside toxic (col 0)
    y = np.array(
        [
            [1, 1, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 1, 1],
        ]
    )

    adjacency = compute_cooccurrence_adjacency(y)

    assert adjacency.shape == (len(LABEL_COLUMNS), len(LABEL_COLUMNS))
    assert torch.allclose(torch.diagonal(adjacency), torch.zeros(len(LABEL_COLUMNS)))
    # P(toxic=1 | severe_toxic=1) should be 1.0 given the toy data above
    assert adjacency[1, 0].item() == pytest.approx(1.0)


def test_adjacency_handles_a_label_with_zero_positives():
    # "threat" (col 3) never occurs -> would divide by zero without clipping
    y = np.zeros((5, len(LABEL_COLUMNS)), dtype=int)
    y[:, 0] = 1

    adjacency = compute_cooccurrence_adjacency(y)

    assert torch.isfinite(adjacency).all()


def test_label_dependency_layer_forward_shape_and_gradients():
    batch_size, hidden_size, dep_dim, num_labels = 4, 32, 8, len(LABEL_COLUMNS)
    adjacency = torch.rand(num_labels, num_labels)
    adjacency.fill_diagonal_(0.0)

    layer = LabelDependencyGraphLayer(hidden_size, dep_dim, num_labels, adjacency)
    pooled = torch.randn(batch_size, hidden_size, requires_grad=True)

    logits = layer(pooled)
    assert logits.shape == (batch_size, num_labels)

    logits.sum().backward()
    assert pooled.grad is not None
    assert not torch.isnan(pooled.grad).any()


def test_label_dependency_layer_neighbors_actually_influence_output():
    """If the adjacency is all-zero, output should differ from a fully-connected adjacency."""

    hidden_size, dep_dim, num_labels = 16, 8, len(LABEL_COLUMNS)
    pooled = torch.randn(2, hidden_size)

    torch.manual_seed(0)
    zero_adjacency = torch.zeros(num_labels, num_labels)
    layer_isolated = LabelDependencyGraphLayer(hidden_size, dep_dim, num_labels, zero_adjacency)

    torch.manual_seed(0)
    full_adjacency = torch.ones(num_labels, num_labels) - torch.eye(num_labels)
    layer_connected = LabelDependencyGraphLayer(hidden_size, dep_dim, num_labels, full_adjacency)

    out_isolated = layer_isolated(pooled)
    out_connected = layer_connected(pooled)

    assert not torch.allclose(out_isolated, out_connected)


def test_classifier_is_clonable_without_being_fit():
    """evaluation.cross_validate_model calls sklearn.base.clone(estimator) per fold."""

    estimator = RobertaLabelDependencyClassifier(dep_dim=32, num_epochs=1, batch_size=4)
    cloned = clone(estimator)

    assert cloned.get_params() == estimator.get_params()
    assert cloned is not estimator

FILE_EOF

cat > requirements.txt << 'FILE_EOF'
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
jupyterlab>=4.0
matplotlib>=3.7
seaborn>=0.13
pytest>=7.4
torch>=2.1
transformers>=4.38

FILE_EOF

cat > pyproject.toml << 'FILE_EOF'
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "toxic-comments"
version = "0.1.0"
description = "Multi-label toxic comment classification project"
requires-python = ">=3.10"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "matplotlib>=3.7",
    "seaborn>=0.13",
    "torch>=2.1",
    "transformers>=4.38",
]

[project.optional-dependencies]
dev = ["pytest>=7.4", "jupyterlab>=4.0"]

[tool.setuptools.packages.find]
where = ["src"]

FILE_EOF

cat > notebooks/view_experiment_results.ipynb << 'FILE_EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Kết quả Cross-Validation — So sánh các Method\n",
    "\n",
    "Notebook này **chỉ đọc lại** các file CSV đã được `python -m toxic_comments` lưu sẵn trong thư mục `results/` — không train lại bất kỳ model nào, kể cả `roberta_label_dependency`. Chạy lại notebook này bao nhiêu lần cũng không tốn thời gian/GPU.\n",
    "\n",
    "File kết quả được ghi **tăng dần theo từng fold**, nên có thể chạy notebook này ngay cả khi terminal khác vẫn đang train — số liệu sẽ phản ánh đúng những fold đã hoàn thành.\n",
    "\n",
    "**Yêu cầu:** đã chạy ít nhất một fold của\n",
    "```\n",
    "python -m toxic_comments --include-transformers --folds 5\n",
    "```\n",
    "và có 2 file trong `results/`: `cross_validation_results.csv`, `summary_results.csv`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "from pathlib import Path\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "# Cùng quy ước với các notebook khác trong repo: nếu đang chạy từ notebooks/\n",
    "# thì repo root là thư mục cha, ngược lại lấy thư mục hiện tại.\n",
    "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == \"notebooks\" else Path.cwd()\n",
    "\n",
    "# Nếu bạn chạy CLI với --output khác mặc định, sửa lại dòng dưới cho khớp.\n",
    "RESULTS_DIR = PROJECT_ROOT / \"results\"\n",
    "\n",
    "FOLD_RESULTS_PATH = RESULTS_DIR / \"cross_validation_results.csv\"\n",
    "SUMMARY_PATH = RESULTS_DIR / \"summary_results.csv\"\n",
    "\n",
    "print(\"PROJECT_ROOT =\", PROJECT_ROOT)\n",
    "print(\"RESULTS_DIR  =\", RESULTS_DIR)\n",
    "print(\"fold results exists:\", FOLD_RESULTS_PATH.exists())\n",
    "print(\"summary exists:     \", SUMMARY_PATH.exists())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load dữ liệu đã lưu\n",
    "\n",
    "Nếu ô dưới báo lỗi `FileNotFoundError`: nghĩa là quá trình train ở bước CLI chưa chạy xong (experiment.py chỉ ghi 2 file này **sau khi tất cả model, tất cả fold đều xong**), hoặc `RESULTS_DIR` ở trên đang trỏ sai thư mục — kiểm tra lại đường dẫn `--output` bạn đã dùng khi chạy CLI."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "if not FOLD_RESULTS_PATH.exists() or not SUMMARY_PATH.exists():\n",
    "    raise FileNotFoundError(\n",
    "        f\"Chưa thấy kết quả trong {RESULTS_DIR}. \"\n",
    "        \"Chạy xong `python -m toxic_comments --include-transformers` trước, \"\n",
    "        \"rồi quay lại chạy notebook này.\"\n",
    "    )\n",
    "\n",
    "fold_results = pd.read_csv(FOLD_RESULTS_PATH)\n",
    "# summary_results.csv có 2 tầng cột (metric, mean/std) — cần header=[0, 1] để đọc đúng\n",
    "summary = pd.read_csv(SUMMARY_PATH, header=[0, 1], index_col=0)\n",
    "\n",
    "print(\"Các model có trong kết quả:\", fold_results['model_name'].unique().tolist())\n",
    "print(\"Số fold mỗi model:\")\n",
    "print(fold_results.groupby('model_name')['fold'].count())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Bảng tổng hợp (mean ± std qua các fold)\n",
    "\n",
    "Sắp xếp theo **macro-F1** giảm dần — đây là metric chính theo research design (macro-F1 coi trọng các label thiểu số như `threat`, `identity_hate` ngang với `toxic`)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "KEY_METRICS = [\"macro_f1\", \"micro_f1\", \"subset_accuracy\", \"hamming_loss\", \"micro_roc_auc\"]\n",
    "\n",
    "def format_mean_std(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:\n",
    "    \"\"\"Gộp (mean, std) thành một chuỗi 'mean ± std' dễ đọc cho từng metric.\"\"\"\n",
    "    formatted = pd.DataFrame(index=frame.index)\n",
    "    for metric in metrics:\n",
    "        if metric not in frame.columns.get_level_values(0):\n",
    "            continue\n",
    "        mean = frame[(metric, \"mean\")]\n",
    "        std = frame[(metric, \"std\")]\n",
    "        formatted[metric] = [f\"{m:.4f} ± {s:.4f}\" for m, s in zip(mean, std)]\n",
    "    return formatted\n",
    "\n",
    "summary_display = format_mean_std(summary, KEY_METRICS)\n",
    "summary_display = summary_display.reindex(\n",
    "    summary[(\"macro_f1\", \"mean\")].sort_values(ascending=False).index\n",
    ")\n",
    "summary_display"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Biểu đồ so sánh Macro-F1 giữa các model\n",
    "\n",
    "Cột càng cao càng tốt; thanh lỗi (error bar) là độ lệch chuẩn giữa các fold — thanh lỗi dài nghĩa là kết quả chưa ổn định qua các fold, nên cẩn thận khi kết luận."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "macro_f1_mean = summary[(\"macro_f1\", \"mean\")].sort_values(ascending=False)\n",
    "macro_f1_std = summary[(\"macro_f1\", \"std\")].reindex(macro_f1_mean.index)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(8, 5))\n",
    "x_positions = np.arange(len(macro_f1_mean))\n",
    "bars = ax.bar(x_positions, macro_f1_mean.values, yerr=macro_f1_std.values, capsize=6)\n",
    "ax.set_ylabel(\"Macro-F1\")\n",
    "ax.set_title(\"Macro-F1 trung bình theo model (± std qua các fold)\")\n",
    "ax.set_xticks(x_positions)\n",
    "ax.set_xticklabels(macro_f1_mean.index, rotation=20, ha=\"right\")\n",
    "for bar, value in zip(bars, macro_f1_mean.values):\n",
    "    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f\"{value:.3f}\",\n",
    "            ha=\"center\", va=\"bottom\")\n",
    "fig.tight_layout()\n",
    "fig.savefig(RESULTS_DIR / \"macro_f1_comparison.png\", dpi=150)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Biến thiên qua từng fold (boxplot)\n",
    "\n",
    "Nhìn độ phân tán thật của macro-F1 qua 5 fold, thay vì chỉ một con số trung bình."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model_order = macro_f1_mean.index.tolist()\n",
    "data_by_model = [\n",
    "    fold_results.loc[fold_results[\"model_name\"] == model, \"macro_f1\"].values\n",
    "    for model in model_order\n",
    "]\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(8, 5))\n",
    "ax.boxplot(data_by_model, tick_labels=model_order)\n",
    "ax.set_ylabel(\"Macro-F1\")\n",
    "ax.set_title(\"Phân bố Macro-F1 qua các fold\")\n",
    "ax.set_xticks(range(1, len(model_order) + 1))\n",
    "ax.set_xticklabels(model_order, rotation=20, ha=\"right\")\n",
    "fig.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Bảng chi tiết theo từng fold\n",
    "\n",
    "Hữu ích khi cần trích số cụ thể vào báo cáo LaTeX, hoặc kiểm tra xem có fold nào bất thường (ví dụ RoBERTa hội tụ kém ở một fold cụ thể do dữ liệu fold đó ít mẫu nhãn hiếm)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pivot = fold_results.pivot(index=\"fold\", columns=\"model_name\", values=\"macro_f1\")\n",
    "pivot.style.format(\"{:.4f}\").background_gradient(axis=1, cmap=\"Greens\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Ghi chú / giới hạn\n",
    "\n",
    "- **Giờ xem được tiến độ giữa chừng**: `experiment.py` đã được cập nhật để ghi `cross_validation_results.csv` và `summary_results.csv` ngay sau **mỗi fold**, không chỉ khi toàn bộ chạy xong. Có thể chạy lại notebook này bất cứ lúc nào trong khi RoBERTa còn đang train ở terminal khác — số liệu sẽ phản ánh đúng những fold đã xong.\n",
    "- `evaluation.py` hiện chỉ tính metric **micro/macro tổng hợp**, chưa có breakdown theo từng nhãn (`threat`, `identity_hate`, ...). Muốn phân tích minority-label theo đúng phần *Evaluation Strategy* trong research design (PR-AUC/F1 riêng từng nhãn), cần bổ sung per-label metrics vào `evaluate_predictions` — hỏi mình nếu muốn làm phần đó."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
FILE_EOF

cat > notebooks/train_and_save_roberta_label_dependency.ipynb << 'FILE_EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Train 1 lần, Lưu model, Evaluate — RoBERTa Label-Dependency (Method 3)\n",
    "\n",
    "Khác với `python -m toxic_comments --include-transformers --folds 5` (train **5 lần**, mỗi fold một bản, chỉ để lấy mean±std macro-F1 so sánh giữa các method rồi vứt hết model) — notebook này:\n",
    "\n",
    "1. Train **đúng 1 lần** trên một train/test split cố định (dùng chung seed với bước CV đã chạy, nên test set ở đây trùng với fold 1 của lần chạy CV trước).\n",
    "2. **Lưu model** xuống đĩa (`models/roberta_label_dependency/`) để dùng lại sau này mà không phải train lại.\n",
    "3. Load lại từ đĩa và evaluate trên test set, để tự kiểm chứng save/load hoạt động đúng.\n",
    "\n",
    "Chạy 1 lần vẫn tốn thời gian tương đương ~1 fold của bước CV trước (nên nếu bước CV mất ~X giờ cho 5 fold, bước train ở đây chỉ mất khoảng X/5)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "from pathlib import Path\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == \"notebooks\" else Path.cwd()\n",
    "SRC_DIR = PROJECT_ROOT / \"src\"\n",
    "if str(SRC_DIR) not in sys.path:\n",
    "    sys.path.insert(0, str(SRC_DIR))\n",
    "\n",
    "from toxic_comments.config import HEAVY_TEXT_COLUMN, LABEL_COLUMNS, MODELS_DIR, RAW_DATA_DIR\n",
    "from toxic_comments.cleaning import process_cleaning\n",
    "from toxic_comments.repositories import CsvFileDatasetRepository, validate_training_data\n",
    "from toxic_comments.folds import make_kfold_splits\n",
    "from toxic_comments.evaluation import evaluate_predictions\n",
    "from toxic_comments.models.roberta_label_dependency import build_roberta_label_dependency\n",
    "\n",
    "print(\"PROJECT_ROOT =\", PROJECT_ROOT)\n",
    "print(\"SRC_DIR exists:\", SRC_DIR.exists())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 0. Kiểm tra GPU (làm trước tiên)\n",
    "\n",
    "Nếu dòng dưới in ra `KHÔNG CÓ GPU`, train sẽ rất chậm (xem ước tính giờ ở phần trước trong chat) — nên dừng lại và chuyển sang Colab (Runtime > Change runtime type > GPU) ngay, trước khi tốn thời gian chạy các bước load/clean data và train bên dưới."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "\n",
    "DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "\n",
    "if DEVICE == \"cuda\":\n",
    "    print(f\"✅ Dùng GPU: {torch.cuda.get_device_name(0)}\")\n",
    "else:\n",
    "    print(\"⚠️  KHÔNG CÓ GPU — đang chạy CPU, sẽ rất chậm (xem ước tính thời gian ở phần chat).\")\n",
    "\n",
    "DEVICE"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load + clean dữ liệu (giống hệt pipeline CLI đang dùng)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "repository = CsvFileDatasetRepository(RAW_DATA_DIR / \"train.csv\")\n",
    "data = process_cleaning(\n",
    "    validate_training_data(repository.load()),\n",
    "    is_train=True,\n",
    "    verbose=False,\n",
    ")\n",
    "data = data[data[\"is_empty_heavy\"] == 0].reset_index(drop=True)\n",
    "print(\"Số dòng sau khi clean:\", len(data))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Train/test split — dùng chung seed với bước CV (fold 1 = test set ở đây)\n",
    "\n",
    "`make_kfold_splits(..., random_state=42)` chính là hàm `experiment.py` đang dùng, nên `splits[0]` (train_index, test_index) tương ứng đúng fold 1 của lần chạy CV trước — con số ở notebook này so sánh được trực tiếp với dòng `fold 1` trong `results/cross_validation_results.csv` của model `roberta_label_dependency`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "splits = make_kfold_splits(data, n_splits=5, random_state=42, text_column=HEAVY_TEXT_COLUMN)\n",
    "train_index, test_index = splits[0]\n",
    "\n",
    "X_train = data[HEAVY_TEXT_COLUMN].iloc[train_index]\n",
    "y_train = data[LABEL_COLUMNS].iloc[train_index].to_numpy()\n",
    "X_test = data[HEAVY_TEXT_COLUMN].iloc[test_index]\n",
    "y_test = data[LABEL_COLUMNS].iloc[test_index].to_numpy()\n",
    "\n",
    "print(\"Train:\", X_train.shape, \" Test:\", X_test.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Train (1 lần)\n",
    "\n",
    "Đổi tham số ở đây nếu cần (vd. `num_epochs`, `batch_size`) — mặc định khớp với những gì `registry.py` đang dùng cho `roberta_label_dependency`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import time\n",
    "\n",
    "model = build_roberta_label_dependency(device=DEVICE)  # ép dùng đúng device đã kiểm tra ở trên\n",
    "\n",
    "start = time.perf_counter()\n",
    "model.fit(X_train, y_train)\n",
    "print(f\"Train xong sau {time.perf_counter() - start:.1f}s\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Lưu model xuống đĩa\n",
    "\n",
    "Lưu vào `models/roberta_label_dependency/` (đã có sẵn trong `config.MODELS_DIR`) — gồm trọng số model (`model_state_dict.pt`, có luôn ma trận co-occurrence vì nó là registered buffer), tokenizer, và `params.json` (hyperparameter để dựng lại đúng kiến trúc lúc load)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "save_path = MODELS_DIR / \"roberta_label_dependency\"\n",
    "model.save(save_path)\n",
    "print(\"Đã lưu tại:\", save_path)\n",
    "print(\"Các file:\", sorted(p.name for p in save_path.iterdir()))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Load lại từ đĩa (kiểm chứng save/load đúng) + Evaluate trên test set\n",
    "\n",
    "Không dùng lại biến `model` ở bước 3 — cố tình load lại từ đĩa bằng `RobertaLabelDependencyClassifier.load(...)` để chắc chắn phần lưu/load hoạt động thật, không phải chỉ đang evaluate model còn nằm trong RAM."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from toxic_comments.models.roberta_label_dependency import RobertaLabelDependencyClassifier\n",
    "\n",
    "loaded_model = RobertaLabelDependencyClassifier.load(save_path)\n",
    "\n",
    "y_pred = loaded_model.predict(X_test)\n",
    "y_score = loaded_model.predict_proba(X_test)\n",
    "\n",
    "metrics = evaluate_predictions(y_test, y_pred, y_score)\n",
    "pd.Series(metrics, name=\"roberta_label_dependency (fold 1 test set)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. So sánh nhanh với số liệu CV đã có (nếu có)\n",
    "\n",
    "Đối chiếu với dòng `fold=1, model_name=roberta_label_dependency` trong `results/cross_validation_results.csv` — hai con số macro_f1 nên gần bằng nhau (không tuyệt đối giống 100% vì thứ tự batch/shuffle trong training có thể khác chút, nhưng cùng train/test split nên phải cùng thang điểm)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "cv_results_path = PROJECT_ROOT / \"results\" / \"cross_validation_results.csv\"\n",
    "if cv_results_path.exists():\n",
    "    cv_results = pd.read_csv(cv_results_path)\n",
    "    reference_row = cv_results[\n",
    "        (cv_results[\"model_name\"] == \"roberta_label_dependency\") & (cv_results[\"fold\"] == 1)\n",
    "    ]\n",
    "    if not reference_row.empty:\n",
    "        print(\"Fold 1 trong lần chạy CV trước:\")\n",
    "        print(reference_row[[\"macro_f1\", \"micro_f1\"]].to_string(index=False))\n",
    "    else:\n",
    "        print(\"Chưa thấy fold 1 của roberta_label_dependency trong CSV — có thể CV chưa chạy xong tới đó.\")\n",
    "else:\n",
    "    print(f\"Chưa có {cv_results_path} — bỏ qua bước so sánh này.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Dùng lại model đã lưu ở lần chạy sau (không cần train lại)\n",
    "\n",
    "```python\n",
    "from toxic_comments.models.roberta_label_dependency import RobertaLabelDependencyClassifier\n",
    "from toxic_comments.config import MODELS_DIR\n",
    "\n",
    "model = RobertaLabelDependencyClassifier.load(MODELS_DIR / \"roberta_label_dependency\")\n",
    "model.predict_proba([\"some comment text here\"])\n",
    "```"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
FILE_EOF

echo "DONE — đã ghi đè xong toàn bộ file + 2 notebook."