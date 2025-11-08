"""Metrics for evaluating CRISPR design model predictions."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        prefix: Optional prefix for metric names

    Returns:
        Dictionary of metric names to values
    """
    metrics = {}

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)

    metrics[f"{prefix}mae"] = mae
    metrics[f"{prefix}mse"] = mse
    metrics[f"{prefix}rmse"] = rmse
    metrics[f"{prefix}r2"] = r2
    metrics[f"{prefix}pearson_r"] = pearson_r
    metrics[f"{prefix}pearson_p"] = pearson_p
    metrics[f"{prefix}spearman_r"] = spearman_r
    metrics[f"{prefix}spearman_p"] = spearman_p

    median_ae = np.median(np.abs(y_true - y_pred))
    metrics[f"{prefix}median_ae"] = median_ae

    return metrics


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth binary labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
        prefix: Optional prefix for metric names

    Returns:
        Dictionary of metric names to values
    """
    metrics = {}

    y_pred = (y_pred_proba >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    try:
        auroc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        auroc = np.nan

    try:
        auprc = average_precision_score(y_true, y_pred_proba)
    except ValueError:
        auprc = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    metrics[f"{prefix}accuracy"] = accuracy
    metrics[f"{prefix}f1"] = f1
    metrics[f"{prefix}auroc"] = auroc
    metrics[f"{prefix}auprc"] = auprc
    metrics[f"{prefix}sensitivity"] = sensitivity
    metrics[f"{prefix}specificity"] = specificity
    metrics[f"{prefix}precision"] = precision
    metrics[f"{prefix}npv"] = npv
    metrics[f"{prefix}tp"] = int(tp)
    metrics[f"{prefix}fp"] = int(fp)
    metrics[f"{prefix}tn"] = int(tn)
    metrics[f"{prefix}fn"] = int(fn)

    return metrics


def compute_roc_curve_data(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute ROC curve data for plotting.

    Args:
        y_true: Ground truth binary labels
        y_pred_proba: Predicted probabilities

    Returns:
        Dictionary with fpr, tpr, thresholds, and auc
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auroc = auc(fpr, tpr)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auroc": auroc,
    }


def compute_pr_curve_data(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute precision-recall curve data for plotting.

    Args:
        y_true: Ground truth binary labels
        y_pred_proba: Predicted probabilities

    Returns:
        Dictionary with precision, recall, thresholds, and auprc
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    auprc = auc(recall, precision)

    return {
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
        "auprc": auprc,
    }


def stratified_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stratification_labels: np.ndarray,
    problem_type: str = "regression",
    y_pred_proba: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate metrics stratified by a grouping variable.

    Args:
        y_true: Ground truth values
        y_pred: Predictions (class labels for classification)
        stratification_labels: Labels defining strata
        problem_type: "regression" or "classification"
        y_pred_proba: Predicted probabilities (required for classification)

    Returns:
        Dictionary mapping stratum labels to metric dictionaries
    """
    results = {}

    unique_labels = np.unique(stratification_labels)

    for label in unique_labels:
        mask = stratification_labels == label
        y_true_stratum = y_true[mask]
        y_pred_stratum = y_pred[mask]

        if len(y_true_stratum) == 0:
            continue

        if problem_type == "regression":
            metrics = compute_regression_metrics(
                y_true_stratum,
                y_pred_stratum,
                prefix=f"{label}_",
            )
        elif problem_type == "classification":
            if y_pred_proba is None:
                raise ValueError("y_pred_proba required for classification")
            y_pred_proba_stratum = y_pred_proba[mask]
            metrics = compute_classification_metrics(
                y_true_stratum,
                y_pred_proba_stratum,
                prefix=f"{label}_",
            )
        else:
            raise ValueError(f"Unknown problem_type: {problem_type}")

        results[str(label)] = metrics

    return results


def top_k_accuracy(
    y_true: np.ndarray,
    y_pred_scores: np.ndarray,
    k: int = 5,
) -> float:
    """
    Compute top-k accuracy for ranking problems.

    Args:
        y_true: Ground truth labels
        y_pred_scores: Predicted scores
        k: Number of top predictions to consider

    Returns:
        Top-k accuracy
    """
    n_samples = len(y_true)
    correct = 0

    for i in range(n_samples):
        top_k_indices = np.argsort(y_pred_scores[i])[-k:]
        if y_true[i] in top_k_indices:
            correct += 1

    return correct / n_samples
