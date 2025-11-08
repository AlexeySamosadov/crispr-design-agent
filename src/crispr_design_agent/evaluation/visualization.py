"""Visualization utilities for model evaluation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure


def plot_regression_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Regression Results",
    figsize: Tuple[int, int] = (12, 4),
) -> Figure:
    """
    Create a comprehensive visualization of regression results.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].scatter(y_true, y_pred, alpha=0.5, s=10)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
    axes[0].set_xlabel("True Values")
    axes[0].set_ylabel("Predicted Values")
    axes[0].set_title(f"{title} - Scatter")
    axes[0].grid(True, alpha=0.3)

    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[1].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[1].set_xlabel("Predicted Values")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residual Plot")
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    axes[2].axvline(x=0, color="r", linestyle="--", lw=2)
    axes[2].set_xlabel("Residuals")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Residual Distribution")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auroc: float,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (6, 6),
) -> Figure:
    """
    Plot ROC curve.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auroc: Area under ROC curve
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, lw=2, label=f"ROC Curve (AUC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    auprc: float,
    title: str = "Precision-Recall Curve",
    figsize: Tuple[int, int] = (6, 6),
) -> Figure:
    """
    Plot precision-recall curve.

    Args:
        precision: Precision values
        recall: Recall values
        auprc: Area under PR curve
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(recall, precision, lw=2, label=f"PR Curve (AUC = {auprc:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (6, 6),
    normalize: bool = False,
) -> Figure:
    """
    Plot confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names of classes
        title: Plot title
        figsize: Figure size
        normalize: Whether to normalize counts

    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names or ["0", "1"],
        yticklabels=class_names or ["0", "1"],
        ax=ax,
    )

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_names: Optional[List[str]] = None,
    title: str = "Metrics Comparison",
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """
    Plot comparison of metrics across different models or conditions.

    Args:
        metrics_dict: Dictionary mapping model names to metric dictionaries
        metric_names: List of metric names to plot (if None, use all)
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if metric_names is None:
        first_key = next(iter(metrics_dict.keys()))
        metric_names = list(metrics_dict[first_key].keys())

    n_metrics = len(metric_names)
    model_names = list(metrics_dict.keys())

    x = np.arange(n_metrics)
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=figsize)

    for i, model_name in enumerate(model_names):
        values = [metrics_dict[model_name].get(metric, 0) for metric in metric_names]
        ax.bar(x + i * width, values, width, label=model_name, alpha=0.8)

    ax.set_xlabel("Metrics")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    figsize: Tuple[int, int] = (6, 6),
) -> Figure:
    """
    Plot calibration curve for probability predictions.

    Args:
        y_true: Ground truth binary labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for calibration
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("True Probability")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (8, 6),
    top_k: Optional[int] = 20,
) -> Figure:
    """
    Plot feature importance scores.

    Args:
        feature_names: Names of features
        importances: Importance scores
        title: Plot title
        figsize: Figure size
        top_k: Show only top k features (None for all)

    Returns:
        Matplotlib figure
    """
    indices = np.argsort(importances)[::-1]

    if top_k is not None:
        indices = indices[:top_k]

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(range(len(indices)), importances[indices], alpha=0.8)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    return fig


def plot_learning_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Optional[Dict[str, List[float]]] = None,
    val_metrics: Optional[Dict[str, List[float]]] = None,
    title: str = "Learning Curves",
    figsize: Tuple[int, int] = (12, 4),
) -> Figure:
    """
    Plot training and validation learning curves.

    Args:
        train_losses: Training loss values over epochs
        val_losses: Validation loss values over epochs
        train_metrics: Optional dict of training metrics
        val_metrics: Optional dict of validation metrics
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_plots = 1
    if train_metrics and val_metrics:
        n_plots = 1 + len(train_metrics)

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, label="Training Loss", marker="o")
    axes[0].plot(epochs, val_losses, label="Validation Loss", marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if train_metrics and val_metrics:
        for idx, metric_name in enumerate(train_metrics.keys(), start=1):
            if idx >= len(axes):
                break
            axes[idx].plot(epochs, train_metrics[metric_name], label=f"Train {metric_name}", marker="o")
            axes[idx].plot(epochs, val_metrics[metric_name], label=f"Val {metric_name}", marker="o")
            axes[idx].set_xlabel("Epoch")
            axes[idx].set_ylabel(metric_name)
            axes[idx].set_title(f"{metric_name} Curves")
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig
