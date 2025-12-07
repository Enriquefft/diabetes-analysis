"""
Machine Learning Model Evaluation Utilities.

This module provides utilities for evaluating regression and classification models.
It supports multiple metrics, learning curves, and advanced visualizations for model comparison.

Included Functions:
- rmse
- accuracy_score
- classification_report_df
- model_performance_df
- compare_models
- select_best_model
- plot_learning_curve
- plot_model_comparison
- plot_confusion_matrix
- plot_roc_curve

"""

from collections.abc import Sequence
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# float

ModelType = Literal["regression", "classification"]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE) for regression."""
    return cast("float", np.sqrt(mean_squared_error(y_true, y_pred)))


def accuracy(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float | np.float16 | np.float32 | np.float64:
    """Compute Accuracy for classification."""
    return accuracy_score(y_true, y_pred)


def classification_report_df(
    y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str] | None = None
) -> pd.DataFrame:
    """
    Generate a classification report as a DataFrame.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.
    target_names : list of str, optional
        Names for target classes.

    Returns
    -------
    pd.DataFrame
        Classification report as a DataFrame.

    """
    from sklearn.metrics import classification_report

    report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True
    )
    return pd.DataFrame(report).transpose()


def model_performance_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_type: ModelType,
    model_name: str = "Model",
) -> pd.DataFrame:
    """
    Convert model performance into a Pandas DataFrame for easy visualization.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.
    model_type : str
        Either 'regression' or 'classification'.
    model_name : str
        Name of the model being evaluated.

    Returns
    -------
    pd.DataFrame
        DataFrame with model performance metrics.

    """
    if model_type == "regression":
        metrics = {"RMSE": rmse(y_true, y_pred)}
    else:
        metrics = {
            "Accuracy": accuracy(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted"),
            "Recall": recall_score(y_true, y_pred, average="weighted"),
            "F1": f1_score(y_true, y_pred, average="weighted"),
            "ROC AUC": roc_auc_score(y_true, y_pred, multi_class="ovr"),
        }

    return pd.DataFrame(metrics, index=[model_name])


def compare_models(
    results: dict[str, pd.DataFrame], output_path: str | None = None
) -> pd.DataFrame:
    """
    Generate a performance table for multiple models. Optionally save to CSV.

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
        Mapping from model names to performance DataFrames.
    output_path : str, optional
        File path to save the comparison table as CSV.

    Returns
    -------
    pd.DataFrame
        Combined comparison table.

    """
    combined_df = pd.concat(results.values(), axis=0)
    if output_path:
        combined_df.to_csv(output_path, index=True)
        print(f"Saved model comparison table to {output_path}")
    return combined_df


def select_best_model(
    results: dict[str, pd.DataFrame], metric: str = "F1"
) -> tuple[str, float]:
    """
    Select the best performing model based on a specified metric.

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
        Mapping from model names to performance DataFrames.
    metric : str
        Metric to use for selecting the best model.

    Returns
    -------
    tuple[str, float]
        Best model name and its metric value.

    """
    best_model, best_score = max(
        ((name, df[metric].to_numpy()[0]) for name, df in results.items()),
        key=lambda kv: kv[1],
    )
    return best_model, best_score


def plot_learning_curve(
    train_sizes: list[int],
    train_scores: list[float],
    valid_scores: list[float],
    title: str = "Learning Curve",
) -> Figure:
    """Plot training and validation scores over different sample sizes."""
    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_scores, marker="o", label="Train Score")
    ax.plot(train_sizes, valid_scores, marker="o", label="Validation Score")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(visible=True)
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
    title: str = "Confusion Matrix",
) -> Figure:
    """Plot a confusion matrix for classification models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax, display_labels=labels, cmap="Blues"
    )
    ax.set_title(title)
    return fig


def plot_roc_curve(
    y_true: np.ndarray, y_pred_proba: np.ndarray, title: str = "ROC Curve"
) -> Figure:
    """Plot ROC curve for binary classification models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax)
    ax.set_title(title)
    return fig


def plot_2d_projection(
    x_2d: np.ndarray | Sequence[Sequence[float]],
    y: Sequence[Any],
    labels: Sequence[str] | None = None,
    title: str = "",
) -> Figure:
    """
    Create a 2D scatter plot of a dimensionality-reduced dataset.

    Parameters
    ----------
    x_2d : array-like of shape (n_samples, 2)
        The 2D coordinates of the samples.
    y : array-like of shape (n_samples,)
        Class labels or group identifiers for each sample.
    labels : sequence of str, optional
        Unique class names corresponding to sorted unique values in `y`.
        If provided, will be used to create an explicit legend. If not,
        legend entries will be derived from `y` values.
    title : str, default=""
        Title for the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib Figure object containing the scatter plot.

    Raises
    ------
    ValueError
        If `X_2d` is not two-dimensional with shape (n_samples, 2) or
        if the length of `X_2d` does not match the length of `y`.

    """
    # Convert X_2d to a NumPy array for consistency
    x_arr = np.asarray(x_2d)
    if x_arr.ndim != 2 or x_arr.shape[1] != 2:
        msg = (
            f"X_2d must be of shape (n_samples, 2); got array with shape {x_arr.shape}"
        )
        raise ValueError(msg)
    if x_arr.shape[0] != len(y):
        msg = f"Number of samples in X_2d ({x_arr.shape[0]}) must match length of y ({len(y)})"
        raise ValueError(msg)

    # Prepare unique classes
    unique_classes = np.unique(y)
    fig, ax = plt.subplots()
    for idx, cls in enumerate(unique_classes):
        mask = np.array(y) == cls
        ax.scatter(
            x_arr[mask, 0],
            x_arr[mask, 1],
            label=(labels[idx] if labels is not None else str(cls)),
            alpha=0.7,
        )

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(title="Classes")
    ax.grid(visible=True)
    fig.tight_layout()
    return fig
