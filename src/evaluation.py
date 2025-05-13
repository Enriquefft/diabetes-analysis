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

from typing import Literal, cast

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


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) ->float | np.float16 | np.float32 | np.float64 :
    """Compute Accuracy for classification."""
    return accuracy_score(y_true, y_pred)


def classification_report_df(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str] | None = None) -> pd.DataFrame:
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
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    return pd.DataFrame(report).transpose()


def model_performance_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_type: ModelType,
    model_name: str = "Model"
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
            "ROC AUC": roc_auc_score(y_true, y_pred, multi_class="ovr")
        }

    return pd.DataFrame(metrics, index=[model_name])


def compare_models(
    results: dict[str, pd.DataFrame],
    output_path: str | None = None
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


def select_best_model(results: dict[str, pd.DataFrame], metric: str = "F1") -> tuple[str, float]:
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
        key=lambda kv: kv[1]
    )
    return best_model, best_score


def plot_learning_curve(
    train_sizes: list[int],
    train_scores: list[float],
    valid_scores: list[float],
    title: str = "Learning Curve"
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
    title: str = "Confusion Matrix"
) -> Figure:
    """Plot a confusion matrix for classification models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, display_labels=labels, cmap="Blues")
    ax.set_title(title)
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    title: str = "ROC Curve"
) -> Figure:
    """Plot ROC curve for binary classification models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax)
    ax.set_title(title)
    return fig
