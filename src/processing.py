"""
Advanced Data Processing Utilities.

This module provides utilities for data preprocessing, including handling missing values,
feature scaling, datetime encoding, data balancing, and feature selection.

Included Functions:
- encode_datetime_features
- encode_wind_direction
- load_data
- fit_imputer
- apply_imputer
- normalize_fit
- normalize_apply
- evaluate_normalization_methods
- balance_classes
- select_features_by_correlation
- select_important_features
"""

from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import pearsonr, spearmanr
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

if TYPE_CHECKING:
    from sklearn.utils import Bunch

NormalizationMethod = Literal["zscore", "minmax", "robust", "power", "quantile"]
ImputationStrategy = Literal["mean", "median", "most_frequent", "constant"]


def encode_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime columns into numeric features."""
    df_copy = df.copy()

    if all(col in df_copy.columns for col in ["year", "month", "day", "hour"]):
        df_copy["year_scaled"] = (df_copy["year"] - df_copy["year"].min()) / (
            df_copy["year"].max() - df_copy["year"].min()
        )

        for col, period in [("month", 12), ("day", 31), ("hour", 24)]:
            radians = 2 * np.pi * df_copy[col] / period
            df_copy[f"{col}_sin"] = np.sin(radians)
            df_copy[f"{col}_cos"] = np.cos(radians)

        df_copy = df_copy.drop(columns=["year", "month", "day", "hour"])
    else:
        error_msg = "The DataFrame must contain 'year', 'month', 'day', and 'hour' columns for datetime encoding."
        raise ValueError(error_msg)

    return df_copy


def encode_wind_direction(df: pd.DataFrame, column: str = "wd") -> pd.DataFrame:
    """Convert a compass-direction column into numeric features (sin/cos)."""
    df_copy = df.copy()

    if column not in df_copy.columns:
        error_msg = f"The specified column '{column}' is not present in the DataFrame."
        raise ValueError(error_msg)

    directions = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]
    deg_map = {d: i * 22.5 for i, d in enumerate(directions)}

    df_copy["WD_deg"] = df_copy[column].map(deg_map)

    radians = np.deg2rad(df_copy["WD_deg"])
    df_copy["WD_sin"] = np.sin(radians)
    df_copy["WD_cos"] = np.cos(radians)

    return df_copy.drop(columns=[column, "WD_deg"])


def load_data(path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    path = Path(path)
    if not path.exists():
        error_msg = f"File not found: {path}"
        raise FileNotFoundError(error_msg)
    return pd.read_csv(path)


def fit_imputer(
    df: pd.DataFrame,
    strategy: ImputationStrategy = "mean",
    fill_value: float | None = None,
) -> tuple[pd.DataFrame, SimpleImputer]:
    """Fit an imputer to fill missing values in numeric columns."""
    df_copy = df.copy()
    num_cols = df_copy.select_dtypes(include=[np.number]).columns

    imputer_kwargs = {}
    if strategy == "constant":
        if fill_value is None:
            error_msg = "fill_value must be provided for constant strategy"
            raise ValueError(error_msg)
        imputer_kwargs["fill_value"] = fill_value

    imputer = SimpleImputer(strategy=strategy, **imputer_kwargs)
    df_copy[num_cols] = imputer.fit_transform(df_copy[num_cols])
    return df_copy, imputer


def normalize_fit(
    df: pd.DataFrame,
    method: NormalizationMethod = "zscore",
    columns: list[str] | None = None,
) -> tuple[
    pd.DataFrame,
    StandardScaler
    | MinMaxScaler
    | RobustScaler
    | PowerTransformer
    | QuantileTransformer,
]:
    """Fit a scaler and normalize specified columns."""
    df_copy = df.copy()
    cols = columns or df_copy.select_dtypes(include=[np.number]).columns.tolist()

    if method == "zscore":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    elif method == "power":
        scaler = PowerTransformer()
    else:
        scaler = QuantileTransformer(output_distribution="normal")

    df_copy[cols] = scaler.fit_transform(df_copy[cols])
    return df_copy, scaler


def balance_classes(
    x: pd.DataFrame,
    y: pd.Series,
    method: Literal["smote", "undersample", "smoteenn", "smotetomek"] = "smote",
) -> tuple[np.ndarray, np.ndarray]:
    """Balance the classes in a dataset."""
    if method == "smote":
        balancer = SMOTE()
    elif method == "undersample":
        balancer = RandomUnderSampler()
    elif method == "smoteenn":
        balancer = SMOTEENN()
    else:
        balancer = SMOTETomek()

    result = balancer.fit_resample(x, y)

    if isinstance(result, tuple) and len(result) == 3:
        x_res, y_res, _ = result
    else:
        x_res, y_res = result

    # Ensure the outputs are numpy arrays
    return np.asarray(x_res), np.asarray(y_res)


def select_features_by_correlation(
    df: pd.DataFrame,
    target: pd.Series,
    threshold: float = 0.1,
    method: Literal["spearman", "pearson"] = "spearman",
) -> pd.DataFrame:
    """
    Select features based on correlation with the target.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.
    target : pd.Series
        Target variable.
    threshold : float
        Minimum absolute correlation required.
    method : str, optional
        Correlation method: 'spearman' (default) or 'pearson'.

    Returns
    -------
    pd.DataFrame
        DataFrame with selected features.

    """
    # Calculate correlations
    correlations = {}
    for col in df.columns:
        if method == "spearman":
            corr_value = spearmanr(df[col], target).statistics  # type: ignore reportAttributeAccessIssue
        elif method == "pearson":
            corr_value = pearsonr(df[col], target)[
                0
            ]  # Pearson returns a tuple (correlation, p-value)

        correlations[col] = corr_value

    # Filter based on the threshold
    selected_features = [
        col for col, corr in correlations.items() if abs(corr) > threshold
    ]

    return df[selected_features]


def select_important_features(
    df: pd.DataFrame,
    y: pd.Series,
    k: int = 10,
    method: Literal["mutual_info", "chi2"] = "mutual_info",
) -> pd.DataFrame:
    """Select the top k most important features."""
    if method == "mutual_info":
        selector = SelectKBest(mutual_info_classif, k=k)
    else:
        selector = SelectKBest(chi2, k=k)

    x_new = selector.fit_transform(df, y)
    selected_columns = df.columns[selector.get_support()]
    return pd.DataFrame(x_new, columns=selected_columns, index=df.index)


def load_iris_dataset(as_frame: bool = True) -> tuple[pd.DataFrame, pd.Series]:  # noqa: FBT001, FBT002
    """
    Carga el Iris dataset y devuelve (X, y).

    Si as_frame=True usa data.frame para incluir 'target';
    si no, construye manualmente X e y a partir de data.data y data.target.
    """
    data = cast("Bunch", load_iris(as_frame=as_frame))

    if as_frame and hasattr(data, "frame"):
        # data.frame tiene tanto las 4 features como la columna "target"
        data_frame = data.frame.copy()
        x = data_frame[data.feature_names]
        y = data_frame["target"].rename("target")
    else:
        x = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")

    return x, y
