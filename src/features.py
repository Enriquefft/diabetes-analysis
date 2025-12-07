"""
Advanced Feature Engineering Utilities.

This module provides utilities for feature engineering, including polynomial expansion,
interaction terms, categorical encoding, dimensionality reduction, and feature selection.

Included Functions:
- combine_features
- drop_combine_features
- add_polynomial_features
- encode_categorical
- log_transform
- drop_low_variance
- select_important_features
- reduce_dimensionality
- balance_classes

"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    chi2,
    mutual_info_classif,
)
from sklearn.preprocessing import PolynomialFeatures

type Operation = Literal["product", "sum", "difference", "ratio"]


def combine_features(
    df: pd.DataFrame,
    combos: list[tuple[str, str]],
    operations: list[Operation],
) -> pd.DataFrame:
    """
    Create new features from pairs of existing columns without dropping the originals.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    combos : list of tuple(str, str)
        Pairs of column names to combine.
    operations : list of str, optional
        Which operations to apply: any of 'product', 'sum', 'difference', 'ratio'.

    Returns
    -------
    pd.DataFrame
        DataFrame with combined features.

    """
    df_copy = df.copy()

    for (col1, col2), operation in zip(combos, operations, strict=True):
        if operation == "product":
            df_copy[f"{col1}_x_{col2}"] = df_copy[col1] * df_copy[col2]
        elif operation == "sum":
            df_copy[f"{col1}_plus_{col2}"] = df_copy[col1] + df_copy[col2]
        elif operation == "difference":
            df_copy[f"{col1}_minus_{col2}"] = df_copy[col1] - df_copy[col2]
        elif operation == "ratio":
            df_copy[f"{col1}_div_{col2}"] = df_copy[col1] / df_copy[col2].replace(
                {0: np.nan}
            )
    return df_copy


def drop_combine_features(
    df: pd.DataFrame,
    combos: list[tuple[str, str]],
    operations: list[Operation],
) -> pd.DataFrame:
    """
    Create new features from pairs of existing columns and drop the originals.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    combos : list of tuple(str, str)
        Pairs of column names to combine.
    operations : list of str, optional
        Which operations to apply: any of 'product', 'sum', 'difference', 'ratio'.

    Returns
    -------
    pd.DataFrame
        DataFrame with combined features, original columns dropped.

    """
    combined_df = combine_features(df, combos, operations)
    cols_to_drop = {c for pair in combos for c in pair}
    return combined_df.drop(columns=list(cols_to_drop))


def add_polynomial_features(
    *,
    df: pd.DataFrame,
    degree: int,
    include_bias: bool = False,
    interaction_only: bool = False,
    drop_original: bool = False,
    features: list[str] | None = None,
    model: PolynomialFeatures | None = None,
) -> tuple[pd.DataFrame, PolynomialFeatures]:
    """
    Expand numeric features to polynomial features up to the given degree.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to expand.
    degree : int
        Maximum degree of polynomial features.
    include_bias : bool
        If True, include a bias (constant 1) term.
    interaction_only : bool
        If True, only include interaction features (no pure powers).
    drop_original : bool
        If True, drop the original input columns.
    features : list of str, optional
        Specific columns to expand; otherwise all numeric columns.
    model : PolynomialFeatures, optional
        Pre-fitted PolynomialFeatures instance; if None, a new one is created.


    Returns
    -------
    pd.DataFrame
        Expanded DataFrame.
    PolynomialFeatures
        Fitted PolynomialFeatures instance.

    """
    df_copy = df.copy()
    cols = features or df_copy.select_dtypes(include=[np.number]).columns.tolist()

    if model is None:
        model = PolynomialFeatures(
            degree=degree, include_bias=include_bias, interaction_only=interaction_only
        )
        arr = model.fit_transform(df_copy[cols])
    else:
        arr = model.transform(df_copy[cols])
    if isinstance(arr, sp.spmatrix):
        arr = arr.todense()

    poly_df = pd.DataFrame(
        arr, columns=model.get_feature_names_out(cols), index=df_copy.index
    )

    if drop_original:
        df_copy = df_copy.drop(columns=cols)

    return pd.concat([df_copy, poly_df], axis=1), model


def encode_categorical(
    *,
    df: pd.DataFrame,
    columns: list[str] | None = None,
    drop_first: bool = True,
    dtype: str = "uint8",
) -> pd.DataFrame:
    """
    One-hot encode categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to encode.
    columns : list of str, optional
        Columns to encode; if None, all object or category dtype columns.
    drop_first : bool
        If True, drop the first level of each category.
    dtype : str
        Dtype for the dummy variables.

    Returns
    -------
    pd.DataFrame
        DataFrame with one-hot encoded columns.

    """
    df_copy = df.copy()
    cols = (
        columns
        or df_copy.select_dtypes(include=["object", "category"]).columns.tolist()
    )
    return pd.get_dummies(df_copy, columns=cols, drop_first=drop_first, dtype=dtype)


def drop_low_variance(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    Remove numeric features with variance <= threshold.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to filter.
    threshold : float
        Variance threshold; features with variance <= this are dropped.

    Returns
    -------
    pd.DataFrame
        DataFrame with low variance features removed.

    """
    df_copy = df.copy()
    selector = VarianceThreshold(threshold=threshold)
    return df_copy.loc[:, selector.fit(df_copy).get_support()]


def select_important_features(
    df: pd.DataFrame,
    y: pd.Series,
    method: Literal["chi2", "mutual_info"] = "mutual_info",
    k: int = 10,
) -> pd.DataFrame:
    """
    Select the top k most important features.

    Parameters
    ----------
    df : pd.DataFrame
        Input feature DataFrame.
    y : pd.Series
        Target variable.
    method : str
        Feature selection method, either 'chi2' or 'mutual_info'.
    k : int
        Number of top features to select.

    Returns
    -------
    pd.DataFrame
        DataFrame with top k features.

    """
    if method == "chi2":
        selector = SelectKBest(chi2, k=k)
    else:
        selector = SelectKBest(mutual_info_classif, k=k)

    selected = selector.fit_transform(df, y)
    selected_columns = df.columns[selector.get_support()]
    return pd.DataFrame(selected, columns=selected_columns, index=df.index)


ArrayLike = np.ndarray | pd.DataFrame


def fit_pca(x: ArrayLike, n_components: int = 2, **kwargs) -> tuple[np.ndarray, PCA]:  # noqa: ANN003
    """
    Fit PCA on the input data and return the reduced representation.

    Parameters
    ----------
    x : np.ndarray or pd.DataFrame
        Feature matrix to reduce. Rows are samples and columns are features.
    n_components : int, default=2
        Number of principal components to compute.
    **kwargs
        Additional keyword arguments for sklearn.decomposition.PCA.

    Returns
    -------
    X_reduced : np.ndarray
        The input data projected onto the top principal components.
    pca_model : PCA
        The fitted PCA estimator.

    Raises
    ------
    ValueError
        If n_components is not a positive integer less than or equal to the number of features.

    """
    # Extract raw array
    data = x.to_numpy() if isinstance(x, pd.DataFrame) else x

    # Validate n_components
    n_features = data.shape[1]
    if not (isinstance(n_components, int) and 1 <= n_components <= n_features):
        msg = f"n_components must be an integer between 1 and {n_features}, got {n_components}."
        raise ValueError(msg)

    # Fit PCA
    pca = PCA(n_components=n_components, **kwargs)
    x_reduced = pca.fit_transform(data)
    return x_reduced, pca


def fit_lda(
    x: ArrayLike,
    y: Sequence,
    n_components: int = 2,
    **kwargs,  # noqa: ANN003
) -> tuple[np.ndarray, LinearDiscriminantAnalysis]:
    """
    Fit LDA on the input data and return the reduced representation.

    Parameters
    ----------
    x : np.ndarray or pd.DataFrame
        Feature matrix to reduce. Rows are samples and columns are features.
    y : Sequence
        Target labels corresponding to each sample in X.
    n_components : int, default=2
        Number of linear discriminants to compute. Must be less than number of classes.
    **kwargs
        Additional keyword arguments for sklearn.discriminant_analysis.LinearDiscriminantAnalysis.

    Returns
    -------
    X_reduced : np.ndarray
        The input data projected onto the linear discriminants.
    lda_model : LinearDiscriminantAnalysis
        The fitted LDA estimator.

    Raises
    ------
    ValueError
        If y is not the same length as X samples or if n_components is invalid.

    """
    # Extract raw array
    data = x.to_numpy() if isinstance(x, pd.DataFrame) else x

    y_arr = np.asarray(y)
    if data.shape[0] != y_arr.shape[0]:
        msg = f"Number of samples in X ({data.shape[0]}) and y ({y_arr.shape[0]}) must match."
        raise ValueError(msg)

    # Validate n_components
    n_classes = np.unique(y_arr).size
    max_components = min(n_classes - 1, data.shape[1])
    if not (isinstance(n_components, int) and 1 <= n_components <= max_components):
        msg = f"n_components must be an integer between 1 and {max_components} (min(n_classes-1, n_features))."
        raise ValueError(msg)

    # Fit LDA
    lda = LinearDiscriminantAnalysis(n_components=n_components, **kwargs)
    x_reduced = lda.fit_transform(data, y_arr)
    return x_reduced, lda


def reduce_dimensionality(
    x: ArrayLike,
    y: Sequence | None = None,
    method: Literal["pca", "lda"] = "pca",
    n_components: int = 2,
    **kwargs,  # noqa: ANN003
) -> tuple[np.ndarray, PCA | LinearDiscriminantAnalysis]:
    """
    Reduce dimensionality of the dataset using PCA or LDA.

    Parameters
    ----------
    x : np.ndarray or pd.DataFrame
        Feature matrix to reduce.
    y : Sequence, optional
        Target labels required for LDA. Ignored if method="pca".
    method : {"pca", "lda"}, default="pca"
        Reduction technique to apply.
    n_components : int, default=2
        Number of components or discriminants to compute.
    **kwargs
        Additional keyword arguments passed to fit_pca or fit_lda.

    Returns
    -------
    X_reduced : np.ndarray
        The data projected into the reduced space.
    model : PCA or LinearDiscriminantAnalysis
        The fitted dimensionality-reduction model.

    Raises
    ------
    ValueError
        If method is not recognized or if y is required but not provided.

    """
    if method == "pca":
        return fit_pca(x, n_components=n_components, **kwargs)
    if y is None:
        msg = "Target labels y must be provided for LDA."
        raise ValueError(msg)
    return fit_lda(x, y, n_components=n_components, **kwargs)
