"""
Robust Machine Learning Modeling Utilities.

This module provides utilities for both regression and classification tasks.
It supports a wide range of models, including both basic and advanced algorithms.
Designed to be part of a robust ML library for academic and production use.

Models Included:
- Regression: Linear, Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting
- Classification: KNN, Logistic Regression, SVM, DecisionTree, RandomForest, XGBoost, LightGBM

Main Features:
- Model instantiation with hyperparameter support
- Training and prediction
- Hyperparameter tuning with GridSearchCV
- Balanced training for imbalanced datasets

"""

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from numpy.typing import ArrayLike
from scipy.sparse import spmatrix
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

# Available models
RegressionModel = Literal[
    "linear",
    "ridge",
    "lasso",
    "elasticnet",
    "random_forest_regressor",
    "gradient_boosting",
    "xgboost_regressor",
    "lightgbm_regressor",
]

ClassificationModel = Literal[
    "knn",
    "logistic_regression",
    "svm",
    "decision_tree",
    "random_forest_classifier",
    "xgboost_classifier",
    "lightgbm_classifier",
]

ModelType = RegressionModel | ClassificationModel

type Model = (
    LinearRegression
    | Ridge
    | Lasso
    | ElasticNet
    | RandomForestRegressor
    | GradientBoostingRegressor
    | XGBRegressor
    | LGBMRegressor
    | KNeighborsClassifier
    | LogisticRegression
    | SVC
    | DecisionTreeClassifier
    | RandomForestClassifier
    | XGBClassifier
    | LGBMClassifier
)


def get_model(  # noqa: C901, PLR0911, PLR0912
    model_type: ModelType, model_params: dict[str, Any] | None = None
) -> Model:
    """
    Instantiate a machine learning model based on the model_type.

    Parameters
    ----------
    model_type : ModelType
        The type of model to instantiate.
    model_params : dict[str, Any], optional
        Hyperparameters to pass to the model.

    Returns
    -------
    Model
        An instantiated model object.

    Examples
    --------
    >>> get_model("linear")
    LinearRegression()

    >>> get_model("knn", {"n_neighbors": 5})
    KNeighborsClassifier(n_neighbors=5)

    """
    params = model_params or {}

    # Regression Models
    if model_type == "linear":
        return LinearRegression(**params)
    if model_type == "ridge":
        return Ridge(**params)
    if model_type == "lasso":
        return Lasso(**params)
    if model_type == "elasticnet":
        return ElasticNet(**params)
    if model_type == "random_forest_regressor":
        return RandomForestRegressor(**params)
    if model_type == "gradient_boosting":
        return GradientBoostingRegressor(**params)
    if model_type == "xgboost_regressor":
        return XGBRegressor(**params)
    if model_type == "lightgbm_regressor":
        return LGBMRegressor(**params)

    # Classification Models
    if model_type == "knn":
        return KNeighborsClassifier(**params)
    if model_type == "logistic_regression":
        return LogisticRegression(**params)
    if model_type == "svm":
        return SVC(**params)
    if model_type == "decision_tree":
        return DecisionTreeClassifier(**params)
    if model_type == "random_forest_classifier":
        return RandomForestClassifier(**params)
    if model_type == "xgboost_classifier":
        return XGBClassifier(**params)
    return LGBMClassifier(**params)


def train_model(
    x: np.ndarray,
    y: np.ndarray | ArrayLike,
    model_type: ModelType,
    model_params: dict[str, Any] | None = None,
) -> Model:
    """Train the specified model on X, y and return the fitted model."""
    model = get_model(model_type, model_params)
    model.fit(x, y)
    return model


def predict(
    model: Model, x: np.ndarray | spmatrix
) -> np.ndarray | ArrayLike | spmatrix | list[spmatrix]:
    """Generate predictions for X using the fitted model."""
    return model.predict(x)


def tune_model(
    x: np.ndarray,
    y: np.ndarray,
    model_type: ModelType,
    param_grid: dict[str, Any],
    cv: int = 5,
    scoring: str = "accuracy",
    n_jobs: int = -1,
) -> GridSearchCV:
    """
    Perform a grid search over param_grid for the specified model_type.

    Returns the fitted GridSearchCV instance.
    """
    base_model: BaseEstimator = cast("BaseEstimator", get_model(model_type))
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    grid_search.fit(x, y)
    return grid_search
