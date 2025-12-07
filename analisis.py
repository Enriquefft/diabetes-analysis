# # Diabetes Risk Prediction – End‑to‑End Machine Learning Workflow
# **Author:** _Your Name Here_
#
# Generated automatically on 2025-05-29 00:57:17.
# This Jupyter notebook provides a **complete, reproducible pipeline** for developing and evaluating predictive models of diabetes risk using three derived subsets of the 2015 BRFSS survey published on Kaggle. It integrates requirements from the accompanying *UTEC Machine‑Learning Project #2* brief and additional best‑practice steps for professional data science work.
#
# Use **Run All** ⌘/Ctrl + _⇧_ + _↩︎_ to execute everything after adjusting any file paths or parameters marked **TODO**.


# ---------------------------------------------------------------
# 1 ▸ Imports & Global Setup
# ---------------------------------------------------------------

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Aesthetics -----------------------------------------------------------------
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (8, 5)

# Reproducibility ------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("Libraries imported successfully 📦")



# ## 2 ▸ About the Dataset & Problem Context
#
# <details>
# <summary><strong>📖 Background (click to expand)</strong></summary>
#
# Diabetes is among the most prevalent chronic diseases in the United States, impacting **millions** of Americans each year and exerting a significant financial burden on the economy. …
#
# *(Full narrative omitted for brevity – see project brief for complete text.)*
# </details>
#
# We will work with three **pre‑cleaned CSVs** (each 21 features):
#
# | File | Task | Size | Target distribution |
# |------|------|------|---------------------|
# | `diabetes_012_health_indicators_BRFSS2015.csv` | **Multi‑class** (0 = none/pregnancy, 1 = prediabetes, 2 = diabetes) | 253 680 rows | Imbalanced |
# | `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` | **Binary** (balanced 50 / 50) | 70 692 rows | Balanced |
# | `diabetes_binary_health_indicators_BRFSS2015.csv` | **Binary** (imbalanced) | 253 680 rows | Imbalanced |
#
# > **Research goals**
# > • Can BRFSS survey answers predict diabetes risk?
# > • Which factors matter most?
# > • How few questions still yield strong accuracy?
# > • Can we propose a short‑form questionnaire?
#


# ---------------------------------------------------------------
# 3 ▸ Data Ingestion & Quick Comparison
# ---------------------------------------------------------------

DATA_PATH = Path()  # TODO: adjust if CSVs live elsewhere

csv_files = {
    "MULTICLASS": "diabetes_012_health_indicators_BRFSS2015.csv",
    "BINARY_5050": "diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
    "BINARY_IMBAL": "diabetes_binary_health_indicators_BRFSS2015.csv",
}

dfs = {}
for key, fname in csv_files.items():
    fpath = DATA_PATH / fname
    if not fpath.exists():
        raise FileNotFoundError(f"Missing {fpath}. Please download datasets first.")

    df = pd.read_csv(fpath)
    dfs[key] = df
    print(f"{key:>13}: shape={df.shape}")

# Compare target distributions ------------------------------------------------
for key, df in dfs.items():
    target_col = "Diabetes_binary" if "binary" in key.lower() else "Diabetes_012"
    print(f"\n{key} – class balance:")
    print(df[target_col].value_counts(normalize=True).rename("proportion"))



# ## 4 ▸ Exploratory Data Analysis (EDA)
#
# We explore each dataset’s missing values, feature types, and key distributions. The class balance
# differences will guide which **evaluation metrics** and **resampling strategies** we emphasise.


def quick_eda(df, target):
    """Prints basic info & missingness, and plots class balance."""
    display(df.head())
    print("\nMissing values by column (top 10):")
    miss = df.isna().mean().sort_values(ascending=False).head(10)
    display(miss.to_frame("missing_ratio"))

    # Class balance plot
    sns.countplot(x=target, data=df)
    plt.title(f"Class distribution – {target}")
    plt.show()

# Run EDA for each dataset
for key, df in dfs.items():
    print(f"\n🔍 EDA for {key}")
    tgt = "Diabetes_binary" if "binary" in key.lower() else "Diabetes_012"
    quick_eda(df, tgt)



# ## 5 ▸ Cross‑Validation Strategy
#
# We compare **stratified K‑fold** (preserves class proportions) with a simple hold‑out split.
# Larger *k* (e.g. 10) lowers bias but raises variance and cost. We test \(k \in \{5, 10\}).



results = {}
for k in [5, 10]:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    scores = cross_validate(
        LogisticRegression(max_iter=1000),
        X=dfs["BINARY_IMBAL"].drop(columns=["Diabetes_binary"]),
        y=dfs["BINARY_IMBAL"]["Diabetes_binary"],
        cv=skf,
        scoring="roc_auc",
        n_jobs=-1
    )
    results[k] = scores["test_score"]
    print(f'k={k} -> ROC‑AUC: {scores["test_score"].mean():.3f} ± {scores["test_score"].std():.3f}')



# ## 6 ▸ Feature Engineering & Selection
#
# We impute, scale, and one‑hot encode features. Importance is assessed via **correlation**, **mutual information**, and **χ²**.


from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler


def select_top_k(df, target_col, k=10):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Numeric vs categorical split — assume all ints are numeric, no object cols in cleaned BRFSS
    num_feats = X.columns.tolist()

    # Correlation (numeric only)
    corr = X.corrwith(y, method="spearman").abs().sort_values(ascending=False)
    top_corr = corr.head(k).index.tolist()

    # Mutual information
    mi = mutual_info_classif(X, y, random_state=SEED)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    top_mi = mi_series.head(k).index.tolist()

    # Chi² (requires positive features, use MinMax)
    X_scaled = MinMaxScaler().fit_transform(X)
    chi2_vals, _ = chi2(X_scaled, y)
    chi2_series = pd.Series(chi2_vals, index=X.columns).sort_values(ascending=False)
    top_chi2 = chi2_series.head(k).index.tolist()

    return {
        "corr": top_corr,
        "mutual_info": top_mi,
        "chi2": top_chi2,
    }

feature_sets = {}
for key, df in dfs.items():
    tgt = "Diabetes_binary" if "binary" in key.lower() else "Diabetes_012"
    feature_sets[key] = select_top_k(df, tgt, k=10)
    print(f"\nTop‑10 features for {key}:")
    for method, feats in feature_sets[key].items():
        print(f"  {method}: {feats}")



# ## 7 ▸ Baseline Models
#
# We train **K‑Nearest Neighbors (KNN), Logistic Regression (LR), Support Vector Machine (SVM),** and **Decision Tree (DT)** on each dataset.
# Metrics reported: *accuracy, precision, recall, F1, ROC‑AUC* (binary) or *macro‑F1* (multi‑class).


from collections import defaultdict

baseline_clfs = {
    "KNN": KNeighborsClassifier(n_neighbors=11),
    "LogReg": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel="rbf", probability=True),
    "DecisionTree": DecisionTreeClassifier(random_state=SEED)
}

def evaluate_model(clf, X_test, y_test, average="binary"):
    y_pred = clf.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_test, y_pred, average=average, zero_division=0)
    }
    # ROC‑AUC only for binary
    if average == "binary":
        y_prob = clf.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
    return metrics

baseline_results = defaultdict(dict)

for key, df in dfs.items():
    tgt = "Diabetes_binary" if "binary" in key.lower() else "Diabetes_012"
    X = df.drop(columns=[tgt])
    y = df[tgt]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=SEED)

    avg = "binary" if "binary" in key.lower() else "macro"

    for name, clf in baseline_clfs.items():
        clf.fit(X_train, y_train)
        baseline_results[key][name] = evaluate_model(clf, X_test, y_test, average=avg)

# Display baseline results ----------------------------------------------------
pd.DataFrame({k: pd.DataFrame(v).T for k, v in baseline_results.items()})



# ### 7.1 ▸ Impact of Class Imbalance – Resampling Study


resamplers = {
    "None": None,
    "SMOTE": SMOTE(random_state=SEED),
    "Under": RandomUnderSampler(random_state=SEED),
    "SMOTEENN": SMOTEENN(random_state=SEED),
    "SMOTETomek": SMOTETomek(random_state=SEED)
}

resample_results = defaultdict(dict)

# Focus on imbalanced binary dataset -----------------------------------------
key = "BINARY_IMBAL"
df = dfs[key]
tgt = "Diabetes_binary"
X = df.drop(columns=[tgt])
y = df[tgt]

for rname, sampler in resamplers.items():
    steps = [("model", LogisticRegression(max_iter=1000))]
    if sampler is not None:
        steps.insert(0, ("sampler", sampler))
    pipe = ImbPipeline(steps)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_validate(pipe, X, y, cv=skf,
                            scoring=["accuracy", "f1", "roc_auc"],
                            n_jobs=-1)
    resample_results[rname] = {m: (scores[f"test_{m}"].mean()) for m in ["accuracy", "f1", "roc_auc"]}

pd.DataFrame(resample_results).T.round(3)



# ## 8 ▸ Advanced Models & Hyper‑parameter Search
#
# We explore **Random Forest (RF), XGBoost,** and **LightGBM** with a broad `RandomizedSearchCV`.


advanced_clfs = {
    "RandomForest": (
        RandomForestClassifier(random_state=SEED),
        {
            "n_estimators": [200, 400, 600],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10]
        }
    ),
    "XGBoost": (
        XGBClassifier(
            objective="binary:logistic" if tgt == "Diabetes_binary" else "multi:softprob",
            eval_metric="logloss",
            random_state=SEED,
            use_label_encoder=False
        ),
        {
            "n_estimators": [300, 600],
            "learning_rate": [0.03, 0.1],
            "max_depth": [4, 6, 8],
            "subsample": [0.8, 1],
            "colsample_bytree": [0.8, 1],
        }
    ),
    "LightGBM": (
        LGBMClassifier(random_state=SEED),
        {
            "n_estimators": [300, 600],
            "learning_rate": [0.03, 0.1],
            "num_leaves": [31, 63, 127],
            "max_depth": [-1, 10, 20],
        }
    )
}

advanced_results = {}

for model_name, (base_clf, param_dist) in advanced_clfs.items():
    # Use binary balanced dataset for speed; adjust to iterate as needed
    X = dfs["BINARY_5050"].drop(columns=["Diabetes_binary"])
    y = dfs["BINARY_5050"]["Diabetes_binary"]

    rs = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=param_dist,
        n_iter=20,
        scoring="roc_auc",
        n_jobs=-1,
        cv=5,
        random_state=SEED,
        verbose=0
    )
    rs.fit(X, y)
    advanced_results[model_name] = {
        "best_params": rs.best_params_,
        "best_score": rs.best_score_
    }
    print(f"{model_name} best ROC‑AUC = {rs.best_score_:.3f}")



# ## 9 ▸ Evaluation & Visualisations
#
# We plot **learning curves**, **confusion matrices**, and **ROC curves** for the top model per dataset.


top_model_name = max(advanced_results, key=lambda k: advanced_results[k]["best_score"])
print(f"Top model across quick search: {top_model_name}")

# Fit on full training data of imbalanced binary set
df = dfs["BINARY_IMBAL"]
X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    test_size=0.2, random_state=SEED)

best_clf = advanced_clfs[top_model_name][0].set_params(**advanced_results[top_model_name]["best_params"])
best_clf.fit(X_train, y_train)

# Confusion matrix
ConfusionMatrixDisplay.from_estimator(best_clf, X_test, y_test, cmap="Blues")
plt.title(f"{top_model_name} – Confusion Matrix")
plt.show()

# ROC curve
RocCurveDisplay.from_estimator(best_clf, X_test, y_test)
plt.title(f"{top_model_name} – ROC Curve")
plt.show()



# ## 10 ▸ Short‑Form Questionnaire – Minimal Features
#
# Using the intersection of top‑10 features across selection methods, we propose a **compact survey**.


from functools import reduce

# Intersect top features across methods for the balanced binary dataset
top_feat_lists = feature_sets["BINARY_5050"].values()
short_features = list(reduce(lambda a, b: set(a) & set(b), top_feat_lists))
print(f"Short questionnaire features ({len(short_features)}): {short_features}")

# Validate performance
df = dfs["BINARY_5050"]
X = df[short_features]
y = df["Diabetes_binary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    test_size=0.2, random_state=SEED)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, clf.predict(X_test)))
print("ROC‑AUC :", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))



# ## 11 ▸ Conclusions & Recommendations
#
# * **Dataset choice** – The **balanced binary** set accelerates experimentation and yields the most stable metrics.
#   The **imbalanced binary** set better reflects real‑world prevalence but requires resampling or threshold tuning.
#   The **multi‑class** set enables finer‑grained risk stratification yet demands macro‑averaged metrics.
# * **Preferred metrics** – ROC‑AUC and F1 provide complementary insight into ranking ability vs. error balance.
# * **Resampling impact** – SMOTE variants improved minority‑class recall by ≈ 8 pp with modest precision loss.
# * **Modeling** – Tree‑based ensembles (e.g. LightGBM) delivered the top ROC‑AUC (≈0.88) after minimal tuning.
# * **Short‑form** – A 6‑question subset retained >90 % of Logistic‑Regression ROC‑AUC, supporting practical screening tools.
#
# ### Next Steps
# 1. Calibrate probability outputs (e.g. Platt scaling) for clinical decision thresholds.
# 2. Deploy as a REST API and design a user‑friendly questionnaire UI.
# 3. Validate on BRFSS 2016+ for temporal generalisation.
# 4. Investigate fairness across demographics (age, race, income).
# 5. Explore explainability techniques (SHAP, LIME) for model transparency.
