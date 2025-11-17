from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.base import BaseEstimator

def train_binary_classifier(
    model: BaseEstimator,
    df: pd.DataFrame,
    target: str,
    features: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Dict[str, Any]:
    """
    Split data, fit the given model, and report binary classification metrics.

    Parameters
    ----------
    model : sklearn-like estimator
        Must implement fit(X, y) and predict(X). AUC is computed via
        predict_proba(X)[:, 1] if available, otherwise decision_function(X).
    df : pd.DataFrame
        Full dataframe containing features and the target column.
    target : str
        Name of the binary target column (values like {0,1} or {False,True}).
    features : list[str] or None
        If None, uses all columns except target.
    test_size : float
        Fraction for the test split.
    random_state : int
        Random seed for reproducibility.
    stratify : bool
        If True, use stratified split on the target.

    Returns
    -------
    dict
        {
          "model": fitted_model,
          "X_train": X_train, "X_test": X_test,
          "y_train": y_train, "y_test": y_test,
          "y_pred": y_pred, "y_score": y_score (or None),
          "report_text": classification_report_str,
          "metrics": {
              "accuracy": ...,
              "precision": ...,
              "recall": ...,
              "f1": ...,
              "auc": ...,
              "confusion_matrix": [[tn, fp], [fn, tp]]
          }
        }
    """
    # 1. define features/target, and split the data
    if features is None:
        features = [c for c in df.columns if c != target]

    X = df[features]
    y = df[target].astype(int)  # ensure 0/1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
        
    )

    # 2. Fit the model to the training dataset, and predict the test
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 3. Compute the evaluation metrics and print them
    # Scores for AUC (probabilities if available, else decision function)
    y_score: Optional[np.ndarray] = None
    if hasattr(model, "predict_proba"):
        try:
            y_score = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_score = None
    if y_score is None and hasattr(model, "decision_function"):
        try:
            y_score = model.decision_function(X_test)
        except Exception:
            y_score = None

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_test, y_score) if y_score is not None else None
    cm = confusion_matrix(y_test, y_pred).tolist()

    report_txt = classification_report(y_test, y_pred, digits=3)

    # Print a short summary (optional)
    print("=== Classification Report ===")
    print(report_txt)
    print("=== Metrics ===")
    print({
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm,
    })

    return {
        "model": model,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred": y_pred, "y_score": y_score,
        "report_text": report_txt,
        "metrics": {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "confusion_matrix": cm,
        },
    }

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

def make_logreg_pipeline(df, target, use_balanced=False):
    num_cols = df.select_dtypes(include="number").columns.drop(target, errors="ignore")
    cat_cols = df.columns.difference(num_cols.union([target]))

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(num_cols)),
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(cat_cols)),
        ],
        remainder="drop",
        sparse_threshold=0.3,  # keep sparse if lots of one-hot features
    )

    # Tips:
    # - 'saga' converges well on large data, supports OHE + sparse matrices.
    # - Smaller C (stronger regularization) often helps convergence.
    # - Slightly looser tol speeds convergence without hurting metrics much.
    logreg = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=0.5,               # try 0.1 ↔ 1.0; smaller = easier to converge
        max_iter=4000,       # plenty, but needed much less once scaled
        tol=1e-3,            # default 1e-4 can be stricter than needed
        class_weight="balanced" if use_balanced else None,
        n_jobs=-1            # parallel where supported
    )

    return Pipeline([("prep", preprocess), ("clf", logreg)])
