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
    # Scores for AUC 
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

    # Print a short summary 
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

