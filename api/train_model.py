"""Train and version the SmartOps demonstration churn model."""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
from sklearn.datasets import make_classification
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgboost_churn_model.pkl"
METRICS_PATH = BASE_DIR / "xgboost_churn_model.metrics.json"
RANDOM_SEED = 42


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=BASE_DIR.parent, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def train() -> dict:
    """Train a reproducible sample model and save evaluation metadata with it."""
    features, labels = make_classification(
        n_samples=1_000,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        weights=[0.7, 0.3],
        random_state=RANDOM_SEED,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=RANDOM_SEED
    )
    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )
    metrics = {
        "model_type": "XGBClassifier",
        "feature_order": ["recency", "frequency", "monetary_value", "refund_rate"],
        "random_seed": RANDOM_SEED,
        "trained_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "test_roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "test_precision": round(float(precision), 4),
        "test_recall": round(float(recall), 4),
        "test_f1": round(float(f1), 4),
    }
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
