"""
Train the EduPredict model and save reusable artifacts.
Run: python3 train.py
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "student_data.csv"
MODEL_PATH = ROOT / "student_model.joblib"
METRICS_PATH = ROOT / "model_metrics.json"

def make_dataset(n=600, seed=42):
    """Create a reproducible teaching dataset; not real student records."""
    rng = np.random.default_rng(seed)
    study_hours = rng.uniform(0.5, 9.0, n).round(1)
    attendance = rng.uniform(45, 100, n).round(1)
    previous_marks = rng.uniform(25, 98, n).round(1)

    # A transparent synthetic rule with a little noise for a learning demo.
    score = (
        4.2 * study_hours
        + 0.22 * attendance
        + 0.31 * previous_marks
        + rng.normal(0, 7.5, n)
    )
    passed = (score >= 43).astype(int)

    return pd.DataFrame({
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks,
        "result": passed
    })

def main():
    df = make_dataset()
    df.to_csv(DATA_PATH, index=False)

    X = df[["study_hours", "attendance", "previous_marks"]]
    y = df["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=5, random_state=42
        ))
    ])
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, target_names=["Needs support", "On track"],
            output_dict=True, zero_division=0
        ),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "features": ["study_hours", "attendance", "previous_marks"]
    }

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print("Training complete.")
    print(f"Model saved: {MODEL_PATH.name}")
    print(f"Metrics saved: {METRICS_PATH.name}")
    print(f"Test accuracy: {metrics['accuracy']:.1%}")
    print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")

if __name__ == "__main__":
    main()
