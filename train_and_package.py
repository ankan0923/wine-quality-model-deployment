"""Train a scikit-learn model on the wine quality data and package it with joblib.

Usage:
    python train_and_package.py [--data data/wine_quality.csv] [--out models/wine_model_v1.joblib]

The saved artifact is a single scikit-learn Pipeline (scaler + model), so the
exact preprocessing used in training is always applied at prediction time.
A small JSON metadata file is written next to the model.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "Quality"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="1. Wine Quality dataset - Sheet1.csv")
    parser.add_argument("--out", default="models/wine_model_v1.joblib")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. Load data (drop exact duplicate rows so they cannot leak across the split)
    df = pd.read_csv(args.data)
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Loaded {n_raw} rows, {len(df)} after removing duplicates")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 2. Split (no stratify: some classes have a single sample in this dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=args.seed
    )

    # 3. Train a pipeline
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=args.seed)),
        ]
    )
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Hold-out accuracy: {accuracy:.3f} (n_test={len(X_test)})")

    # 4. Package with joblib
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path, compress=3)

    # 5. Write metadata alongside the model
    metadata = {
        "model_file": out_path.name,
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "features": list(X.columns),
        "target": TARGET,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "holdout_accuracy": round(float(accuracy), 4),
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved model    -> {out_path}")
    print(f"Saved metadata -> {meta_path}")

    # Pickle alternative (joblib is preferred for sklearn models with numpy arrays):
    #   import pickle
    #   with open("models/wine_model_v1.pkl", "wb") as f:
    #       pickle.dump(model, f)


if __name__ == "__main__":
    main()
