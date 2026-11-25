"""Validate that the packaged model loads and can predict on a sample dataset.

Usage:
    python validate_model.py [--model models/wine_model_v1.joblib] [--data data/wine_quality.csv]

Exits with code 0 if every check passes, 1 otherwise (CI-friendly).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/wine_model_v1.joblib")
    parser.add_argument("--data", default="data/wine_quality.csv")
    parser.add_argument("--n", type=int, default=10, help="sample size")
    args = parser.parse_args()

    model_path = Path(args.model)
    meta = json.loads(model_path.with_suffix(".json").read_text())
    results = []

    # 1. Artifact integrity and environment compatibility
    sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    results.append(check("Checksum matches metadata", sha == meta["sha256"]))
    results.append(
        check(
            "scikit-learn version matches training",
            sklearn.__version__ == meta["sklearn_version"],
            f"trained={meta['sklearn_version']} runtime={sklearn.__version__}",
        )
    )

    # 2. Load
    model = joblib.load(model_path)
    results.append(check("Model loads", hasattr(model, "predict")))

    # 3. Predict on a sample dataset
    df = pd.read_csv(args.data)
    sample = df.sample(n=min(args.n, len(df)), random_state=0)
    X = sample[meta["features"]]  # enforces expected columns and order
    preds = model.predict(X)
    results.append(check("Prediction count matches input", len(preds) == len(X)))
    results.append(check("Predictions are finite", bool(np.all(np.isfinite(preds)))))
    allowed = set(model.classes_)
    results.append(
        check("Predictions are known classes", set(preds) <= allowed, f"classes={sorted(allowed)}")
    )

    # 4. Probabilities, if supported
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        results.append(
            check("Probabilities sum to 1", bool(np.allclose(proba.sum(axis=1), 1.0)))
        )

    # 5. Determinism
    results.append(check("Predictions are deterministic", bool(np.array_equal(preds, model.predict(X)))))

    # 6. Bad input should fail loudly
    try:
        model.predict(X.iloc[:, :-1])
        results.append(check("Rejects input with missing feature", False))
    except Exception:
        results.append(check("Rejects input with missing feature", True))

    print("\nSample predictions (first 5):")
    print(pd.DataFrame({"actual": sample[meta["target"]].values, "predicted": preds}).head())

    if all(results):
        print("\nAll checks passed.")
        return 0
    print("\nSome checks failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
