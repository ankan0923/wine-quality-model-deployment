import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import subprocess
import sys

# All paths are relative to this file, so it works from any terminal folder
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "wine_model_v1.joblib"
DATA_PATH = BASE / "1. Wine Quality dataset - Sheet1.csv"
TRAIN_SCRIPT = BASE / "train_and_package.py"

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷")
st.title("🍷 Wine Quality Predictor")


@st.cache_resource
def load_model():
    meta = json.loads(MODEL_PATH.with_suffix(".json").read_text())
    return joblib.load(MODEL_PATH), meta


if not MODEL_PATH.exists() or not MODEL_PATH.with_suffix(".json").exists():
    st.error("Model not found yet.")
    st.write(f"The app is looking in: `{MODEL_PATH.parent}`")
    st.write(f"Dataset found: **{DATA_PATH.exists()}** (`{DATA_PATH}`)")
    st.write(f"Training script found: **{TRAIN_SCRIPT.exists()}** (`{TRAIN_SCRIPT}`)")
    if DATA_PATH.exists() and TRAIN_SCRIPT.exists():
        if st.button("Train the model now"):
            result = subprocess.run(
                [sys.executable, str(TRAIN_SCRIPT), "--data", str(DATA_PATH), "--out", str(MODEL_PATH)],
                capture_output=True, text=True, cwd=str(BASE),
            )
            if result.returncode == 0:
                st.success("Model trained. Reload this page (press R).")
                st.code(result.stdout)
            else:
                st.error("Training failed:")
                st.code(result.stderr or result.stdout)
    else:
        st.info("Put `train_and_package.py` next to this file and the CSV at `data/wine_quality.csv`.")
    st.stop()

model, meta = load_model()

# Defaults and ranges taken from the training data
data = pd.read_csv(DATA_PATH)
features = meta["features"]

st.write("Enter the wine's properties and click **Predict**.")

values = {}
cols = st.columns(2)
for i, name in enumerate(features):
    lo, hi = float(data[name].min()), float(data[name].max())
    default = float(data[name].median())
    with cols[i % 2]:
        values[name] = st.number_input(
            name, min_value=min(lo, 0.0), value=default, format="%.4f", step=(hi - lo) / 100 or 0.01
        )

if st.button("Predict"):
    row = pd.DataFrame([values], columns=features)
    pred = int(model.predict(row)[0])
    st.success(f"Predicted quality: {pred}")
    if hasattr(model, "predict_proba"):
        proba = pd.Series(model.predict_proba(row)[0], index=model.classes_, name="probability")
        st.bar_chart(proba)

st.caption("Note: the sample dataset is tiny and almost all quality 5, so predictions are mostly 5.")
