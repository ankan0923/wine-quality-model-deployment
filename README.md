# Wine Quality Model: Packaging and Production Deployment

A scikit-learn pipeline (StandardScaler + RandomForestClassifier) trained on the wine quality data, packaged with joblib, validated, and ready to deploy.

## Project layout

```
wine-quality-model/
├── data/wine_quality.csv          # training data
├── models/
│   ├── wine_model_v1.joblib       # packaged model (Pipeline)
│   └── wine_model_v1.json         # metadata: sha256, features, sklearn version
├── train_and_package.py           # train + save
├── validate_model.py              # smoke test for the packaged model
├── requirements.txt
└── docs/
    ├── architecture.svg
    └── GIT_VERSIONING.md
```

## 1. Local setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_and_package.py
python validate_model.py             # must exit 0 before you deploy
```

## 2. Pin the environment

A joblib/pickle file is only guaranteed to load with the same scikit-learn version that created it. Pin versions in `requirements.txt` (the one included matches the versions used here) and use the same file in training and production. `validate_model.py` fails if the runtime version differs from the one recorded in the metadata.

**Security:** never load a joblib/pickle file from an untrusted source; loading can execute arbitrary code. Only deploy artifacts your own pipeline produced, and verify the `sha256` in the metadata.

## 3. Serve the model

Create `app.py` (needs `fastapi`, `uvicorn`, `pydantic` added to requirements):

```python
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path("models/wine_model_v1.joblib")
meta = json.loads(MODEL_PATH.with_suffix(".json").read_text())
model = joblib.load(MODEL_PATH)          # load once at startup, not per request
app = FastAPI(title="Wine Quality API")


class Wine(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    ph: float
    sulphates: float
    alcohol: float


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name}


@app.post("/predict")
def predict(wine: Wine):
    row = pd.DataFrame([list(wine.model_dump().values())], columns=meta["features"])
    return {"quality": int(model.predict(row)[0]), "model_version": MODEL_PATH.name}
```

Run locally: `uvicorn app:app --host 0.0.0.0 --port 8000`

## 4. Containerize

`Dockerfile`:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY models/ models/
RUN useradd -m appuser
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t wine-quality:1.0.0 .
docker run -p 8000:8000 wine-quality:1.0.0
curl http://localhost:8000/health
```

## 5. Deploy

1. Push the image to a registry (Docker Hub, ECR, GCR, ACR).
2. Run it on your platform of choice (Kubernetes, ECS, Cloud Run, a VM).
3. Run 2 or more replicas behind a load balancer; keep the container stateless.
4. Put the service behind HTTPS and add authentication (API key or OAuth).
5. Configure a readiness probe on `/health`.

## 6. Release checklist

- [ ] `python validate_model.py` exits 0 in the same image or environment that will serve
- [ ] Hold-out metrics reviewed against the currently deployed model
- [ ] Model file and metadata committed or registered (see `docs/GIT_VERSIONING.md`)
- [ ] Image tagged with the model version, e.g. `wine-quality:1.0.0`
- [ ] Rollout plan: canary or blue/green, with a one-step rollback to the previous tag

## 7. Monitoring and retraining

- Log request counts, latency and error rates.
- Log inputs and predictions (respecting privacy rules) to detect feature drift, for example shifts in alcohol or pH distributions.
- When ground-truth labels arrive, track accuracy over time.
- Retrain on a schedule or when drift or accuracy alerts fire, re-run validation, and ship as a new version.

## Known limitations of the sample data

The provided CSV has 52 rows, only 27 unique, and 51 of the 52 are quality 5 (one is quality 6). The hold-out accuracy of 1.0 is therefore not meaningful: the model mostly learns to predict "5". Treat this project as a working template for packaging and deployment, and retrain on a fuller dataset (for example the UCI Wine Quality data, about 1,600 red wines) before relying on predictions.
