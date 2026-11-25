# Version Control for the Model with Git

Goal: be able to answer "which code, data and settings produced the model running in production?" and to roll back quickly.

## 1. Initialize the repository

```bash
cd wine-quality-model
git init -b main
git config user.name  "Your Name"
git config user.email "you@example.com"
```

## 2. Decide what goes in Git

| Item | Where it lives |
|------|----------------|
| Code (`train_and_package.py`, `validate_model.py`, `app.py`) | Git |
| `requirements.txt`, `Dockerfile`, docs | Git |
| Small metadata (`*.json`) | Git |
| Small datasets (like this 52-row CSV) | Git |
| Large datasets and model files (tens of MB or more) | Git LFS or DVC (step 4) |
| Virtual environments, caches, secrets | Never (`.gitignore`) |

## 3. Add a `.gitignore` and make the first commit

```bash
cat > .gitignore <<'EOF'
.venv/
__pycache__/
*.pyc
.env
.ipynb_checkpoints/
EOF

git add .
git commit -m "Initial commit: wine quality training, validation and docs"
```

This model is small, so `models/wine_model_v1.joblib` can be committed directly.

## 4. Handling larger models and data

**Option A, Git LFS** (stores big files outside the normal history):

```bash
git lfs install
git lfs track "*.joblib" "*.pkl"
git add .gitattributes
git commit -m "Track model binaries with Git LFS"
```

**Option B, DVC** (Git tracks small pointer files; data and models go to S3, GCS, Azure or a shared disk):

```bash
pip install dvc
dvc init
dvc add data/wine_quality.csv models/wine_model_v1.joblib
git add data/*.dvc models/*.dvc .gitignore
git commit -m "Track data and model with DVC"
dvc remote add -d storage s3://my-bucket/wine-model
dvc push
```

## 5. Branching workflow

```bash
git switch -c experiment/gradient-boosting   # try something new
# ...edit, retrain, validate...
git add -A
git commit -m "Try GradientBoosting; accuracy 0.xx"
git switch main
git merge --no-ff experiment/gradient-boosting
```

Suggested conventions:

- `main` always holds the version that is deployed or deployable.
- `experiment/*` branches for modeling trials; `fix/*` and `feature/*` for the rest.
- Open a pull request for anything merged to `main`, and require `python validate_model.py` to pass in CI.

## 6. Tag every release

Use semantic versioning: MAJOR for a new feature set or incompatible API, MINOR for retraining with new data or better tuning, PATCH for bug fixes.

```bash
python train_and_package.py --out models/wine_model_v1.joblib
python validate_model.py
git add models/ && git commit -m "Release model v1.0.0 (holdout acc recorded in metadata)"
git tag -a model-v1.0.0 -m "Wine model 1.0.0, sklearn 1.9.1"
git push origin main --tags
```

Keep the same tag on the Docker image (`wine-quality:1.0.0`) so code, model and image stay linked.

## 7. Record lineage in the metadata

Add the commit hash to the metadata JSON at training time so the artifact points back to its source:

```python
import subprocess
metadata["git_commit"] = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True
).strip()
```

## 8. Everyday commands

```bash
git status                      # what changed
git log --oneline --decorate    # history with tags
git diff model-v1.0.0 HEAD      # what changed since a release
git tag -l "model-v*"           # list model versions
```

## 9. Roll back

```bash
# Inspect an old release without changing anything
git switch --detach model-v1.0.0

# Restore just the old model files onto the current branch
git restore --source model-v1.0.0 models/

# Undo a bad release commit safely (keeps history)
git revert <commit-sha>
```

Then redeploy the matching image tag.

## 10. Connect to a remote

```bash
git remote add origin git@github.com:<you>/wine-quality-model.git
git push -u origin main --tags
```

## Good habits

- One logical change per commit, with a message saying why.
- Never rewrite history on `main` or on tagged commits.
- Never commit credentials; keep them in environment variables or a secrets manager.
- Retrain and tag on every data or code change that affects predictions.
