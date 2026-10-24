# Project: Personalized Stress Monitoring Platform
Multi-user platform that classifies stress from wearable physiological signals (HRV, EDA, etc.), with per-user calibration. Wellness tool, not a diagnostic device.

## Stack (all free-tier)
Python 3.11, scikit-learn, PyTorch, NumPy/Pandas/SciPy, MLflow, FastAPI, Pydantic, PostgreSQL, Redis, Docker/docker-compose, pytest, ruff, GitHub Actions, Prometheus/Grafana, Locust, ONNX.

## Structure
src/stress/{data,features,models,personalization,api,workers}, tests/, notebooks/, configs/, docs/, data/ (gitignored)

## Rules
- Before writing code for any task, explain the approach and trade-offs in plain language, then wait for my approval.
- One task at a time. Small commits with clear messages.
- Every module ships with pytest tests. Run them before saying a task is done.
- Never commit data, secrets, or .env files. Use .env.example.
- ML: all evaluation is LOSO (leave-one-subject-out). No data leakage: fit scalers/feature selection only on training subjects. Fixed seeds. Log every experiment to MLflow.
- Type hints and docstrings on public functions. Lint with ruff.
- Report real measured numbers only. Never invent metrics.
- Ask me before making architecture or ML design decisions.

## Commands

All commands use the venv interpreter explicitly to avoid running against global Python:

```bash
# Install package and dependencies in editable mode
.venv/Scripts/python.exe -m pip install -e ".[dev]"

# Run tests
.venv/Scripts/python.exe -m pytest tests/ -v

# Run linter (check only)
.venv/Scripts/python.exe -m ruff check src/ tests/ scripts/

# Format code (check without modifying)
.venv/Scripts/python.exe -m ruff format --check src/ tests/ scripts/

# Format code (apply changes)
.venv/Scripts/python.exe -m ruff format src/ tests/ scripts/

# Run inspection script
.venv/Scripts/python.exe scripts/inspect_wesad.py data/raw/WESAD

# Run window report
.venv/Scripts/python.exe scripts/report_windows.py data/raw/WESAD

# Run API (when implemented)
.venv/Scripts/python.exe -m uvicorn stress.api.main:app --reload
```

**Convention:** Always use `.venv/Scripts/python.exe -m` for package commands.

## Working protocol (added at handoff)
- Never trust an assistant's summary or typed numbers; ask for raw command output and re-run key checks yourself.
- Every key invariant needs a mutation check: break the code on purpose, show the test fail, restore from a backup in $env:TEMP, and show equal (Get-FileHash).Hash for both files.
- Git belongs to the user: explicit `git add <paths>`, check the branch first, one branch per task, PRs into master with a merge commit after green CI, never push to master.
- Assistants must not run installs, Docker, or git. Use `.venv\Scripts\python.exe -m ...` only.
- API run command: `.venv\Scripts\python.exe -m uvicorn stress.api.main:create_app --factory`.
- CI lints only src/, tests/, scripts/. loadtest/ is not linted by CI.
- Claims: measured numbers with sources; exploratory vs pre-registered kept separate; research demo, not a medical device.
