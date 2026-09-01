# Recoverly

Recoverly is a four-day MVP for payment-failure recovery decisioning. It
classifies a failed payment, chooses a constrained recovery action, explains
that decision, and compares the result against a fixed-delay baseline.

## Scope and data disclosure

All data in this repository is synthetic and generated from documented,
inspectable rules. Reported metrics will be computed only from the frozen
held-out `backend/data/test_set_v1.csv`. The separate
`backend/data/demo_cases.csv` file will contain hand-picked illustrative
transactions only and must never contribute to reported metrics.

The MVP demonstrates a decisioning architecture on simulated data; it is not
production validation or a payment-gateway replacement.

## Technology

- FastAPI provides the recovery-action API.
- scikit-learn GradientBoosting is the tabular failure-reason classifier.
- Streamlit provides the dashboard.
- Gemini is used only for live explanation narration. Set `GEMINI_API_KEY` in
  the environment; never commit a key. Deterministic templates remain the
  required fallback.
- pytest verifies each module.

## Local setup

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

Implementation follows the fixed module contracts in `ARCHITECTURE.md` and
the evaluation rules in `MVP_SCOPE.md`.
