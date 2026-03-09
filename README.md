# Airline Passenger Satisfaction Dashboard

This project builds and serves a Streamlit dashboard for airline passenger satisfaction analysis and prediction.

## Project Contents

- `app.py`: Streamlit dashboard app
- `train_and_save_models.py`: Offline model training script
- `train.csv`, `test.csv`: Input datasets
- `artifacts/model_results.joblib`: Saved trained model artifacts (created by training script)
- `requirements.txt`: Python dependencies

## Prerequisites

- Python 3.12 (see `runtime.txt`)
- `pip` available in your shell

## 1. Setup Environment

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you are on macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Launch the Dashboard

Start Streamlit:

```powershell
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## 3. Typical Workflow

1. Install dependencies.
2. Run `python train_and_save_models.py`.
3. Run `streamlit run app.py`.
4. In the dashboard, use **Reload saved artifacts** in the Modeling tab if you retrain models while the app is already open.

## Troubleshooting

- `Saved model artifact not found`:
  Run `python train_and_save_models.py` first.
- Missing package errors:
  Re-activate your virtual environment and run `pip install -r requirements.txt`.
- Streamlit command not found:
  Use `python -m streamlit run app.py` instead.
