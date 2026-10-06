# Heart Disease Predictor — Streamlit App

A small web app to try the trained Heart Disease model in a browser, with no install required. Built from the model trained in the `Heart_Disease_Prediction.ipynb` notebook (adapted from the Titanic Passenger Survival Prediction notebook, same steps).

## Dataset and Attributes

Source: `heart.csv` (1025 instances). Only **4 input attributes** are used:

| Input attribute  | Original column | Values |
| ---------------- | --------------- | ------ |
| Sex              | `sex`   | Female, Male |
| ChestPain        | `cp`    | Typical Angina, Atypical Angina, Non-Anginal Pain, Asymptomatic |
| ExerciseAngina   | `exang` | No, Yes |
| MajorVessels     | `ca`    | Zero, One, Two, Three, Four |

Output: `HeartDisease` (`target`) — Yes / No. `sample-data.csv` is `heart.csv` with these 4 columns converted to readable text labels.

## Running Locally

```bash
python -m venv venv
venv\Scripts\activate   # Windows; use "source venv/bin/activate" on macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

This opens the app at `http://localhost:8501`.

## Deploying to Streamlit Community Cloud (free)

1. Push this folder to a public GitHub repository (needs `app.py`, `requirements.txt`, `runtime.txt`, `svc_trained_model.pkl`).
2. Go to **https://share.streamlit.io** → **Create app** → **Deploy a public app from GitHub**.
3. Select your repository, branch `main`, main file path `app.py`, then click **Deploy**.

## How It Works

- `app.py` loads `svc_trained_model.pkl` (the same file the notebook produces) and recreates the exact same `LabelEncoder`s used during training, so predictions match the notebook exactly.
- Users pick Sex, Chest Pain, Exercise-Induced Angina and Major Vessels from dropdowns and click **Predict Heart Disease**.
- If you retrain the model in the notebook, copy the new `svc_trained_model.pkl` into this folder and push.

> Educational demo only — not for medical diagnosis.
