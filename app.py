"""
Heart Disease Prediction - Student Demo App
----------------------------------------------------------
A small Streamlit web app that lets students interact with the trained
Support Vector Classifier from the "Heart Disease Prediction" notebook,
without needing to install Python or run Jupyter.

It loads the same svc_trained_model.pkl produced by that notebook, and
recreates the exact same LabelEncoders (fit on the same fixed category
lists) used during training, so predictions match the notebook exactly.

Only 4 input attributes from heart.csv are used:
Sex (sex), ChestPain (cp), ExerciseAngina (exang), MajorVessels (ca).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# ---------------------------------------------------------------------------
# Load the trained model (same file the notebook produced)
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"

SEX_VALUES = ["Female", "Male"]
CHEST_PAIN_VALUES = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
EXERCISE_ANGINA_VALUES = ["No", "Yes"]
MAJOR_VESSELS_VALUES = ["Zero", "One", "Two", "Three", "Four"]

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def build_label_encoders():
    """
    Recreates the exact same LabelEncoders used in the notebook.
    LabelEncoder assigns codes based on the sorted order of the category
    list, so fitting on the same fixed lists reproduces the same codes
    deterministically - no need to pickle/ship the encoders separately.
    """
    encoders = {}
    encoders["Sex"] = LabelEncoder().fit(SEX_VALUES)
    encoders["ChestPain"] = LabelEncoder().fit(CHEST_PAIN_VALUES)
    encoders["ExerciseAngina"] = LabelEncoder().fit(EXERCISE_ANGINA_VALUES)
    encoders["MajorVessels"] = LabelEncoder().fit(MAJOR_VESSELS_VALUES)
    return encoders

model = load_model()
encoders = build_label_encoders()

# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

st.title("❤️ Heart Disease Predictor")
st.markdown(
    """
    This is a teaching demo built from the **Heart Disease Prediction**
    notebook. Enter a patient's details below and the trained Support Vector
    Classifier (SVC) will predict whether the patient has heart disease.

    *Model accuracy on held-out test data: **81%** (heart.csv, 1025 rows, 4 input attributes).*
    """
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Sex", SEX_VALUES, index=1)
    chest_pain = st.selectbox("Chest Pain Type", CHEST_PAIN_VALUES)
with col2:
    exercise_angina = st.selectbox("Exercise-Induced Angina", EXERCISE_ANGINA_VALUES)
    major_vessels = st.selectbox("Major Vessels Coloured by Fluoroscopy", MAJOR_VESSELS_VALUES)

if st.button("Predict Heart Disease", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "Sex": [sex],
        "ChestPain": [chest_pain],
        "ExerciseAngina": [exercise_angina],
        "MajorVessels": [major_vessels],
    })

    encoded = pd.DataFrame({
        "Sex": encoders["Sex"].transform(user_input["Sex"]),
        "ChestPain": encoders["ChestPain"].transform(user_input["ChestPain"]),
        "ExerciseAngina": encoders["ExerciseAngina"].transform(user_input["ExerciseAngina"]),
        "MajorVessels": encoders["MajorVessels"].transform(user_input["MajorVessels"]),
    })

    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ HEART DISEASE")
    else:
        st.success("### ✅ NO HEART DISEASE")

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption(
    "Built for teaching purposes from the heart.csv dataset using only 4 input "
    "attributes. This is NOT a medical tool and must not be used for diagnosis."
)
