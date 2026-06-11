import streamlit as st
import joblib
import numpy as np

# =========================
# Load Model and Scaler
# =========================

model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Risk Predictor")

st.markdown("""
Predict the likelihood of heart disease using a Machine Learning model trained on clinical patient data.
""")

st.markdown("---")

# =========================
# User Inputs
# =========================

age = st.number_input(
    "Age",
    min_value=20,
    max_value=100,
    value=50
)

sex = st.selectbox(
    "Sex",
    ["Female", "Male"]
)

sex = 1 if sex == "Male" else 0

# Chest Pain Type
cp = st.selectbox(
    "Chest Pain Type",
    [
        "Category 0",
        "Category 1",
        "Category 2",
        "Category 3"
    ]
)

cp_mapping = {
    "Category 0": 0,
    "Category 1": 1,
    "Category 2": 2,
    "Category 3": 3
}

cp = cp_mapping[cp]

# Resting Blood Pressure
trestbps = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=80,
    max_value=250,
    value=120
)

# Cholesterol
chol = st.number_input(
    "Cholesterol (mg/dL)",
    min_value=100,
    max_value=600,
    value=200
)

# Fasting Blood Sugar
fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dL",
    ["No", "Yes"]
)

fbs = 1 if fbs == "Yes" else 0

# Rest ECG
restecg = st.selectbox(
    "Rest ECG",
    [
        "Category 0",
        "Category 1",
        "Category 2"
    ]
)

restecg_mapping = {
    "Category 0": 0,
    "Category 1": 1,
    "Category 2": 2
}

restecg = restecg_mapping[restecg]

# Maximum Heart Rate
thalach = st.number_input(
    "Maximum Heart Rate",
    min_value=60,
    max_value=250,
    value=150
)

# Exercise Induced Angina
exang = st.selectbox(
    "Exercise Induced Angina",
    ["No", "Yes"]
)

exang = 1 if exang == "Yes" else 0

# Old Peak
oldpeak = st.number_input(
    "Old Peak",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

# ST Segment Slope
slope = st.selectbox(
    "ST Segment Slope",
    [
        "Category 0",
        "Category 1",
        "Category 2"
    ]
)

slope_mapping = {
    "Category 0": 0,
    "Category 1": 1,
    "Category 2": 2
}

slope = slope_mapping[slope]

# Number of Major Vessels
ca = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3]
)

# Thalassemia
thal = st.selectbox(
    "Thalassemia",
    [
        "Category 0",
        "Category 1",
        "Category 2",
        "Category 3"
    ]
)

thal_mapping = {
    "Category 0": 0,
    "Category 1": 1,
    "Category 2": 2,
    "Category 3": 3
}

thal = thal_mapping[thal]

# =========================
# Prediction
# =========================

if st.button("Predict Risk"):

    patient = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    patient_scaled = scaler.transform(patient)

    prediction = model.predict(patient_scaled)

    probability = model.predict_proba(patient_scaled)[0][1]

    st.markdown("---")

    st.subheader(
        f"Risk Probability: {probability*100:.2f}%"
    )

    st.progress(float(probability))

    if prediction[0] == 1:
        st.error("⚠️ High Risk")
    else:
        st.success("✅ Low Risk")

# =========================
# Footer
# =========================

st.markdown("---")

st.caption(
    "Built using Logistic Regression, SHAP Explainability, Cross-Validation, and Streamlit"
)