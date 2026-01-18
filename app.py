import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Menstrual Health Disorder Predictor",
    layout="wide"
)

st.title("🩺 AI-Based Menstrual Health Disorder Predictor")
st.write(
    "Predict **PCOS, Anemia, and Hormonal Imbalance** using menstrual, lifestyle, "
    "and health indicators with explainable AI (SHAP)."
)

# -----------------------------
# Load model and feature columns
# -----------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_columns():
    with open("columns.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
feature_columns = load_columns()

# -----------------------------
# Sidebar user inputs
# -----------------------------
st.sidebar.header("🧾 Enter Health Details")

user_input = {}
for col in feature_columns:
    user_input[col] = st.sidebar.number_input(
        label=col,
        value=0.0,
        step=0.1
    )

input_df = pd.DataFrame([user_input])

# -----------------------------
# Risk level function
# -----------------------------
def risk_level(confidence):
    if confidence < 0.40:
        return "🟢 Low Risk"
    elif confidence < 0.70:
        return "🟡 Medium Risk"
    else:
        return "🔴 High Risk"

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Health Risks"):

    st.subheader("📌 Prediction Results")

    # Output order used during training
    targets = {
        0: "PCOS",
        1: "Anemia",
        2: "Hormonal Imbalance"
    }

    results = {}

    for idx, label in targets.items():

        estimator = model.estimators_[idx]

        # Prediction
        pred = int(estimator.predict(input_df)[0])

        # Safe probability handling
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(input_df)[0]
            confidence = float(np.max(proba))
        else:
            confidence = 1.0

        results[label] = (pred, confidence)

        # Display
        if pred == 1:
            st.error(
                f"⚠ **{label} Detected**  \n"
                f"Confidence: **{confidence:.2f}**  \n"
                f"Risk Level: **{risk_level(confidence)}**"
            )
        else:
            st.success(
                f"✅ **No {label} Detected**  \n"
                f"Confidence: **{confidence:.2f}**  \n"
                f"Risk Level: **{risk_level(confidence)}**"
            )

    # -----------------------------
    # SHAP Explainability
    # Only for Hormonal Imbalance
    # -----------------------------
    st.subheader("📊 Why was Hormonal Imbalance predicted?")

    try:
        hormonal_model = model.estimators_[2]

        explainer = shap.Explainer(hormonal_model, input_df)
        shap_values = explainer(input_df)

        st.write("### 🔍 Top Contributing Factors")

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(
            shap_values[0],
            max_display=10,
            show=False
        )
        st.pyplot(fig)

    except Exception as e:
        st.warning("⚠ SHAP explanation could not be generated.")
        st.text(str(e))
