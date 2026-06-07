import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide"
)

# ==================================================
# LOAD MODEL
# ==================================================

try:
    model = joblib.load("models/random_forest_model.pkl")
except Exception as e:
    st.error(f"Model could not be loaded: {e}")
    st.stop()

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("🩺 Breast Cancer Classifier")

st.sidebar.markdown("""
### About

This application predicts whether a breast tumor is:

- **Benign (B)**
- **Malignant (M)**

using a trained Random Forest Machine Learning model.

### Dataset
Breast Cancer Wisconsin Diagnostic Dataset

### Model
Random Forest Classifier
""")

# ==================================================
# TITLE
# ==================================================

st.title("🩺 Breast Cancer Classification System")

st.markdown("""
This application uses machine learning to predict whether a tumor is **Benign** or **Malignant** based on diagnostic measurements.

Choose a prediction mode below.
""")

# ==================================================
# MODE SELECTION
# ==================================================

mode = st.radio(
    "Select Prediction Mode",
    [
        "Single Prediction",
        "Batch Prediction (CSV Upload)"
    ]
)

# ==================================================
# FEATURE LIST
# ==================================================

features = [
    'radius_mean',
    'texture_mean',
    'perimeter_mean',
    'area_mean',
    'smoothness_mean',
    'compactness_mean',
    'concavity_mean',
    'concave points_mean',
    'symmetry_mean',
    'fractal_dimension_mean',

    'radius_se',
    'texture_se',
    'perimeter_se',
    'area_se',
    'smoothness_se',
    'compactness_se',
    'concavity_se',
    'concave points_se',
    'symmetry_se',
    'fractal_dimension_se',

    'radius_worst',
    'texture_worst',
    'perimeter_worst',
    'area_worst',
    'smoothness_worst',
    'compactness_worst',
    'concavity_worst',
    'concave points_worst',
    'symmetry_worst',
    'fractal_dimension_worst'
]

# ==================================================
# SINGLE PREDICTION
# ==================================================

if mode == "Single Prediction":

    st.subheader("Patient Measurements")

    col1, col2 = st.columns(2)

    user_inputs = {}

    with col1:
        for feature in features[:15]:
            user_inputs[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.6f"
            )

    with col2:
        for feature in features[15:]:
            user_inputs[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.6f"
            )

    if st.button("Predict Diagnosis"):

        input_df = pd.DataFrame([user_inputs])

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0]

        benign_prob = probability[0] * 100
        malignant_prob = probability[1] * 100

        st.divider()

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("🔴 Malignant Tumor Detected")
        else:
            st.success("🟢 Benign Tumor Detected")

        st.subheader("Prediction Probability")

        st.write(
            f"Benign Probability: **{benign_prob:.2f}%**"
        )

        st.write(
            f"Malignant Probability: **{malignant_prob:.2f}%**"
        )

        st.progress(float(max(benign_prob, malignant_prob)) / 100)

# ==================================================
# BATCH PREDICTION
# ==================================================

else:

    st.subheader("Upload CSV File")

    uploaded_file = st.file_uploader(
        "Choose CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            data = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data")

            st.dataframe(data.head())

            prediction = model.predict(data)

            probability = model.predict_proba(data)

            results = data.copy()

            results["Prediction"] = prediction

            results["Prediction"] = results[
                "Prediction"
            ].map({
                0: "Benign",
                1: "Malignant"
            })

            results["Benign_Probability"] = (
                probability[:, 0] * 100
            )

            results["Malignant_Probability"] = (
                probability[:, 1] * 100
            )

            st.subheader("Prediction Results")

            st.dataframe(results)

            csv = results.to_csv(index=False)

            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="prediction_results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error: {e}")

# ==================================================
# MODEL INFO
# ==================================================

st.divider()

with st.expander("Model Information"):

    st.markdown("""
### Machine Learning Pipeline

1. Data Cleaning
2. Label Encoding
3. Train-Test Split
4. Random Forest Training
5. Cross Validation
6. Model Evaluation

### Metrics Used

- Accuracy
- Precision
- Recall
- F1 Score
- Cross Validation

### Target Classes

| Class | Meaning |
|---------|---------|
| 0 | Benign |
| 1 | Malignant |
""")

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "Built with Streamlit and Scikit-Learn"
)