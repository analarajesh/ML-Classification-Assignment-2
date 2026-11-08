
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Machine Learning Classification Dashboard")
st.write("BITS M.Tech DSE - Machine Learning Assignment 2")

st.info(
    "Compare five classification models using the "
    "Wisconsin Breast Cancer dataset."
)

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

selected_model = st.selectbox(
    "Select Machine Learning Model",
    list(model_files.keys())
)

uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.dataframe(df.head())

    if "target" not in df.columns:
        st.error(
            "The uploaded CSV must contain a target column "
            "to calculate evaluation metrics."
        )
        st.stop()

    X = df.drop(columns=["target"])
    y = df["target"]

    model_path = (
        Path(__file__).resolve().parent
        / "model"
        / model_files[selected_model]
    )

    model = joblib.load(model_path)

    expected_features = list(model.feature_names_in_)

    if set(X.columns) != set(expected_features):
        st.error("CSV feature columns do not match the trained model.")
        st.stop()

    X = X[expected_features]

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    st.subheader("Evaluation Metrics")

    accuracy = accuracy_score(y, predictions)
    auc = roc_auc_score(y, probabilities)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)
    mcc = matthews_corrcoef(y, predictions)

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("AUC", f"{auc:.4f}")
    col3.metric("Precision", f"{precision:.4f}")

    col4, col5, col6 = st.columns(3)

    col4.metric("Recall", f"{recall:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots()

    ConfusionMatrixDisplay.from_predictions(
        y,
        predictions,
        labels=[0, 1],
        display_labels=["Malignant", "Benign"],
        cmap="Blues",
        ax=ax
    )

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Prediction Results")

    output = df.copy()
    output["Predicted Target"] = predictions

    st.dataframe(output)

else:
    st.info("Upload test_data.csv to view model performance.")
