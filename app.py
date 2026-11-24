
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from huggingface_hub import hf_hub_download


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Banking Risk Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Banking & Financial Risk Prediction System")
st.caption("Machine Learning Based Financial Analytics Dashboard")


# ==================================================
# HUGGING FACE CONFIGURATION
# ==================================================

REPO_ID = "AbhiNeural/banking-financial-risk-artifacts"

ROOT = Path(__file__).resolve().parent


def get_file_path(relative_path):

    local_path = ROOT / relative_path

    # Use local file if available
    if local_path.exists():
        return str(local_path)

    # Otherwise download from Hugging Face
    try:
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=relative_path,
            repo_type="model"
        )

        return downloaded_path

    except Exception as e:
        st.error(f"Unable to load file: {relative_path}")
        st.error(str(e))
        st.stop()


# ==================================================
# LOAD DATASET
# ==================================================

@st.cache_data
def load_data():

    data_path = get_file_path(
        "data/processed/cleaned_data.csv"
    )

    return pd.read_csv(data_path)


@st.cache_data
def load_segment_data():

    segment_path = get_file_path(
        "data/processed/customer_segments.csv"
    )

    return pd.read_csv(segment_path)


# ==================================================
# LOAD TRAINED MODELS
# ==================================================

@st.cache_resource
def load_models():

    with st.spinner("Loading trained machine learning models..."):

        regression = joblib.load(
            get_file_path("models/regression_model.pkl")
        )

        classification = joblib.load(
            get_file_path("models/classification_model.pkl")
        )

        clustering = joblib.load(
            get_file_path("models/clustering_model.pkl")
        )

        metadata = joblib.load(
            get_file_path("models/clustering_metadata.pkl")
        )

    return regression, classification, clustering, metadata


# ==================================================
# INITIALIZE APPLICATION DATA
# ==================================================

try:

    df = load_data()

    regression_data, classification_data, clustering_data, metadata = (
        load_models()
    )

except Exception as e:

    st.error("Application initialization failed.")
    st.exception(e)
    st.stop()


# Regression model details
regression_model = regression_data["model"]
regression_features = regression_data["features"]


# Classification model details
classification_model = classification_data["model"]
classification_features = classification_data["features"]


# Clustering model details
clustering_model = clustering_data["model"]
clustering_scaler = clustering_data["scaler"]
clustering_features = clustering_data["features"]


# Clustering metadata
cluster_risk_map = metadata["cluster_risk_map"]
cluster_default_rates = metadata["cluster_default_rates"]


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Dashboard",
    [
        "Overview",
        "Classification",
        "Regression",
        "Clustering"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Banking Financial Risk Prediction | ML Project"
)


# ==================================================
# OVERVIEW DASHBOARD
# ==================================================

if page == "Overview":

    st.header("📊 Banking Analytics Overview")

    total_customers = len(df)

    total_defaults = int(df["DEFAULT"].sum())

    non_defaults = total_customers - total_defaults

    default_rate = df["DEFAULT"].mean() * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Defaulters",
        f"{total_defaults:,}"
    )

    col3.metric(
        "Non-Defaulters",
        f"{non_defaults:,}"
    )

    col4.metric(
        "Default Rate",
        f"{default_rate:.2f}%"
    )

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        st.subheader("Default Distribution")

        counts = df["DEFAULT"].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.bar(
            ["Non-Default", "Default"],
            counts.values
        )

        ax.set_ylabel("Customers")
        ax.set_title("Customer Default Distribution")

        st.pyplot(fig)
        plt.close(fig)

    with right:

        st.subheader("Customer Age Distribution")

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.histplot(
            df["AGE"],
            bins=30,
            kde=True,
            ax=ax
        )

        ax.set_title("Age Distribution")

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ==================================================
# CUSTOMER SELECTION
# ==================================================

elif page in ["Classification", "Regression", "Clustering"]:

    st.sidebar.markdown("### Customer Selection")

    customer_index = st.sidebar.number_input(
        "Select Customer Record",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
    )

    customer = df.iloc[[int(customer_index)]]

    st.header(
        f"Customer Record #{int(customer_index) + 1}"
    )


# ==================================================
# CLASSIFICATION
# ==================================================

if page == "Classification":

    st.subheader("🔍 Customer Default Prediction")

    X = customer[classification_features]

    prediction = classification_model.predict(X)[0]

    probability = classification_model.predict_proba(X)[0][1]

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Default Probability",
            f"{probability * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Predicted Class",
            "Default" if prediction == 1 else "No Default"
        )

    if prediction == 1:

        st.error(
            "Model predicts possible payment default."
        )

    else:

        st.success(
            "Model predicts no payment default."
        )

    st.progress(float(probability))

    st.caption(
        "This is an educational model prediction, "
        "not a lending decision."
    )

    st.subheader("Customer Financial Information")

    st.dataframe(
        customer,
        use_container_width=True
    )


# ==================================================
# REGRESSION
# ==================================================

elif page == "Regression":

    st.subheader("📈 Historical Bill Amount Estimation")

    X = customer[regression_features]

    predicted_bill = regression_model.predict(X)[0]

    actual_bill = customer["BILL_AMT6"].iloc[0]

    col1, col2 = st.columns(2)

    col1.metric(
        "Estimated Historical Bill Amount",
        f"{predicted_bill:,.2f}"
    )

    col2.metric(
        "Actual Historical Bill Amount",
        f"{actual_bill:,.2f}"
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        ["Actual", "Estimated"],
        [actual_bill, predicted_bill]
    )

    ax.set_title("Historical Bill Amount Comparison")
    ax.set_ylabel("Bill Amount")

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "This model estimates the historical BILL_AMT6 value "
        "from other customer features. "
        "It is not a future billing forecast."
    )


# ==================================================
# CLUSTERING
# ==================================================

elif page == "Clustering":

    st.subheader("👥 Customer Risk Segmentation")

    X = customer[clustering_features]

    X_scaled = clustering_scaler.transform(X)

    cluster = int(
        clustering_model.predict(X_scaled)[0]
    )

    risk_segment = cluster_risk_map[cluster]

    default_rate = cluster_default_rates[cluster] * 100

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Cluster",
        cluster
    )

    col2.metric(
        "Risk Segment",
        risk_segment
    )

    col3.metric(
        "Observed Default Rate",
        f"{default_rate:.2f}%"
    )

    if risk_segment == "High Risk":

        st.warning(
            "Customer belongs to the higher observed-default-rate segment."
        )

    elif risk_segment == "Medium Risk":

        st.info(
            "Customer belongs to the medium observed-default-rate segment."
        )

    else:

        st.success(
            "Customer belongs to the lower observed-default-rate segment."
        )

    st.subheader("Customer Segment Distribution")

    segment_data = load_segment_data()

    segment_counts = (
        segment_data["RISK_SEGMENT"].value_counts()
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    segment_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Risk Segment")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customer Segmentation")

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "Risk labels are assigned using historical default rates "
        "within each cluster. They are exploratory, "
        "not definitive risk decisions."
    )


# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "Banking & Financial Risk Prediction System | "
    "Python | Scikit-learn | Pandas | Streamlit | Hugging Face"
)
