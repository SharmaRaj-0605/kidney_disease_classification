import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("KIDNEY_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Kidney CT Classifier",
    page_icon="🩺",
    layout="centered",
)

st.title("Kidney CT Image Classifier")
st.write(
    "Upload a kidney CT image to see the model's predicted class and confidence."
)
st.warning(
    "Educational project only. This model is not clinically validated and must not "
    "be used to diagnose a condition or make treatment decisions."
)

with st.sidebar:
    st.subheader("API status")
    st.caption(API_URL)
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=8)
        if health_response.ok:
            health = health_response.json()
            if health.get("model_available"):
                st.success("API is ready")
            else:
                st.warning(health.get("detail", "The trained model is not available."))
        else:
            st.error(f"API returned HTTP {health_response.status_code}.")
    except requests.RequestException:
        st.error("Cannot reach the API. Start the FastAPI service and check KIDNEY_API_URL.")

uploaded_file = st.file_uploader(
    "Choose a CT image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG and PNG.",
)

if uploaded_file is not None:
    st.image(uploaded_file.getvalue(), caption=uploaded_file.name, width="stretch")

if st.button("Classify image", type="primary", disabled=uploaded_file is None):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/octet-stream",
                )
            },
            timeout=(8, 90),
        )

        if response.status_code == 503:
            detail = response.json().get("detail", "The trained model is not available.")
            st.error(detail)
        else:
            response.raise_for_status()
            result = response.json()
            st.subheader("Prediction")
            left, right = st.columns(2)
            left.metric("Predicted class", result["prediction"])
            right.metric("Model confidence", f"{result['confidence']:.1%}")

            probability_frame = pd.DataFrame(
                {
                    "Class": list(result["probabilities"].keys()),
                    "Probability": list(result["probabilities"].values()),
                }
            ).set_index("Class")
            st.write("Class probabilities")
            st.bar_chart(probability_frame)
    except requests.Timeout:
        st.error("The prediction request timed out. Try again or check the API logs.")
    except requests.RequestException as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            try:
                detail = exc.response.json().get("detail", "")
            except ValueError:
                detail = exc.response.text
        st.error(detail or f"Could not complete the prediction request: {exc}")
