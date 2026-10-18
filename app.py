"""
FraudGuard-ML — Streamlit application for batch fraud prediction.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "models" / "fraud_model.pkl"

REQUIRED_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifact(path: Path) -> dict:
    """Load the saved model artifact (cached across Streamlit reruns)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{path}'. "
            "Run `python train.py` first to train and save the model."
        )
    return joblib.load(path)


# ---------------------------------------------------------------------------
# CSV validation
# ---------------------------------------------------------------------------
def validate_upload(df: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    """
    Validate uploaded transaction CSV.

    Returns (prepared_dataframe, error_message).
    On success error_message is None; on failure prepared_dataframe is None.
    """
    if df is None or df.empty:
        return None, "The uploaded file is empty."

    missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if missing:
        return None, (
            "Missing required columns: "
            + ", ".join(missing)
            + ". Expected features: Time, V1–V28, Amount."
        )

    prepared = df[REQUIRED_FEATURES].copy()

    for col in REQUIRED_FEATURES:
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce")

    invalid_mask = prepared.isna().any(axis=1)
    n_invalid = int(invalid_mask.sum())
    if n_invalid == len(prepared):
        return None, (
            "All rows contain non-numeric or missing values in required features."
        )
    if n_invalid > 0:
        # Drop invalid rows but continue with the rest
        prepared = prepared.loc[~invalid_mask].reset_index(drop=True)

    return prepared, (
        None
        if n_invalid == 0
        else f"Dropped {n_invalid} row(s) with invalid numeric values."
    )


def run_predictions(
    model,
    features: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Generate fraud probabilities and labels using the saved threshold."""
    probabilities = model.predict_proba(features)[:, 1]
    labels = np.where(
        probabilities >= threshold,
        "Suspicious",
        "Legitimate",
    )
    results = features.copy()
    results["Fraud_Probability"] = np.round(probabilities, 6)
    results["Prediction"] = labels
    return results


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="FraudGuard-ML",
        layout="wide",
    )

    st.title("FraudGuard-ML")
    st.markdown("*Machine Learning-Based Credit Card Fraud Detection*")
    st.divider()

    # Load model
    try:
        artifact = load_artifact(ARTIFACT_PATH)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info(
            "Train the model with:\n\n```powershell\npython train.py\n```"
        )
        return
    except Exception:
        st.error(
            "Failed to load the model artifact. "
            "Try retraining with `python train.py`."
        )
        return

    model = artifact["model"]
    threshold = float(artifact["threshold"])
    model_name = artifact.get("model_name", "Unknown")
    feature_names = artifact.get("feature_names", REQUIRED_FEATURES)

    st.sidebar.header("Model Info")
    st.sidebar.write(f"**Selected model:** {model_name}")
    st.sidebar.write(f"**Decision threshold:** {threshold:.4f}")
    st.sidebar.caption(
        "Threshold was chosen on the validation set to maximize F1-Score."
    )

    st.subheader("Project Overview")
    st.write(
        "Upload a CSV of credit card transactions to estimate fraud risk. "
        "The application applies the trained model and a validation-tuned "
        "classification threshold, then lets you download the results."
    )

    uploaded = st.file_uploader("Upload Transaction CSV", type=["csv"])

    if uploaded is None:
        st.info(
            "Upload a CSV containing columns: **Time**, **V1–V28**, **Amount**. "
            "A `Class` column is optional and will be ignored."
        )
        # Empty dashboard metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Selected Model", model_name)
        c2.metric("Transactions Analyzed", 0)
        c3.metric("Suspicious", 0)
        c4.metric("Flagged %", "0.00%")
        return

    # Parse CSV
    try:
        raw_df = pd.read_csv(uploaded)
    except Exception:
        st.error(
            "Could not read the uploaded file. "
            "Please upload a valid CSV file."
        )
        return

    prepared, warning = validate_upload(raw_df)
    if prepared is None:
        st.error(warning)
        return
    if warning:
        st.warning(warning)

    # Ensure column order matches training
    prepared = prepared[feature_names]

    try:
        results = run_predictions(model, prepared, threshold)
    except Exception:
        st.error(
            "Prediction failed. Ensure the CSV columns match the "
            "training feature schema (Time, V1–V28, Amount)."
        )
        return

    n_total = len(results)
    n_suspicious = int((results["Prediction"] == "Suspicious").sum())
    n_legitimate = n_total - n_suspicious
    pct_flagged = (n_suspicious / n_total * 100) if n_total else 0.0

    # Metrics dashboard
    st.subheader("Analysis Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected Model", model_name)
    m2.metric("Transactions Analyzed", f"{n_total:,}")
    m3.metric("Suspicious", f"{n_suspicious:,}")
    m4.metric("Flagged %", f"{pct_flagged:.2f}%")

    # Results table
    st.subheader("Prediction Results")
    display_df = pd.DataFrame(
        {
            "Transaction": np.arange(1, n_total + 1),
            "Fraud Probability": results["Fraud_Probability"],
            "Prediction": results["Prediction"],
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Visualization
    st.subheader("Prediction Distribution")
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.metric("Legitimate", f"{n_legitimate:,}")
        st.metric("Suspicious", f"{n_suspicious:,}")

    with col_right:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        counts = [n_legitimate, n_suspicious]
        colors = ["#2E7D32", "#C62828"]
        bars = ax.bar(["Legitimate", "Suspicious"], counts, color=colors)
        ax.set_ylabel("Count")
        ax.set_title("Fraud Prediction Distribution")
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{count:,}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Download
    st.subheader("Download Results")
    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions CSV",
        data=csv_bytes,
        file_name="fraudguard_predictions.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
