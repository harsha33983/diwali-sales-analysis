"""
Model Loader Utility Module with Streamlit Caching.
Loads trained scikit-learn pipelines, model evaluation metrics, and handles live prediction queries.
"""

import os
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np

BEST_MODEL_PATH = os.path.join("models", "best_model.pkl")
ALL_MODELS_PATH = os.path.join("models", "all_models.pkl")
COMPARISON_PATH = os.path.join("models", "model_comparison.json")
IMPORTANCE_PATH = os.path.join("models", "feature_importance.json")


@st.cache_resource(show_spinner=False)
def load_best_model():
    """Load the persisted best model pipeline."""
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"Model artifact not found at {BEST_MODEL_PATH}")
    return joblib.load(BEST_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_all_models():
    """Load all trained candidate pipelines for comparative testing."""
    if not os.path.exists(ALL_MODELS_PATH):
        return {}
    return joblib.load(ALL_MODELS_PATH)


@st.cache_data(show_spinner=False)
def load_model_metadata():
    """Load model evaluation comparison and feature importances."""
    comp = {}
    feats = []
    if os.path.exists(COMPARISON_PATH):
        with open(COMPARISON_PATH, "r") as f:
            comp = json.load(f)
    if os.path.exists(IMPORTANCE_PATH):
        with open(IMPORTANCE_PATH, "r") as f:
            feats = json.load(f)
    return comp, feats


def predict_purchase_amount(
    pipeline,
    gender: str,
    age: int,
    age_group: str,
    marital_status: int,
    state: str,
    zone: str,
    occupation: str,
    product_category: str,
    orders: int
) -> dict:
    """
    Format user input, run inference through the trained pipeline,
    and return point prediction plus realistic statistical confidence intervals.
    """
    input_df = pd.DataFrame([{
        "Age": int(age),
        "Orders": int(orders),
        "Gender": gender,
        "Age Group": age_group,
        "Marital_Status": int(marital_status),
        "State": state,
        "Zone": zone,
        "Occupation": occupation,
        "Product_Category": product_category
    }])

    # Run pipeline inference
    pred_val = float(pipeline.predict(input_df)[0])
    
    # Clip prediction to sensible bounds (minimum price in dataset is Rs 188)
    clean_pred = max(188.0, round(pred_val, 2))

    # Standard model error margin (using test MAE ~ Rs 2372)
    error_margin = 2372.0
    lower_bound = max(188.0, round(clean_pred - error_margin, 2))
    upper_bound = round(clean_pred + error_margin, 2)

    return {
        "predicted_amount": clean_pred,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "input_features": input_df.to_dict(orient="records")[0]
    }

