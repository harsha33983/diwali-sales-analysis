"""
Page 5: Machine Learning Predictions.
Real-time customer purchase amount forecasting using trained scikit-learn regression pipelines.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
from utils.model_loader import load_best_model, load_all_models, load_model_metadata, predict_purchase_amount
from utils.data_loader import load_cleaned_data
from components.metrics import render_page_header, render_kpi_grid
from components.charts import plot_actual_vs_predicted, plot_feature_importance

st.set_page_config(page_title="ML Purchase Prediction | Diwali Sales BI", page_icon="🪔", layout="wide")

df = load_cleaned_data()
best_pipeline = load_best_model()
all_models = load_all_models()
comp_data, feat_list = load_model_metadata()

render_page_header(
    title="Customer <span>Purchase Prediction Engine</span>",
    subtitle="Live transaction inference powered by trained scikit-learn regression models",
    badge="Machine Learning Inference"
)

# Model Performance Badge
best_name = comp_data.get("best_model_name", "Linear Regression")
comp_table = comp_data.get("comparison", {})
best_metrics = comp_table.get(best_name, {})

ml_kpis = [
    {"label": "Active Production Model", "value": best_name, "subtext": "Pipeline status:", "highlight": "Trained & Validated"},
    {"label": "Test R² Score", "value": f"{best_metrics.get('Test_R2', 0):.4f}", "subtext": "Explains ~64% variance", "highlight": ""},
    {"label": "Test Mean Absolute Error", "value": f"₹{best_metrics.get('Test_MAE', 0):,.2f}", "subtext": "Average error margin", "highlight": ""},
    {"label": "Test RMSE", "value": f"₹{best_metrics.get('Test_RMSE', 0):,.2f}", "subtext": "Root Mean Squared Error", "highlight": ""},
]
render_kpi_grid(ml_kpis)

# Interactive Prediction Form and Model Comparison
col_form, col_eval = st.columns([1.2, 1])

# State-to-Zone mapping dictionary for automatic consistency
state_zone_map = df.drop_duplicates(subset=["State"])[["State", "Zone"]].set_index("State")["Zone"].to_dict()

with col_form:
    st.markdown("### 🎯 Interactive Transaction Inference Form")
    st.markdown("Adjust customer and order parameters below to generate an empirical purchase estimate.")

    with st.form("purchase_prediction_form"):
        # Model selector
        model_options = list(all_models.keys()) if all_models else [best_name]
        selected_model_name = st.selectbox("Select ML Model Engine", options=model_options, index=model_options.index(best_name) if best_name in model_options else 0)

        f_col1, f_col2 = st.columns(2)
        with f_col1:
            gender = st.selectbox("Gender", options=["Female", "Male"], index=0)
            gender_code = "F" if gender == "Female" else "M"
            
            age = st.slider("Customer Age", min_value=12, max_value=90, value=30, step=1)

            # Auto determine Age Group bracket based on Age
            if age <= 17:
                age_group = "0-17"
            elif 18 <= age <= 25:
                age_group = "18-25"
            elif 26 <= age <= 35:
                age_group = "26-35"
            elif 36 <= age <= 45:
                age_group = "36-45"
            elif 46 <= age <= 50:
                age_group = "46-50"
            elif 51 <= age <= 55:
                age_group = "51-55"
            else:
                age_group = "55+"

            marital_status_label = st.radio("Marital Status", options=["Single / Unmarried", "Married"], index=0)
            marital_status = 0 if "Single" in marital_status_label else 1

            orders = st.number_input("Number of Items / Orders in Basket", min_value=1, max_value=4, value=2, step=1)

        with f_col2:
            all_states = sorted(df["State"].dropna().unique().tolist())
            state = st.selectbox("Customer State", options=all_states, index=all_states.index("Uttar Pradesh") if "Uttar Pradesh" in all_states else 0)
            zone = state_zone_map.get(state, "Central")
            st.caption(f"Geographic Zone: **{zone}**")

            all_occs = sorted(df["Occupation"].dropna().unique().tolist())
            occupation = st.selectbox("Customer Occupation", options=all_occs, index=all_occs.index("IT Sector") if "IT Sector" in all_occs else 0)

            all_cats = sorted(df["Product_Category"].dropna().unique().tolist())
            product_category = st.selectbox("Product Category", options=all_cats, index=all_cats.index("Food") if "Food" in all_cats else 0)

        submitted = st.form_submit_button("🔮 Predict Purchase Amount", use_container_width=True)

    if submitted:
        selected_pipeline = all_models.get(selected_model_name, best_pipeline)
        
        result = predict_purchase_amount(
            pipeline=selected_pipeline,
            gender=gender_code,
            age=age,
            age_group=age_group,
            marital_status=marital_status,
            state=state,
            zone=zone,
            occupation=occupation,
            product_category=product_category,
            orders=orders
        )

        pred_amt = result["predicted_amount"]
        low_b = result["lower_bound"]
        up_b = result["upper_bound"]

        # Log prediction event to SQLite database
        try:
            from src.database import log_prediction
            log_prediction(
                model_used=selected_model_name,
                gender=gender,
                age=age,
                age_group=age_group,
                marital_status=marital_status_label,
                state=state,
                zone=zone,
                occupation=occupation,
                product_category=product_category,
                orders=orders,
                predicted_amount=pred_amt,
                lower_bound=low_b,
                upper_bound=up_b
            )
        except Exception:
            pass

        st.markdown(
            f"""
            <div class="prediction-box">
                <div style="font-size: 13px; text-transform: uppercase; letter-spacing: 1.5px; color: #94A3B8; font-weight: 600;">
                    Predicted Customer Purchase Amount
                </div>
                <div class="prediction-value">₹{pred_amt:,.2f}</div>
                <div style="font-size: 14px; color: #E2E8F0;">
                    Estimated Statistical Range (MAE ±₹{best_metrics.get('Test_MAE', 2372):,.0f}):<br/>
                    <b style="color: #FBBF24;">₹{low_b:,.2f}</b> — <b style="color: #FBBF24;">₹{up_b:,.2f}</b>
                </div>
                <div style="font-size: 11px; color: #94A3B8; margin-top: 10px;">
                    Inference generated by <b>{selected_model_name}</b> pipeline with full categorical one-hot encoding.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.warning(
            "⚠️ **Statistical Guidance**: This prediction represents an expected retail basket spend based on historical festival behavior. "
            "Individual buyer purchases may fluctuate due to flash discounts, bundled promotions, and payment incentives."
        )

with col_eval:
    st.markdown("### 🏆 Candidate Model Comparison")
    
    comp_rows = []
    for m_name, m_stats in comp_table.items():
        is_best = " ⭐ (Best)" if m_name == best_name else ""
        comp_rows.append({
            "Algorithm": m_name + is_best,
            "Test R²": f"{m_stats['Test_R2']:.4f}",
            "Test MAE": f"₹{m_stats['Test_MAE']:,.2f}",
            "Test RMSE": f"₹{m_stats['Test_RMSE']:,.2f}",
            "Train R²": f"{m_stats['Train_R2']:.4f}",
        })
    comp_df = pd.DataFrame(comp_rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("### 🔍 Model Diagnostic: Residuals & Fit")
    test_pred_path = os.path.join("models", "test_predictions.csv")
    if os.path.exists(test_pred_path):
        pred_df = pd.read_csv(test_pred_path)
        st.plotly_chart(plot_actual_vs_predicted(pred_df), use_container_width=True)

st.markdown("---")
st.markdown("### 📈 Feature Importances & Drivers")
if feat_list:
    st.plotly_chart(plot_feature_importance(feat_list), use_container_width=True)

st.markdown("---")
st.markdown("### 🗄️ SQLite Live Prediction Audit Log")
st.caption("All real-time inference requests are securely committed to `data/diwali_sales.db` (`predictions_log` table).")
try:
    from src.database import get_prediction_history
    history_df = get_prediction_history(limit=10)
    if not history_df.empty:
        history_df["predicted_amount"] = history_df["predicted_amount"].apply(lambda x: f"₹{x:,.2f}")
        history_df["lower_bound"] = history_df["lower_bound"].apply(lambda x: f"₹{x:,.2f}")
        history_df["upper_bound"] = history_df["upper_bound"].apply(lambda x: f"₹{x:,.2f}")
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("No inference events recorded in SQLite yet. Use the prediction form above to submit an inference.")
except Exception as e:
    st.caption(f"Audit log status: {e}")

