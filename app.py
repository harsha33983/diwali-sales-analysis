"""
Diwali Sales Analysis & Customer Purchase Prediction System.
Main Application Entrypoint.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
from utils.data_loader import load_cleaned_data
from components.metrics import render_page_header, render_hero_box, apply_custom_css
from components.charts import plot_revenue_by_state, plot_category_revenue, plot_sales_by_gender

st.set_page_config(
    page_title="Diwali Sales Intelligence | Enterprise BI & ML",
    page_icon="🪔",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

# Load cleaned data
try:
    df = load_cleaned_data()
except Exception as e:
    st.error(f"Error loading data: {e}. Please ensure data/processed/diwali_sales_cleaned.csv exists.")
    st.stop()

# Header
render_page_header(
    title="Diwali Festival <span>Sales Intelligence</span> & ML Platform",
    subtitle="End-to-End Enterprise Retail Analytics, Customer Segmentation & Machine Learning Purchase Forecasting",
    badge="Production Analytics Suite"
)

# Hero Metric Box
total_rev = df["Amount"].sum()
total_orders = df["Orders"].sum()
total_customers = df["User_ID"].nunique()
top_state = df.groupby("State")["Amount"].sum().idxmax()
top_cat = df.groupby("Product_Category")["Amount"].sum().idxmax()

render_hero_box(
    title="GROSS FESTIVAL TURNOVER",
    value=f"₹{total_rev/1e7:.2f} Cr",
    total_raw=f"₹{total_rev:,.0f}",
    subtitle=f"Aggregated sales across {len(df):,} verified festive transactions and {total_customers:,} unique shoppers.",
    highlights=[
        f"🏆 Top State: {top_state} (18.2% share)",
        f"🛍️ Top Category: {top_cat} (32.0% share)",
        "👩 Dominant Demographic: Female 26–35 (29.1% share)",
        f"📦 Total Units Sold: {total_orders:,} units"
    ]
)

# Executive Welcome & Platform Architecture
st.markdown("---")
col_intro1, col_intro2 = st.columns([3, 2])

with col_intro1:
    st.markdown(
        """
        ### 🎯 Platform Capabilities & Business Scope
        This enterprise platform integrates real-world festive retail transaction logs to provide data-driven commercial intelligence:
        
        - **Phase 1-3: Rigorous Data Quality & EDA**: 11,231 verified transaction records across 16 Indian states and 18 product categories.
        - **Phase 4: Customer Persona Clustering**: Unsupervised K-Means clustering identifying 4 high-value customer segments (from VIP Festive Whales to Young Value Shoppers).
        - **Phase 5: Demand & Merchandise Analytics**: Empirical category demand trends, SKU velocity, and sales-driven inventory allocation.
        - **Phase 6-7: Production Machine Learning Engine**: Scikit-Learn regression pipelines predicting transaction purchase amounts (`Amount`) with transparent evaluation (R² = 0.6385, MAE = ₹2,372).
        - **Phase 8: Multi-Page Business Intelligence UI**: Interactive multi-page exploration, dynamic cross-filtering, and on-demand CSV exports.
        """
    )

with col_intro2:
    st.markdown(
        f"""
        <div style="background: #1E293B; border: 1px solid #F59E0B; border-radius: 12px; padding: 20px;">
            <h4 style="color: #F59E0B; margin-top: 0;">⚡ Fast Navigation</h4>
            <p style="color: #94A3B8; font-size: 13px;">Explore specialized analytic modules via the sidebar or direct links below:</p>
            <ul style="color: #E2E8F0; font-size: 13px; line-height: 1.8;">
                <li><b>1. Executive Overview</b>: High-level KPI summaries and market distribution</li>
                <li><b>2. Sales Analytics</b>: Multi-dimensional cohort filtering</li>
                <li><b>3. Customer Segmentation</b>: RFM & K-Means persona clustering</li>
                <li><b>4. Product & Demand Analysis</b>: SKU performance & demand allocation</li>
                <li><b>5. ML Purchase Prediction</b>: Interactive real-time regression inference</li>
                <li><b>6. Data Explorer</b>: Tabular dataset search, inspection & CSV download</li>
                <li><b>7. Strategic Insights</b>: Dynamically generated executive intelligence</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")
st.subheader("📊 Executive Festival Pulse")
pulse_col1, pulse_col2, pulse_col3 = st.columns([1.2, 1, 1.2])

with pulse_col1:
    st.plotly_chart(plot_revenue_by_state(df, top_n=7), use_container_width=True)

with pulse_col2:
    st.plotly_chart(plot_sales_by_gender(df), use_container_width=True)

with pulse_col3:
    st.plotly_chart(plot_category_revenue(df, top_n=7), use_container_width=True)

st.info("💡 **Navigation Tip**: Select any page from the sidebar on the left to dive into specific analytics, test ML predictions, or inspect data quality.")

