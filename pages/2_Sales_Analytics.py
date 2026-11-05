"""
Page 2: Sales Analytics.
Deep-dive multi-dimensional sales analysis with interactive cross-filtering.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
from utils.data_loader import load_cleaned_data
from components.metrics import render_page_header, render_kpi_grid
from components.sidebar import render_sidebar
from components.charts import (
    plot_revenue_by_state,
    plot_sales_by_gender,
    plot_sales_by_age_gender,
    plot_occupation_sales,
    plot_category_revenue,
    plot_top_products
)

st.set_page_config(page_title="Sales Analytics | Diwali Sales BI", page_icon="🪔", layout="wide")

raw_cleaned_df = load_cleaned_data()
filtered_df = render_sidebar(raw_cleaned_df)

render_page_header(
    title="Festival <span>Sales Analytics</span>",
    subtitle="Interactive cohort exploration across geography, demographics, and merchandise hierarchies",
    badge="Multi-Dimensional Deep Dive"
)

if filtered_df.empty:
    st.warning("⚠️ No records match your selected filter criteria. Please adjust or reset the sidebar filters.")
    st.stop()

# Responsive KPI Cards for the active cohort
total_rev = filtered_df["Amount"].sum()
total_orders = filtered_df["Orders"].sum()
trans_count = len(filtered_df)
avg_spend = filtered_df["Amount"].mean()
top_state = filtered_df.groupby("State")["Amount"].sum().idxmax()
top_cat = filtered_df.groupby("Product_Category")["Amount"].sum().idxmax()

cohort_kpis = [
    {"label": "Filtered Revenue", "value": f"₹{total_rev:,.0f}", "subtext": "Cohort share:", "highlight": f"{(total_rev/raw_cleaned_df['Amount'].sum()*100):.1f}%"},
    {"label": "Filtered Transactions", "value": f"{trans_count:,}", "subtext": "Total records:", "highlight": f"{len(raw_cleaned_df):,}"},
    {"label": "Total Units Ordered", "value": f"{total_orders:,}", "subtext": "Avg per basket:", "highlight": f"{(total_orders/trans_count):.1f} units"},
    {"label": "Mean Basket Size", "value": f"₹{avg_spend:,.0f}", "subtext": "Median basket:", "highlight": f"₹{filtered_df['Amount'].median():,.0f}"},
    {"label": "Top State (Filtered)", "value": top_state, "subtext": "Cohort Top State", "highlight": ""},
    {"label": "Top Category (Filtered)", "value": top_cat, "subtext": "Cohort Top Category", "highlight": ""},
]
render_kpi_grid(cohort_kpis)

# Tabs for Structured Exploration
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Geographic & Demographics", "👔 Occupation & Lifestyle", "🛍️ Product & Merchandise", "📋 Cohort Data Summary"])

with tab1:
    col_g1, col_g2 = st.columns([1.2, 1])
    with col_g1:
        st.plotly_chart(plot_revenue_by_state(filtered_df, top_n=10), use_container_width=True)
    with col_g2:
        st.plotly_chart(plot_sales_by_gender(filtered_df), use_container_width=True)

    st.markdown("---")
    st.plotly_chart(plot_sales_by_age_gender(filtered_df), use_container_width=True)

    # Marital Status analysis
    st.markdown("#### 💍 Marital Status & Gender Purchasing Breakdown")
    ms_agg = filtered_df.groupby(["Marital_Status_Desc", "Gender_Desc"]).agg(
        Transactions=("User_ID", "count"),
        Total_Revenue=("Amount", "sum"),
        Mean_Spend=("Amount", "mean")
    ).reset_index()
    ms_agg["Revenue_Formatted"] = ms_agg["Total_Revenue"].apply(lambda x: f"₹{x:,.2f}")
    ms_agg["Mean_Spend_Formatted"] = ms_agg["Mean_Spend"].apply(lambda x: f"₹{x:,.2f}")
    st.dataframe(ms_agg[["Marital_Status_Desc", "Gender_Desc", "Transactions", "Revenue_Formatted", "Mean_Spend_Formatted"]], use_container_width=True)

with tab2:
    st.markdown("#### 💼 Occupation-Based Purchasing Power")
    col_occ1, col_occ2 = st.columns([1.3, 1])
    with col_occ1:
        st.plotly_chart(plot_occupation_sales(filtered_df), use_container_width=True)
    with col_occ2:
        occ_table = filtered_df.groupby("Occupation").agg(
            Transactions=("User_ID", "count"),
            Total_Spend=("Amount", "sum"),
            Avg_Ticket=("Amount", "mean")
        ).sort_values(by="Total_Spend", ascending=False).reset_index()
        occ_table["Total_Spend_INR"] = occ_table["Total_Spend"].apply(lambda x: f"₹{x:,.0f}")
        occ_table["Avg_Ticket_INR"] = occ_table["Avg_Ticket"].apply(lambda x: f"₹{x:,.0f}")
        st.dataframe(occ_table[["Occupation", "Transactions", "Total_Spend_INR", "Avg_Ticket_INR"]], use_container_width=True, height=420)

with tab3:
    st.markdown("#### 📦 Merchandise Category & SKU Velocity")
    col_cat1, col_cat2 = st.columns([1.2, 1])
    with col_cat1:
        st.plotly_chart(plot_category_revenue(filtered_df, top_n=10), use_container_width=True)
    with col_cat2:
        st.plotly_chart(plot_top_products(filtered_df, top_n=10), use_container_width=True)

with tab4:
    st.markdown("#### 📊 Filtered Cohort Data Matrix")
    st.dataframe(
        filtered_df[[
            "User_ID", "Cust_name", "Gender_Desc", "Age Group", "State", 
            "Occupation", "Product_Category", "Orders", "Amount"
        ]].head(100),
        use_container_width=True
    )
    st.caption(f"Displaying top 100 rows out of {len(filtered_df):,} filtered transactions.")

