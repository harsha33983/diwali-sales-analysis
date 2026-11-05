"""
Page 1: Executive Overview.
High-level strategic KPIs, revenue distributions, and dynamic executive briefing.
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
from components.charts import (
    plot_revenue_by_state,
    plot_sales_by_gender,
    plot_sales_by_age_gender,
    plot_category_revenue,
    plot_amount_distribution
)
from src.business_insights import compute_comprehensive_insights

st.set_page_config(page_title="Executive Overview | Diwali Sales BI", page_icon="🪔", layout="wide")

df = load_cleaned_data()
ins = compute_comprehensive_insights(df)

render_page_header(
    title="Executive <span>Festival Overview</span>",
    subtitle="Macro performance metrics, revenue drivers, and high-level behavioral distributions",
    badge="Executive Briefing"
)

# 1. KPI Cards Row
kpis = [
    {"label": "Total Revenue", "value": f"₹{ins.get('total_revenue', 0)/1e7:.2f} Cr", "subtext": "Gross INR:", "highlight": f"₹{ins.get('total_revenue', 0):,.0f}"},
    {"label": "Total Orders Placed", "value": f"{ins.get('total_orders', 0):,}", "subtext": "Units ordered in:", "highlight": f"{ins.get('total_transactions', 0):,} transactions"},
    {"label": "Total Unique Customers", "value": f"{ins.get('unique_customers', 0):,}", "subtext": "Repeat rate:", "highlight": f"{(ins.get('total_transactions', 1)/max(1, ins.get('unique_customers', 1))):.2f} trans/cust"},
    {"label": "Average Order Value", "value": f"₹{df['Amount'].mean():,.0f}", "subtext": "Median basket:", "highlight": f"₹{df['Amount'].median():,.0f}"},
    {"label": "Top Performing State", "value": ins.get("top_state", "N/A"), "subtext": "Revenue share:", "highlight": f"{ins.get('top_state_pct', 0):.1f}%"},
    {"label": "Top Product Category", "value": ins.get("top_cat", "N/A"), "subtext": "Revenue share:", "highlight": f"{ins.get('top_cat_pct', 0):.1f}%"},
]
render_kpi_grid(kpis)

# 2. Dynamic Business Summary
st.markdown("### 📋 Executive Analytical Summary")
st.markdown(
    f"""
    <div style="background: #1E293B; border-left: 4px solid #E67E22; border-radius: 8px; padding: 18px 22px; margin-bottom: 24px; color: #E2E8F0; font-size: 14px; line-height: 1.7;">
        <b>Dynamic Operational Synthesis:</b><br/>
        During this Diwali festival cycle, total gross revenue reached <b>₹{ins.get('total_revenue', 0):,.2f}</b> across <b>{ins.get('total_transactions', 0):,} transactions</b>. 
        Commercial momentum was primarily anchored by <b>{ins.get('top_state', 'N/A')}</b> (<b>₹{ins.get('top_state_rev', 0):,.2f}</b> or <b>{ins.get('top_state_pct', 0):.1f}%</b> of national sales) and the <b>{ins.get('top_zone', 'N/A')} Zone</b>. 
        Demographically, <b>Female consumers drove {ins.get('female_pct', 0):.1f}% of total spend</b>, with the <b>{ins.get('top_demo_segment', 'N/A')}</b> cohort serving as the primary commercial engine (<b>{ins.get('top_demo_pct', 0):.1f}%</b> of total turnover). 
        The top merchandise department was <b>{ins.get('top_cat', 'N/A')}</b>, delivering <b>₹{ins.get('top_cat_rev', 0):,.2f}</b>, followed closely by Clothing & Apparel and Electronics.
    </div>
    """,
    unsafe_allow_html=True
)

# 3. Macro Visualizations
col_row1_1, col_row1_2 = st.columns([1.2, 1])
with col_row1_1:
    st.plotly_chart(plot_revenue_by_state(df, top_n=10), use_container_width=True)
with col_row1_2:
    st.plotly_chart(plot_sales_by_gender(df), use_container_width=True)

col_row2_1, col_row2_2 = st.columns([1.2, 1])
with col_row2_1:
    st.plotly_chart(plot_sales_by_age_gender(df), use_container_width=True)
with col_row2_2:
    st.plotly_chart(plot_category_revenue(df, top_n=8), use_container_width=True)

st.markdown("### 💰 Purchase Amount Distribution & Spread")
st.plotly_chart(plot_amount_distribution(df), use_container_width=True)

