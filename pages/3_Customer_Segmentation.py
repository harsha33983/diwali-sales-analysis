"""
Page 3: Customer Segmentation.
Identifies and profiles customer groups using demographic analytics and unsupervised K-Means clustering.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
from utils.data_loader import load_customer_segments, load_cleaned_data
from components.metrics import render_page_header, render_kpi_grid
from components.charts import plot_customer_clusters

st.set_page_config(page_title="Customer Segmentation | Diwali Sales BI", page_icon="🪔", layout="wide")

df = load_cleaned_data()
cust_df = load_customer_segments()

render_page_header(
    title="Customer <span>Segmentation & Personas</span>",
    subtitle="Behavioral clustering and demographic profiling for targeted festive marketing and retention",
    badge="Machine Learning Clustering"
)

# High-Level Customer KPIs
total_customers = len(cust_df)
avg_spend_per_cust = cust_df["Total_Spend"].mean()
vip_count = (cust_df["Segment_Name"].str.contains("VIP", case=False)).sum()
vip_rev = cust_df[cust_df["Segment_Name"].str.contains("VIP", case=False)]["Total_Spend"].sum()
vip_rev_pct = (vip_rev / cust_df["Total_Spend"].sum()) * 100

seg_kpis = [
    {"label": "Unique Customer Base", "value": f"{total_customers:,}", "subtext": "Active festival shoppers", "highlight": ""},
    {"label": "Mean Customer Value", "value": f"₹{avg_spend_per_cust:,.0f}", "subtext": "Average total festival spend", "highlight": ""},
    {"label": "Identified Segments", "value": "4 Distinct Personas", "subtext": "K-Means model", "highlight": "k=4"},
    {"label": "VIP Festive Whales", "value": f"{vip_count:,} Shoppers", "subtext": f"Drives {vip_rev_pct:.1f}% total spend:", "highlight": f"₹{vip_rev/1e6:.1f}M"},
]
render_kpi_grid(seg_kpis)

# 1. Interactive Cluster Visualization
st.markdown("### 🧬 Customer Segmentation Map (K-Means)")
st.plotly_chart(plot_customer_clusters(cust_df), use_container_width=True)

# 2. Detailed Cluster Persona Profiles
st.markdown("### 👥 Detailed Customer Persona Profiles & Commercial Strategy")

col_c1, col_c2 = st.columns(2)

with col_c1:
    st.markdown(
        """
        <div style="background: #1E293B; border-top: 4px solid #EF4444; border-radius: 8px; padding: 18px; margin-bottom: 16px;">
            <h4 style="color: #F87171; margin: 0 0 8px 0;">👑 VIP Festive Whales (Ultra High Spend)</h4>
            <p style="font-size: 13px; color: #94A3B8; margin-bottom: 10px;">
                <b>Profile:</b> 192 customers (~5.1% of customer base) accounting for over ₹21.7 Million in festive turnover.<br/>
                <b>Average Spend:</b> ₹113,247 | <b>Average Transactions:</b> 11.6 | <b>Mean Age:</b> 35.7 years<br/>
                <b>Key Sectors:</b> Aviation, IT Sector, Healthcare, Banking.
            </p>
            <div style="background: rgba(239, 68, 68, 0.1); border-radius: 6px; padding: 10px; font-size: 12px; color: #FCA5A5;">
                <b>Actionable Playbook:</b> Provide white-glove VIP concierge services, preview access to limited-edition festive hampers, and personalized relationship manager outreach.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="background: #1E293B; border-top: 4px solid #F59E0B; border-radius: 8px; padding: 18px; margin-bottom: 16px;">
            <h4 style="color: #FBBF24; margin: 0 0 8px 0;">🛍️ High-Value Frequent Shoppers</h4>
            <p style="font-size: 13px; color: #94A3B8; margin-bottom: 10px;">
                <b>Profile:</b> 805 customers (~21.5% of customer base) delivering ₹40.7 Million in gross revenue.<br/>
                <b>Average Spend:</b> ₹50,664 | <b>Average Transactions:</b> 5.3 | <b>Mean Age:</b> 33.2 years<br/>
                <b>Key Sectors:</b> IT Sector, Healthcare, Banking, Media.
            </p>
            <div style="background: rgba(245, 158, 11, 0.1); border-radius: 6px; padding: 10px; font-size: 12px; color: #FDE68A;">
                <b>Actionable Playbook:</b> Deploy loyalty tier upgrades, multi-buy bundle rewards (e.g. "Buy ₹40,000 get ₹5,000 voucher"), and express delivery incentives.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_c2:
    st.markdown(
        """
        <div style="background: #1E293B; border-top: 4px solid #38BDF8; border-radius: 8px; padding: 18px; margin-bottom: 16px;">
            <h4 style="color: #38BDF8; margin: 0 0 8px 0;">🏡 Mature Moderate Buyers</h4>
            <p style="font-size: 13px; color: #94A3B8; margin-bottom: 10px;">
                <b>Profile:</b> 992 customers (~26.4% of base) with stable festive gift purchasing.<br/>
                <b>Average Spend:</b> ₹16,883 | <b>Average Transactions:</b> 1.8 | <b>Mean Age:</b> 52.5 years<br/>
                <b>Key Sectors:</b> Government, Healthcare, Banking, Agriculture.
            </p>
            <div style="background: rgba(56, 189, 248, 0.1); border-radius: 6px; padding: 10px; font-size: 12px; color: #BAE6FD;">
                <b>Actionable Playbook:</b> Target with traditional festive home decor, food sweets, religious puja gift sets, and family-oriented bundle packs.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="background: #1E293B; border-top: 4px solid #10B981; border-radius: 8px; padding: 18px; margin-bottom: 16px;">
            <h4 style="color: #34D399; margin: 0 0 8px 0;">🚀 Young Value Shoppers</h4>
            <p style="font-size: 13px; color: #94A3B8; margin-bottom: 10px;">
                <b>Profile:</b> 1,763 customers (~47.0% of base) representing the largest volume entry tier.<br/>
                <b>Average Spend:</b> ₹15,260 | <b>Average Transactions:</b> 1.7 | <b>Mean Age:</b> 28.1 years<br/>
                <b>Key Sectors:</b> IT Sector, Retail, Hospitality, Media.
            </p>
            <div style="background: rgba(16, 185, 129, 0.1); border-radius: 6px; padding: 10px; font-size: 12px; color: #A7F3D0;">
                <b>Actionable Playbook:</b> Attract with flash mobile sales, app-exclusive instant bank discounts (e.g. 10% credit card cashback), and trendy festive apparel.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")
st.markdown("### 📊 Customer Segment Statistical Summary")
seg_summary = cust_df.groupby("Segment_Name").agg(
    Customer_Count=("User_ID", "count"),
    Total_Segment_Spend=("Total_Spend", "sum"),
    Mean_Customer_Spend=("Total_Spend", "mean"),
    Mean_Orders_Count=("Total_Orders", "mean"),
    Mean_Transactions=("Transactions_Count", "mean"),
    Average_Age=("Avg_Age", "mean")
).reset_index()

seg_summary["Total_Spend_INR"] = seg_summary["Total_Segment_Spend"].apply(lambda x: f"₹{x:,.2f}")
seg_summary["Mean_Spend_INR"] = seg_summary["Mean_Customer_Spend"].apply(lambda x: f"₹{x:,.2f}")
seg_summary["Avg_Age_Years"] = seg_summary["Average_Age"].round(1)

st.dataframe(
    seg_summary[[
        "Segment_Name", "Customer_Count", "Total_Spend_INR", 
        "Mean_Spend_INR", "Mean_Orders_Count", "Mean_Transactions", "Avg_Age_Years"
    ]],
    use_container_width=True
)

