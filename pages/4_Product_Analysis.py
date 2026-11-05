"""
Page 4: Product & Demand Analysis.
Merchandise category velocity, SKU demand patterns, and sales-driven inventory allocation.
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
from components.charts import plot_category_revenue, plot_top_products

st.set_page_config(page_title="Product Analysis | Diwali Sales BI", page_icon="🪔", layout="wide")

df = load_cleaned_data()

render_page_header(
    title="Product & <span>Demand Analytics</span>",
    subtitle="Merchandise revenue distribution, SKU velocity, and empirical stock allocation guidance",
    badge="Merchandise Intelligence"
)

# Category and Product aggregations
cat_agg = df.groupby("Product_Category").agg(
    Total_Revenue=("Amount", "sum"),
    Total_Units=("Orders", "sum"),
    Transactions=("User_ID", "count"),
    Avg_Unit_Price=("Unit_Amount", "mean")
).sort_values(by="Total_Revenue", ascending=False).reset_index()

total_rev = df["Amount"].sum()
cat_agg["Revenue_Share_Pct"] = (cat_agg["Total_Revenue"] / total_rev * 100).round(2)

top_cat = cat_agg.iloc[0]["Product_Category"]
top_cat_rev = cat_agg.iloc[0]["Total_Revenue"]
top_sku_row = df.groupby(["Product_ID", "Product_Category"]).agg(
    Revenue=("Amount", "sum"),
    Orders=("Orders", "sum")
).sort_values(by="Revenue", ascending=False).iloc[0]

prod_kpis = [
    {"label": "Total Active Categories", "value": f"{len(cat_agg)} Categories", "subtext": "Catalog coverage:", "highlight": "100%"},
    {"label": "Top Category", "value": top_cat, "subtext": "Generated:", "highlight": f"₹{top_cat_rev/1e6:.1f}M ({cat_agg.iloc[0]['Revenue_Share_Pct']}%)"},
    {"label": "Total Unique SKUs", "value": f"{df['Product_ID'].nunique():,} SKUs", "subtext": "Active products", "highlight": ""},
    {"label": "Highest Grossing SKU", "value": top_sku_row.name[0], "subtext": f"Category: {top_sku_row.name[1]}", "highlight": f"₹{top_sku_row['Revenue']:,.0f}"},
]
render_kpi_grid(prod_kpis)

# 1. Product Category Charts
col_p1, col_p2 = st.columns([1.2, 1])
with col_p1:
    st.plotly_chart(plot_category_revenue(df, top_n=10), use_container_width=True)
with col_p2:
    st.plotly_chart(plot_top_products(df, top_n=10), use_container_width=True)

# 2. Demand Velocity & Category Table
st.markdown("---")
st.markdown("### 📊 Comprehensive Category Demand & Sales Matrix")

cat_display = cat_agg.copy()
cat_display["Total_Revenue_INR"] = cat_display["Total_Revenue"].apply(lambda x: f"₹{x:,.0f}")
cat_display["Total_Units_Sold"] = cat_display["Total_Units"].apply(lambda x: f"{x:,}")
cat_display["Avg_Unit_Price_INR"] = cat_display["Avg_Unit_Price"].apply(lambda x: f"₹{x:,.0f}")
cat_display["Share_of_Total"] = cat_display["Revenue_Share_Pct"].apply(lambda x: f"{x:.2f}%")

st.dataframe(
    cat_display[[
        "Product_Category", "Total_Revenue_INR", "Share_of_Total", 
        "Total_Units_Sold", "Transactions", "Avg_Unit_Price_INR"
    ]],
    use_container_width=True
)

# 3. Top vs Underperforming Product Deep Dive
st.markdown("---")
col_perf1, col_perf2 = st.columns(2)

with col_perf1:
    st.markdown("#### 🌟 Top 5 Powerhouse Categories")
    top_5 = cat_agg.head(5)
    for _, row in top_5.iterrows():
        st.markdown(
            f"""
            <div style="background: #1E293B; border-left: 3px solid #10B981; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between;">
                    <b>{row['Product_Category']}</b>
                    <span style="color: #10B981; font-weight: 700;">₹{row['Total_Revenue']:,.0f} ({row['Revenue_Share_Pct']}%)</span>
                </div>
                <div style="font-size: 11px; color: #94A3B8;">{row['Total_Units']:,} units across {row['Transactions']:,} transactions</div>
            </div>
            """,
            unsafe_allow_html=True
        )

with col_perf2:
    st.markdown("#### ⚠️ Low-Velocity / Underperforming Categories")
    bottom_5 = cat_agg.tail(5)
    for _, row in bottom_5.iterrows():
        st.markdown(
            f"""
            <div style="background: #1E293B; border-left: 3px solid #EF4444; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between;">
                    <b>{row['Product_Category']}</b>
                    <span style="color: #F87171; font-weight: 700;">₹{row['Total_Revenue']:,.0f} ({row['Revenue_Share_Pct']}%)</span>
                </div>
                <div style="font-size: 11px; color: #94A3B8;">{row['Total_Units']:,} units across {row['Transactions']:,} transactions</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# 4. Practical Stock Allocation Recommendations
st.markdown("---")
st.markdown("### 📦 Demand-Driven Stock Allocation Recommendations")
st.info(
    "📌 **Demand Attribution Notice**: These recommendations reflect observed sales velocity and transaction demand patterns. "
    "Because physical warehouse stock-level logs are not included in the dataset, guidance focuses on festive replenishment prioritization."
)

st.markdown(
    """
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-top: 12px;">
        <div style="background: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 16px;">
            <h5 style="color: #F59E0B; margin: 0 0 8px 0;">1. Food & Confectionery Staging</h5>
            <p style="font-size: 13px; color: #CBD5E1; margin: 0;">
                Food represents <b>32.0% of total revenue</b>. Implement weekly cycle inventory checks with perishable-certified regional hubs (UP, Maharashtra, Karnataka) 15-20 days prior to Dhanteras.
            </p>
        </div>
        <div style="background: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 16px;">
            <h5 style="color: #38BDF8; margin: 0 0 8px 0;">2. Apparel & Footwear Buffer</h5>
            <p style="font-size: 13px; color: #CBD5E1; margin: 0;">
                Clothing & Footwear generate <b>34.5% of combined spend</b>. Maintain 25% safety stock in high-density metropolitan fulfillment centers to prevent size and style stockouts during Diwali weekend.
            </p>
        </div>
        <div style="background: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 16px;">
            <h5 style="color: #A855F7; margin: 0 0 8px 0;">3. High-Ticket Electronics</h5>
            <p style="font-size: 13px; color: #CBD5E1; margin: 0;">
                Electronics command high unit prices. Secure consignment financing with OEM electronics suppliers to fulfill surges without tying up excessive pre-season working capital.
            </p>
        </div>
        <div style="background: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 16px;">
            <h5 style="color: #10B981; margin: 0 0 8px 0;">4. Tail Category Optimization</h5>
            <p style="font-size: 13px; color: #CBD5E1; margin: 0;">
                Reduce standalone inventory holding for Hand Tools, Veterinary, and Office supplies by 40% during festive periods; transition to cross-category festive bundles.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

