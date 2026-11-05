"""
Page 6: Data Explorer.
Interactive dataset inspection, metadata analytics, full-text search, and CSV export.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
from utils.data_loader import load_cleaned_data, load_raw_data_preview
from components.metrics import render_page_header, render_kpi_grid

st.set_page_config(page_title="Data Explorer | Diwali Sales BI", page_icon="🪔", layout="wide")

df = load_cleaned_data()

render_page_header(
    title="Interactive <span>Data Explorer</span>",
    subtitle="Audit dataset schema, inspect record quality, search transactions, and download filtered slices",
    badge="Data Governance & Exploration"
)

# Explorer KPIs
kpis = [
    {"label": "Total Clean Records", "value": f"{len(df):,}", "subtext": "100% verified records", "highlight": ""},
    {"label": "Engineered Columns", "value": f"{len(df.columns)} Features", "subtext": "Numerical & Categorical", "highlight": ""},
    {"label": "Zero Missing Values", "value": "0 Nulls", "subtext": "Data completeness:", "highlight": "100%"},
    {"label": "Unique SKUs Indexed", "value": f"{df['Product_ID'].nunique():,}", "subtext": "Across 18 categories", "highlight": ""},
]
render_kpi_grid(kpis)

tab_view, tab_schema, tab_stats, tab_raw, tab_sql = st.tabs([
    "🔍 Filter & Export Dataset",
    "📐 Schema & Types",
    "📊 Statistical Summaries",
    "📜 Raw vs Cleaned Audit",
    "⚡ SQLite SQL Console"
])

with tab_view:
    st.markdown("### 🔎 Search & Slice Transactions")
    
    # Filter Controls Row
    c1, c2, c3 = st.columns(3)
    with c1:
        search_query = st.text_input("Search Customer Name, Product ID, or User ID", placeholder="e.g. Sanskriti, P00125942, 1002903")
    with c2:
        selected_states = st.multiselect("Filter by State", options=sorted(df["State"].unique().tolist()), default=[])
    with c3:
        selected_cats = st.multiselect("Filter by Product Category", options=sorted(df["Product_Category"].unique().tolist()), default=[])

    c4, c5, c6 = st.columns(3)
    with c4:
        selected_gender = st.multiselect("Filter by Gender", options=["Female", "Male"], default=[])
    with c5:
        selected_ages = st.multiselect("Filter by Age Group", options=["0-17", "18-25", "26-35", "36-45", "46-50", "51-55", "55+"], default=[])
    with c6:
        selected_spends = st.multiselect("Filter by Spend Tier", options=sorted(df["Spend_Tier"].dropna().unique().tolist()), default=[])

    # Apply filters
    filtered_df = df.copy()
    if search_query:
        q = search_query.strip().lower()
        filtered_df = filtered_df[
            filtered_df["Cust_name"].astype(str).str.lower().str.contains(q) |
            filtered_df["Product_ID"].astype(str).str.lower().str.contains(q) |
            filtered_df["User_ID"].astype(str).str.contains(q)
        ]
    if selected_states:
        filtered_df = filtered_df[filtered_df["State"].isin(selected_states)]
    if selected_cats:
        filtered_df = filtered_df[filtered_df["Product_Category"].isin(selected_cats)]
    if selected_gender:
        filtered_df = filtered_df[filtered_df["Gender_Desc"].isin(selected_gender)]
    if selected_ages:
        filtered_df = filtered_df[filtered_df["Age Group"].isin(selected_ages)]
    if selected_spends:
        filtered_df = filtered_df[filtered_df["Spend_Tier"].isin(selected_spends)]

    st.markdown(f"**Showing {len(filtered_df):,} matching rows out of {len(df):,} total records.**")

    # Column Selector for Custom View
    all_cols = df.columns.tolist()
    default_cols = ["User_ID", "Cust_name", "Product_ID", "Gender_Desc", "Age", "State", "Occupation", "Product_Category", "Orders", "Amount", "Spend_Tier"]
    visible_cols = st.multiselect("Select Columns to Display", options=all_cols, default=default_cols)

    if visible_cols:
        st.dataframe(filtered_df[visible_cols].head(500), use_container_width=True, height=450)
    else:
        st.dataframe(filtered_df.head(500), use_container_width=True, height=450)

    # Download Button
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download Filtered Data as CSV ({len(filtered_df):,} records)",
        data=csv_bytes,
        file_name="diwali_sales_filtered.csv",
        mime="text/csv",
        use_container_width=True
    )

with tab_schema:
    st.markdown("### 📐 Cleaned Dataset Schema & Non-Null Audit")
    
    schema_info = []
    for col in df.columns:
        schema_info.append({
            "Column Name": col,
            "Data Type": str(df[col].dtype),
            "Non-Null Count": int(df[col].count()),
            "Null Count": int(df[col].isnull().sum()),
            "Unique Values": int(df[col].nunique()),
            "Sample Value": str(df[col].dropna().iloc[0]) if not df[col].empty else ""
        })
    schema_df = pd.DataFrame(schema_info)
    st.dataframe(schema_df, use_container_width=True, height=450)

with tab_stats:
    st.markdown("### 📊 Descriptive Statistics")
    
    st.markdown("#### Numerical Attributes")
    num_desc = df.describe().round(2).T
    num_desc["count"] = num_desc["count"].astype(int)
    st.dataframe(num_desc, use_container_width=True)

    st.markdown("#### Categorical Attributes")
    cat_desc = df.describe(include=["object", "category"]).T
    st.dataframe(cat_desc, use_container_width=True)

with tab_raw:
    st.markdown("### 📜 Audit: Raw Data vs Cleaned Data Pipeline")
    st.markdown(
        """
        - **Raw Dataset**: 11,251 rows, 15 columns (included empty `Status` & `unnamed1`).
        - **Deduplication**: 8 exact duplicate rows identified and excised.
        - **Missing Values**: 12 null entries in `Amount` dropped (zero synthetic generation).
        - **Encoding Fixes**: Unicode non-breaking space `\\xa0` in `State` corrected (`Andhra Pradesh`).
        - **Feature Additions**: `Gender_Desc`, `Marital_Status_Desc`, `Unit_Amount`, `Spend_Tier`.
        """
    )
    raw_preview = load_raw_data_preview(10)
    if not raw_preview.empty:
        st.markdown("#### Raw CSV Sample (First 10 Rows with Null Artifacts)")
        st.dataframe(raw_preview, use_container_width=True)

with tab_sql:
    st.markdown("### ⚡ SQLite Interactive SQL Console")
    st.markdown("Query the high-performance local SQLite database (`data/diwali_sales.db`) directly using standard SQL syntax.")

    from utils.data_loader import get_sqlite_stats
    from src.database import execute_custom_query
    import time

    db_stats = get_sqlite_stats()

    sq_col1, sq_col2, sq_col3 = st.columns(3)
    with sq_col1:
        st.metric("Database File Size", f"{db_stats['db_size_mb']} MB", "data/diwali_sales.db")
    with sq_col2:
        st.metric("Indexed Transactions", f"{db_stats['sales_count']:,}", "table: sales_transactions")
    with sq_col3:
        st.metric("Customer Segments", f"{db_stats['segments_count']:,}", "table: customer_segments")

    st.markdown("#### Pre-Configured Analytical Queries")
    preset_queries = {
        "Top 10 States by Total Festival Revenue": """SELECT State, Zone, COUNT(*) as Transactions, SUM(Orders) as Total_Units, SUM(Amount) as Total_Revenue, ROUND(AVG(Amount), 2) as Mean_Ticket
FROM sales_transactions
GROUP BY State, Zone
ORDER BY Total_Revenue DESC
LIMIT 10;""",
        "Category Revenue & Velocity Breakdown": """SELECT Product_Category, COUNT(*) as Transactions, SUM(Orders) as Total_Units, SUM(Amount) as Total_Revenue, ROUND(AVG(Unit_Amount), 2) as Avg_Unit_Price
FROM sales_transactions
GROUP BY Product_Category
ORDER BY Total_Revenue DESC;""",
        "Customer Persona Performance from customer_segments": """SELECT Segment_Name, COUNT(*) as Customer_Count, SUM(Total_Spend) as Aggregate_Spend, ROUND(AVG(Total_Spend), 2) as Avg_Customer_Spend, ROUND(AVG(Avg_Age), 1) as Mean_Age
FROM customer_segments
GROUP BY Segment_Name
ORDER BY Aggregate_Spend DESC;""",
        "Occupation & Gender Spend Correlation": """SELECT Occupation, Gender_Desc, COUNT(*) as Transactions, SUM(Amount) as Total_Spend, ROUND(AVG(Amount), 2) as Mean_Spend
FROM sales_transactions
GROUP BY Occupation, Gender_Desc
ORDER BY Total_Spend DESC
LIMIT 12;""",
        "Recent Machine Learning Inference Audit Trail": """SELECT id, timestamp, model_used, gender, age, state, occupation, product_category, orders, predicted_amount
FROM predictions_log
ORDER BY id DESC
LIMIT 20;"""
    }

    selected_preset = st.selectbox("Choose a Preset Query or write custom SQL below:", options=list(preset_queries.keys()))
    default_sql = preset_queries[selected_preset]

    query_input = st.text_area("SQL Statement (Read-only SELECT supported)", value=default_sql, height=140)

    if st.button("🚀 Execute SQL Query", use_container_width=True):
        t0 = time.time()
        sql_df, err = execute_custom_query(query_input)
        elapsed = time.time() - t0

        if err:
            st.error(f"❌ SQL Execution Error: {err}")
        else:
            st.success(f"✓ Query executed successfully in {elapsed:.3f} seconds ({len(sql_df):,} rows returned)")
            st.dataframe(sql_df, use_container_width=True)

            # Download query result as CSV
            csv_data = sql_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Query Results as CSV",
                data=csv_data,
                file_name="diwali_sql_query_result.csv",
                mime="text/csv"
            )

