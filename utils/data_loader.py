"""
Data Loader Utility Module with Streamlit Caching.
Loads and caches preprocessed datasets, connects to SQLite database, and executes SQL queries.
"""

import os
import sqlite3
import streamlit as st
import pandas as pd
from src.database import get_db_connection, init_database, DB_PATH

RAW_DATA_PATH = os.path.join("data", "raw", "Diwali_Sales_Data.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "diwali_sales_cleaned.csv")
SEGMENTS_DATA_PATH = os.path.join("data", "processed", "customer_segments.csv")


@st.cache_data(show_spinner=False)
def load_cleaned_data() -> pd.DataFrame:
    """Load cleaned Diwali sales transactions from SQLite database or fallback CSV."""
    if os.path.exists(DB_PATH):
        try:
            conn = get_db_connection(DB_PATH)
            df = pd.read_sql_query("SELECT * FROM sales_transactions;", conn)
            conn.close()
            if "Age_Group" in df.columns and "Age Group" not in df.columns:
                df = df.rename(columns={"Age_Group": "Age Group"})
            return df
        except Exception:
            pass

    if os.path.exists(PROCESSED_DATA_PATH):
        df = pd.read_csv(PROCESSED_DATA_PATH)
        return df

    raise FileNotFoundError("Neither SQLite database nor processed CSV found.")


@st.cache_data(show_spinner=False)
def load_customer_segments() -> pd.DataFrame:
    """Load customer-level segmented dataset from SQLite database or fallback CSV."""
    if os.path.exists(DB_PATH):
        try:
            conn = get_db_connection(DB_PATH)
            df = pd.read_sql_query("SELECT * FROM customer_segments;", conn)
            conn.close()
            return df
        except Exception:
            pass

    if os.path.exists(SEGMENTS_DATA_PATH):
        return pd.read_csv(SEGMENTS_DATA_PATH)

    raise FileNotFoundError("Customer segments data not found.")


@st.cache_data(show_spinner=False)
def load_raw_data_preview(limit: int = 100) -> pd.DataFrame:
    """Load raw dataset head for data explorer comparison."""
    if not os.path.exists(RAW_DATA_PATH):
        return pd.DataFrame()
    try:
        return pd.read_csv(RAW_DATA_PATH, encoding="utf-8", nrows=limit)
    except UnicodeDecodeError:
        return pd.read_csv(RAW_DATA_PATH, encoding="unicode_escape", nrows=limit)


def get_sqlite_stats() -> dict:
    """Return database file size and row counts."""
    if not os.path.exists(DB_PATH):
        init_database()

    conn = get_db_connection(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sales_transactions;")
    sales_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM customer_segments;")
    seg_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions_log;")
    pred_count = cursor.fetchone()[0]
    conn.close()

    return {
        "db_path": DB_PATH,
        "db_size_mb": round(os.path.getsize(DB_PATH) / (1024 * 1024), 2),
        "sales_count": sales_count,
        "segments_count": seg_count,
        "predictions_count": pred_count
    }


def filter_sales_data(
    df: pd.DataFrame,
    states: list = None,
    genders: list = None,
    age_groups: list = None,
    occupations: list = None,
    categories: list = None,
    spend_tiers: list = None
) -> pd.DataFrame:
    """Filter dataset dynamically based on user selections."""
    filtered = df.copy()

    if states:
        filtered = filtered[filtered["State"].isin(states)]
    if genders:
        filtered = filtered[filtered["Gender_Desc"].isin(genders)]
    if age_groups:
        filtered = filtered[filtered["Age Group"].isin(age_groups)]
    if occupations:
        filtered = filtered[filtered["Occupation"].isin(occupations)]
    if categories:
        filtered = filtered[filtered["Product_Category"].isin(categories)]
    if spend_tiers:
        filtered = filtered[filtered["Spend_Tier"].isin(spend_tiers)]

    return filtered
