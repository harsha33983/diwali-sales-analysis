"""
Exploratory Data Analysis (EDA) Module for Diwali Sales Data.
Calculates statistical summaries and generates production-quality visual charts.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED_DATA_PATH = os.path.join("data", "processed", "diwali_sales_cleaned.csv")
VISUALS_DIR = "visualizations"


def ensure_visuals_dir():
    os.makedirs(VISUALS_DIR, exist_ok=True)


def set_custom_style():
    """Set professional, clean publication styling."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#CCCCCC"
    plt.rcParams["axes.linewidth"] = 0.8


def analyze_overall_metrics(df: pd.DataFrame) -> dict:
    """Compute high-level executive KPIs."""
    total_revenue = df["Amount"].sum()
    total_orders = df["Orders"].sum()
    total_transactions = len(df)
    unique_customers = df["User_ID"].nunique()
    unique_products = df["Product_ID"].nunique()
    avg_ticket_size = df["Amount"].mean()
    median_ticket_size = df["Amount"].median()

    top_state = df.groupby("State")["Amount"].sum().idxmax()
    top_state_revenue = df.groupby("State")["Amount"].sum().max()

    top_category = df.groupby("Product_Category")["Amount"].sum().idxmax()
    top_category_revenue = df.groupby("Product_Category")["Amount"].sum().max()

    return {
        "total_revenue": int(total_revenue),
        "total_orders": int(total_orders),
        "total_transactions": int(total_transactions),
        "unique_customers": int(unique_customers),
        "unique_products": int(unique_products),
        "avg_ticket_size": round(avg_ticket_size, 2),
        "median_ticket_size": round(median_ticket_size, 2),
        "top_state": top_state,
        "top_state_revenue": int(top_state_revenue),
        "top_category": top_category,
        "top_category_revenue": int(top_category_revenue),
    }


def analyze_gender_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Sales and transaction metrics by gender."""
    gender_agg = df.groupby("Gender_Desc").agg(
        Transactions=("User_ID", "count"),
        Unique_Customers=("User_ID", "nunique"),
        Total_Orders=("Orders", "sum"),
        Total_Revenue=("Amount", "sum"),
        Mean_Spend=("Amount", "mean"),
    ).reset_index()
    total_rev = gender_agg["Total_Revenue"].sum()
    gender_agg["Revenue_Pct"] = (gender_agg["Total_Revenue"] / total_rev * 100).round(2)
    return gender_agg


def analyze_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Sales performance across age brackets."""
    age_agg = df.groupby(["Age Group", "Gender_Desc"]).agg(
        Transactions=("User_ID", "count"),
        Total_Orders=("Orders", "sum"),
        Total_Revenue=("Amount", "sum"),
        Mean_Spend=("Amount", "mean")
    ).reset_index()
    return age_agg


def analyze_geography(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sales by State and Zone."""
    state_agg = df.groupby("State").agg(
        Transactions=("User_ID", "count"),
        Total_Orders=("Orders", "sum"),
        Total_Revenue=("Amount", "sum"),
        Avg_Order_Value=("Amount", "mean")
    ).sort_values(by="Total_Revenue", ascending=False).reset_index()

    zone_agg = df.groupby("Zone").agg(
        Transactions=("User_ID", "count"),
        Total_Orders=("Orders", "sum"),
        Total_Revenue=("Amount", "sum"),
        Avg_Order_Value=("Amount", "mean")
    ).sort_values(by="Total_Revenue", ascending=False).reset_index()

    return state_agg, zone_agg


def analyze_occupations(df: pd.DataFrame) -> pd.DataFrame:
    """Sales by Customer Occupation."""
    occ_agg = df.groupby("Occupation").agg(
        Transactions=("User_ID", "count"),
        Total_Orders=("Orders", "sum"),
        Total_Revenue=("Amount", "sum"),
        Avg_Spend=("Amount", "mean")
    ).sort_values(by="Total_Revenue", ascending=False).reset_index()
    return occ_agg


def analyze_marital_status(df: pd.DataFrame) -> pd.DataFrame:
    """Sales by Marital Status."""
    ms_agg = df.groupby(["Marital_Status_Desc", "Gender_Desc"]).agg(
        Transactions=("User_ID", "count"),
        Total_Orders=("Orders", "sum"),
        Total_Revenue=("Amount", "sum"),
        Avg_Spend=("Amount", "mean")
    ).reset_index()
    return ms_agg


def analyze_products(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Product category analysis and top SKUs."""
    cat_agg = df.groupby("Product_Category").agg(
        Transactions=("User_ID", "count"),
        Total_Orders=("Orders", "sum"),
        Total_Revenue=("Amount", "sum"),
        Avg_Spend=("Amount", "mean")
    ).sort_values(by="Total_Revenue", ascending=False).reset_index()

    top_products = df.groupby(["Product_ID", "Product_Category"]).agg(
        Orders_Count=("Orders", "sum"),
        Transactions=("User_ID", "count"),
        Total_Revenue=("Amount", "sum"),
        Avg_Spend=("Amount", "mean")
    ).sort_values(by="Total_Revenue", ascending=False).reset_index().head(10)

    return cat_agg, top_products


def generate_static_eda_charts(df: pd.DataFrame):
    """Generate high-resolution PNG charts for EDA documentation."""
    ensure_visuals_dir()
    set_custom_style()

    # 1. Gender Sales Distribution
    plt.figure(figsize=(9, 5))
    gender_df = df.groupby("Gender_Desc")["Amount"].sum().reset_index()
    colors = ["#E67E22", "#2E4053"]
    bars = plt.bar(gender_df["Gender_Desc"], gender_df["Amount"] / 1e6, color=colors, width=0.5)
    plt.title("Total Revenue by Gender (₹ Millions)", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Revenue in ₹ Millions", fontsize=11)
    plt.xlabel("Gender", fontsize=11)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 1, f"₹{h:.2f}M", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "01_gender_revenue.png"), dpi=200)
    plt.close()

    # 2. Revenue by Age Group & Gender
    plt.figure(figsize=(11, 5))
    age_gender = df.groupby(["Age Group", "Gender_Desc"])["Amount"].sum().reset_index()
    sns.barplot(data=age_gender, x="Age Group", y="Amount", hue="Gender_Desc", palette=["#E67E22", "#2E4053"])
    plt.title("Diwali Purchase Revenue by Age Bracket and Gender", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Total Revenue (INR)", fontsize=11)
    plt.xlabel("Age Group", fontsize=11)
    plt.legend(title="Gender")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "02_age_gender_revenue.png"), dpi=200)
    plt.close()

    # 3. Top 10 States by Revenue
    plt.figure(figsize=(12, 6))
    top_states = df.groupby("State")["Amount"].sum().sort_values(ascending=False).head(10).reset_index()
    sns.barplot(data=top_states, y="State", x="Amount", hue="State", palette="Oranges_r", legend=False)
    plt.title("Top 10 States by Diwali Festival Revenue", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Total Revenue (INR)", fontsize=11)
    plt.ylabel("State", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "03_top_states_revenue.png"), dpi=200)
    plt.close()

    # 4. Top Product Categories
    plt.figure(figsize=(12, 6))
    top_cats = df.groupby("Product_Category")["Amount"].sum().sort_values(ascending=False).head(10).reset_index()
    sns.barplot(data=top_cats, y="Product_Category", x="Amount", hue="Product_Category", palette="YlOrBr_r", legend=False)
    plt.title("Top 10 Revenue-Generating Product Categories", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Total Revenue (INR)", fontsize=11)
    plt.ylabel("Product Category", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "04_product_category_revenue.png"), dpi=200)
    plt.close()

    # 5. Purchase Amount Distribution & Boxplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["Amount"], kde=True, ax=ax1, color="#D35400", bins=30)
    ax1.set_title("Purchase Amount Distribution (Histogram & KDE)", fontweight="bold")
    ax1.set_xlabel("Amount (INR)")
    
    sns.boxplot(x=df["Amount"], ax=ax2, color="#F39C12")
    ax2.set_title("Purchase Amount Outlier & Spread (Boxplot)", fontweight="bold")
    ax2.set_xlabel("Amount (INR)")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "05_amount_distribution.png"), dpi=200)
    plt.close()

    # 6. Occupation Revenue
    plt.figure(figsize=(12, 6))
    occ_df = df.groupby("Occupation")["Amount"].sum().sort_values(ascending=False).reset_index()
    sns.barplot(data=occ_df, y="Occupation", x="Amount", hue="Occupation", palette="Blues_r", legend=False)
    plt.title("Total Spend by Customer Occupation", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Total Revenue (INR)", fontsize=11)
    plt.ylabel("Occupation", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "06_occupation_revenue.png"), dpi=200)
    plt.close()

    print("Static EDA charts successfully created in visualizations/ directory.")


def main():
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Cleaned dataset not found at {PROCESSED_DATA_PATH}")

    df = pd.read_csv(PROCESSED_DATA_PATH)
    metrics = analyze_overall_metrics(df)
    print("\n--- Diwali Sales EDA Key Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    generate_static_eda_charts(df)


if __name__ == "__main__":
    main()
