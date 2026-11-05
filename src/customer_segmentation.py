"""
Customer Segmentation Module for Diwali Sales Dataset.
Implements RFM/Demographic aggregation and K-Means Clustering to identify actionable customer personas.
"""

import os
import sys
import joblib

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

PROCESSED_DATA_PATH = os.path.join("data", "processed", "diwali_sales_cleaned.csv")
SEGMENTS_DATA_PATH = os.path.join("data", "processed", "customer_segments.csv")
MODEL_PATH = os.path.join("models", "kmeans_model.pkl")
VISUALS_DIR = "visualizations"


def build_customer_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction logs into customer-level RFM and demographic profiles."""
    cust_df = df.groupby("User_ID").agg(
        Total_Spend=("Amount", "sum"),
        Total_Orders=("Orders", "sum"),
        Transactions_Count=("Orders", "count"),
        Avg_Order_Value=("Amount", "mean"),
        Avg_Age=("Age", "mean"),
        Primary_Gender=("Gender_Desc", "first"),
        Primary_Marital_Status=("Marital_Status_Desc", "first"),
        Primary_State=("State", "first"),
        Primary_Occupation=("Occupation", "first"),
        Primary_Category=("Product_Category", "first")
    ).reset_index()

    cust_df["Avg_Age"] = cust_df["Avg_Age"].round(1)
    cust_df["Avg_Order_Value"] = cust_df["Avg_Order_Value"].round(2)
    return cust_df


def find_optimal_clusters(X_scaled: np.ndarray, k_range=range(2, 9)) -> dict:
    """Evaluate Inertia and Silhouette Scores across multiple k values with sample optimization."""
    inertias = []
    silhouettes = []
    sample_size = min(5000, len(X_scaled))
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels, sample_size=sample_size, random_state=42)
        silhouettes.append(sil)

    return {
        "k_values": list(k_range),
        "inertias": inertias,
        "silhouettes": silhouettes
    }


def perform_customer_clustering(cust_df: pd.DataFrame, n_clusters: int = 4) -> tuple[pd.DataFrame, KMeans, StandardScaler, dict]:
    """Train K-Means clustering model and assign interpretable segment personas."""
    features = ["Total_Spend", "Total_Orders", "Transactions_Count", "Avg_Age"]
    X = cust_df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    eval_stats = find_optimal_clusters(X_scaled)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
    cluster_labels = km.fit_predict(X_scaled)
    cust_df["Cluster_ID"] = cluster_labels

    # Dynamically map clusters based on average total spend to ensure consistent persona labeling
    cluster_spend = cust_df.groupby("Cluster_ID")["Total_Spend"].mean().sort_values(ascending=False)
    sorted_cluster_ids = cluster_spend.index.tolist()

    # Identify cluster with highest age among the lower spenders
    lower_two = sorted_cluster_ids[2:]
    age_rank = cust_df[cust_df["Cluster_ID"].isin(lower_two)].groupby("Cluster_ID")["Avg_Age"].mean()
    older_low_spend_id = age_rank.idxmax()
    younger_low_spend_id = [c for c in lower_two if c != older_low_spend_id][0]

    cluster_name_map = {
        sorted_cluster_ids[0]: "VIP Festive Whales (Ultra High Spend)",
        sorted_cluster_ids[1]: "High-Value Frequent Shoppers",
        older_low_spend_id: "Mature Moderate Buyers",
        younger_low_spend_id: "Young Value Shoppers"
    }

    cust_df["Segment_Name"] = cust_df["Cluster_ID"].map(cluster_name_map)

    # Save artifacts
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": km, "scaler": scaler, "features": features, "name_map": cluster_name_map}, MODEL_PATH)

    os.makedirs(os.path.dirname(SEGMENTS_DATA_PATH), exist_ok=True)
    cust_df.to_csv(SEGMENTS_DATA_PATH, index=False)

    return cust_df, km, scaler, eval_stats


def generate_segmentation_visuals(cust_df: pd.DataFrame, eval_stats: dict):
    """Plot Elbow/Silhouette diagnostics and 2D cluster scatter plot."""
    os.makedirs(VISUALS_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Elbow and Silhouette Diagnostics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(eval_stats["k_values"], eval_stats["inertias"], marker="o", color="#E67E22", linewidth=2)
    ax1.set_title("K-Means Elbow Curve (Inertia)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (Sum of Squared Distances)")

    ax2.plot(eval_stats["k_values"], eval_stats["silhouettes"], marker="s", color="#2E4053", linewidth=2)
    ax2.set_title("Silhouette Score by Cluster Count", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Mean Silhouette Coefficient")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "08_elbow_silhouette.png"), dpi=200)
    plt.close()

    # 2. Cluster Profiles Scatter
    plt.figure(figsize=(10, 6))
    palette = ["#C0392B", "#D35400", "#2980B9", "#27AE60"]
    sns.scatterplot(
        data=cust_df,
        x="Transactions_Count",
        y="Total_Spend",
        hue="Segment_Name",
        palette="tab10",
        alpha=0.8,
        s=60
    )
    plt.title("Diwali Customer Segments: Total Spend vs Order Frequency", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Total Transactions per Customer", fontsize=11)
    plt.ylabel("Total Spend (INR)", fontsize=11)
    plt.yscale("log")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Customer Persona")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "07_customer_clusters.png"), dpi=200)
    plt.close()

    print("Segmentation visual plots successfully saved to visualizations/ directory.")


def main():
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Cleaned dataset not found at {PROCESSED_DATA_PATH}")

    df = pd.read_csv(PROCESSED_DATA_PATH)
    print("Building customer level profiles...")
    cust_df = build_customer_profiles(df)
    print(f"Customer profile count: {len(cust_df)}")

    print("Executing K-Means clustering...")
    cust_df, km, scaler, eval_stats = perform_customer_clustering(cust_df, n_clusters=4)

    summary = cust_df.groupby("Segment_Name").agg(
        Customers=("User_ID", "count"),
        Avg_Spend=("Total_Spend", "mean"),
        Avg_Orders=("Total_Orders", "mean"),
        Avg_Transactions=("Transactions_Count", "mean"),
        Mean_Age=("Avg_Age", "mean")
    ).reset_index()

    print("\n--- Customer Segmentation Summary ---")
    print(summary.to_string(index=False))

    generate_segmentation_visuals(cust_df, eval_stats)
    print(f"Customer segmentation completed. Processed data saved to {SEGMENTS_DATA_PATH}")


if __name__ == "__main__":
    main()

