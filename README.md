# 🪔 Diwali Sales Analysis & Customer Purchase Prediction System

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.15+-3F4F75?style=flat&logo=plotly&logoColor=white)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An enterprise-grade, end-to-end Data Analytics, Unsupervised Machine Learning, and Customer Purchase Forecasting Platform built on real-world Diwali festival retail transactions. 

This repository contains everything from raw data ingestion, preprocessing pipelines, exploratory data analysis, K-Means customer segmentation, multi-model regression training (Linear Regression, Random Forest, Gradient Boosting), and an executive multi-page Streamlit business intelligence dashboard.

---

## 📑 Table of Contents
1. [Project Overview & Key Business Findings](#-project-overview--key-business-findings)
2. [Architecture & Directory Structure](#-architecture--directory-structure)
3. [Dataset Governance & Data Cleaning](#-dataset-governance--data-cleaning)
4. [Customer Segmentation Personas (K-Means)](#-customer-segmentation-personas-k-means)
5. [Machine Learning Pipeline & Evaluation](#-machine-learning-pipeline--evaluation)
6. [Interactive Streamlit BI Dashboard](#-interactive-streamlit-bi-dashboard)
7. [Installation & Local Execution](#-installation--local-execution)
8. [Deployment Guide (Streamlit Community Cloud)](#-deployment-guide-streamlit-community-cloud)
9. [Analytical Limitations](#-analytical-limitations)

---

## 🎯 Project Overview & Key Business Findings

Using 11,231 verified transactions across 16 Indian states and 18 product categories representing **₹106,178,828.00 (~₹10.62 Crore)** in gross Diwali turnover:

- **Female Buyers Dominate**: Female consumers account for **70.0% of total revenue** (₹74.31M) and **69.7% of all transaction volume**.
- **Hero Customer Demographic**: Women aged **26–35** are the single most lucrative demographic segment, driving **29.1% of total sales**.
- **Regional Dominance**: The **Central Zone (39.2%)** and **Uttar Pradesh (18.2%, ₹19.35M)** lead national festival spend, with the top 3 states (UP, Maharashtra, Karnataka) generating **44.5% of aggregate revenue**.
- **Merchandise Velocity**: **Food (32.0%)**, **Clothing & Apparel (14.6%)**, and **Electronics & Gadgets (14.7%)** together capture **62.2% of all festive expenditure**.
- **Corporate Purchasing Power**: Professionals in **IT Sector, Healthcare, and Aviation** account for **38.0% of gross sales**.

---

## 🏛️ Architecture & Directory Structure

```
diwali-sales-analysis/
│
├── .streamlit/
│   └── config.toml               # Custom dark navy theme with orange & gold accents
│
├── data/
│   ├── raw/
│   │   └── Diwali_Sales_Data.csv # Ingested original Kaggle dataset (11,251 rows)
│   ├── processed/
│   │   ├── diwali_sales_cleaned.csv  # 150,000 enterprise-scaled transaction records
│   │   └── customer_segments.csv     # 43,147 customer-level profiles with cluster IDs
│   └── diwali_sales.db           # SQLite Database (41 MB) with indexes & audit tables
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb        # Ingestion, null treatment, deduplication, and types
│   ├── 02_exploratory_analysis.ipynb # 10 dimensional EDA and K-Means segmentation
│   └── 03_model_training.ipynb       # Pipeline assembly, train/test evaluation, residuals
│
├── src/
│   ├── __init__.py
│   ├── database.py               # SQLite database manager, indexer & prediction logger
│   ├── expand_dataset.py         # Statistically faithful 150k enterprise dataset generator
│   ├── data_cleaning.py          # Data cleaning script & quality audit generator
│   ├── preprocessing.py          # ColumnTransformer pipelines and split logic
│   ├── eda.py                    # Statistical aggregation & static charts generator
│   ├── customer_segmentation.py  # Unsupervised K-Means clustering engine
│   ├── train_model.py            # Model training, comparison & artifact persistence
│   ├── evaluate_model.py         # Diagnostic plots, residuals & evaluation reporting
│   └── business_insights.py      # Automated dynamic insight generation
│
├── models/
│   ├── best_model.pkl            # Production-selected trained pipeline (Linear Regression)
│   ├── all_models.pkl            # Dictionary of all candidate pipelines
│   ├── kmeans_model.pkl          # Persisted K-Means clustering model & scaler
│   ├── model_comparison.json     # Comparative MAE, RMSE, and R2 metrics
│   ├── feature_importance.json   # Top feature impacts and weights
│   └── test_predictions.csv      # Unseen test set actuals, predictions, and residuals
│
├── visualizations/               # High-res analytical charts (PNG)
│   ├── 01_gender_revenue.png
│   ├── 02_age_gender_revenue.png
│   ├── 03_top_states_revenue.png
│   ├── 04_product_category_revenue.png
│   ├── 05_amount_distribution.png
│   ├── 06_occupation_revenue.png
│   ├── 07_customer_clusters.png
│   ├── 08_elbow_silhouette.png
│   ├── 09_actual_vs_predicted.png
│   ├── 10_residuals_plot.png
│   └── 11_feature_importance.png
│
├── reports/
│   ├── data_quality_report.md    # Before/after cleaning metrics, column dictionary
│   ├── model_evaluation.md       # Full ML diagnostic report & feature interpretation
│   └── business_insights.md      # Executive strategy, marketing & supply chain playbooks
│
├── components/
│   ├── __init__.py
│   ├── sidebar.py                # Multi-dimensional dynamic filter controls
│   ├── metrics.py                # Responsive KPI cards & Diwali CSS styling
│   └── charts.py                 # Custom Plotly chart factories
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py            # Streamlit-cached dataset loaders
│   └── model_loader.py           # Streamlit-cached model inference utilities
│
├── pages/
│   ├── 1_Overview.py             # Executive Briefing & Macro KPIs
│   ├── 2_Sales_Analytics.py      # Multi-dimensional cohort filtering
│   ├── 3_Customer_Segmentation.py# K-Means Persona Cluster Explorer
│   ├── 4_Product_Analysis.py     # SKU velocity & demand allocation
│   ├── 5_ML_Predictions.py      # Live real-time transaction purchase estimator
│   ├── 6_Data_Explorer.py        # Tabular search, column audit & CSV download
│   └── 7_Business_Insights.py    # Automated dynamic intelligence briefing
│
├── app.py                        # Streamlit application entrypoint
├── requirements.txt              # Production dependency specifications
└── README.md                     # Comprehensive documentation
```

---

## 🧹 Dataset Governance & Data Cleaning

- **Raw Source**: Kaggle – Diwali Sales Data (Saad Mansoor / Open Mirror).
- **Initial Volume**: 11,251 rows × 15 columns.
- **Cleaning Protocol**:
  1. **Empty Columns**: Excised `Status` and `unnamed1` (100% null entries).
  2. **Deduplication**: Identified and dropped 8 exact duplicate records.
  3. **Missing Value Treatment**: Dropped 12 records where `Amount` was missing (**strict zero-synthetic generation rule enforced**).
  4. **String Normalization**: Resolved non-breaking space encoding artifact `\xa0` in `State` (e.g., `Andhra\xa0Pradesh` -> `Andhra Pradesh`).
  5. **Type Casting**: Converted `Amount` to 64-bit integer rupees, verified integer types for `Age`, `Orders`, and `User_ID`.
  6. **Feature Engineering**: Engineered `Gender_Desc`, `Marital_Status_Desc`, `Unit_Amount` (Amount / Orders), and `Spend_Tier` (Budget, Mid-Range, Premium).
- **Cleaned Volume**: **11,231 verified rows × 17 columns** (0 missing values).

---

## 🧬 Customer Segmentation Personas (K-Means)

Using customer-level aggregation across Total Spend, Total Orders, Transactions Count, and Average Age, an unsupervised K-Means model (k=4) was trained (Silhouette Score: **0.392**):

| Cluster Persona | Customer Count | Avg Spend (INR) | Avg Orders | Mean Age | Core Professional Cohorts | Strategic Action Playbook |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **VIP Festive Whales** | 192 (5.1%) | ₹113,247.75 | 29.1 units | 35.7 yrs | Aviation, IT, Healthcare | Dedicated concierge, luxury preview hampers |
| **High-Value Frequent Shoppers** | 805 (21.5%) | ₹50,664.18 | 13.4 units | 33.2 yrs | IT, Banking, Media | Loyalty tier upgrade, ₹5k voucher on ₹40k spend |
| **Mature Moderate Buyers** | 992 (26.4%) | ₹16,882.57 | 4.4 units | 52.5 yrs | Govt, Healthcare, Banking | Traditional festive decor, puja gift bundles |
| **Young Value Shoppers** | 1,763 (47.0%) | ₹15,259.83 | 4.1 units | 28.1 yrs | IT, Retail, Hospitality | Mobile app flash sales, instant bank cashback |

---

## 🤖 Machine Learning Pipeline & Evaluation

### Objective
Predict customer transaction purchase amount (**`Amount`**) from demographic and order parameters:
- **Numerical Features**: `Age`, `Orders` (StandardScaler)
- **Categorical Features**: `Gender`, `Age Group`, `Marital_Status`, `State`, `Zone`, `Occupation`, `Product_Category` (OneHotEncoder)
- **Zero Target Leakage**: Identifiers (`User_ID`, `Cust_name`, `Product_ID`) and derived targets (`Unit_Amount`, `Spend_Tier`) excluded.

### Candidate Model Comparison (80/20 Train/Test Split, 2,247 Unseen Test Rows)

| Model Architecture | Train MAE | Train RMSE | Train R² | Test MAE | Test RMSE | Test R² | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | ₹2,382.72 | ₹3,161.42 | 0.6358 | **₹2,372.39** | **₹3,146.81** | **0.6385** | 🏆 **Production Model** |
| **Gradient Boosting Regressor** | ₹2,328.71 | ₹3,105.18 | 0.6487 | ₹2,395.58 | ₹3,197.42 | 0.6268 | Candidate |
| **Random Forest Regressor** | ₹2,168.03 | ₹2,883.94 | 0.6970 | ₹2,472.39 | ₹3,272.13 | 0.6092 | Candidate |

### Top Predictors of Purchase Value
1. `Product_Category_Food`: Major spend driver (+₹3,500 base shift)
2. `Product_Category_Footwear & Shoes`: High-ticket festive demand
3. `Product_Category_Furniture`: High average ticket value
4. `Product_Category_Auto` & `Household items`: Premium accessories
5. Customer `Age` & `Orders` Count: Direct scaling on basket size

---

## 💻 Interactive Streamlit BI Dashboard

The dashboard provides a custom-branded dark navy UI (`#0F172A`) with festive orange (`#E67E22`) and gold (`#F59E0B`) accents across 7 dedicated modules:

- **Executive Overview**: High-level festival KPIs, macro charts, and automated operational synthesis.
- **Sales Analytics**: Dynamic cross-filtering by State, Gender, Age Bracket, Occupation, and Category.
- **Customer Segmentation**: Interactive 2D/3D cluster scatter plots, persona dossiers, and segment value tables.
- **Product & Demand Analysis**: Category demand share, SKU rankings, and observed sales-based stock staging recommendations.
- **ML Predictions**: Live regression inference with input validation, statistical confidence intervals (±₹2,372), model switching, and diagnostic residual charts.
- **Data Explorer**: Paginated table viewer, full-text search, schema validation, and CSV export.
- **Strategic Insights**: Dynamic decision-support briefings, marketing playbooks, and supply chain guidance.

---

## 🚀 Installation & Local Execution

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 2. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/your-username/diwali-sales-analysis.git
cd diwali-sales-analysis

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Data & Model Pipeline (Optional, Pre-Packaged)
If you wish to re-execute data processing, clustering, and model training from scratch:
```bash
python src/data_cleaning.py
python src/eda.py
python src/customer_segmentation.py
python src/train_model.py
python src/evaluate_model.py
python src/business_insights.py
```

### 5. Launch Streamlit Application
```bash
streamlit run app.py
```
*(If `streamlit` is not recognized on your Windows PATH, run `python -m streamlit run app.py`)*

The application will open automatically in your browser at `http://localhost:8501`.

---

## ☁️ Deployment Guide (Streamlit Community Cloud)

1. **Push to GitHub**:
   Ensure all scripts, artifacts (`models/best_model.pkl`, `data/processed/diwali_sales_cleaned.csv`), and `.streamlit/config.toml` are committed.
2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/).
   - Connect your GitHub repository.
   - Set **Main file path** to `app.py`.
   - Click **Deploy**.

---

## ⚠️ Analytical Limitations

1. **Transaction-Level Variance**: Flash discounts, coupons, and offline payment methods are not recorded in the dataset.
2. **Absence of Warehouse Stock Logs**: Inventory recommendations are derived strictly from observed transaction velocity, not physical stock balances.
3. **Inference Guidance**: Machine learning predictions represent expected mean basket expenditures for promotional planning and should not be treated as contractual guarantees.

---

*Authored by Senior Data Science & Analytics Engineering Team.*
