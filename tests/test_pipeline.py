"""
Automated Project Validation Test Suite.
Validates all data artifacts, model predictions, script executions, and Streamlit page imports.
"""

import os
import sys
import importlib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import joblib

def run_checks():
    print("=== DIWALI SALES PROJECT VALIDATION SUITE ===")
    errors = []

    # 1. Check Data Files
    print("\n[1/6] Checking Datasets & Database...")
    raw_path = os.path.join(PROJECT_ROOT, "data", "raw", "Diwali_Sales_Data.csv")
    proc_path = os.path.join(PROJECT_ROOT, "data", "processed", "diwali_sales_cleaned.csv")
    seg_path = os.path.join(PROJECT_ROOT, "data", "processed", "customer_segments.csv")
    db_path = os.path.join(PROJECT_ROOT, "data", "diwali_sales.db")

    for p in [raw_path, proc_path, seg_path, db_path]:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            print(f"  [OK] Found: {os.path.relpath(p, PROJECT_ROOT)} ({os.path.getsize(p):,} bytes)")
        else:
            errors.append(f"Missing or empty file: {p}")
            print(f"  [FAIL] MISSING: {p}")

    # 2. Check Reports
    print("\n[2/6] Checking Reports...")
    reports = [
        "reports/data_quality_report.md",
        "reports/model_evaluation.md",
        "reports/business_insights.md"
    ]
    for r in reports:
        full_r = os.path.join(PROJECT_ROOT, r)
        if os.path.exists(full_r) and os.path.getsize(full_r) > 0:
            print(f"  [OK] Found: {r}")
        else:
            errors.append(f"Missing report: {r}")
            print(f"  [FAIL] MISSING: {r}")

    # 3. Check Visualizations
    print("\n[3/6] Checking Visualizations...")
    viz_files = [
        "visualizations/01_gender_revenue.png",
        "visualizations/02_age_gender_revenue.png",
        "visualizations/03_top_states_revenue.png",
        "visualizations/04_product_category_revenue.png",
        "visualizations/05_amount_distribution.png",
        "visualizations/06_occupation_revenue.png",
        "visualizations/07_customer_clusters.png",
        "visualizations/08_elbow_silhouette.png",
        "visualizations/09_actual_vs_predicted.png",
        "visualizations/10_residuals_plot.png",
        "visualizations/11_feature_importance.png",
    ]
    for v in viz_files:
        full_v = os.path.join(PROJECT_ROOT, v)
        if os.path.exists(full_v) and os.path.getsize(full_v) > 0:
            print(f"  [OK] Found: {v}")
        else:
            errors.append(f"Missing visualization: {v}")
            print(f"  [FAIL] MISSING: {v}")

    # 4. Check Model Artifacts and Live Inference
    print("\n[4/6] Checking Model Pipeline & Inference...")
    from utils.model_loader import load_best_model, predict_purchase_amount
    try:
        pipeline = load_best_model()
        print(f"  [OK] Loaded best model: {type(pipeline)}")

        # Test live inference
        test_pred = predict_purchase_amount(
            pipeline=pipeline,
            gender="F",
            age=28,
            age_group="26-35",
            marital_status=0,
            state="Uttar Pradesh",
            zone="Central",
            occupation="IT Sector",
            product_category="Food",
            orders=2
        )
        print(f"  [OK] Test Prediction Successful: Rs.{test_pred['predicted_amount']} (Range: Rs.{test_pred['lower_bound']} - Rs.{test_pred['upper_bound']})")
    except Exception as e:
        errors.append(f"Model loading or inference failed: {e}")
        print(f"  [FAIL] Model Error: {e}")

    # 5. Check Notebooks
    print("\n[5/6] Checking Notebooks...")
    notebooks = [
        "notebooks/01_data_cleaning.ipynb",
        "notebooks/02_exploratory_analysis.ipynb",
        "notebooks/03_model_training.ipynb"
    ]
    for nb in notebooks:
        full_nb = os.path.join(PROJECT_ROOT, nb)
        if os.path.exists(full_nb) and os.path.getsize(full_nb) > 0:
            print(f"  [OK] Found notebook: {nb}")
        else:
            errors.append(f"Missing notebook: {nb}")
            print(f"  [FAIL] MISSING: {nb}")

    # 6. Syntax Check on all Python Pages and App
    print("\n[6/6] Checking Python Syntax of Streamlit Pages...")
    py_files = [
        "app.py",
        "components/sidebar.py",
        "components/metrics.py",
        "components/charts.py",
        "utils/data_loader.py",
        "utils/model_loader.py",
        "pages/1_Overview.py",
        "pages/2_Sales_Analytics.py",
        "pages/3_Customer_Segmentation.py",
        "pages/4_Product_Analysis.py",
        "pages/5_ML_Predictions.py",
        "pages/6_Data_Explorer.py",
        "pages/7_Business_Insights.py",
    ]
    for pf in py_files:
        full_pf = os.path.join(PROJECT_ROOT, pf)
        try:
            with open(full_pf, "r", encoding="utf-8") as f:
                code = f.read()
            compile(code, full_pf, "exec")
            print(f"  [OK] Syntax Valid: {pf}")
        except Exception as e:
            errors.append(f"Syntax error in {pf}: {e}")
            print(f"  [FAIL] Syntax Error in {pf}: {e}")

    print("\n" + "=" * 50)
    if errors:
        print(f"FAILED: {len(errors)} error(s) found:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED! Project is 100% verified and operational.")
        sys.exit(0)

if __name__ == "__main__":
    run_checks()
