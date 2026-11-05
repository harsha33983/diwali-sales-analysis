"""
Model Training Script for Diwali Sales Purchase Amount Prediction.
Trains Linear Regression, Random Forest, and Gradient Boosting models, evaluates them,
and persists best model artifacts.
"""

import os
import sys
import json
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.preprocessing import get_preprocessor, prepare_train_test_data, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

DATA_PATH = os.path.join("data", "processed", "diwali_sales_cleaned.csv")
MODELS_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
ALL_MODELS_PATH = os.path.join(MODELS_DIR, "all_models.pkl")
COMPARISON_PATH = os.path.join(MODELS_DIR, "model_comparison.json")
IMPORTANCE_PATH = os.path.join(MODELS_DIR, "feature_importance.json")


def train_and_compare_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset with {len(df)} records for training.")

    X_train, X_test, y_train, y_test = prepare_train_test_data(df, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")

    candidate_algorithms = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=50, max_depth=12, max_samples=0.5, random_state=42, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=60, max_depth=4, subsample=0.5, learning_rate=0.1, random_state=42)
    }

    comparison_results = {}
    trained_pipelines = {}

    for name, regressor in candidate_algorithms.items():
        print(f"\nTraining {name}...", flush=True)
        preprocessor = get_preprocessor()
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])

        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        train_preds = pipeline.predict(X_train)
        test_preds = pipeline.predict(X_test)

        train_mae = float(mean_absolute_error(y_train, train_preds))
        train_rmse = float(root_mean_squared_error(y_train, train_preds))
        train_r2 = float(r2_score(y_train, train_preds))

        test_mae = float(mean_absolute_error(y_test, test_preds))
        test_rmse = float(root_mean_squared_error(y_test, test_preds))
        test_r2 = float(r2_score(y_test, test_preds))

        comparison_results[name] = {
            "Train_MAE": round(train_mae, 2),
            "Train_RMSE": round(train_rmse, 2),
            "Train_R2": round(train_r2, 4),
            "Test_MAE": round(test_mae, 2),
            "Test_RMSE": round(test_rmse, 2),
            "Test_R2": round(test_r2, 4),
        }

        print(f"  {name} Test Metrics -> MAE: Rs.{test_mae:.2f} | RMSE: Rs.{test_rmse:.2f} | R2: {test_r2:.4f}")

    # Determine best model based on highest test R² score
    best_model_name = max(comparison_results, key=lambda k: comparison_results[k]["Test_R2"])
    best_pipeline = trained_pipelines[best_model_name]
    print(f"\n--> Selected Best Model: {best_model_name} (R2 = {comparison_results[best_model_name]['Test_R2']})")

    # Extract feature importance or coefficients
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    cat_feature_names = fitted_preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    all_feature_names = NUMERICAL_FEATURES + list(cat_feature_names)

    best_regressor = best_pipeline.named_steps["regressor"]
    if hasattr(best_regressor, "feature_importances_"):
        importance_vals = best_regressor.feature_importances_.tolist()
    elif hasattr(best_regressor, "coef_"):
        importance_vals = np.abs(best_regressor.coef_).tolist()
    else:
        importance_vals = [0.0] * len(all_feature_names)

    feature_importance_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Importance": importance_vals
    }).sort_values(by="Importance", ascending=False)

    feature_importance_dict = feature_importance_df.head(20).to_dict(orient="records")

    # Save artifacts
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    joblib.dump(trained_pipelines, ALL_MODELS_PATH)

    output_data = {
        "best_model_name": best_model_name,
        "comparison": comparison_results,
        "features": {
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "all_encoded_count": len(all_feature_names)
        }
    }

    with open(COMPARISON_PATH, "w") as f:
        json.dump(output_data, f, indent=4)

    with open(IMPORTANCE_PATH, "w") as f:
        json.dump(feature_importance_dict, f, indent=4)

    # Save test set predictions for evaluation plotting
    test_eval_df = X_test.copy()
    test_eval_df["Actual_Amount"] = y_test.values
    test_eval_df["Predicted_Amount"] = best_pipeline.predict(X_test)
    test_eval_df["Residual"] = test_eval_df["Actual_Amount"] - test_eval_df["Predicted_Amount"]
    test_eval_df.to_csv(os.path.join(MODELS_DIR, "test_predictions.csv"), index=False)

    print(f"\nArtifacts successfully persisted:")
    print(f"  - Best Model: {BEST_MODEL_PATH}")
    print(f"  - All Models: {ALL_MODELS_PATH}")
    print(f"  - Comparison JSON: {COMPARISON_PATH}")
    print(f"  - Feature Importance JSON: {IMPORTANCE_PATH}")

    return best_model_name, comparison_results


if __name__ == "__main__":
    train_and_compare_models()
