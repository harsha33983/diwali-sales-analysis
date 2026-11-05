"""
Preprocessing Module for Diwali Sales Analysis.
Defines feature schema, ColumnTransformer pipelines, and train/test splitting utilities.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Feature definition
NUMERICAL_FEATURES = ["Age", "Orders"]
CATEGORICAL_FEATURES = [
    "Gender",
    "Age Group",
    "Marital_Status",
    "State",
    "Zone",
    "Occupation",
    "Product_Category"
]
TARGET = "Amount"


def get_preprocessor() -> ColumnTransformer:
    """Build ColumnTransformer for numerical scaling and categorical one-hot encoding."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor


def prepare_train_test_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Extracts features and target, ensuring no leakage, and performs reproducible train/test split.
    """
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test

