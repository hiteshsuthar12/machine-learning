"""
Preprocessing module for the Diabetes Risk Prediction dataset.

This module defines the preprocessing pipeline using scikit-learn's
ColumnTransformer and Pipeline to avoid data leakage. All transformations
are fit only on training data during cross-validation.

Dataset columns:
- Age: numerical (integer)
- Gender: categorical (Male/Female)
- Polyuria, Polydipsia, sudden weight loss, weakness, Polyphagia,
  Genital thrush, visual blurring, delayed healing, muscle stiffness:
  binary categorical (Yes/No)
- class: target (Positive/Negative)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer


# Feature definitions based on dataset inspection
NUMERICAL_FEATURES = ["Age"]

BINARY_FEATURES = [
    "Polyuria",
    "Polydipsia",
    "sudden weight loss",
    "weakness",
    "Polyphagia",
    "Genital thrush",
    "visual blurring",
    "delayed healing",
    "muscle stiffness",
]

GENDER_FEATURE = ["Gender"]

ALL_FEATURES = NUMERICAL_FEATURES + GENDER_FEATURE + BINARY_FEATURES

TARGET_COLUMN = "class"
TARGET_POSITIVE = "Positive"
TARGET_NEGATIVE = "Negative"

RANDOM_SEED = 42


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """Encode Yes/No binary features to 1/0.

    This transformer maps 'Yes' -> 1 and 'No' -> 0 for binary symptom
    features. It handles case-insensitive matching and strips whitespace.
    Unknown values are mapped to 0 with a warning logged.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X)
        result = np.zeros(X.shape, dtype=np.float64)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                val = str(X[i, j]).strip().lower()
                if val == "yes":
                    result[i, j] = 1.0
                elif val == "no":
                    result[i, j] = 0.0
                else:
                    # Treat unknown as 0 (absent)
                    result[i, j] = 0.0
        return result


class GenderEncoder(BaseEstimator, TransformerMixin):
    """Encode Gender feature: Male -> 1, Female -> 0.

    Handles case-insensitive matching and strips whitespace.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X).ravel()
        result = np.zeros((len(X), 1), dtype=np.float64)
        for i, val in enumerate(X):
            val_str = str(val).strip().lower()
            if val_str == "male":
                result[i, 0] = 1.0
            elif val_str == "female":
                result[i, 0] = 0.0
            else:
                result[i, 0] = 0.0
        return result


def build_preprocessor():
    """Build the ColumnTransformer preprocessing pipeline.

    Returns a ColumnTransformer that:
    - Scales the Age feature using StandardScaler
    - Encodes Gender (Male=1, Female=0)
    - Encodes binary symptom features (Yes=1, No=0)

    The preprocessor must be fit only on training data to avoid data leakage.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num_age", StandardScaler(), NUMERICAL_FEATURES),
            ("cat_gender", GenderEncoder(), GENDER_FEATURE),
            ("cat_binary", BinaryEncoder(), BINARY_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names():
    """Return the ordered list of feature names after preprocessing."""
    return NUMERICAL_FEATURES + GENDER_FEATURE + BINARY_FEATURES


def encode_target(y):
    """Encode target column: Positive -> 1, Negative -> 0."""
    return (y.str.strip() == TARGET_POSITIVE).astype(int)


def decode_target(y_encoded):
    """Decode target: 1 -> 'Positive', 0 -> 'Negative'."""
    return TARGET_POSITIVE if y_encoded == 1 else TARGET_NEGATIVE


def load_dataset(filepath):
    """Load and validate the diabetes dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Tuple of (X DataFrame with features, y Series with encoded target).

    Raises:
        ValueError: If required columns are missing or data is invalid.
    """
    df = pd.read_csv(filepath)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Validate required columns
    missing_cols = set(ALL_FEATURES + [TARGET_COLUMN]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Drop rows with all NaN
    df = df.dropna(how="all")

    # Separate features and target
    X = df[ALL_FEATURES].copy()
    y = encode_target(df[TARGET_COLUMN])

    return X, y, df
