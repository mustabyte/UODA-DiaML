from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

TARGET_COLUMN = "Outcome"

INVALID_ZERO_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]


# Validate the dataset structure and values
def validate_dataset(df):
    expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    if df.empty:
        raise ValueError("The dataset is empty.")

    if df.columns.tolist() != expected_columns:
        raise ValueError("Invalid dataset column structure.")

    if not all(pd.api.types.is_numeric_dtype(df[column])
               for column in expected_columns):
        raise ValueError("Dataset must contain numeric values.")

    if np.isinf(df.to_numpy(dtype=float)).any():
        raise ValueError("Dataset contains infinite values.")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")

    if not df[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Outcome must contain only 0 or 1.")

    if (df[FEATURE_COLUMNS] < 0).any().any():
        raise ValueError("Dataset contains negative measurements.")

    return df


# Load and validate the original dataset
def load_dataset(path=None):
    if path is None:
        path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "diabetes.csv"
        )

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    try:
        df = pd.read_csv(path, header=None)
    except pd.errors.EmptyDataError as error:
        raise ValueError("The dataset is empty.") from error

    expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    if df.shape[1] != len(expected_columns):
        raise ValueError(
            f"Expected {len(expected_columns)} columns, "
            f"but found {df.shape[1]}."
        )

    df.columns = expected_columns

    return validate_dataset(df)


# Replace invalid zero measurements with missing values
class InvalidZeroTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=FEATURE_COLUMNS).copy()

        X[INVALID_ZERO_COLUMNS] = (
            X[INVALID_ZERO_COLUMNS].replace(0, np.nan)
        )

        return X


# Build preprocessing steps for each model
def create_preprocessor(scale=True):
    steps = [
        ("invalid_zeros", InvalidZeroTransformer()),
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale:
        steps.append(("scaler", StandardScaler()))

    return Pipeline(steps)


# Combine preprocessing with a machine learning model
def create_model_pipeline(model, scale=True):
    return Pipeline([
        ("preprocessing", create_preprocessor(scale=scale)),
        ("model", model)
    ])