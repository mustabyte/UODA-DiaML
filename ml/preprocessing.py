
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


# Load the original dataset with its column names
def load_dataset(path=None):
    if path is None:
        path = Path(__file__).resolve().parents[1] / "data" / "diabetes.csv"

    df = pd.read_csv(
        path,
        header=None,
        names=FEATURE_COLUMNS + [TARGET_COLUMN]
    )

    return df


# Replace invalid zero measurements with missing values
class InvalidZeroTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=FEATURE_COLUMNS).copy()
        X[INVALID_ZERO_COLUMNS] = X[INVALID_ZERO_COLUMNS].replace(0, np.nan)
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
