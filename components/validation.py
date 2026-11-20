
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real

import pandas as pd


# Keep feature names consistent with the trained model.
FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


# Define input ranges for data validation, not medical diagnosis.
FEATURE_RULES = {
    "Pregnancies": {
        "label": "Pregnancies",
        "unit": "",
        "minimum": 0,
        "maximum": 20,
        "warning_low": None,
        "warning_high": 12,
        "integer": True,
        "zero_missing": False,
    },
    "Glucose": {
        "label": "Glucose",
        "unit": "mg/dL",
        "minimum": 1,
        "maximum": 600,
        "warning_low": 50,
        "warning_high": 250,
        "integer": True,
        "zero_missing": True,
    },
    "BloodPressure": {
        "label": "Diastolic Blood Pressure",
        "unit": "mmHg",
        "minimum": 1,
        "maximum": 200,
        "warning_low": 40,
        "warning_high": 120,
        "integer": True,
        "zero_missing": True,
    },
    "SkinThickness": {
        "label": "Skin Thickness",
        "unit": "mm",
        "minimum": 1,
        "maximum": 150,
        "warning_low": None,
        "warning_high": 80,
        "integer": True,
        "zero_missing": True,
    },
    "Insulin": {
        "label": "2-Hour Serum Insulin",
        "unit": "µU/mL",
        "minimum": 1,
        "maximum": 1500,
        "warning_low": None,
        "warning_high": 500,
        "integer": True,
        "zero_missing": True,
    },
    "BMI": {
        "label": "Body Mass Index",
        "unit": "kg/m²",
        "minimum": 5,
        "maximum": 100,
        "warning_low": 15,
        "warning_high": 50,
        "integer": False,
        "zero_missing": True,
    },
    "DiabetesPedigreeFunction": {
        "label": "Diabetes Pedigree Function",
        "unit": "",
        "minimum": 0,
        "maximum": 5,
        "warning_low": None,
        "warning_high": 2.5,
        "integer": False,
        "zero_missing": False,
    },
    "Age": {
        "label": "Age",
        "unit": "years",
        "minimum": 21,
        "maximum": 120,
        "warning_low": None,
        "warning_high": 81,
        "integer": True,
        "zero_missing": False,
    },
}


# Store validation results in one reusable structure.
@dataclass
class ValidationResult:
    is_valid: bool = True
    cleaned_data: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    warnings: dict = field(default_factory=dict)

    @property
    def has_warnings(self):
        return bool(self.warnings)

    @property
    def status(self):
        if not self.is_valid:
            return "invalid"

        if self.has_warnings:
            return "warning"

        return "valid"


# Check whether a value is missing or unusable.
def is_missing_value(value):
    if value is None:
        return True

    if isinstance(value, str):
        return not value.strip()

    try:
        result = pd.isna(value)

        if isinstance(result, bool):
            return result

        return False
    except (TypeError, ValueError):
        return False


# Convert a submitted value into a finite number.
def parse_numeric_value(value):
    if isinstance(value, bool):
        raise ValueError("Boolean values are not accepted.")

    if not isinstance(value, (str, Real)):
        raise ValueError("Enter a valid numeric value.")

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Enter a valid numeric value.")

    if not isfinite(number):
        raise ValueError("Enter a finite numeric value.")

    return number


# Validate one health measurement.
def validate_feature(feature, value):
    if feature not in FEATURE_RULES:
        raise ValueError(f"Unknown feature: {feature}")

    rule = FEATURE_RULES[feature]
    label = rule["label"]
    unit = rule["unit"]

    if is_missing_value(value):
        return {
            "is_valid": False,
            "value": None,
            "error": f"{label} is required.",
            "warning": None,
        }

    try:
        number = parse_numeric_value(value)
    except ValueError as error:
        return {
            "is_valid": False,
            "value": None,
            "error": f"{label}: {error}",
            "warning": None,
        }

    # Prevent decimal values in integer-only measurements.
    if rule["integer"] and not number.is_integer():
        return {
            "is_valid": False,
            "value": None,
            "error": f"{label} must be a whole number.",
            "warning": None,
        }

    minimum = rule["minimum"]
    maximum = rule["maximum"]

    # Reject measurements outside the accepted input range.
    if number < minimum or number > maximum:
        unit_text = f" {unit}" if unit else ""

        return {
            "is_valid": False,
            "value": None,
            "error": (
                f"{label} must be between "
                f"{minimum} and {maximum}{unit_text}."
            ),
            "warning": None,
        }

    # Identify unusual measurements without rejecting them.
    warning = None

    warning_low = rule["warning_low"]
    warning_high = rule["warning_high"]

    if warning_low is not None and number < warning_low:
        warning = (
            f"{label} is unusually low for this application's "
            "input screening rules. Please verify the measurement."
        )

    elif warning_high is not None and number > warning_high:
        warning = (
            f"{label} is unusually high for this application's "
            "input screening rules. Please verify the measurement."
        )

    cleaned_value = (
        int(number)
        if rule["integer"]
        else float(number)
    )

    return {
        "is_valid": True,
        "value": cleaned_value,
        "error": None,
        "warning": warning,
    }


# Validate all eight measurements together.
def validate_prediction_input(data):
    result = ValidationResult()

    if not isinstance(data, dict):
        result.is_valid = False
        result.errors["input"] = (
            "Prediction input must be a dictionary."
        )
        return result

    # Reject unexpected fields before preparing model input.
    unexpected_features = set(data) - set(FEATURE_COLUMNS)

    if unexpected_features:
        result.is_valid = False
        result.errors["input"] = (
            "Unexpected feature(s): "
            + ", ".join(sorted(unexpected_features))
        )

    for feature in FEATURE_COLUMNS:
        value = data.get(feature)

        validation = validate_feature(feature, value)

        if not validation["is_valid"]:
            result.is_valid = False
            result.errors[feature] = validation["error"]
            continue

        result.cleaned_data[feature] = validation["value"]

        if validation["warning"]:
            result.warnings[feature] = validation["warning"]

    return result


# Prepare a single validated record for the ML pipeline.
def prepare_prediction_dataframe(data):
    result = validate_prediction_input(data)

    if not result.is_valid:
        raise ValueError(
            "Invalid prediction input: "
            + "; ".join(result.errors.values())
        )

    return pd.DataFrame(
        [result.cleaned_data],
        columns=FEATURE_COLUMNS,
    )


# Provide metadata for building reusable Streamlit input fields.
def get_feature_rules():
    return {
        feature: rule.copy()
        for feature, rule in FEATURE_RULES.items()
    }
