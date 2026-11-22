
from pathlib import Path
from math import isfinite

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from database.repository import save_prediction
from components.validation import (
    FEATURE_COLUMNS,
    FEATURE_RULES,
    prepare_prediction_dataframe,
    validate_prediction_input,
)


# Locate the saved model and configuration files.
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_pipeline.joblib"
CONFIG_PATH = BASE_DIR / "models" / "model_config.joblib"

POSITIVE_COLOR = "#F59E0B"
NEGATIVE_COLOR = "#0D9488"

FEATURE_HELP = {
    "Pregnancies": "Number of pregnancies recorded for the individual.",
    "Glucose": "Plasma glucose concentration in mg/dL.",
    "BloodPressure": "Diastolic blood pressure in mmHg.",
    "SkinThickness": "Triceps skinfold thickness in millimeters.",
    "Insulin": "2-hour serum insulin measurement in µU/mL.",
    "BMI": "Body mass index calculated from height and weight.",
    "DiabetesPedigreeFunction": (
        "A numerical indicator of family-related diabetes history "
        "used in the Pima Indians Diabetes Dataset."
    ),
    "Age": "Age in completed years.",
}

FEATURE_PLACEHOLDERS = {
    "Pregnancies": "Enter number of pregnancies",
    "Glucose": "Enter glucose level",
    "BloodPressure": "Enter diastolic blood pressure",
    "SkinThickness": "Enter skin thickness",
    "Insulin": "Enter insulin level",
    "BMI": "Enter body mass index",
    "DiabetesPedigreeFunction": "Enter pedigree function value",
    "Age": "Enter age",
}


# Load the saved machine learning artifacts.
@st.cache_resource
def load_prediction_artifacts():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "The trained Random Forest pipeline was not found."
        )

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            "The model configuration file was not found."
        )

    pipeline = joblib.load(MODEL_PATH)
    config = joblib.load(CONFIG_PATH)

    if not hasattr(pipeline, "predict_proba"):
        raise ValueError(
            "The saved model does not support probability prediction."
        )

    if not isinstance(config, dict):
        raise ValueError(
            "The model configuration must be a dictionary."
        )

    return pipeline, config


# Read the classification threshold from the model configuration.
def get_decision_threshold(config):
    possible_keys = [
        "threshold",
        "decision_threshold",
        "optimal_threshold",
        "best_threshold",
        "classification_threshold",
    ]

    for key in possible_keys:
        if key in config:
            try:
                threshold = float(config[key])
            except (TypeError, ValueError):
                continue

            if isfinite(threshold) and 0 < threshold < 1:
                return threshold

    return 0.40


# Create the probability gauge.
def create_probability_gauge(probability, threshold):
    percentage = probability * 100
    threshold_percentage = threshold * 100

    result_color = (
        POSITIVE_COLOR
        if probability >= threshold
        else NEGATIVE_COLOR
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=percentage,
            number={
                "suffix": "%",
                "font": {
                    "size": 44,
                    "color": result_color,
                },
            },
            title={
                "text": "Model-Estimated Positive-Class Probability",
                "font": {
                    "size": 15,
                    "color": "#CBD5E1",
                },
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#94A3B8",
                },
                "bar": {
                    "color": result_color,
                    "thickness": 0.3,
                },
                "bgcolor": "#172337",
                "borderwidth": 0,
                "steps": [
                    {
                        "range": [0, threshold_percentage],
                        "color": "rgba(13,148,136,0.18)",
                    },
                    {
                        "range": [threshold_percentage, 100],
                        "color": "rgba(245,158,11,0.14)",
                    },
                ],
                "threshold": {
                    "line": {
                        "color": "#F8FAFC",
                        "width": 3,
                    },
                    "thickness": 0.8,
                    "value": threshold_percentage,
                },
            },
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=310,
        margin=dict(l=25, r=25, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E1"),
    )

    return fig


# Render an empty measurement field with a placeholder.
def render_feature_input(feature):
    rule = FEATURE_RULES[feature]

    label = rule["label"]
    unit = rule["unit"]

    display_label = f"{label} ({unit})" if unit else label
    is_integer = rule["integer"]

    return st.number_input(
        display_label,
        min_value=None,
        max_value=None,
        value=None,
        step=1 if is_integer else 0.01,
        format="%d" if is_integer else "%.3f",
        placeholder=FEATURE_PLACEHOLDERS[feature],
        help=FEATURE_HELP[feature],
        key=f"prediction_input_{feature}",
    )


# Display validation errors and unusual-value warnings.
def display_validation_feedback(result):
    if result.errors:
        st.error(
            "Some measurements need to be corrected "
            "before prediction can continue."
        )

        for message in result.errors.values():
            st.markdown(f"- {message}")

    if result.warnings:
        st.warning(
            "Some measurements are unusual. "
            "Please verify them before interpreting the result."
        )

        for message in result.warnings.values():
            st.markdown(f"- {message}")


# Generate the model prediction.
def run_prediction(pipeline, input_dataframe, threshold):
    probabilities = pipeline.predict_proba(input_dataframe)
    classes = list(pipeline.classes_)

    if 1 not in classes:
        raise ValueError(
            "The trained model does not contain the positive class."
        )

    positive_index = classes.index(1)
    probability = float(probabilities[0][positive_index])

    if not isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(
            "The model returned an invalid probability."
        )

    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "probability": probability,
        "threshold": threshold,
    }


# Display the prediction classification and probability.
def display_prediction_result(result):
    prediction = result["prediction"]
    probability = result["probability"]
    threshold = result["threshold"]

    is_positive = prediction == 1

    label = (
        "Positive Classification"
        if is_positive
        else "Negative Classification"
    )

    st.subheader("Prediction Results")

    with st.container(border=True):
        st.markdown(f"### {label}")

        gauge = create_probability_gauge(
            probability,
            threshold,
        )

        st.plotly_chart(
            gauge,
            use_container_width=True,
            key="prediction_probability_gauge",
        )

        metric_columns = st.columns(2)

        with metric_columns[0]:
            st.metric(
                "Positive-Class Probability",
                f"{probability * 100:.1f}%",
            )

        with metric_columns[1]:
            st.metric(
                "Decision Threshold",
                f"{threshold * 100:.0f}%",
            )

        if is_positive:
            st.markdown(
                "**Model Interpretation:** The estimated "
                "positive-class probability meets or exceeds "
                "the selected classification threshold."
            )
        else:
            st.markdown(
                "**Model Interpretation:** The estimated "
                "positive-class probability is below "
                "the selected classification threshold."
            )

        st.caption(
            "This probability is generated by a machine learning "
            "model trained on the Pima Indians Diabetes Dataset. "
            "It is not a clinically calibrated estimate of an "
            "individual's diabetes risk."
        )

        # Display the save confirmation inside the result box.
        saved_prediction_id = st.session_state["diaml_prediction_id"]
        save_error = st.session_state["diaml_prediction_save_error"]

        if saved_prediction_id is not None:
            st.success(
                f"Prediction saved to history — ID #{saved_prediction_id}"
            )
        elif save_error:
            st.warning(save_error)

    # Keep the detailed explanation hidden until requested.
    with st.expander("Understanding Your Result", expanded=False):
        st.markdown("**What does the classification mean?**")

        st.write(
            "The Random Forest model analyzes the eight supplied "
            "measurements and estimates how strongly the input "
            "resembles records labeled positive or negative "
            "in its training dataset."
        )

        st.markdown(
            "**Does a positive classification mean diabetes?**"
        )

        st.write(
            "No. A positive classification is not a diagnosis, "
            "and a negative classification does not rule out diabetes. "
            "Clinical assessment requires appropriate medical testing "
            "and professional interpretation."
        )

        st.markdown(
            f"**Why is the threshold {threshold * 100:.0f}%?**"
        )

        st.write(
            f"The model uses a decision threshold of "
            f"{threshold * 100:.0f}%. This threshold was selected "
            "during model development to balance classification "
            "performance rather than automatically using 50%."
        )


# Display the prediction page heading.
st.title("Diabetes Prediction")


# Load the saved Random Forest model.
try:
    pipeline, model_config = load_prediction_artifacts()
    decision_threshold = get_decision_threshold(model_config)

except (
    FileNotFoundError,
    ValueError,
    TypeError,
    OSError,
) as error:
    st.error(f"Unable to load the prediction model: {error}")
    st.stop()


# Initialize prediction state.
if "diaml_prediction_result" not in st.session_state:
    st.session_state["diaml_prediction_result"] = None

if "diaml_prediction_inputs" not in st.session_state:
    st.session_state["diaml_prediction_inputs"] = None

if "diaml_prediction_warnings" not in st.session_state:
    st.session_state["diaml_prediction_warnings"] = {}

# Track the saved database record for the latest prediction.
if "diaml_prediction_id" not in st.session_state:
    st.session_state["diaml_prediction_id"] = None

if "diaml_prediction_save_error" not in st.session_state:
    st.session_state["diaml_prediction_save_error"] = None


# Build the health measurement form.
with st.container(border=True):
    st.subheader("Health Measurements")

    st.caption(
        "All eight measurements are required. "
        "Values outside the accepted application ranges "
        "will be flagged before prediction."
    )

    with st.form("diabetes_prediction_form"):
        left_column, right_column = st.columns(2)

        input_data = {}

        with left_column:
            for feature in FEATURE_COLUMNS[:4]:
                input_data[feature] = render_feature_input(feature)

        with right_column:
            for feature in FEATURE_COLUMNS[4:]:
                input_data[feature] = render_feature_input(feature)

        submitted = st.form_submit_button(
            "Generate Prediction",
            type="primary",
            use_container_width=True,
        )


# Validate submitted measurements and generate a prediction.
if submitted:
    # A new submission replaces the previous record status.
    st.session_state["diaml_prediction_id"] = None
    st.session_state["diaml_prediction_save_error"] = None
    validation_result = validate_prediction_input(input_data)

    if not validation_result.is_valid:
        st.session_state["diaml_prediction_result"] = None
        st.session_state["diaml_prediction_inputs"] = None
        st.session_state["diaml_prediction_warnings"] = {}

        display_validation_feedback(validation_result)

    else:
        st.session_state["diaml_prediction_warnings"] = (
            validation_result.warnings.copy()
        )

        try:
            input_dataframe = prepare_prediction_dataframe(input_data)

            prediction_result = run_prediction(
                pipeline,
                input_dataframe,
                decision_threshold,
            )

            st.session_state["diaml_prediction_result"] = (
                prediction_result
            )

            st.session_state["diaml_prediction_inputs"] = (
                validation_result.cleaned_data
            )

            # Save only after a new, successful prediction submission.
            try:
                prediction_id = save_prediction(
                    measurements=validation_result.cleaned_data,
                    probability=prediction_result["probability"],
                    classification=(
                        "Positive"
                        if prediction_result["prediction"] == 1
                        else "Negative"
                    ),
                    threshold=prediction_result["threshold"],
                    model_name=model_config.get("model_name", "Random Forest"),
                    model_version=model_config.get("model_version", "v1.0"),
                )
                st.session_state["diaml_prediction_id"] = prediction_id
            except Exception:
                # Keep the model result visible if database saving fails.
                st.session_state["diaml_prediction_save_error"] = (
                    "The prediction was generated, but it could not be "
                    "saved to Prediction History. Please check the "
                    "database connection before trying again."
                )

        except (ValueError, TypeError, KeyError) as error:
            st.session_state["diaml_prediction_result"] = None
            st.session_state["diaml_prediction_inputs"] = None
            st.session_state["diaml_prediction_warnings"] = {}

            st.error(
                f"Prediction could not be completed: {error}"
            )


# Display the latest successful prediction.
saved_result = st.session_state["diaml_prediction_result"]

if saved_result is not None:
    saved_warnings = st.session_state["diaml_prediction_warnings"]

    if saved_warnings:
        st.warning(
            "Some submitted measurements were unusual. "
            "Please verify them before interpreting the result."
        )

        for message in saved_warnings.values():
            st.markdown(f"- {message}")

    display_prediction_result(saved_result)

    # Keep submitted measurements in a separate collapsible panel.
    with st.expander("View Submitted Measurements", expanded=False):
        saved_inputs = st.session_state["diaml_prediction_inputs"]

        if saved_inputs is not None:
            input_table = pd.DataFrame(
                [
                    {
                        "Measurement": FEATURE_RULES[feature]["label"],
                        "Value": saved_inputs[feature],
                        "Unit": FEATURE_RULES[feature]["unit"] or "—",
                    }
                    for feature in FEATURE_COLUMNS
                ]
            )

            st.dataframe(
                input_table,
                hide_index=True,
                use_container_width=True,
            )


# Keep the educational disclaimer at the bottom.
st.divider()

st.caption(
    "UODA DiaML | Educational Diabetes Prediction System | "
    "Not a substitute for professional medical evaluation."
)
