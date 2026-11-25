
from pathlib import Path
from math import isfinite
from html import escape

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.report_generator import generate_what_if_report, generate_report_filename

from components.validation import (
    FEATURE_COLUMNS,
    FEATURE_RULES,
    prepare_prediction_dataframe,
    validate_prediction_input,
)

from database.repository import (
    get_all_predictions,
    get_prediction_by_id,
)


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_pipeline.joblib"
CONFIG_PATH = BASE_DIR / "models" / "model_config.joblib"

DATABASE_COLUMNS = {
    "Pregnancies": "pregnancies",
    "Glucose": "glucose",
    "BloodPressure": "blood_pressure",
    "SkinThickness": "skin_thickness",
    "Insulin": "insulin",
    "BMI": "bmi",
    "DiabetesPedigreeFunction": "diabetes_pedigree_function",
    "Age": "age",
}

FEATURE_LABELS = {
    "Pregnancies": "Pregnancies",
    "Glucose": "Glucose",
    "BloodPressure": "Diastolic Blood Pressure",
    "SkinThickness": "Skin Thickness",
    "Insulin": "Insulin",
    "BMI": "Body Mass Index",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
    "Age": "Age",
}

FEATURE_UNITS = {
    "Pregnancies": "",
    "Glucose": "mg/dL",
    "BloodPressure": "mmHg",
    "SkinThickness": "mm",
    "Insulin": "µU/mL",
    "BMI": "kg/m²",
    "DiabetesPedigreeFunction": "",
    "Age": "years",
}

FEATURE_STEPS = {
    "Pregnancies": 1,
    "Glucose": 1,
    "BloodPressure": 1,
    "SkinThickness": 1,
    "Insulin": 1,
    "BMI": 0.1,
    "DiabetesPedigreeFunction": 0.001,
    "Age": 1,
}

INTEGER_FEATURES = {
    feature
    for feature in FEATURE_COLUMNS
    if FEATURE_RULES[feature]["integer"]
}

TEAL = "#14b8a6"
RED = "#ef4444"
AMBER = "#f59e0b"
SLATE = "#94a3b8"


# Apply consistent simulator styling.
def apply_what_if_styles():
    st.html(
        """
        <style>
            .what-if-card {
                background: rgba(128, 150, 175, 0.055);
                border: 1px solid rgba(128, 150, 175, 0.22);
                border-radius: 12px;
                padding: 22px;
                min-height: 155px;
                width: 100%;
                min-width: 0;
                box-sizing: border-box;
                overflow: hidden;
                transition: border-color 0.2s ease;
            }

            .what-if-card:hover {
                border-color: rgba(20, 184, 166, 0.55);
            }

            .what-if-card-label {
                font-size: 14px;
                font-weight: 500;
                opacity: 0.85;
                line-height: 1.4;
            }

            .what-if-card-value {
                font-size: 36px;
                font-weight: 650;
                line-height: 1.2;
                letter-spacing: -0.8px;
                margin-top: 12px;
                font-variant-numeric: tabular-nums;
                overflow-wrap: anywhere;
            }

            .what-if-card-description {
                font-size: 12px;
                opacity: 0.7;
                line-height: 1.5;
                margin-top: 10px;
            }

            .what-if-card-status {
                display: inline-block;
                border-radius: 6px;
                padding: 5px 10px;
                margin-top: 12px;
                font-size: 12px;
                font-weight: 600;
            }

            .what-if-baseline {
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 14px;
                padding: 16px 20px;
                border: 1px solid rgba(20, 184, 166, 0.28);
                border-radius: 10px;
                background: rgba(20, 184, 166, 0.045);
            }

            .what-if-baseline-label {
                font-size: 12px;
                opacity: 0.7;
                margin-bottom: 5px;
            }

            .what-if-baseline-value {
                font-size: 16px;
                font-weight: 600;
            }

            .what-if-baseline-status {
                font-size: 12px;
                font-weight: 600;
                color: #14b8a6;
                background: rgba(20, 184, 166, 0.12);
                border-radius: 6px;
                padding: 6px 10px;
            }

            .what-if-measurement {
                padding-top: 4px;
                padding-bottom: 2px;
            }

            .what-if-measurement-label {
                font-size: 14px;
                font-weight: 550;
                margin-bottom: 7px;
            }

            .what-if-measurement-original {
                font-size: 12px;
                opacity: 0.65;
                line-height: 1.4;
            }

            .what-if-measurement-change {
                font-size: 12px;
                font-weight: 500;
                margin-top: -5px;
                margin-bottom: 12px;
            }

            .what-if-progress-track {
                height: 8px;
                background: rgba(128, 150, 175, 0.16);
                border-radius: 6px;
                overflow: hidden;
                margin-top: 16px;
            }

            .what-if-progress-fill {
                height: 100%;
                border-radius: 6px;
            }

            .what-if-transition {
                display: flex;
                align-items: center;
                flex-wrap: wrap;
                gap: 12px;
                padding: 14px 0;
                font-size: 18px;
                font-weight: 600;
            }

            .what-if-transition-arrow {
                opacity: 0.5;
                font-weight: 400;
            }

            .what-if-insight-label {
                font-size: 12px;
                opacity: 0.7;
                margin-bottom: 5px;
            }

            .what-if-insight-value {
                font-size: 25px;
                font-weight: 650;
                font-variant-numeric: tabular-nums;
            }

            .what-if-empty-state {
                border: 1px dashed rgba(128, 150, 175, 0.3);
                border-radius: 12px;
                padding: 36px 20px;
                text-align: center;
                background: rgba(128, 150, 175, 0.025);
            }

            .what-if-empty-title {
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 10px;
            }

            .what-if-empty-description {
                font-size: 13px;
                opacity: 0.7;
                max-width: 480px;
                margin: 0 auto;
                line-height: 1.6;
            }

            @media (max-width: 768px) {
                .what-if-card {
                    padding: 16px;
                    min-height: 140px;
                }

                .what-if-card-value {
                    font-size: 30px;
                }

                .what-if-baseline {
                    padding: 14px;
                }
            }
        </style>
        """
    )


# Load the trained model and configuration.
@st.cache_resource
def load_model():
    pipeline = joblib.load(MODEL_PATH)
    config = joblib.load(CONFIG_PATH)

    return pipeline, config


# Use the same measurement limits as prediction validation.
def get_measurement_range(feature):
    rules = FEATURE_RULES[feature]

    minimum = rules["minimum"]
    maximum = rules["maximum"]
    step = FEATURE_STEPS[feature]

    if feature in INTEGER_FEATURES:
        return int(minimum), int(maximum), int(step)

    return float(minimum), float(maximum), float(step)


# Return the appropriate number format.
def get_measurement_format(feature):
    if feature in INTEGER_FEATURES:
        return "%d"

    if feature == "DiabetesPedigreeFunction":
        return "%.3f"

    return "%.1f"


# Format measurement values for display.
def format_measurement(feature, value, include_unit=True):
    if feature in INTEGER_FEATURES:
        formatted = f"{int(value):,}"

    elif feature == "DiabetesPedigreeFunction":
        formatted = f"{float(value):.3f}"

    else:
        formatted = f"{float(value):.1f}"

    unit = FEATURE_UNITS[feature]

    if include_unit and unit:
        return f"{formatted} {unit}"

    return formatted


# Normalize measurements into their expected numerical types.
def normalize_measurements(measurements):
    normalized = {}

    for feature in FEATURE_COLUMNS:
        value = measurements[feature]

        if feature in INTEGER_FEATURES:
            normalized[feature] = int(value)
        else:
            normalized[feature] = float(value)

    return normalized


# Generate predictions using the existing trained model.
def calculate_prediction(pipeline, measurements, threshold):
    dataframe = prepare_prediction_dataframe(measurements)

    probabilities = pipeline.predict_proba(dataframe)

    positive_index = list(pipeline.classes_).index(1)

    probability = float(
        probabilities[0][positive_index]
    )

    if not isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(
            "The model returned an invalid probability."
        )

    return {
        "probability": probability,
        "classification": (
            "Positive"
            if probability >= threshold
            else "Negative"
        ),
    }


# Convert a database record into model measurements.
def measurements_from_record(record):
    return {
        feature: record[database_column]
        for feature, database_column
        in DATABASE_COLUMNS.items()
    }


# Initialize simulator state.
def initialize_simulator_state():
    defaults = {
        "what_if_baseline": None,
        "what_if_source": None,
        "what_if_baseline_warnings": {},
        "what_if_baseline_source": "Manual Measurements",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# Clear the current simulation.
def clear_simulation():
    st.session_state["what_if_baseline"] = None
    st.session_state["what_if_source"] = None
    st.session_state["what_if_baseline_warnings"] = {}

    for feature in FEATURE_COLUMNS:
        st.session_state.pop(
            f"what_if_slider_{feature}",
            None,
        )


# Store a validated baseline.
def set_baseline(measurements, source):
    validation = validate_prediction_input(
        measurements
    )

    if not validation.is_valid:
        raise ValueError(
            " ".join(validation.errors.values())
        )

    normalized = normalize_measurements(
        validation.cleaned_data
    )

    st.session_state["what_if_baseline"] = (
        normalized.copy()
    )

    st.session_state["what_if_source"] = source

    st.session_state["what_if_baseline_warnings"] = (
        validation.warnings.copy()
    )

    for feature, value in normalized.items():
        st.session_state[
            f"what_if_slider_{feature}"
        ] = value


# Reset sliders safely before the next rerun.
def reset_simulation():
    baseline = st.session_state.get(
        "what_if_baseline"
    )

    if baseline is None:
        return

    for feature, value in baseline.items():
        st.session_state[
            f"what_if_slider_{feature}"
        ] = value


# Calculate the number of modified measurements.
def count_modified_measurements(original, simulated):
    return sum(
        abs(
            float(simulated[feature])
            - float(original[feature])
        ) > 1e-8
        for feature in FEATURE_COLUMNS
    )


# Return the classification display color.
def get_classification_color(classification):
    return RED if classification == "Positive" else TEAL


# Render a reusable analytical summary card.
def render_summary_card(
    label,
    value,
    description,
    color=None,
    status=None,
    probability=None,
):
    color_style = (
        f"color: {color};"
        if color
        else ""
    )

    status_html = ""

    if status:
        status_color = get_classification_color(
            status
        )

        status_html = f"""
            <div
                class="what-if-card-status"
                style="
                    color: {status_color};
                    background: {status_color}18;
                "
            >
                {escape(status)} Classification
            </div>
        """

    progress_html = ""

    if probability is not None:
        percentage = max(
            0,
            min(100, probability * 100),
        )

        progress_html = f"""
            <div class="what-if-progress-track">
                <div
                    class="what-if-progress-fill"
                    style="
                        width: {percentage:.2f}%;
                        background: {color or TEAL};
                    "
                ></div>
            </div>
        """

    st.html(
        f"""
        <div class="what-if-card">
            <div class="what-if-card-label">
                {escape(str(label))}
            </div>

            <div
                class="what-if-card-value"
                style="{color_style}"
            >
                {escape(str(value))}
            </div>

            {status_html}

            <div class="what-if-card-description">
                {escape(str(description))}
            </div>

            {progress_html}
        </div>
        """
    )


# Display the manual baseline form.
def render_manual_baseline():
    with st.container(border=True):
        st.markdown("**Original Measurements**")

        st.caption(
            "Enter all eight measurements to establish "
            "the starting point for your simulation."
        )

        with st.form(
            "what_if_baseline_form",
            border=False,
        ):
            columns = st.columns(2)

            manual_inputs = {}

            for index, feature in enumerate(
                FEATURE_COLUMNS
            ):
                minimum, maximum, step = (
                    get_measurement_range(feature)
                )

                label = FEATURE_LABELS[feature]
                unit = FEATURE_UNITS[feature]

                if unit:
                    label = f"{label} ({unit})"

                with columns[index % 2]:
                    manual_inputs[feature] = (
                        st.number_input(
                            label,
                            min_value=minimum,
                            max_value=maximum,
                            value=None,
                            step=step,
                            format=get_measurement_format(
                                feature
                            ),
                            placeholder="Enter value",
                            key=f"what_if_manual_{feature}",
                        )
                    )

            st.divider()

            spacer, action_column = st.columns(
                [2, 1]
            )

            with action_column:
                submitted = st.form_submit_button(
                    "Set Baseline",
                    type="primary",
                    width="stretch",
                )

        if not submitted:
            return

        validation = validate_prediction_input(
            manual_inputs
        )

        if not validation.is_valid:
            st.error(
                "Please review the measurements below."
            )

            for message in validation.errors.values():
                st.caption(f"• {message}")

            return

        set_baseline(
            validation.cleaned_data,
            "Manual Measurements",
        )

        st.rerun()


# Load a saved baseline from prediction history.
def render_saved_baseline():
    with st.container(border=True):
        st.markdown("**Saved Prediction**")

        st.caption(
            "Use measurements from a previously "
            "saved prediction as your baseline."
        )

        try:
            recent_records = get_all_predictions(
                limit=50,
                offset=0,
            )

        except Exception:
            st.error(
                "Unable to retrieve prediction history. "
                "Please check the database connection."
            )

            return

        if not recent_records:
            st.info(
                "No saved predictions are available."
            )

            st.page_link(
                "pages/prediction.py",
                label="Create a Prediction",
                icon=":material/arrow_forward:",
            )

            return

        record_options = {
            int(record["id"]): record
            for record in recent_records
        }

        def format_record_option(record_id):
            record = record_options[record_id]

            classification = record[
                "classification"
            ]

            probability = float(
                record["probability"]
            ) * 100

            return (
                f"Prediction #{record_id} — "
                f"{classification} · "
                f"{probability:.1f}%"
            )

        select_column, action_column = st.columns(
            [3, 1],
            vertical_alignment="bottom",
        )

        with select_column:
            selected_id = st.selectbox(
                "Select a Saved Prediction",
                options=list(record_options),
                index=None,
                placeholder="Choose a saved prediction",
                format_func=format_record_option,
                key="what_if_saved_prediction_id",
            )

        with action_column:
            load_clicked = st.button(
                "Load Prediction",
                type="primary",
                width="stretch",
                disabled=selected_id is None,
                key="what_if_load_prediction",
            )

        st.caption(
            "Showing up to 50 recent saved predictions."
        )

        if selected_id is not None:
            record = record_options[selected_id]

            preview_columns = st.columns(3)

            with preview_columns[0]:
                st.metric(
                    "Prediction ID",
                    f"#{selected_id}",
                )

            with preview_columns[1]:
                st.metric(
                    "Classification",
                    record["classification"],
                )

            with preview_columns[2]:
                st.metric(
                    "Saved Probability",
                    f"{float(record['probability']) * 100:.1f}%",
                )

            with st.expander("Preview Measurements"):
                measurements = measurements_from_record(
                    record
                )

                preview = pd.DataFrame([
                    {
                        "Measurement": FEATURE_LABELS[
                            feature
                        ],
                        "Value": format_measurement(
                            feature,
                            measurements[feature],
                        ),
                    }
                    for feature in FEATURE_COLUMNS
                ])

                st.dataframe(
                    preview,
                    width="stretch",
                    hide_index=True,
                )

        if not load_clicked:
            return

        try:
            record = get_prediction_by_id(
                selected_id
            )

        except Exception:
            st.error(
                "Unable to load the selected prediction."
            )

            return

        if record is None:
            st.error(
                "The selected prediction could not be found."
            )

            return

        try:
            measurements = measurements_from_record(
                record
            )

            set_baseline(
                measurements,
                f"Saved Prediction #{selected_id}",
            )

        except ValueError as error:
            st.error(
                f"Unable to use this baseline: {error}"
            )

            return

        st.rerun()


# Render the baseline selection controls.
def render_baseline_selector():
    st.subheader("Simulation Baseline")

    source = st.segmented_control(
        "Baseline Source",
        options=[
            "Manual Measurements",
            "Saved Prediction",
        ],
        selection_mode="single",
        key="what_if_baseline_source",
        label_visibility="collapsed",
        width="stretch",
        on_change=clear_simulation,
    )

    if source == "Saved Prediction":
        render_saved_baseline()

    else:
        render_manual_baseline()


# Display the active baseline information.
def render_active_baseline(baseline):
    source = st.session_state.get(
        "what_if_source",
        "Unknown",
    )

    st.html(
        f"""
        <div class="what-if-baseline">
            <div>
                <div class="what-if-baseline-label">
                    Active Baseline
                </div>

                <div class="what-if-baseline-value">
                    {escape(str(source))}
                </div>
            </div>

            <div class="what-if-baseline-status">
                Baseline Ready
            </div>
        </div>
        """
    )

    with st.expander("View Original Measurements"):
        rows = []

        for feature in FEATURE_COLUMNS:
            rows.append({
                "Measurement": FEATURE_LABELS[
                    feature
                ],
                "Original Value": format_measurement(
                    feature,
                    baseline[feature],
                ),
            })

        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
        )

        warnings = st.session_state.get(
            "what_if_baseline_warnings",
            {},
        )

        if warnings:
            st.markdown("**Baseline Warnings**")

            for message in warnings.values():
                st.warning(message)


# Display an empty simulator state.
def render_empty_state():
    st.html(
        """
        <div class="what-if-empty-state">
            <div class="what-if-empty-title">
                No Active Simulation
            </div>

            <div class="what-if-empty-description">
                Establish a baseline using manual measurements
                or a saved prediction. You can then adjust
                individual measurements and compare the
                resulting model predictions in real time.
            </div>
        </div>
        """
    )


# Render an individual simulation slider.
def render_measurement_slider(feature, baseline_value):
    minimum, maximum, step = (
        get_measurement_range(feature)
    )

    label = FEATURE_LABELS[feature]
    unit = FEATURE_UNITS[feature]

    if unit:
        label = f"{label} ({unit})"

    original_text = format_measurement(
        feature,
        baseline_value,
    )

    st.html(
        f"""
        <div class="what-if-measurement">
            <div class="what-if-measurement-label">
                {escape(label)}
            </div>

            <div class="what-if-measurement-original">
                Original: {escape(original_text)}
            </div>
        </div>
        """
    )

    slider_key = f"what_if_slider_{feature}"

    if slider_key not in st.session_state:
        st.session_state[slider_key] = baseline_value

    value = st.slider(
        label,
        min_value=minimum,
        max_value=maximum,
        step=step,
        format=get_measurement_format(feature),
        key=slider_key,
        label_visibility="collapsed",
    )

    difference = round(
        float(value) - float(baseline_value),
        8,
    )

    if abs(difference) < 1e-8:
        change_text = "Unchanged"
        color = SLATE

    else:
        if feature in INTEGER_FEATURES:
            formatted_change = f"{difference:+.0f}"

        elif feature == "DiabetesPedigreeFunction":
            formatted_change = f"{difference:+.3f}"

        else:
            formatted_change = f"{difference:+.1f}"

        change_text = (
            f"Change: {formatted_change}"
            f"{' ' + unit if unit else ''}"
        )

        color = AMBER

    st.html(
        f"""
        <div
            class="what-if-measurement-change"
            style="color: {color};"
        >
            {escape(change_text)}
        </div>
        """
    )

    return value


# Render all interactive measurement controls.
def render_simulation_controls(baseline):
    st.subheader("Adjust Measurements")

    st.caption(
        "Explore hypothetical scenarios by adjusting "
        "the measurements below. Predictions update automatically."
    )

    simulated_inputs = {}

    with st.container(border=True):
        heading_column, reset_column = st.columns(
            [2, 1],
            vertical_alignment="center",
        )

        with heading_column:
            st.markdown("**Simulation Controls**")

        with reset_column:
            st.button(
                "Reset Measurements",
                icon=":material/restart_alt:",
                width="stretch",
                on_click=reset_simulation,
                key="what_if_reset_measurements",
            )

        st.divider()

        columns = st.columns(
            2,
            gap="large",
        )

        left_features = [
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
        ]

        right_features = [
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
            "Pregnancies",
        ]

        with columns[0]:
            st.markdown("**Health Measurements**")

            for feature in left_features:
                simulated_inputs[feature] = (
                    render_measurement_slider(
                        feature,
                        baseline[feature],
                    )
                )

        with columns[1]:
            st.markdown("**Participant Measurements**")

            for feature in right_features:
                simulated_inputs[feature] = (
                    render_measurement_slider(
                        feature,
                        baseline[feature],
                    )
                )

        st.divider()

        modified_count = count_modified_measurements(
            baseline,
            simulated_inputs,
        )

        st.caption(
            f"{modified_count} of "
            f"{len(FEATURE_COLUMNS)} measurements modified."
        )

    return simulated_inputs


# Display validation errors and unusual measurements.
def render_simulation_validation(validation):
    if not validation.is_valid:
        with st.container(border=True):
            st.error(
                "The simulated measurements contain "
                "invalid values."
            )

            for message in validation.errors.values():
                st.caption(f"• {message}")

        return False

    if validation.warnings:
        warning_count = len(
            validation.warnings
        )

        with st.expander(
            f"Measurement Warnings · {warning_count}",
            expanded=False,
        ):
            for message in validation.warnings.values():
                st.warning(message)

    return True


# Render the primary simulation result cards.
def render_result_cards(
    original,
    simulated,
):
    probability_change = (
        simulated["probability"]
        - original["probability"]
    ) * 100

    if abs(probability_change) < 1e-8:
        change_color = SLATE
        change_description = (
            "No change in model probability"
        )
        change_value = "0.00"

    elif probability_change > 0:
        change_color = RED
        change_description = (
            "Increase in model probability"
        )
        change_value = f"{probability_change:+.2f}"

    else:
        change_color = TEAL
        change_description = (
            "Decrease in model probability"
        )
        change_value = f"{probability_change:+.2f}"

    columns = st.columns(3)

    with columns[0]:
        render_summary_card(
            label="Original Prediction",
            value=f"{original['probability'] * 100:.2f}%",
            description="Baseline positive-class probability",
            color=get_classification_color(
                original["classification"]
            ),
            status=original["classification"],
            probability=original["probability"],
        )

    with columns[1]:
        render_summary_card(
            label="Simulated Prediction",
            value=f"{simulated['probability'] * 100:.2f}%",
            description="Simulated positive-class probability",
            color=get_classification_color(
                simulated["classification"]
            ),
            status=simulated["classification"],
            probability=simulated["probability"],
        )

    with columns[2]:
        render_summary_card(
            label="Probability Change",
            value=f"{change_value} pp",
            description=change_description,
            color=change_color,
        )


# Build a probability comparison chart.
def create_comparison_chart(
    original,
    simulated,
    threshold,
):
    labels = [
        "Original",
        "Simulated",
    ]

    values = [
        original["probability"] * 100,
        simulated["probability"] * 100,
    ]

    colors = [
        get_classification_color(
            original["classification"]
        ),
        get_classification_color(
            simulated["classification"]
        ),
    ]

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=colors,
                line=dict(width=0),
            ),
            width=0.38,
            text=[
                f"{value:.2f}%"
                for value in values
            ],
            textposition="outside",
            textfont=dict(size=14),
            cliponaxis=False,
            hovertemplate=(
                "%{x}<br>"
                "Probability: %{y:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    figure.add_hline(
        y=threshold * 100,
        line_dash="dash",
        line_color=AMBER,
        line_width=1.5,
        annotation_text=(
            f"Threshold: {threshold * 100:.0f}%"
        ),
        annotation_position="top right",
        annotation_font=dict(
            size=11,
            color=AMBER,
        ),
    )

    figure.update_layout(
        height=330,
        margin=dict(
            l=20,
            r=25,
            t=25,
            b=20,
        ),
        yaxis=dict(
            title="Positive-Class Probability",
            range=[0, 110],
            tickvals=[0, 20, 40, 60, 80, 100],
            ticksuffix="%",
            gridcolor="rgba(128,150,175,0.12)",
            zeroline=False,
        ),
        xaxis=dict(
            title="",
            showgrid=False,
        ),
        font=dict(size=12),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return figure


# Explain the simulated classification transition.
def render_simulation_insights(
    original,
    simulated,
    baseline,
    simulated_measurements,
    threshold,
):
    original_class = original["classification"]
    simulated_class = simulated["classification"]

    original_color = get_classification_color(
        original_class
    )

    simulated_color = get_classification_color(
        simulated_class
    )

    changed = original_class != simulated_class

    modified_count = count_modified_measurements(
        baseline,
        simulated_measurements,
    )

    probability_change = (
        simulated["probability"]
        - original["probability"]
    ) * 100

    st.markdown("**Classification Transition**")

    st.html(
        f"""
        <div class="what-if-transition">
            <span style="color: {original_color};">
                {original_class}
            </span>

            <span class="what-if-transition-arrow">
                &rarr;
            </span>

            <span style="color: {simulated_color};">
                {simulated_class}
            </span>
        </div>
        """
    )

    if changed:
        st.caption(
            "The simulated prediction crossed "
            "the classification threshold."
        )

    else:
        st.caption(
            "The model classification remains unchanged."
        )

    st.divider()

    first, second = st.columns(2)

    with first:
        st.html(
            f"""
            <div class="what-if-insight-label">
                Modified Measurements
            </div>

            <div class="what-if-insight-value">
                {modified_count}/{len(FEATURE_COLUMNS)}
            </div>
            """
        )

    with second:
        st.html(
            f"""
            <div class="what-if-insight-label">
                Model Threshold
            </div>

            <div class="what-if-insight-value">
                {threshold * 100:.0f}%
            </div>
            """
        )

    st.divider()

    st.markdown("**Simulation Insight**")

    if abs(probability_change) < 1e-8:
        description = (
            "The model's positive-class probability "
            "has not changed."
        )

    elif probability_change > 0:
        description = (
            "The simulated measurements increased "
            "the model's positive-class probability "
            f"by {probability_change:.2f} percentage points."
        )

    else:
        description = (
            "The simulated measurements decreased "
            "the model's positive-class probability "
            f"by {abs(probability_change):.2f} percentage points."
        )

    st.caption(description)


# Render the complete prediction comparison section.
def render_prediction_comparison(
    original,
    simulated,
    baseline,
    simulated_measurements,
    threshold,
):
    st.subheader("Simulation Results")

    render_result_cards(
        original,
        simulated,
    )

    st.caption(
        "pp = percentage points. Probability changes "
        "describe model outputs, not changes in actual "
        "diabetes risk."
    )

    chart_column, insight_column = st.columns(
        [1.2, 1],
        gap="medium",
    )

    with chart_column:
        with st.container(border=True):
            st.markdown("**Probability Comparison**")

            st.caption(
                "Original and simulated model probabilities."
            )

            figure = create_comparison_chart(
                original,
                simulated,
                threshold,
            )

            st.plotly_chart(
                figure,
                width="stretch",
                theme="streamlit",
                config={
                    "displaylogo": False,
                    "responsive": True,
                },
                key="what_if_probability_chart",
            )

    with insight_column:
        with st.container(border=True):
            render_simulation_insights(
                original,
                simulated,
                baseline,
                simulated_measurements,
                threshold,
            )


# Build the measurement comparison table.
def build_measurement_comparison(
    original,
    simulated,
):
    rows = []

    for feature in FEATURE_COLUMNS:
        original_value = original[feature]
        simulated_value = simulated[feature]

        difference = round(
            float(simulated_value)
            - float(original_value),
            8,
        )

        changed = abs(difference) > 1e-8

        if not changed:
            change_text = "—"

        elif feature in INTEGER_FEATURES:
            change_text = f"{difference:+.0f}"

        elif feature == "DiabetesPedigreeFunction":
            change_text = f"{difference:+.3f}"

        else:
            change_text = f"{difference:+.1f}"

        rows.append({
            "Measurement": FEATURE_LABELS[
                feature
            ],
            "Original": format_measurement(
                feature,
                original_value,
                include_unit=False,
            ),
            "Simulated": format_measurement(
                feature,
                simulated_value,
                include_unit=False,
            ),
            "Change": change_text,
            "Unit": FEATURE_UNITS[feature] or "—",
            "Status": (
                "Modified"
                if changed
                else "Unchanged"
            ),
        })

    return pd.DataFrame(rows)


# Apply subtle styling to modified measurements.
def style_measurement_comparison(dataframe):
    def style_status(value):
        if value == "Modified":
            return (
                f"color: {AMBER}; "
                "font-weight: 600;"
            )

        return (
            f"color: {SLATE};"
        )

    def style_change(value):
        if value != "—":
            return (
                f"color: {AMBER}; "
                "font-weight: 600;"
            )

        return (
            f"color: {SLATE};"
        )

    return (
        dataframe.style
        .map(
            style_status,
            subset=["Status"],
        )
        .map(
            style_change,
            subset=["Change"],
        )
    )


# Render original and simulated measurements.
def render_measurement_comparison(
    original,
    simulated,
):
    st.subheader("Measurement Comparison")

    with st.container(border=True):
        heading_column, filter_column = st.columns(
            [2, 1],
            vertical_alignment="center",
        )

        with heading_column:
            st.markdown(
                "**Original vs. Simulated Measurements**"
            )

            modified_count = (
                count_modified_measurements(
                    original,
                    simulated,
                )
            )

            st.caption(
                f"{modified_count} of "
                f"{len(FEATURE_COLUMNS)} measurements modified."
            )

        with filter_column:
            comparison_filter = st.segmented_control(
                "Measurement Filter",
                options=[
                    "All",
                    "Modified",
                ],
                default="All",
                selection_mode="single",
                label_visibility="collapsed",
                width="stretch",
                key="what_if_comparison_filter",
            )

        dataframe = build_measurement_comparison(
            original,
            simulated,
        )

        if comparison_filter == "Modified":
            dataframe = dataframe[
                dataframe["Status"] == "Modified"
            ].copy()

        if dataframe.empty:
            st.info(
                "No measurements have been modified."
            )

            return

        styled = style_measurement_comparison(
            dataframe
        )

        st.dataframe(
            styled,
            width="stretch",
            hide_index=True,
            height=38 + len(dataframe) * 35,
            column_config={
                "Measurement": st.column_config.TextColumn(
                    width="large",
                    pinned=True,
                ),
                "Original": st.column_config.TextColumn(
                    width="small",
                ),
                "Simulated": st.column_config.TextColumn(
                    width="small",
                ),
                "Change": st.column_config.TextColumn(
                    width="small",
                ),
                "Unit": st.column_config.TextColumn(
                    width="small",
                ),
                "Status": st.column_config.TextColumn(
                    width="medium",
                ),
            },
        )

        st.caption(
            "Highlighted values indicate modified "
            "measurements, not clinical improvement "
            "or deterioration."
        )


# Export the current validated simulation without saving it to history.
def render_simulation_report(baseline, simulated, original_result, simulated_result, threshold, model_config, warnings):
    st.subheader("Simulation Report")
    with st.container(border=True):
        st.caption("Download the current comparison, including all eight measurements and any validation warnings.")
        try:
            pdf = generate_what_if_report(
                baseline, simulated, original_result, simulated_result,
                threshold=threshold,
                source=st.session_state.get("what_if_source") or "Manual Measurements",
                model_name=model_config.get("model_name", "Random Forest"),
                model_version=model_config.get("model_version", "v1.0"),
                baseline_warnings=st.session_state.get("what_if_baseline_warnings", {}),
                simulation_warnings=warnings,
            )
        except Exception:
            st.error("Unable to generate the simulation report.")
            return
        st.download_button(
            "Download Simulation Report (PDF)", data=pdf,
            file_name=generate_report_filename("What_If_Simulation"),
            mime="application/pdf", icon=":material/download:",
            on_click="ignore", width="stretch", key="what_if_pdf_download",
        )


# Render model information and simulator limitations.
def render_simulator_information(
    model_config,
    threshold,
):
    with st.expander("Model Information & Limitations"):
        columns = st.columns(3)

        with columns[0]:
            st.metric(
                "Model",
                model_config.get(
                    "model_name",
                    "Random Forest",
                ),
            )

        with columns[1]:
            st.metric(
                "Classification Threshold",
                f"{threshold * 100:.0f}%",
            )

        with columns[2]:
            st.metric(
                "Input Measurements",
                len(FEATURE_COLUMNS),
            )

        st.divider()

        st.markdown("**How the Simulator Works**")

        st.write(
            "The trained classification model evaluates "
            "the original measurements and the modified "
            "measurements separately. The resulting "
            "positive-class probabilities are compared "
            "using the same classification threshold."
        )

        st.markdown("**Interpretation**")

        st.write(
            "A change in predicted probability reflects "
            "the model's response to different input "
            "values. It does not establish a causal "
            "relationship or predict the effect of "
            "medical treatment."
        )

        st.markdown("**Prediction History**")

        st.write(
            "Simulated predictions are not automatically "
            "saved to Prediction History. The original "
            "saved prediction remains unchanged."
        )


# Render the educational disclaimer.
def render_disclaimer():
    st.divider()

    st.caption(
        "UODA DiaML · Educational What-If Prediction Simulator. "
        "Hypothetical model outputs are not medical "
        "advice, diagnoses or clinical recommendations."
    )


# Initialize the simulator page.
apply_what_if_styles()

initialize_simulator_state()

st.title("What-If Prediction Simulator")

st.caption(
    "Explore hypothetical measurement changes and "
    "compare their effects on the trained model's predictions."
)

st.divider()

try:
    pipeline, model_config = load_model()

    threshold = float(
        model_config.get("threshold", 0.40)
    )

    if not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(
            "Invalid classification threshold."
        )

except Exception:
    st.error(
        "Unable to load the trained model. "
        "Please check the model files and configuration."
    )

    st.stop()


baseline = st.session_state.get(
    "what_if_baseline"
)

if baseline is None:
    render_baseline_selector()

    render_empty_state()

    render_simulator_information(
        model_config,
        threshold,
    )

    render_disclaimer()

    st.stop()


render_active_baseline(baseline)

with st.expander(
    "Change Simulation Baseline",
    expanded=False,
):
    render_baseline_selector()


st.divider()

simulated_inputs = render_simulation_controls(
    baseline
)

simulation_validation = validate_prediction_input(
    simulated_inputs
)

if not render_simulation_validation(
    simulation_validation
):
    render_disclaimer()

    st.stop()


try:
    original_result = calculate_prediction(
        pipeline,
        baseline,
        threshold,
    )

    simulated_result = calculate_prediction(
        pipeline,
        simulation_validation.cleaned_data,
        threshold,
    )

except Exception:
    st.error(
        "Unable to calculate the simulation results."
    )

    render_disclaimer()

    st.stop()


st.divider()

render_prediction_comparison(
    original_result,
    simulated_result,
    baseline,
    simulation_validation.cleaned_data,
    threshold,
)

st.divider()

render_measurement_comparison(
    baseline,
    simulation_validation.cleaned_data,
)

render_simulation_report(
    baseline, simulation_validation.cleaned_data, original_result,
    simulated_result, threshold, model_config, simulation_validation.warnings,
)

render_simulator_information(
    model_config,
    threshold,
)

render_disclaimer()