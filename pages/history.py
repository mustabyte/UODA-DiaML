
from datetime import date, datetime, timezone
from math import ceil

import pandas as pd
import streamlit as st

from database.repository import (
    get_all_predictions,
    get_prediction_by_id,
    get_prediction_count,
)


# Define the measurements stored with each prediction.
MEASUREMENT_LABELS = {
    "pregnancies": ("Pregnancies", "—"),
    "glucose": ("Glucose", "mg/dL"),
    "blood_pressure": ("Diastolic Blood Pressure", "mmHg"),
    "skin_thickness": ("Skin Thickness", "mm"),
    "insulin": ("Insulin", "µU/mL"),
    "bmi": ("Body Mass Index", "kg/m²"),
    "diabetes_pedigree_function": ("Diabetes Pedigree Function", "—"),
    "age": ("Age", "years"),
}

PAGE_SIZE = 10


# Format database timestamps consistently in UTC.
def format_timestamp(value):
    if value is None:
        return "—"

    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc).strftime("%d %b %Y, %I:%M %p UTC")


# Convert a stored prediction into a readable table row.
def format_prediction_row(record):
    return {
        "ID": int(record["id"]),
        "Date & Time": format_timestamp(record["created_at"]),
        "Classification": record["classification"],
        "Probability": f'{float(record["probability"]) * 100:.1f}%',
        "Glucose": f'{float(record["glucose"]):.1f}',
        "BMI": f'{float(record["bmi"]):.1f}',
        "Age": int(record["age"]),
        "Model": record["model_name"],
    }


# Display the overview metrics.
def render_history_overview():
    total = get_prediction_count()
    positive = get_prediction_count("Positive")
    negative = get_prediction_count("Negative")

    columns = st.columns(3)

    with columns[0]:
        st.metric("Total Predictions", f"{total:,}")

    with columns[1]:
        st.metric("Positive Predictions", f"{positive:,}")

    with columns[2]:
        st.metric("Negative Predictions", f"{negative:,}")

    return total


# Build the history search and filter controls.
def render_history_filters():
    st.subheader("Search & Filters")

    with st.container(border=True):
        first_column, second_column = st.columns(2)

        with first_column:
            search_id = st.text_input(
                "Prediction ID",
                placeholder="Search by ID",
                key="history_search_id",
            )

        with second_column:
            classification = st.selectbox(
                "Classification",
                options=["All", "Positive", "Negative"],
                key="history_classification",
            )

        date_column_1, date_column_2 = st.columns(2)

        with date_column_1:
            start_date = st.date_input(
                "From Date (UTC)",
                value=None,
                key="history_start_date",
            )

        with date_column_2:
            end_date = st.date_input(
                "To Date (UTC)",
                value=None,
                key="history_end_date",
            )

    if search_id.strip():
        if not search_id.strip().isdigit():
            st.warning("Prediction ID must contain numbers only.")
            return None

        parsed_id = int(search_id.strip())

        if parsed_id <= 0:
            st.warning("Prediction ID must be greater than zero.")
            return None
    else:
        parsed_id = None

    if start_date is not None and end_date is not None:
        if start_date > end_date:
            st.warning("The start date cannot be after the end date.")
            return None

    return {
        "classification": (
            None if classification == "All" else classification
        ),
        "search_id": parsed_id,
        "start_date": start_date,
        "end_date": end_date,
    }


# Retrieve filtered records in manageable pages.
def load_history_page(filters, page):
    return get_all_predictions(
        classification=filters["classification"],
        search_id=filters["search_id"],
        start_date=filters["start_date"],
        end_date=filters["end_date"],
        limit=PAGE_SIZE + 1,
        offset=(page - 1) * PAGE_SIZE,
    )


# Show all eight measurements for a selected prediction.
def render_prediction_details(record):
    st.subheader(f'Prediction Details — #{record["id"]}')

    with st.container(border=True):
        st.caption(format_timestamp(record["created_at"]))

        result_column, probability_column, threshold_column = st.columns(3)

        with result_column:
            st.metric("Classification", record["classification"])

        with probability_column:
            st.metric(
                "Positive-Class Probability",
                f'{float(record["probability"]) * 100:.1f}%',
            )

        with threshold_column:
            st.metric(
                "Decision Threshold",
                f'{float(record["threshold"]) * 100:.0f}%',
            )

        st.divider()
        st.markdown("#### Submitted Measurements")

        measurement_rows = []

        for column, (label, unit) in MEASUREMENT_LABELS.items():
            value = record[column]

            measurement_rows.append(
                {
                    "Measurement": label,
                    "Value": value,
                    "Unit": unit,
                }
            )

        st.dataframe(
            pd.DataFrame(measurement_rows),
            hide_index=True,
            use_container_width=True,
        )

        st.divider()

        model_column, version_column = st.columns(2)

        with model_column:
            st.markdown(f'**Model:** {record["model_name"]}')

        with version_column:
            st.markdown(f'**Version:** {record["model_version"]}')

        st.caption(
            "This record contains an educational machine learning "
            "prediction and must not be interpreted as a diagnosis."
        )


# Initialize the history page.
st.title("Prediction History")

st.caption(
    "Explore saved diabetes predictions, inspect their measurements, "
    "and review model-generated results."
)


# Initialize pagination and selected-record state.
if "history_page_number" not in st.session_state:
    st.session_state["history_page_number"] = 1

if "history_selected_id" not in st.session_state:
    st.session_state["history_selected_id"] = None


# Load the overview counts.
try:
    total_predictions = render_history_overview()

except Exception:
    st.error(
        "Unable to load prediction history. "
        "Please check the database connection."
    )
    st.stop()


st.divider()

filters = render_history_filters()

if filters is None:
    st.stop()


# Reset pagination whenever the search filters change.
filter_signature = (
    filters["classification"],
    filters["search_id"],
    filters["start_date"],
    filters["end_date"],
)

if st.session_state.get("history_previous_filters") != filter_signature:
    st.session_state["history_page_number"] = 1
    st.session_state["history_selected_id"] = None
    st.session_state["history_previous_filters"] = filter_signature


# Load the current page of matching records.
page_number = st.session_state["history_page_number"]

try:
    records = load_history_page(filters, page_number)

except Exception:
    st.error(
        "Unable to retrieve saved predictions. "
        "Please check the database connection."
    )
    st.stop()


# Fetch one extra row to determine whether a next page exists.
has_next_page = len(records) > PAGE_SIZE
visible_records = records[:PAGE_SIZE]


st.subheader("Saved Predictions")

if not visible_records:
    st.info(
        "No predictions match the selected filters. "
        "Try changing your search criteria."
    )

else:
    history_dataframe = pd.DataFrame(
        [format_prediction_row(record) for record in visible_records]
    )

    st.dataframe(
        history_dataframe,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        f"Showing page {page_number} "
        f"({len(visible_records)} predictions on this page)."
    )

    # Provide pagination controls without loading every database row.
    previous_column, page_column, next_column = st.columns([1, 2, 1])

    with previous_column:
        if st.button(
            "Previous",
            disabled=page_number == 1,
            use_container_width=True,
        ):
            st.session_state["history_page_number"] -= 1
            st.rerun()

    with page_column:
        st.markdown(
            f"<div style='text-align:center;padding-top:8px;'>"
            f"Page {page_number}</div>",
            unsafe_allow_html=True,
        )

    with next_column:
        if st.button(
            "Next",
            disabled=not has_next_page,
            use_container_width=True,
        ):
            st.session_state["history_page_number"] += 1
            st.rerun()

    st.divider()

    # Let users inspect a saved prediction by its ID.
    st.subheader("View Prediction Details")

    available_ids = [int(record["id"]) for record in visible_records]

    selected_id = st.selectbox(
        "Select a prediction",
        options=available_ids,
        index=None,
        placeholder="Choose a prediction ID",
        format_func=lambda prediction_id: f"Prediction #{prediction_id}",
        key="history_detail_selection",
    )

    if selected_id is not None:
        try:
            selected_record = get_prediction_by_id(selected_id)

            if selected_record is None:
                st.warning("The selected prediction no longer exists.")
            else:
                render_prediction_details(selected_record)

        except Exception:
            st.error(
                "Unable to load the selected prediction details."
            )


st.divider()

st.caption(
    "UODA DiaML | Educational Diabetes Prediction System | "
    "Prediction records are stored in Supabase PostgreSQL."
)
