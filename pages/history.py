
from datetime import datetime, timedelta, timezone
from math import ceil

import pandas as pd
import streamlit as st

from components.report_generator import (
    generate_individual_report,
    generate_saved_batch_report,
    generate_report_filename,
)

from database.repository import (
    FEATURE_COLUMNS,
    get_all_predictions,
    get_prediction_by_id,
    get_prediction_count,
    get_batch_predictions,
    get_available_batches,
)


PAGE_SIZE = 10

MEASUREMENT_LABELS = {
    "pregnancies": ("Pregnancies", ""),
    "glucose": ("Glucose", "mg/dL"),
    "blood_pressure": ("Diastolic Blood Pressure", "mmHg"),
    "skin_thickness": ("Skin Thickness", "mm"),
    "insulin": ("Insulin", "µU/mL"),
    "bmi": ("Body Mass Index", "kg/m²"),
    "diabetes_pedigree_function": (
        "Diabetes Pedigree Function",
        "",
    ),
    "age": ("Age", "years"),
}


# Apply styles specifically to the history page.
def apply_history_styles():
    st.html("""
<style>
.history-summary-card {
    background: rgba(128, 150, 175, 0.055);
    border: 1px solid rgba(128, 150, 175, 0.22);
    border-radius: 12px;
    padding: 22px;
    min-height: 142px;
    width: 100%;
    max-width: 100%;
    min-width: 0;
    overflow: hidden;
    box-sizing: border-box;
    transition: border-color 0.2s ease;
}

.history-summary-card:hover {
    border-color: rgba(20, 184, 166, 0.65);
}

.history-card-label {
    font-size: 14px;
    font-weight: 500;
    opacity: 0.85;
    line-height: 1.4;
    overflow-wrap: anywhere;
}

.history-card-value {
    font-size: 36px;
    font-weight: 650;
    line-height: 1.2;
    margin-top: 14px;
    letter-spacing: -0.8px;
    font-variant-numeric: tabular-nums;
}

.history-card-description {
    font-size: 12px;
    opacity: 0.65;
    margin-top: 10px;
    line-height: 1.4;
    overflow-wrap: anywhere;
}

.history-pagination {
    text-align: center;
    padding-top: 9px;
    font-size: 14px;
}

@media (max-width: 768px) {
    .history-summary-card {
        padding: 16px;
        min-height: 135px;
    }

    .history-card-value {
        font-size: 30px;
    }

    .history-card-label {
        font-size: 12px;
    }
}
</style>
""")


# Format database timestamps consistently in UTC.
def format_timestamp(value):
    if value is None:
        return "—"

    if isinstance(value, str):
        value = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc).strftime(
        "%d %b %Y, %I:%M %p UTC"
    )


# Format measurements without unnecessary decimal zeros.
def format_measurement(value):
    if value is None:
        return "—"

    return f"{float(value):g}"


# Convert a saved prediction into a compact table row.
def format_prediction_row(record, include_source=True):
    row = {
        "ID": int(record["id"]),
        "Date & Time": format_timestamp(record["created_at"]),
    }

    if include_source:
        row["Source"] = record["source"].capitalize()

    row.update({
        "Classification": record["classification"],
        "Probability": float(record["probability"]),
        "Glucose": float(record["glucose"]),
        "BMI": float(record["bmi"]),
        "Age": int(record["age"]),
    })

    return row


# Render a clean summary card without icons.
def render_summary_card(label, value, description):
    html = f"""
<div class="history-summary-card">
    <div class="history-card-label">{label}</div>
    <div class="history-card-value">{value}</div>
    <div class="history-card-description">{description}</div>
</div>
"""

    st.html(html)


# Load all overview statistics.
def load_history_statistics(batches):
    total = get_prediction_count()
    positive = get_prediction_count("Positive")
    negative = get_prediction_count("Negative")
    individual = get_prediction_count(source="individual")
    batch = get_prediction_count(source="batch")

    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "individual": individual,
        "batch": batch,
        "saved_batches": len(batches),
    }


# Display six summary cards in two balanced rows.
def render_history_overview(statistics):
    total = statistics["total"]
    positive = statistics["positive"]
    negative = statistics["negative"]

    positive_rate = positive / total * 100 if total else 0
    negative_rate = negative / total * 100 if total else 0

    cards = [
        (
            "Total Predictions",
            f"{total:,}",
            "All saved prediction records",
        ),
        (
            "Positive Predictions",
            f"{positive:,}",
            f"{positive_rate:.1f}% of all records",
        ),
        (
            "Negative Predictions",
            f"{negative:,}",
            f"{negative_rate:.1f}% of all records",
        ),
        (
            "Individual Predictions",
            f'{statistics["individual"]:,}',
            "Manually submitted records",
        ),
        (
            "Batch Predictions",
            f'{statistics["batch"]:,}',
            "Records from file uploads",
        ),
        (
            "Saved Batches",
            f'{statistics["saved_batches"]:,}',
            "Saved batch groups",
        ),
    ]

    first_row = st.columns(3, gap="medium")

    for column, card in zip(first_row, cards[:3]):
        with column:
            render_summary_card(*card)

    second_row = st.columns(3, gap="medium")

    for column, card in zip(second_row, cards[3:]):
        with column:
            render_summary_card(*card)


# Reset the filters belonging to one records view.
def reset_record_filters(prefix):
    st.session_state[f"{prefix}_search"] = ""
    st.session_state[f"{prefix}_classification"] = "All"
    st.session_state[f"{prefix}_date_range"] = "All time"
    st.session_state[f"{prefix}_start_date"] = None
    st.session_state[f"{prefix}_end_date"] = None

    if prefix == "history_all":
        st.session_state[f"{prefix}_source"] = "All"

    st.session_state[f"{prefix}_page"] = 1

    st.session_state[f"{prefix}_table_version"] = (
        st.session_state.get(
            f"{prefix}_table_version",
            0,
        ) + 1
    )


# Build compact search and filter controls.
def render_record_filters(prefix, source=None):
    with st.container(border=True):
        header_column, reset_column = st.columns([4, 1])

        with header_column:
            st.markdown("**Search & Filters**")

        with reset_column:
            st.button(
                "Reset",
                key=f"{prefix}_reset",
                on_click=reset_record_filters,
                args=(prefix,),
                width="stretch",
                icon=":material/restart_alt:",
            )

        if source is None:
            columns = st.columns([1, 1, 1, 1.2])
        else:
            columns = st.columns([1, 1, 1.2])

        with columns[0]:
            search_id = st.text_input(
                "Prediction ID",
                placeholder="Search by ID",
                key=f"{prefix}_search",
            )

        with columns[1]:
            classification = st.selectbox(
                "Classification",
                options=["All", "Positive", "Negative"],
                key=f"{prefix}_classification",
            )

        if source is None:
            with columns[2]:
                source_selection = st.selectbox(
                    "Source",
                    options=["All", "Individual", "Batch"],
                    key=f"{prefix}_source",
                )

            selected_source = (
                None
                if source_selection == "All"
                else source_selection.lower()
            )

        else:
            selected_source = source

        with columns[-1]:
            date_range = st.selectbox(
                "Date Range",
                options=[
                    "All time",
                    "Today",
                    "Last 7 days",
                    "Last 30 days",
                    "Custom range",
                ],
                key=f"{prefix}_date_range",
            )

        today = datetime.now(timezone.utc).date()

        start_date = None
        end_date = None

        if date_range == "Today":
            start_date = today
            end_date = today

        elif date_range == "Last 7 days":
            start_date = today - timedelta(days=6)
            end_date = today

        elif date_range == "Last 30 days":
            start_date = today - timedelta(days=29)
            end_date = today

        elif date_range == "Custom range":
            date_column_1, date_column_2 = st.columns(2)

            with date_column_1:
                start_date = st.date_input(
                    "From Date (UTC)",
                    value=None,
                    key=f"{prefix}_start_date",
                )

            with date_column_2:
                end_date = st.date_input(
                    "To Date (UTC)",
                    value=None,
                    key=f"{prefix}_end_date",
                )

        st.caption("All date filters use UTC.")

    parsed_id = None

    if search_id.strip():
        try:
            parsed_id = int(search_id.strip())

            if not 1 <= parsed_id <= 9223372036854775807:
                raise ValueError

        except ValueError:
            st.warning("Enter a valid positive prediction ID.")
            return None

    if start_date is not None and end_date is not None:
        if start_date > end_date:
            st.warning(
                "The start date cannot be after the end date."
            )
            return None

    return {
        "classification": (
            None
            if classification == "All"
            else classification
        ),
        "search_id": parsed_id,
        "start_date": start_date,
        "end_date": end_date,
        "source": selected_source,
    }


# Prepare a report only when requested, using session-local storage.
def render_saved_report(key, label, signature, generator):
    cache_key = "history_prepared_pdf"
    cached = st.session_state.get(cache_key)
    if st.button("Prepare PDF Report", key=f"{key}_prepare",
                 icon=":material/picture_as_pdf:", width="stretch"):
        st.session_state.pop(cache_key, None)
        cached = None
        try:
            with st.spinner("Preparing saved report..."):
                data = generator()
            if not isinstance(data, bytes) or not data.startswith(b"%PDF-"):
                raise ValueError("Invalid PDF report.")
            cached = (signature, data, generate_report_filename(label))
            st.session_state[cache_key] = cached
        except Exception:
            st.error("Unable to prepare the saved report. Please try again.")
    if cached is not None and cached[0] == signature:
        st.download_button("Download PDF", data=cached[1], file_name=cached[2],
                           mime="application/pdf", icon=":material/download:",
                           width="stretch", on_click="ignore", key=f"{key}_download")


# Render the details of a selected prediction.
def render_prediction_details(record):
    with st.container(border=True):
        title_column, source_column = st.columns([3, 1])

        with title_column:
            st.markdown(
                f'### Prediction #{record["id"]}'
            )

            st.caption(
                format_timestamp(record["created_at"])
            )

        with source_column:
            st.caption("Prediction Source")
            st.markdown(
                f'**{record["source"].capitalize()}**'
            )

        result_column, probability_column, threshold_column = (
            st.columns(3)
        )

        with result_column:
            st.metric(
                "Classification",
                record["classification"],
            )

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

        measurements = list(MEASUREMENT_LABELS.items())

        for index in range(0, len(measurements), 4):
            columns = st.columns(4)

            for column, (name, details) in zip(
                columns,
                measurements[index:index + 4],
            ):
                label, unit = details
                value = format_measurement(record[name])

                with column:
                    st.caption(label)
                    st.markdown(f"**{value} {unit}**")

        if record["source"] == "batch":
            st.divider()
            st.markdown("#### Batch Information")

            batch_column, row_column = st.columns([3, 1])

            with batch_column:
                st.caption("Batch ID")
                st.code(
                    str(record["batch_id"]),
                    language=None,
                )

            with row_column:
                st.caption("Original File Row")
                st.markdown(
                    f'**{record["batch_row_number"]}**'
                )

        st.divider()

        model_column, version_column = st.columns(2)

        with model_column:
            st.caption("Prediction Model")
            st.markdown(
                f'**{record["model_name"]}**'
            )

        with version_column:
            st.caption("Model Version")
            st.markdown(
                f'**{record["model_version"]}**'
            )

        if record["source"] == "individual":
            st.divider()
            st.markdown("**Saved Prediction Report**")
            st.caption("Original validation warnings are not stored in history.")
            render_saved_report(
                f"history_individual_pdf_{record['id']}",
                f"Saved_Prediction_{record['id']}",
                ("individual", tuple(sorted((k, str(v)) for k, v in record.items()))),
                lambda: generate_individual_report(
                    measurements={feature: float(record[column]) for feature, column in FEATURE_COLUMNS.items()},
                    probability=float(record["probability"]),
                    classification=record["classification"],
                    threshold=float(record["threshold"]),
                    prediction_id=record["id"],
                    prediction_date=record["created_at"],
                    model_name=record["model_name"],
                    model_version=record["model_version"],
                    saved_history=True,
                ),
            )

        st.caption(
            "This is an educational machine learning prediction "
            "and must not be interpreted as a medical diagnosis."
        )


# Define consistent formatting for prediction tables.
def get_prediction_column_config():
    return {
        "ID": st.column_config.NumberColumn(
            "ID",
            format="%d",
            width="small",
        ),
        "Date & Time": st.column_config.TextColumn(
            "Date & Time",
            width="medium",
        ),
        "Source": st.column_config.TextColumn(
            "Source",
            width="small",
        ),
        "Classification": st.column_config.TextColumn(
            "Classification",
            width="small",
        ),
        "Probability": st.column_config.ProgressColumn(
            "Positive Probability",
            min_value=0.0,
            max_value=1.0,
            format="percent",
            width="medium",
        ),
        "Glucose": st.column_config.NumberColumn(
            "Glucose",
            format="%.1f",
        ),
        "BMI": st.column_config.NumberColumn(
            "BMI",
            format="%.1f",
        ),
        "Age": st.column_config.NumberColumn(
            "Age",
            format="%d",
        ),
        "File Row": st.column_config.NumberColumn(
            "File Row",
            format="%d",
        ),
    }


# Retrieve and display a prediction selected from a table.
def show_selected_prediction(prediction_id):
    try:
        record = get_prediction_by_id(prediction_id)

        if record is None:
            st.warning(
                "The selected prediction no longer exists."
            )
            return

        render_prediction_details(record)

    except Exception:
        st.error(
            "Unable to load the selected prediction details."
        )


# Display pagination and clear old table selections.
def render_pagination(
    prefix,
    page_number,
    total_pages,
):
    previous_column, page_column, next_column = (
        st.columns([1, 2, 1])
    )

    with previous_column:
        if st.button(
            "Previous",
            key=f"{prefix}_previous",
            disabled=page_number <= 1,
            width="stretch",
            icon=":material/chevron_left:",
        ):
            st.session_state[f"{prefix}_page"] -= 1

            st.session_state[f"{prefix}_table_version"] = (
                st.session_state.get(
                    f"{prefix}_table_version",
                    0,
                ) + 1
            )

            st.rerun()

    with page_column:
        st.html(
            f"""
<div class="history-pagination">
    Page {page_number} of {total_pages}
</div>
"""
        )

    with next_column:
        if st.button(
            "Next",
            key=f"{prefix}_next",
            disabled=page_number >= total_pages,
            width="stretch",
            icon=":material/chevron_right:",
        ):
            st.session_state[f"{prefix}_page"] += 1

            st.session_state[f"{prefix}_table_version"] = (
                st.session_state.get(
                    f"{prefix}_table_version",
                    0,
                ) + 1
            )

            st.rerun()


# Display a searchable and selectable records table.
def render_records_view(
    prefix,
    title,
    source=None,
):
    st.subheader(title)

    filters = render_record_filters(
        prefix,
        source=source,
    )

    if filters is None:
        return

    page_key = f"{prefix}_page"
    signature_key = f"{prefix}_filter_signature"
    version_key = f"{prefix}_table_version"

    if page_key not in st.session_state:
        st.session_state[page_key] = 1

    if version_key not in st.session_state:
        st.session_state[version_key] = 0

    filter_signature = tuple(filters.items())

    if st.session_state.get(signature_key) != filter_signature:
        st.session_state[page_key] = 1
        st.session_state[signature_key] = filter_signature
        st.session_state[version_key] += 1

    try:
        filtered_count = get_prediction_count(**filters)

        total_pages = max(
            1,
            ceil(filtered_count / PAGE_SIZE),
        )

        page_number = min(
            max(1, st.session_state[page_key]),
            total_pages,
        )

        st.session_state[page_key] = page_number

        records = get_all_predictions(
            **filters,
            limit=PAGE_SIZE,
            offset=(page_number - 1) * PAGE_SIZE,
        )

    except Exception:
        st.error(
            "Unable to retrieve prediction records. "
            "Please check the database connection."
        )
        return

    if not records:
        with st.container(border=True):
            st.info(
                "No predictions match the selected filters."
            )

            st.caption(
                "Try changing the search criteria or "
                "resetting the filters."
            )

        return

    include_source = filters["source"] is None

    dataframe = pd.DataFrame([
        format_prediction_row(
            record,
            include_source=include_source,
        )
        for record in records
    ])

    record_ids = tuple(
        int(record["id"])
        for record in records
    )

    records_signature_key = f"{prefix}_visible_ids"

    if (
        st.session_state.get(records_signature_key)
        != record_ids
    ):
        st.session_state[records_signature_key] = record_ids
        st.session_state[version_key] += 1

    with st.container(border=True):
        heading_column, count_column = st.columns([3, 1])

        with heading_column:
            st.markdown("**Saved Predictions**")

        with count_column:
            st.caption(
                f"{filtered_count:,} matching records"
            )

        st.caption(
            "Select a table row to inspect its complete details."
        )

        selection = st.dataframe(
            dataframe,
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            column_config=get_prediction_column_config(),
            key=(
                f"{prefix}_table_"
                f"{st.session_state[version_key]}"
            ),
        )

        first_record = (
            (page_number - 1) * PAGE_SIZE + 1
        )

        last_record = (
            first_record + len(records) - 1
        )

        st.caption(
            f"Showing {first_record:,}–{last_record:,} "
            f"of {filtered_count:,} records"
        )

        render_pagination(
            prefix,
            page_number,
            total_pages,
        )

    selected_rows = selection.selection.rows

    if selected_rows:
        row_index = selected_rows[0]

        if 0 <= row_index < len(records):
            prediction_id = int(
                records[row_index]["id"]
            )

            show_selected_prediction(prediction_id)

    elif filters["search_id"] is not None:
        if len(records) == 1:
            show_selected_prediction(
                int(records[0]["id"])
            )


# Convert saved batch summaries into a selectable table.
def format_batch_row(batch):
    total = int(batch["total_predictions"])
    positive = int(batch["positive_count"])
    negative = int(batch["negative_count"])

    return {
        "Batch": str(batch["batch_id"])[:8],
        "Saved On": format_timestamp(batch["created_at"]),
        "Records": total,
        "Positive": positive,
        "Negative": negative,
        "Positive Rate": positive / total if total else 0,
    }


# Display the complete overview of a selected batch.
def render_batch_overview(batch):
    batch_id = str(batch["batch_id"])

    total = int(batch["total_predictions"])
    positive = int(batch["positive_count"])
    negative = int(batch["negative_count"])

    average_probability = float(
        batch["average_probability"]
    )

    positive_rate = positive / total if total else 0
    negative_rate = negative / total if total else 0

    with st.container(border=True):
        st.markdown(
            f"### Batch {batch_id[:8]}"
        )

        st.caption(
            f'Saved on {format_timestamp(batch["created_at"])}'
        )

        first_row = st.columns(3)

        with first_row[0]:
            st.metric(
                "Total Records",
                f"{total:,}",
            )

        with first_row[1]:
            st.metric(
                "Positive Predictions",
                f"{positive:,}",
            )

        with first_row[2]:
            st.metric(
                "Negative Predictions",
                f"{negative:,}",
            )

        st.markdown("**Classification Distribution**")

        st.html(
            f"""
<div style="
    display:flex;
    width:100%;
    height:10px;
    overflow:hidden;
    border-radius:6px;
    background:rgba(128,150,175,0.15);
">
    <div style="
        width:{positive_rate * 100:.4f}%;
        background:#F59E0B;
    "></div>
    <div style="
        width:{negative_rate * 100:.4f}%;
        background:#14B8A6;
    "></div>
</div>
"""
        )

        positive_column, negative_column = st.columns(2)

        with positive_column:
            st.caption(
                f"Positive: {positive_rate * 100:.1f}%"
            )

        with negative_column:
            st.caption(
                f"Negative: {negative_rate * 100:.1f}%"
            )

        st.metric(
            "Average Positive-Class Probability",
            f"{average_probability * 100:.1f}%",
        )

        with st.expander("Batch Information"):
            st.caption("Full Batch ID")

            st.code(
                batch_id,
                language=None,
            )

            st.caption(
                "All records in this batch share the same "
                "unique batch identifier."
            )


# Display predictions belonging to a selected batch.
def render_batch_records(batch):
    batch_id = str(batch["batch_id"])

    try:
        records = get_batch_predictions(batch_id)

    except Exception:
        st.error(
            "Unable to retrieve the selected batch records."
        )
        return

    if not records:
        st.info("This batch contains no saved predictions.")
        return

    with st.container(border=True):
        st.markdown("**Saved Batch Report**")
        st.caption("Includes every saved record in this batch, regardless of the table filter. Original validation details are unavailable.")
        render_saved_report(
            f"history_batch_pdf_{batch_id}",
            f"Saved_Batch_{batch_id[:8]}",
            ("batch", batch_id, tuple(tuple(sorted((k, str(v)) for k, v in record.items())) for record in records)),
            lambda: generate_saved_batch_report(records, batch_id),
        )

    st.markdown("### Batch Predictions")

    classification = st.segmented_control(
        "Filter batch predictions",
        options=["All", "Positive", "Negative"],
        default="All",
        key=f"history_batch_classification_{batch_id}",
    )

    classification = classification or "All"

    filtered_records = [
        record
        for record in records
        if classification == "All"
        or record["classification"] == classification
    ]

    if not filtered_records:
        st.info(
            "No batch predictions match this classification."
        )
        return

    dataframe_rows = []

    for record in filtered_records:
        row = format_prediction_row(
            record,
            include_source=False,
        )

        row.pop("Date & Time")

        row = {
            "File Row": int(record["batch_row_number"]),
            **row,
        }

        dataframe_rows.append(row)

    dataframe = pd.DataFrame(dataframe_rows)

    signature_key = f"history_batch_records_signature_{batch_id}"
    version_key = f"history_batch_records_version_{batch_id}"

    signature = (
        classification,
        tuple(
            int(record["id"])
            for record in filtered_records
        ),
    )

    if st.session_state.get(signature_key) != signature:
        st.session_state[signature_key] = signature

        st.session_state[version_key] = (
            st.session_state.get(version_key, 0) + 1
        )

    table_version = st.session_state.get(version_key, 0)

    with st.container(border=True):
        st.caption(
            f"{len(filtered_records):,} matching records "
            f"out of {len(records):,} batch predictions."
        )

        st.caption(
            "Select a row to inspect its complete prediction."
        )

        selection = st.dataframe(
            dataframe,
            hide_index=True,
            width="stretch",
            height=380,
            on_select="rerun",
            selection_mode="single-row",
            column_config=get_prediction_column_config(),
            key=(
                f"history_batch_records_"
                f"{batch_id}_{table_version}"
            ),
        )

    selected_rows = selection.selection.rows

    if selected_rows:
        row_index = selected_rows[0]

        if 0 <= row_index < len(filtered_records):
            prediction_id = int(
                filtered_records[row_index]["id"]
            )

            show_selected_prediction(prediction_id)


# Reset saved batch search and sorting.
def reset_batch_filters():
    st.session_state["history_batch_search"] = ""
    st.session_state["history_batch_sort"] = "Newest first"

    st.session_state["history_batch_table_version"] = (
        st.session_state.get(
            "history_batch_table_version",
            0,
        ) + 1
    )


# Display saved batches as grouped prediction records.
def render_saved_batches(batches):
    st.subheader("Saved Batches")

    st.caption(
        "Explore batch uploads and their complete "
        "prediction results."
    )

    if not batches:
        with st.container(border=True):
            st.html("""
<div style="
    text-align:center;
    padding:28px 16px;
">
    <h3 style="margin:0 0 12px;">
        No saved batches yet
    </h3>
    <p style="
        opacity:0.65;
        font-size:14px;
        max-width:420px;
        margin:0 auto;
        line-height:1.6;
    ">
        Saved batch predictions will appear here after
        you save your first batch from the Batch Prediction page.
    </p>
</div>
""")

        return

    with st.container(border=True):
        search_column, sort_column, reset_column = (
            st.columns([2, 1.5, 1])
        )

        with search_column:
            search_batch = st.text_input(
                "Search Batch",
                placeholder="Enter a full or partial Batch ID",
                key="history_batch_search",
            )

        with sort_column:
            sort_order = st.selectbox(
                "Sort By",
                options=[
                    "Newest first",
                    "Oldest first",
                    "Most records",
                    "Fewest records",
                ],
                key="history_batch_sort",
            )

        with reset_column:
            st.write("")

            st.button(
                "Reset",
                key="history_batch_reset",
                on_click=reset_batch_filters,
                width="stretch",
                icon=":material/restart_alt:",
            )

    filtered_batches = [
        batch
        for batch in batches
        if search_batch.strip().lower()
        in str(batch["batch_id"]).lower()
    ]

    if sort_order == "Newest first":
        filtered_batches.sort(
            key=lambda batch: batch["created_at"],
            reverse=True,
        )

    elif sort_order == "Oldest first":
        filtered_batches.sort(
            key=lambda batch: batch["created_at"],
        )

    elif sort_order == "Most records":
        filtered_batches.sort(
            key=lambda batch: batch["total_predictions"],
            reverse=True,
        )

    elif sort_order == "Fewest records":
        filtered_batches.sort(
            key=lambda batch: batch["total_predictions"],
        )

    if not filtered_batches:
        st.info("No batches match your search.")
        return

    batch_signature = (
        search_batch.strip().lower(),
        sort_order,
        tuple(
            str(batch["batch_id"])
            for batch in filtered_batches
        ),
    )

    if (
        st.session_state.get("history_batch_signature")
        != batch_signature
    ):
        st.session_state["history_batch_signature"] = (
            batch_signature
        )

        st.session_state["history_batch_table_version"] = (
            st.session_state.get(
                "history_batch_table_version",
                0,
            ) + 1
        )

    table_version = st.session_state.get(
        "history_batch_table_version",
        0,
    )

    batch_dataframe = pd.DataFrame([
        format_batch_row(batch)
        for batch in filtered_batches
    ])

    with st.container(border=True):
        st.markdown("**Available Batches**")

        st.caption(
            f"{len(filtered_batches):,} saved batches. "
            "Select a batch to explore its results."
        )

        selection = st.dataframe(
            batch_dataframe,
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            key=f"history_saved_batches_{table_version}",
            column_config={
                "Batch": st.column_config.TextColumn(
                    "Batch ID",
                    width="small",
                ),
                "Saved On": st.column_config.TextColumn(
                    "Saved On",
                    width="medium",
                ),
                "Records": st.column_config.NumberColumn(
                    "Records",
                    format="%d",
                ),
                "Positive": st.column_config.NumberColumn(
                    "Positive",
                    format="%d",
                ),
                "Negative": st.column_config.NumberColumn(
                    "Negative",
                    format="%d",
                ),
                "Positive Rate": st.column_config.ProgressColumn(
                    "Positive Rate",
                    min_value=0.0,
                    max_value=1.0,
                    format="percent",
                ),
            },
        )

    selected_rows = selection.selection.rows

    selected_batch = None

    if selected_rows:
        row_index = selected_rows[0]

        if 0 <= row_index < len(filtered_batches):
            selected_batch = filtered_batches[row_index]

    elif search_batch.strip():
        if len(filtered_batches) == 1:
            selected_batch = filtered_batches[0]

    if selected_batch is not None:
        render_batch_overview(selected_batch)
        render_batch_records(selected_batch)


# Initialize the history page.
apply_history_styles()

st.title("Prediction History")

st.caption(
    "Manage, explore, and review individual and batch "
    "diabetes prediction records."
)


# Load the overview data.
try:
    available_batches = get_available_batches()

    statistics = load_history_statistics(
        available_batches
    )

except Exception:
    st.error(
        "Unable to load prediction history. "
        "Please check the database connection."
    )
    st.stop()


render_history_overview(statistics)

st.write("")


# Navigate between the three history views.
history_view = st.segmented_control(
    "History View",
    options=[
        "All Records",
        "Individual Predictions",
        "Saved Batches",
    ],
    default="All Records",
    key="history_active_view",
    label_visibility="collapsed",
    width="stretch",
)

history_view = history_view or "All Records"


# Render only the active history view.
if history_view == "All Records":
    render_records_view(
        prefix="history_all",
        title="All Prediction Records",
    )

elif history_view == "Individual Predictions":
    render_records_view(
        prefix="history_individual",
        title="Individual Predictions",
        source="individual",
    )

elif history_view == "Saved Batches":
    render_saved_batches(
        available_batches
    )


st.divider()

st.caption(
    "UODA DiaML | Educational Diabetes Prediction System | "
    "Prediction records are stored in Supabase PostgreSQL."
)
