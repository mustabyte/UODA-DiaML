
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.batch_actions import render_batch_actions

from components.validation import (
    FEATURE_COLUMNS,
    FEATURE_RULES,
)


POSITIVE_COLOR = "#EF4444"
NEGATIVE_COLOR = "#14B8A6"
WARNING_COLOR = "#F59E0B"
NEUTRAL_COLOR = "#94A3B8"


# Apply consistent styling to the results dashboard.
def apply_results_styles():
    st.html("""
<style>
.batch-result-card {
    background: rgba(128, 150, 175, 0.055);
    border: 1px solid rgba(128, 150, 175, 0.22);
    border-radius: 12px;
    padding: 22px;
    min-height: 142px;
    box-sizing: border-box;
    transition: border-color 0.2s ease;
}

.batch-result-card:hover {
    border-color: rgba(20, 184, 166, 0.65);
}

.batch-result-label {
    font-size: 14px;
    font-weight: 500;
    opacity: 0.85;
}

.batch-result-value {
    font-size: 36px;
    font-weight: 650;
    margin-top: 12px;
    line-height: 1.2;
    letter-spacing: -0.8px;
    font-variant-numeric: tabular-nums;
}

.batch-result-description {
    font-size: 12px;
    opacity: 0.65;
    margin-top: 10px;
    line-height: 1.4;
}

.batch-record-measurement {
    background: rgba(128, 150, 175, 0.055);
    border: 1px solid rgba(128, 150, 175, 0.18);
    border-radius: 10px;
    padding: 16px;
    min-height: 105px;
    margin-bottom: 8px;
}

.batch-measurement-label {
    font-size: 13px;
    opacity: 0.75;
    overflow-wrap: anywhere;
}

.batch-measurement-value {
    font-size: 24px;
    font-weight: 600;
    margin-top: 10px;
}

.batch-measurement-unit {
    font-size: 12px;
    opacity: 0.65;
}

@media (max-width: 768px) {
    .batch-result-card {
        padding: 16px;
        min-height: 135px;
    }

    .batch-result-value {
        font-size: 28px;
    }

    .batch-measurement-value {
        font-size: 20px;
    }
}
</style>
""")


# Render an individual summary card.
def render_result_card(
    label,
    value,
    description,
    color=None,
):
    color_style = (
        f"color:{color};" if color else ""
    )

    st.html(f"""
<div class="batch-result-card">
    <div class="batch-result-label">
        {escape(str(label))}
    </div>
    <div class="batch-result-value"
         style="{color_style}">
        {escape(str(value))}
    </div>
    <div class="batch-result-description">
        {escape(str(description))}
    </div>
</div>
""")


# Display the six prediction summary cards.
def render_prediction_summary(result):
    cards = [
        (
            "Processed Records",
            f"{result.total_processed:,}",
            "Successfully predicted records",
            NEGATIVE_COLOR,
        ),
        (
            "Positive Predictions",
            f"{result.positive_count:,}",
            f"{result.positive_rate:.1f}% of processed records",
            POSITIVE_COLOR,
        ),
        (
            "Negative Predictions",
            f"{result.negative_count:,}",
            f"{result.negative_rate:.1f}% of processed records",
            NEGATIVE_COLOR,
        ),
        (
            "Excluded Records",
            f"{result.total_excluded:,}",
            "No predictions generated",
            None,
        ),
        (
            "Warning Records",
            f"{result.total_warnings:,}",
            "Processed with validation warnings",
            WARNING_COLOR,
        ),
        (
            "Average Probability",
            f"{result.average_probability * 100:.2f}%",
            "Mean positive-class model probability",
            None,
        ),
    ]

    for start in range(0, len(cards), 3):
        columns = st.columns(3, gap="medium")

        for column, card in zip(
            columns,
            cards[start:start + 3],
        ):
            with column:
                render_result_card(*card)


# Apply shared Plotly chart settings.
def configure_chart(figure, height=320):
    figure.update_layout(
        height=height,
        margin=dict(
            l=15,
            r=15,
            t=35,
            b=20,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        hoverlabel=dict(
            font_size=13,
        ),
    )

    return figure


# Create the classification distribution chart.
def create_classification_chart(result):
    figure = go.Figure(
        data=[
            go.Pie(
                labels=[
                    "Positive",
                    "Negative",
                ],
                values=[
                    result.positive_count,
                    result.negative_count,
                ],
                hole=0.72,
                sort=False,
                direction="clockwise",
                marker=dict(
                    colors=[
                        POSITIVE_COLOR,
                        NEGATIVE_COLOR,
                    ],
                    line=dict(width=0),
                ),
                textinfo="percent",
                textposition="inside",
                hovertemplate=(
                    "%{label}<br>"
                    "Records: %{value}<br>"
                    "Percentage: %{percent}"
                    "<extra></extra>"
                ),
            )
        ]
    )

    figure.add_annotation(
        x=0.5,
        y=0.5,
        text=(
            f"<b>{result.total_processed:,}</b>"
            "<br><span style='font-size:12px'>"
            "Predictions</span>"
        ),
        showarrow=False,
        font=dict(size=26),
    )

    figure.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.08,
            yanchor="top",
        ),
    )

    return configure_chart(figure)


# Create the positive-class probability histogram.
def create_probability_chart(result):
    dataframe = result.results

    figure = go.Figure()

    for classification, color in [
        ("Negative", NEGATIVE_COLOR),
        ("Positive", POSITIVE_COLOR),
    ]:
        probabilities = (
            dataframe.loc[
                dataframe["Classification"]
                == classification,
                "Probability",
            ] * 100
        )

        figure.add_trace(
            go.Histogram(
                x=probabilities,
                name=classification,
                marker_color=color,
                opacity=0.85,
                xbins=dict(
                    start=0,
                    end=100,
                    size=10,
                ),
                hovertemplate=(
                    "Probability range: %{x}%<br>"
                    "Records: %{y}"
                    "<extra></extra>"
                ),
            )
        )

    threshold_percent = result.threshold * 100

    figure.add_vline(
        x=threshold_percent,
        line_width=2,
        line_dash="dash",
        line_color=WARNING_COLOR,
        annotation_text=(
            f"Threshold: {threshold_percent:g}%"
        ),
        annotation_position="top right",
        annotation_font_color=WARNING_COLOR,
    )

    figure.update_layout(
        barmode="stack",
        bargap=0.12,
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.25,
        ),
        xaxis=dict(
            title="Positive Probability",
            range=[0, 100],
            ticksuffix="%",
            dtick=20,
            showgrid=False,
        ),
        yaxis=dict(
            title="Records",
            rangemode="tozero",
            dtick=1 if result.total_processed <= 10 else None,
            gridcolor="rgba(128,150,175,0.15)",
        ),
    )

    return configure_chart(figure)


# Display the two primary prediction charts.
def render_prediction_charts(result):
    st.subheader("Prediction Analytics")

    left, right = st.columns(
        2,
        gap="medium",
    )

    with left:
        with st.container(border=True):
            st.markdown(
                "**Classification Distribution**"
            )

            st.caption(
                "Predicted positive and negative records."
            )

            st.plotly_chart(
                create_classification_chart(result),
                width="stretch",
                key="diaml_batch_classification_chart",
                config={
                    "displaylogo": False,
                    "displayModeBar": False,
                },
            )

    with right:
        with st.container(border=True):
            st.markdown(
                "**Probability Distribution**"
            )

            st.caption(
                "Model probabilities relative to the "
                "classification threshold."
            )

            st.plotly_chart(
                create_probability_chart(result),
                width="stretch",
                key="diaml_batch_probability_chart",
                config={
                    "displaylogo": False,
                    "displayModeBar": False,
                },
            )


# Format a measurement according to its data type.
def format_measurement(feature, value):
    rule = FEATURE_RULES[feature]

    if rule["integer"]:
        return f"{int(value):,}"

    if feature == "DiabetesPedigreeFunction":
        return f"{float(value):.3f}"

    return f"{float(value):.2f}"


# Compare measurements across predicted groups.
def render_measurement_analysis(result):
    with st.container(border=True):
        st.subheader("Measurement Analysis")

        st.caption(
            "Compare average measurements between "
            "predicted positive and negative records."
        )

        feature = st.selectbox(
            "Select measurement",
            options=FEATURE_COLUMNS,
            index=FEATURE_COLUMNS.index("Glucose"),
            format_func=lambda item: (
                FEATURE_RULES[item]["label"]
            ),
            key="diaml_batch_analysis_feature",
        )

        dataframe = result.results

        rule = FEATURE_RULES[feature]
        unit = rule["unit"]

        groups = []

        for classification, color in [
            ("Positive", POSITIVE_COLOR),
            ("Negative", NEGATIVE_COLOR),
        ]:
            values = dataframe.loc[
                dataframe["Classification"]
                == classification,
                feature,
            ]

            if not values.empty:
                groups.append({
                    "classification": classification,
                    "average": float(values.mean()),
                    "count": len(values),
                    "color": color,
                })

        if len(groups) < 2:
            st.info(
                "Only one predicted classification is "
                "present in this batch. A comparison "
                "between both groups is unavailable."
            )

        figure = go.Figure()

        for group in groups:
            average = group["average"]

            figure.add_trace(
                go.Bar(
                    x=[average],
                    y=[group["classification"]],
                    orientation="h",
                    name=group["classification"],
                    marker_color=group["color"],
                    width=0.45,
                    text=[
                        f"{average:.2f} {unit}".strip()
                    ],
                    textposition="auto",
                    customdata=[group["count"]],
                    hovertemplate=(
                        "Classification: %{y}<br>"
                        "Average: %{x:.2f}<br>"
                        "Records: %{customdata}"
                        "<extra></extra>"
                    ),
                )
            )

        figure.update_layout(
            showlegend=False,
            xaxis=dict(
                title=(
                    f"Average {rule['label']}"
                    + (f" ({unit})" if unit else "")
                ),
                rangemode="tozero",
                gridcolor="rgba(128,150,175,0.15)",
            ),
            yaxis=dict(
                categoryorder="array",
                categoryarray=[
                    "Negative",
                    "Positive",
                ],
                showgrid=False,
            ),
        )

        configure_chart(figure, height=240)

        st.plotly_chart(
            figure,
            width="stretch",
            key="diaml_batch_measurement_chart",
            config={
                "displaylogo": False,
                "displayModeBar": False,
            },
        )

        summary_columns = st.columns(2)

        for column, classification in zip(
            summary_columns,
            ["Positive", "Negative"],
        ):
            group = next(
                (
                    item for item in groups
                    if item["classification"]
                    == classification
                ),
                None,
            )

            with column:
                if group is None:
                    st.metric(
                        f"{classification} Group",
                        "N/A",
                    )

                else:
                    st.metric(
                        f"{classification} Group",
                        (
                            f"{group['average']:.2f} "
                            f"{unit}"
                        ).strip(),
                    )

                    st.caption(
                        f"{group['count']:,} records"
                    )

        st.caption(
            "These comparisons describe the uploaded "
            "records and model predictions. They do not "
            "establish medical causes or clinical risk."
        )


# Attach validation status to prediction records.
def build_results_dataframe(result, validation):
    dataframe = result.results.copy()

    warning_rows = set()

    if not validation.warnings.empty:
        warning_rows = set(
            validation.warnings["File Row"].tolist()
        )

    dataframe["Validation"] = (
        dataframe["File Row"]
        .isin(warning_rows)
        .map({
            True: "Warning",
            False: "Clean",
        })
    )

    return dataframe


# Clear stale table selections when filters change.
def reset_record_selection():
    version = st.session_state.get(
        "diaml_batch_record_table_version",
        0,
    )

    st.session_state[
        "diaml_batch_record_table_version"
    ] = version + 1


# Restore the default prediction table filters.
def reset_record_filters():
    st.session_state[
        "diaml_batch_record_search"
    ] = ""

    st.session_state[
        "diaml_batch_classification_filter"
    ] = "All"

    st.session_state[
        "diaml_batch_warning_filter"
    ] = "All"

    st.session_state[
        "diaml_batch_sort"
    ] = "Original File Order"

    reset_record_selection()


# Apply search, classification and validation filters.
def filter_prediction_records(
    dataframe,
    search,
    classification,
    validation_status,
    sort_order,
):
    filtered = dataframe.copy()

    search = search.strip()

    if search:
        filtered = filtered.loc[
            filtered["File Row"]
            .astype(str)
            .str.contains(
                search,
                regex=False,
            )
        ]

    if classification != "All":
        filtered = filtered.loc[
            filtered["Classification"]
            == classification
        ]

    if validation_status != "All":
        filtered = filtered.loc[
            filtered["Validation"]
            == validation_status
        ]

    if sort_order == "Highest Probability":
        filtered = filtered.sort_values(
            "Probability",
            ascending=False,
            kind="stable",
        )

    elif sort_order == "Lowest Probability":
        filtered = filtered.sort_values(
            "Probability",
            ascending=True,
            kind="stable",
        )

    else:
        filtered = filtered.sort_values(
            "File Row",
            kind="stable",
        )

    return filtered.reset_index(drop=True)


# Display the compact prediction table filters.
def render_record_filters(dataframe):
    search_column, class_column, warning_column = (
        st.columns(3)
    )

    with search_column:
        search = st.text_input(
            "Search file row",
            placeholder="Enter row number",
            key="diaml_batch_record_search",
            on_change=reset_record_selection,
        )

    with class_column:
        classification = st.selectbox(
            "Classification",
            options=[
                "All",
                "Positive",
                "Negative",
            ],
            key="diaml_batch_classification_filter",
            on_change=reset_record_selection,
        )

    with warning_column:
        validation_status = st.selectbox(
            "Validation status",
            options=[
                "All",
                "Clean",
                "Warning",
            ],
            key="diaml_batch_warning_filter",
            on_change=reset_record_selection,
        )

    sort_column, reset_column = st.columns(
        [3, 1],
        vertical_alignment="bottom",
    )

    with sort_column:
        sort_order = st.selectbox(
            "Sort records",
            options=[
                "Original File Order",
                "Highest Probability",
                "Lowest Probability",
            ],
            key="diaml_batch_sort",
            on_change=reset_record_selection,
        )

    with reset_column:
        st.button(
            "Reset Filters",
            icon=":material/restart_alt:",
            width="stretch",
            key="diaml_batch_reset_filters",
            on_click=reset_record_filters,
        )

    return filter_prediction_records(
        dataframe,
        search,
        classification,
        validation_status,
        sort_order,
    )


# Render an individual measurement card.
def render_measurement_card(feature, value):
    rule = FEATURE_RULES[feature]

    label = escape(rule["label"])
    unit = escape(rule["unit"])
    formatted_value = escape(
        format_measurement(feature, value)
    )

    st.html(f"""
<div class="batch-record-measurement">
    <div class="batch-measurement-label">
        {label}
    </div>
    <div class="batch-measurement-value">
        {formatted_value}
        <span class="batch-measurement-unit">
            {unit}
        </span>
    </div>
</div>
""")


# Display all information for the selected record.
def render_selected_record(
    record,
    validation,
    result,
):
    row_number = int(record["File Row"])

    classification = record["Classification"]

    probability = float(
        record["Probability"]
    )

    classification_color = (
        POSITIVE_COLOR
        if classification == "Positive"
        else NEGATIVE_COLOR
    )

    with st.container(border=True):
        st.subheader(
            f"Record Details — File Row {row_number}"
        )

        st.caption(
            f"Model: {result.model_name} · "
            f"Threshold: {result.threshold:.0%}"
        )

        first, second, third = st.columns(3)

        with first:
            render_result_card(
                "Classification",
                classification,
                "Predicted outcome",
                classification_color,
            )

        with second:
            render_result_card(
                "Positive Probability",
                f"{probability * 100:.2f}%",
                "Model positive-class probability",
            )

        with third:
            status = record["Validation"]

            render_result_card(
                "Validation Status",
                status,
                (
                    "Accepted with warnings"
                    if status == "Warning"
                    else "No validation issues"
                ),
                (
                    WARNING_COLOR
                    if status == "Warning"
                    else NEGATIVE_COLOR
                ),
            )

        st.divider()

        st.markdown("**Prediction Probability**")

        figure = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number=dict(
                    suffix="%",
                    valueformat=".2f",
                ),
                gauge=dict(
                    axis=dict(
                        range=[0, 100],
                        ticksuffix="%",
                    ),
                    bar=dict(
                        color=classification_color,
                        thickness=0.25,
                    ),
                    bgcolor="rgba(128,150,175,0.08)",
                    borderwidth=0,
                    steps=[
                        dict(
                            range=[
                                0,
                                result.threshold * 100,
                            ],
                            color="rgba(20,184,166,0.12)",
                        ),
                        dict(
                            range=[
                                result.threshold * 100,
                                100,
                            ],
                            color="rgba(239,68,68,0.12)",
                        ),
                    ],
                    threshold=dict(
                        line=dict(
                            color=WARNING_COLOR,
                            width=3,
                        ),
                        thickness=0.8,
                        value=result.threshold * 100,
                    ),
                ),
            )
        )

        configure_chart(figure, height=260)

        st.plotly_chart(
            figure,
            width="stretch",
            key=f"diaml_batch_record_gauge_{row_number}",
            config={
                "displaylogo": False,
                "displayModeBar": False,
            },
        )

        st.caption(
            f"The model classifies probabilities at or "
            f"above {result.threshold:.0%} as positive."
        )

        st.divider()

        st.markdown("**Measurement Details**")

        for start in range(0, len(FEATURE_COLUMNS), 4):
            columns = st.columns(4)

            for column, feature in zip(
                columns,
                FEATURE_COLUMNS[start:start + 4],
            ):
                with column:
                    render_measurement_card(
                        feature,
                        record[feature],
                    )

        if record["Validation"] == "Warning":
            st.divider()

            st.markdown("**Validation Warnings**")

            warnings = validation.warnings.loc[
                validation.warnings["File Row"]
                == row_number
            ]

            for _, warning in warnings.iterrows():
                st.warning(
                    f"{warning['Field']}: "
                    f"{warning['Warning']}"
                )

        st.caption(
            "This prediction is generated for educational "
            "purposes and is not a medical diagnosis."
        )


# Display the interactive prediction records table.
def render_prediction_records(
    result,
    validation,
):
    dataframe = build_results_dataframe(
        result,
        validation,
    )

    selected_record = None

    with st.container(border=True):
        st.subheader("Prediction Records")

        st.caption(
            "Filter, sort or select an individual record "
            "to inspect its complete prediction."
        )

        filtered = render_record_filters(
            dataframe
        )

        st.divider()

        st.caption(
            f"Showing {len(filtered):,} of "
            f"{len(dataframe):,} predicted records."
        )

        if filtered.empty:
            st.info(
                "No records match the selected filters."
            )

            return

        display_columns = [
            "File Row",
            "Classification",
            "Probability (%)",
            "Validation",
            *FEATURE_COLUMNS,
        ]

        display_data = filtered.loc[
            :, display_columns
        ].copy()

        table_version = st.session_state.get(
            "diaml_batch_record_table_version",
            0,
        )

        table_height = min(
            450,
            38 + len(display_data) * 35,
        )

        selection = st.dataframe(
            display_data,
            hide_index=True,
            width="stretch",
            height=max(110, table_height),
            key=(
                "diaml_batch_prediction_table_"
                f"{table_version}"
            ),
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "File Row": (
                    st.column_config.NumberColumn(
                        format="%d",
                        width="small",
                        pinned=True,
                    )
                ),
                "Classification": (
                    st.column_config.TextColumn(
                        width="medium",
                    )
                ),
                "Probability (%)": (
                    st.column_config.ProgressColumn(
                        label="Positive Probability",
                        min_value=0.0,
                        max_value=100.0,
                        format="%.2f%%",
                        width="medium",
                    )
                ),
                "Validation": (
                    st.column_config.TextColumn(
                        width="small",
                    )
                ),
                "BMI": (
                    st.column_config.NumberColumn(
                        format="%.2f",
                    )
                ),
                "DiabetesPedigreeFunction": (
                    st.column_config.NumberColumn(
                        format="%.3f",
                    )
                ),
            },
        )

        selected_rows = selection.selection.rows

        if selected_rows:
            position = selected_rows[0]

            if 0 <= position < len(filtered):
                selected_record = filtered.iloc[
                    position
                ]

        else:
            st.caption(
                "Select a row to view its full prediction "
                "and measurement details."
            )

    if selected_record is not None:
        render_selected_record(
            selected_record,
            validation,
            result,
        )


# Render the complete batch results dashboard.
def render_batch_results(result, validation):
    if result is None:
        return

    apply_results_styles()

    st.divider()

    st.subheader("Batch Prediction Results")

    st.caption(
        f"Model: {result.model_name} · "
        f"Classification threshold: {result.threshold:.0%}"
    )

    st.success(
        f"Successfully generated predictions for "
        f"{result.total_processed:,} accepted records."
    )

    render_prediction_summary(result)

    st.divider()

    render_prediction_charts(result)

    render_measurement_analysis(result)

    st.divider()

    render_prediction_records(
        result,
        validation,
    )

    render_batch_actions(result, validation)

    st.caption(
        "All analytics describe model-generated "
        "predictions, not confirmed medical outcomes."
    )
