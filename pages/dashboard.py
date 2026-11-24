
from pathlib import Path

import pandas as pd
import streamlit as st

from components.charts import (
    EXPLORER_FEATURES,
    calculate_age_group_statistics,
    create_age_group_chart,
    create_box_plot,
    create_correlation_heatmap,
    create_distribution_chart,
    create_outcome_chart,
    create_parallel_coordinates_chart,
    create_scatter_chart,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"

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

REQUIRED_COLUMNS = FEATURE_COLUMNS + ["Outcome"]

MISSING_ZERO_FEATURES = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

FEATURE_LABELS = {
    "Pregnancies": "Pregnancies",
    "Glucose": "Glucose",
    "BloodPressure": "Blood Pressure",
    "SkinThickness": "Skin Thickness",
    "Insulin": "Insulin",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
    "Age": "Age",
    "Outcome": "Diabetes Outcome",
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
    "Outcome": "",
}

FEATURE_DESCRIPTIONS = {
    "Pregnancies": "Number of pregnancies",
    "Glucose": "Plasma glucose concentration",
    "BloodPressure": "Diastolic blood pressure",
    "SkinThickness": "Triceps skinfold thickness",
    "Insulin": "2-hour serum insulin concentration",
    "BMI": "Body mass index",
    "DiabetesPedigreeFunction": "Diabetes pedigree function",
    "Age": "Age of the participant",
    "Outcome": "Binary diabetes outcome label (0 or 1)",
}

TEAL = "#14b8a6"
RED = "#ef4444"
AMBER = "#f59e0b"


# Apply consistent dashboard styling.
def apply_dashboard_styles():
    st.html(
        """
        <style>
            .dashboard-summary-card {
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

            .dashboard-summary-card:hover {
                border-color: rgba(20, 184, 166, 0.55);
            }

            .dashboard-card-label {
                font-size: 14px;
                font-weight: 500;
                opacity: 0.85;
                line-height: 1.4;
                overflow-wrap: anywhere;
            }

            .dashboard-card-value {
                font-size: 36px;
                font-weight: 650;
                line-height: 1.2;
                margin-top: 14px;
                letter-spacing: -0.8px;
                font-variant-numeric: tabular-nums;
            }

            .dashboard-card-description {
                font-size: 12px;
                opacity: 0.65;
                margin-top: 10px;
                line-height: 1.4;
                overflow-wrap: anywhere;
            }

            .dashboard-progress-track {
                height: 9px;
                width: 100%;
                background: rgba(128, 150, 175, 0.15);
                border-radius: 6px;
                overflow: hidden;
            }

            .dashboard-progress-fill {
                height: 100%;
                border-radius: 6px;
                transition: width 0.3s ease;
            }

            .dashboard-progress-label {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 12px;
                font-size: 13px;
                margin-bottom: 9px;
            }

            .dashboard-progress-value {
                font-size: 13px;
                font-weight: 600;
                font-variant-numeric: tabular-nums;
                white-space: nowrap;
            }

            .dashboard-progress-item {
                margin-top: 20px;
                margin-bottom: 20px;
            }

            .dashboard-ratio-value {
                font-size: 32px;
                font-weight: 650;
                line-height: 1.2;
                letter-spacing: -0.5px;
                margin-top: 8px;
            }

            .dashboard-ratio-label {
                font-size: 13px;
                opacity: 0.7;
                margin-top: 6px;
                margin-bottom: 10px;
            }

            .dashboard-filter-summary {
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
                padding: 13px 16px;
                border: 1px solid rgba(128, 150, 175, 0.18);
                border-radius: 8px;
                background: rgba(128, 150, 175, 0.045);
                font-size: 13px;
            }

            .dashboard-filter-count {
                font-weight: 600;
                color: #14b8a6;
            }

            .dashboard-filter-context {
                opacity: 0.7;
            }

            .dashboard-quality-value {
                font-size: 28px;
                font-weight: 650;
                letter-spacing: -0.5px;
                margin-top: 6px;
            }

            .dashboard-quality-label {
                font-size: 13px;
                opacity: 0.75;
                margin-bottom: 6px;
            }

            @media (max-width: 768px) {
                .dashboard-summary-card {
                    padding: 16px;
                    min-height: 135px;
                }

                .dashboard-card-value {
                    font-size: 30px;
                }

                .dashboard-card-label {
                    font-size: 12px;
                }

                .dashboard-ratio-value {
                    font-size: 26px;
                }
            }
        </style>
        """
    )


# Load and validate the original diabetes dataset.
@st.cache_data(show_spinner=False)
def load_dashboard_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            "The diabetes dataset could not be found."
        )

    dataframe = pd.read_csv(
        DATA_PATH,
        header=None,
    )

    if dataframe.empty:
        raise ValueError(
            "The diabetes dataset is empty."
        )

    if dataframe.shape[1] != len(REQUIRED_COLUMNS):
        raise ValueError(
            "The dataset has an unexpected number of columns."
        )

    dataframe.columns = REQUIRED_COLUMNS

    for column in REQUIRED_COLUMNS:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    if dataframe[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(
            "The dataset contains missing numeric values."
        )

    if not dataframe["Outcome"].isin([0, 1]).all():
        raise ValueError(
            "Outcome must contain only 0 and 1."
        )

    return dataframe


# Replace missing measurement placeholders for analysis.
def clean_measurements(dataframe):
    cleaned = dataframe.copy()

    for feature in MISSING_ZERO_FEATURES:
        cleaned[feature] = cleaned[feature].replace(
            0,
            float("nan"),
        )

    return cleaned


# Return readable measurement names.
def get_feature_label(feature):
    return FEATURE_LABELS.get(feature, feature)


# Return measurement labels with their units.
def get_feature_axis_label(feature):
    label = get_feature_label(feature)
    unit = FEATURE_UNITS.get(feature, "")

    return f"{label} ({unit})" if unit else label


# Format numeric statistics safely.
def format_statistic(value, decimals=1):
    if pd.isna(value):
        return "N/A"

    return f"{value:,.{decimals}f}"


# Calculate dataset-level summary statistics.
def calculate_dashboard_statistics(dataframe):
    total = len(dataframe)

    positive = int(
        (dataframe["Outcome"] == 1).sum()
    )

    negative = int(
        (dataframe["Outcome"] == 0).sum()
    )

    cleaned = clean_measurements(dataframe)

    missing_counts = {
        feature: int(
            (dataframe[feature] == 0).sum()
        )
        for feature in MISSING_ZERO_FEATURES
    }

    incomplete_records = int(
        dataframe[MISSING_ZERO_FEATURES]
        .eq(0)
        .any(axis=1)
        .sum()
    )

    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "positive_rate": positive / total * 100,
        "negative_rate": negative / total * 100,
        "average_glucose": cleaned["Glucose"].mean(),
        "average_bmi": cleaned["BMI"].mean(),
        "average_age": cleaned["Age"].mean(),
        "missing_counts": missing_counts,
        "total_missing": sum(missing_counts.values()),
        "incomplete_records": incomplete_records,
        "complete_records": total - incomplete_records,
        "class_ratio": (
            negative / positive
            if positive > 0
            else None
        ),
    }


# Render a reusable summary card.
def render_summary_card(
    label,
    value,
    description,
    value_color=None,
):
    color_style = (
        f"color: {value_color};"
        if value_color
        else ""
    )

    st.html(
        f"""
        <div class="dashboard-summary-card">
            <div class="dashboard-card-label">
                {label}
            </div>

            <div
                class="dashboard-card-value"
                style="{color_style}"
            >
                {value}
            </div>

            <div class="dashboard-card-description">
                {description}
            </div>
        </div>
        """
    )


# Render a horizontal statistical progress bar.
def render_progress_bar(
    label,
    value_text,
    percentage,
    color,
):
    percentage = max(
        0,
        min(100, float(percentage)),
    )

    st.html(
        f"""
        <div class="dashboard-progress-item">
            <div class="dashboard-progress-label">
                <span>{label}</span>

                <span class="dashboard-progress-value">
                    {value_text}
                </span>
            </div>

            <div class="dashboard-progress-track">
                <div
                    class="dashboard-progress-fill"
                    style="
                        width: {percentage:.2f}%;
                        background: {color};
                    "
                ></div>
            </div>
        </div>
        """
    )


# Apply consistent sizing to existing Plotly charts.
def render_chart(
    figure,
    key,
    height=350,
    hide_title=False,
):
    figure.update_layout(
        height=height,
        margin=dict(
            l=20,
            r=20,
            t=20 if hide_title else 55,
            b=20,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        legend=dict(
            font=dict(size=11),
        ),
    )

    if hide_title:
        figure.update_layout(title_text=None)

    st.plotly_chart(
        figure,
        width="stretch",
        theme="streamlit",
        config={
            "displaylogo": False,
            "responsive": True,
        },
        key=key,
    )


# Render the six primary dataset summary cards.
def render_overview_cards(statistics):
    st.subheader("Dataset Overview")

    first_row = st.columns(3)

    with first_row[0]:
        render_summary_card(
            "Total Records",
            f"{statistics['total']:,}",
            "Observations in the dataset",
        )

    with first_row[1]:
        render_summary_card(
            "Positive Cases",
            f"{statistics['positive']:,}",
            f"{statistics['positive_rate']:.1f}% of dataset",
            RED,
        )

    with first_row[2]:
        render_summary_card(
            "Negative Cases",
            f"{statistics['negative']:,}",
            f"{statistics['negative_rate']:.1f}% of dataset",
            TEAL,
        )

    second_row = st.columns(3)

    with second_row[0]:
        render_summary_card(
            "Average Glucose",
            format_statistic(
                statistics["average_glucose"]
            ),
            "mg/dL · Valid measurements",
        )

    with second_row[1]:
        render_summary_card(
            "Average BMI",
            format_statistic(
                statistics["average_bmi"]
            ),
            "kg/m² · Valid measurements",
        )

    with second_row[2]:
        render_summary_card(
            "Average Age",
            format_statistic(
                statistics["average_age"]
            ),
            "Years · All participants",
        )

    st.caption(
        "Missing zero placeholders are excluded from "
        "glucose and BMI averages."
    )


# Render the outcome distribution and class balance.
def render_outcome_analysis(dataframe, statistics):
    st.subheader("Outcome Analysis")

    chart_column, balance_column = st.columns(
        [1.15, 1],
        gap="medium",
    )

    with chart_column:
        with st.container(border=True):
            st.markdown("**Outcome Distribution**")

            st.caption(
                "Observed diabetes outcome labels."
            )

            figure = create_outcome_chart(dataframe)

            render_chart(
                figure,
                key="dashboard_outcome_chart",
                height=310,
                hide_title=True,
            )

    with balance_column:
        with st.container(border=True):
            st.markdown("**Class Balance**")

            st.caption(
                "Relative representation of outcome groups."
            )

            render_progress_bar(
                "Negative Records",
                (
                    f"{statistics['negative']:,} · "
                    f"{statistics['negative_rate']:.1f}%"
                ),
                statistics["negative_rate"],
                TEAL,
            )

            render_progress_bar(
                "Positive Records",
                (
                    f"{statistics['positive']:,} · "
                    f"{statistics['positive_rate']:.1f}%"
                ),
                statistics["positive_rate"],
                RED,
            )

            st.divider()

            ratio = statistics["class_ratio"]

            ratio_text = (
                f"{ratio:.2f} : 1"
                if ratio is not None
                else "N/A"
            )

            st.html(
                f"""
                <div class="dashboard-ratio-value">
                    {ratio_text}
                </div>

                <div class="dashboard-ratio-label">
                    Negative-to-positive class ratio
                </div>
                """
            )

            st.caption(
                "Class balance is important when "
                "interpreting classification performance."
            )

    st.caption(
        "Outcome percentages describe this dataset, "
        "not diabetes prevalence in the general population."
    )


# Build a detailed missing-measurement table.
def build_quality_table(statistics):
    records = []

    for feature, count in sorted(
        statistics["missing_counts"].items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        records.append({
            "Measurement": get_feature_label(feature),
            "Missing Values": count,
            "Available Values": (
                statistics["total"] - count
            ),
            "Missing Rate": (
                count / statistics["total"] * 100
            ),
        })

    return pd.DataFrame(records)


# Render data completeness and missing measurements.
def render_data_quality(statistics):
    st.subheader("Data Quality")

    quality_column, completeness_column = st.columns(
        [1.4, 1],
        gap="medium",
    )

    with quality_column:
        with st.container(border=True):
            st.markdown(
                "**Missing Measurement Analysis**"
            )

            st.caption(
                "Zero placeholders in the original dataset."
            )

            sorted_counts = sorted(
                statistics["missing_counts"].items(),
                key=lambda item: item[1],
                reverse=True,
            )

            for feature, count in sorted_counts:
                percentage = (
                    count / statistics["total"] * 100
                )

                render_progress_bar(
                    get_feature_label(feature),
                    f"{count:,} · {percentage:.1f}%",
                    percentage,
                    AMBER,
                )

    with completeness_column:
        with st.container(border=True):
            st.markdown("**Record Completeness**")

            st.caption(
                "Availability of the five affected measurements."
            )

            complete_rate = (
                statistics["complete_records"]
                / statistics["total"]
                * 100
            )

            incomplete_rate = (
                statistics["incomplete_records"]
                / statistics["total"]
                * 100
            )

            st.html(
                f"""
                <div class="dashboard-quality-label">
                    Complete Records
                </div>

                <div
                    class="dashboard-quality-value"
                    style="color: {TEAL};"
                >
                    {statistics['complete_records']:,}
                </div>
                """
            )

            render_progress_bar(
                "Complete",
                f"{complete_rate:.1f}%",
                complete_rate,
                TEAL,
            )

            st.divider()

            st.html(
                f"""
                <div class="dashboard-quality-label">
                    Records with Missing Measurements
                </div>

                <div
                    class="dashboard-quality-value"
                    style="color: {AMBER};"
                >
                    {statistics['incomplete_records']:,}
                </div>
                """
            )

            render_progress_bar(
                "Incomplete",
                f"{incomplete_rate:.1f}%",
                incomplete_rate,
                AMBER,
            )

            st.caption(
                f"{statistics['total_missing']:,} missing "
                "measurement values across the dataset."
            )

    st.caption(
        "A record may contain multiple missing measurements. "
        "Missing measurement counts are not missing record counts."
    )

    with st.expander("View Detailed Data Quality Statistics"):
        st.dataframe(
            build_quality_table(statistics),
            width="stretch",
            hide_index=True,
            column_config={
                "Missing Values": st.column_config.NumberColumn(
                    format="%d",
                ),
                "Available Values": st.column_config.NumberColumn(
                    format="%d",
                ),
                "Missing Rate": st.column_config.NumberColumn(
                    format="%.2f%%",
                ),
            },
        )


# Render dataset background and limitations.
def render_dataset_information(statistics):
    with st.expander("About the Dataset"):
        st.markdown(
            "**Pima Indians Diabetes Database**"
        )

        st.write(
            "UODA DiaML uses the Pima Indians Diabetes "
            "Database for educational data analysis and "
            "machine learning classification."
        )

        first, second, third = st.columns(3)

        with first:
            st.metric(
                "Records",
                f"{statistics['total']:,}",
            )

        with second:
            st.metric(
                "Input Features",
                len(FEATURE_COLUMNS),
            )

        with third:
            st.metric(
                "Outcome Classes",
                2,
            )

        st.divider()

        st.markdown("**Dataset Limitations**")

        st.write(
            "The dataset represents a specific study "
            "population. Its observed statistics and "
            "model predictions should not be generalized "
            "to all individuals or populations."
        )

        st.markdown("**Missing Measurements**")

        st.write(
            "Zero values in glucose, blood pressure, "
            "skin thickness, insulin and BMI are treated "
            "as missing measurement placeholders during "
            "exploratory analysis."
        )


# Render the complete dashboard overview.
def render_overview(dataframe):
    statistics = calculate_dashboard_statistics(
        dataframe
    )

    render_overview_cards(statistics)

    st.divider()

    render_outcome_analysis(
        dataframe,
        statistics,
    )

    st.divider()

    render_data_quality(statistics)

    render_dataset_information(statistics)


# Reset all Data Explorer controls.
def reset_explorer_filters():
    defaults = {
        "explorer_outcome_filter": "All Records",
        "dashboard_explorer_view": "Distributions",
        "explorer_distribution_feature": "Glucose",
        "explorer_histogram_bins": 25,
        "explorer_scatter_x": "Glucose",
        "explorer_scatter_y": "BMI",
        "explorer_box_feature": "Glucose",
        "explorer_parallel_features": [
            "Glucose",
            "BMI",
            "Age",
            "Insulin",
        ],
    }

    for key, value in defaults.items():
        st.session_state[key] = value


# Preserve explorer selections between analytical views.
def initialize_explorer_state():
    defaults = {
        "explorer_outcome_filter": "All Records",
        "dashboard_explorer_view": "Distributions",
        "explorer_distribution_feature": "Glucose",
        "explorer_histogram_bins": 25,
        "explorer_scatter_x": "Glucose",
        "explorer_scatter_y": "BMI",
        "explorer_box_feature": "Glucose",
        "explorer_parallel_features": [
            "Glucose",
            "BMI",
            "Age",
            "Insulin",
        ],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
        else:
            st.session_state[key] = st.session_state[key]


# Filter dataset records by their outcome label.
def filter_dataset(dataframe, outcome_filter):
    if outcome_filter == "Positive Only":
        return dataframe[
            dataframe["Outcome"] == 1
        ].copy()

    if outcome_filter == "Negative Only":
        return dataframe[
            dataframe["Outcome"] == 0
        ].copy()

    return dataframe.copy()


# Render compact global explorer filters.
def render_explorer_filters(dataframe):
    with st.container(border=True):
        st.markdown("**Explorer Filters**")

        filter_column, reset_column = st.columns(
            [3, 1],
            vertical_alignment="bottom",
        )

        with filter_column:
            outcome_filter = st.selectbox(
                "Diabetes Outcome",
                options=[
                    "All Records",
                    "Negative Only",
                    "Positive Only",
                ],
                key="explorer_outcome_filter",
            )

        with reset_column:
            st.button(
                "Reset Filters",
                icon=":material/restart_alt:",
                width="stretch",
                on_click=reset_explorer_filters,
                key="dashboard_reset_explorer",
            )

        filtered = filter_dataset(
            dataframe,
            outcome_filter,
        )

        positive = int(
            (filtered["Outcome"] == 1).sum()
        )

        negative = int(
            (filtered["Outcome"] == 0).sum()
        )

        st.html(
            f"""
            <div class="dashboard-filter-summary">
                <span>
                    Showing
                    <span class="dashboard-filter-count">
                        {len(filtered):,}
                    </span>
                    of {len(dataframe):,} records
                </span>

                <span class="dashboard-filter-context">
                    {positive:,} positive ·
                    {negative:,} negative
                </span>
            </div>
            """
        )

    return filtered


# Render summary statistics for a selected measurement.
def render_measurement_statistics(dataframe, feature):
    values = dataframe[feature].copy()

    if feature in MISSING_ZERO_FEATURES:
        values = values.replace(
            0,
            float("nan"),
        )

    values = values.dropna()

    columns = st.columns(4)

    statistics = [
        ("Valid Records", f"{len(values):,}"),
        ("Mean", format_statistic(values.mean(), 2)),
        ("Median", format_statistic(values.median(), 2)),
        (
            "Std. Deviation",
            format_statistic(values.std(), 2),
        ),
    ]

    for column, (label, value) in zip(
        columns,
        statistics,
    ):
        with column:
            st.metric(label, value)


# Render interactive measurement distributions.
def render_distributions(dataframe):
    st.subheader("Measurement Distributions")

    with st.container(border=True):
        st.markdown("**Measurement Distribution**")

        st.caption(
            "Explore the distribution and descriptive "
            "statistics of any input measurement."
        )

        feature_column, bin_column = st.columns(
            [1.3, 1],
        )

        with feature_column:
            feature = st.selectbox(
                "Measurement",
                options=EXPLORER_FEATURES,
                format_func=get_feature_label,
                key="explorer_distribution_feature",
            )

        with bin_column:
            bin_count = st.slider(
                "Histogram Bins",
                min_value=10,
                max_value=50,
                step=5,
                key="explorer_histogram_bins",
            )

        render_measurement_statistics(
            dataframe,
            feature,
        )

        st.divider()

        figure = create_distribution_chart(
            dataframe,
            feature=feature,
            title=f"{get_feature_label(feature)} Distribution",
            x_label=get_feature_axis_label(feature),
            bins=bin_count,
        )

        render_chart(
            figure,
            key="dashboard_selected_distribution",
            height=370,
            hide_title=True,
        )

        if feature in MISSING_ZERO_FEATURES:
            st.caption(
                "Zero placeholders are excluded from "
                "the distribution and summary statistics."
            )

    st.markdown("**Additional Measurement Distributions**")

    supplementary_features = [
        item
        for item in ["Glucose", "BMI", "Age"]
        if item != feature
    ]

    if len(supplementary_features) == 3:
        supplementary_features = (
            supplementary_features[:2]
        )

    columns = st.columns(2)

    for column, supplementary_feature in zip(
        columns,
        supplementary_features,
    ):
        with column:
            with st.container(border=True):
                label = get_feature_label(
                    supplementary_feature
                )

                st.markdown(
                    f"**{label} Distribution**"
                )

                figure = create_distribution_chart(
                    dataframe,
                    feature=supplementary_feature,
                    title=f"{label} Distribution",
                    x_label=get_feature_axis_label(
                        supplementary_feature
                    ),
                    bins=bin_count,
                )

                render_chart(
                    figure,
                    key=(
                        "dashboard_supplementary_"
                        f"{supplementary_feature}"
                    ),
                    height=290,
                    hide_title=True,
                )


# Render correlation and scatter plot analysis.
def render_relationships(dataframe):
    st.subheader("Feature Relationships")

    with st.container(border=True):
        st.markdown("**Feature Correlation Heatmap**")

        st.caption(
            "Pearson correlation between the eight "
            "input measurements."
        )

        figure = create_correlation_heatmap(
            dataframe
        )

        render_chart(
            figure,
            key="dashboard_correlation_heatmap",
            height=460,
            hide_title=True,
        )

        st.caption(
            "Missing zero placeholders are excluded "
            "pairwise. Correlation does not establish "
            "causation."
        )

    with st.container(border=True):
        st.markdown(
            "**Interactive Relationship Explorer**"
        )

        st.caption(
            "Compare two measurements across "
            "the selected outcome groups."
        )

        x_column, y_column = st.columns(2)

        with x_column:
            x_feature = st.selectbox(
                "X-axis Measurement",
                options=EXPLORER_FEATURES,
                format_func=get_feature_label,
                key="explorer_scatter_x",
            )

        with y_column:
            y_feature = st.selectbox(
                "Y-axis Measurement",
                options=EXPLORER_FEATURES,
                format_func=get_feature_label,
                key="explorer_scatter_y",
            )

        if x_feature == y_feature:
            st.warning(
                "Select two different measurements "
                "to explore their relationship."
            )

            return

        cleaned = clean_measurements(dataframe)

        paired = cleaned[
            [x_feature, y_feature]
        ].dropna()

        correlation = paired[x_feature].corr(
            paired[y_feature]
        )

        first, second = st.columns(2)

        with first:
            st.metric(
                "Valid Measurement Pairs",
                f"{len(paired):,}",
            )

        with second:
            st.metric(
                "Pearson Correlation",
                format_statistic(correlation, 3),
            )

        st.divider()

        figure = create_scatter_chart(
            dataframe,
            x_feature=x_feature,
            y_feature=y_feature,
        )

        render_chart(
            figure,
            key="dashboard_scatter_chart",
            height=390,
            hide_title=True,
        )

        st.caption(
            "Each point represents one dataset record. "
            "Records missing either selected measurement "
            "are excluded."
        )


# Build descriptive statistics for outcome groups.
def build_comparison_statistics(
    dataframe,
    feature,
):
    cleaned = clean_measurements(dataframe)

    records = []

    for outcome, label in [
        (0, "Negative"),
        (1, "Positive"),
    ]:
        values = cleaned.loc[
            cleaned["Outcome"] == outcome,
            feature,
        ].dropna()

        if values.empty:
            continue

        records.append({
            "Outcome": label,
            "Records": len(values),
            "Mean": values.mean(),
            "Median": values.median(),
            "Std. Deviation": values.std(),
            "Minimum": values.min(),
            "Q1": values.quantile(0.25),
            "Q3": values.quantile(0.75),
            "Maximum": values.max(),
        })

    return pd.DataFrame(records)


# Render box plots and group-level statistics.
def render_comparisons(dataframe):
    st.subheader("Statistical Comparison")

    with st.container(border=True):
        st.markdown(
            "**Outcome Group Comparison**"
        )

        st.caption(
            "Compare measurement distributions, "
            "medians and variability."
        )

        feature = st.selectbox(
            "Measurement to Compare",
            options=EXPLORER_FEATURES,
            format_func=get_feature_label,
            key="explorer_box_feature",
        )

        figure = create_box_plot(
            dataframe,
            feature=feature,
        )

        render_chart(
            figure,
            key="dashboard_box_plot",
            height=390,
            hide_title=True,
        )

        st.caption(
            "The box represents the middle 50% of "
            "observed values. The central line is the "
            "median, and points beyond the whiskers "
            "indicate potential outliers."
        )

    with st.container(border=True):
        st.markdown(
            "**Descriptive Statistics by Outcome**"
        )

        statistics = build_comparison_statistics(
            dataframe,
            feature,
        )

        if statistics.empty:
            st.info(
                "No valid measurements are available "
                "for the selected outcome group."
            )

            return

        numeric_columns = [
            "Mean",
            "Median",
            "Std. Deviation",
            "Minimum",
            "Q1",
            "Q3",
            "Maximum",
        ]

        column_config = {
            column: st.column_config.NumberColumn(
                format="%.2f",
            )
            for column in numeric_columns
        }

        column_config["Outcome"] = (
            st.column_config.TextColumn(
                pinned=True,
            )
        )

        st.dataframe(
            statistics,
            width="stretch",
            hide_index=True,
            column_config=column_config,
        )

        st.caption(
            "Q1 and Q3 represent the 25th and "
            "75th percentiles respectively."
        )


# Render age-group outcome analysis.
def render_age_groups(dataframe):
    st.subheader("Age-Group Analysis")

    if dataframe["Outcome"].nunique() < 2:
        st.info(
            "Only one outcome class is currently selected. "
            "Choose All Records to compare positive outcome "
            "proportions across age groups."
        )

    with st.container(border=True):
        st.markdown(
            "**Outcome Distribution by Age Group**"
        )

        st.caption(
            "Observed diabetes outcome proportions "
            "across age categories."
        )

        figure = create_age_group_chart(
            dataframe
        )

        render_chart(
            figure,
            key="dashboard_age_group_chart",
            height=390,
            hide_title=True,
        )

        st.caption(
            "Percentages are calculated within each "
            "age group using the selected records. "
            "They are dataset proportions, not "
            "individual risk estimates."
        )

    with st.container(border=True):
        st.markdown("**Age-Group Statistics**")

        statistics = calculate_age_group_statistics(
            dataframe
        )

        display_statistics = statistics.copy()

        if "Positive Rate" in display_statistics.columns:
            display_statistics["Positive Rate"] = (
                display_statistics["Positive Rate"].map(
                    lambda value: (
                        f"{value:.1f}%"
                        if pd.notna(value)
                        else "N/A"
                    )
                )
            )

        st.dataframe(
            display_statistics,
            width="stretch",
            hide_index=True,
        )


# Render parallel coordinates analysis.
def render_multivariable(dataframe):
    st.subheader("Multivariable Analysis")

    with st.container(border=True):
        st.markdown(
            "**Parallel Coordinates Explorer**"
        )

        st.caption(
            "Investigate patterns across multiple "
            "measurements simultaneously."
        )

        selected_features = st.multiselect(
            "Select 3 to 6 Measurements",
            options=EXPLORER_FEATURES,
            format_func=get_feature_label,
            max_selections=6,
            key="explorer_parallel_features",
        )

        if len(selected_features) < 3:
            st.warning(
                "Select at least three measurements "
                "to display the parallel coordinates chart."
            )

            return

        cleaned = clean_measurements(dataframe)

        valid_records = cleaned[
            selected_features
        ].dropna()

        first, second = st.columns(2)

        with first:
            st.metric(
                "Selected Measurements",
                len(selected_features),
            )

        with second:
            st.metric(
                "Complete Records",
                f"{len(valid_records):,}",
            )

        if valid_records.empty:
            st.info(
                "No complete records are available "
                "for the selected measurements."
            )

            return

        st.divider()

        figure = create_parallel_coordinates_chart(
            dataframe,
            selected_features,
        )

        render_chart(
            figure,
            key="dashboard_parallel_coordinates",
            height=440,
            hide_title=True,
        )

        st.caption(
            "Each line represents one dataset record. "
            "Records missing any selected measurement "
            "are excluded."
        )

        st.caption(
            "Tip: Drag along an axis to highlight "
            "records within a selected range."
        )


# Render dataset reference information and raw records.
def render_explorer_resources(dataframe):
    with st.expander("Dataset Feature Descriptions"):
        feature_table = pd.DataFrame([
            {
                "Feature": get_feature_label(feature),
                "Description": description,
                "Unit": FEATURE_UNITS.get(
                    feature,
                    "",
                ) or "—",
            }
            for feature, description
            in FEATURE_DESCRIPTIONS.items()
        ])

        st.dataframe(
            feature_table,
            width="stretch",
            hide_index=True,
            column_config={
                "Feature": st.column_config.TextColumn(
                    width="medium",
                ),
                "Description": st.column_config.TextColumn(
                    width="large",
                ),
            },
        )

    with st.expander(
        f"Dataset Preview · {len(dataframe):,} Records"
    ):
        st.caption(
            "Browse the original measurement records "
            "matching the selected outcome filter."
        )

        preview_size = st.selectbox(
            "Records to Display",
            options=[10, 25, 50, 100, "All Records"],
            key="dashboard_preview_size",
        )

        preview = (
            dataframe.copy()
            if preview_size == "All Records"
            else dataframe.head(preview_size).copy()
        )

        preview.insert(
            0,
            "Record ID",
            preview.index + 1,
        )

        preview["Outcome"] = preview["Outcome"].map({
            0: "Negative",
            1: "Positive",
        })

        st.dataframe(
            preview,
            width="stretch",
            hide_index=True,
            height=min(
                420,
                38 + len(preview) * 35,
            ),
            column_config={
                "Record ID": st.column_config.NumberColumn(
                    pinned=True,
                    format="%d",
                ),
                "BMI": st.column_config.NumberColumn(
                    format="%.2f",
                ),
                "DiabetesPedigreeFunction": (
                    st.column_config.NumberColumn(
                        "Diabetes Pedigree Function",
                        format="%.3f",
                    )
                ),
                "BloodPressure": "Blood Pressure",
                "SkinThickness": "Skin Thickness",
            },
        )

        st.caption(
            f"Displaying {len(preview):,} of "
            f"{len(dataframe):,} selected records. "
            "Original zero placeholders are retained."
        )


# Render the interactive Data Explorer.
def render_data_explorer(dataframe):
    initialize_explorer_state()

    filtered = render_explorer_filters(
        dataframe
    )

    if filtered.empty:
        st.info(
            "No records match the selected filters."
        )

        return

    analytical_view = st.segmented_control(
        "Analytical View",
        options=[
            "Distributions",
            "Relationships",
            "Comparisons",
            "Age Groups",
            "Multivariable",
        ],
        selection_mode="single",
        key="dashboard_explorer_view",
        label_visibility="collapsed",
        width="stretch",
    )

    analytical_view = (
        analytical_view or "Distributions"
    )

    if analytical_view == "Distributions":
        render_distributions(filtered)

    elif analytical_view == "Relationships":
        render_relationships(filtered)

    elif analytical_view == "Comparisons":
        render_comparisons(filtered)

    elif analytical_view == "Age Groups":
        render_age_groups(filtered)

    elif analytical_view == "Multivariable":
        render_multivariable(filtered)

    st.divider()

    render_explorer_resources(filtered)


# Initialize the dashboard page.
apply_dashboard_styles()

st.title("Diabetes Analytics Dashboard")

st.caption(
    "Explore dataset characteristics, measurement "
    "distributions and statistical relationships."
)

st.divider()

try:
    dataframe = load_dashboard_data()

except (
    FileNotFoundError,
    ValueError,
    pd.errors.ParserError,
) as error:
    st.error(
        f"Unable to load dashboard data: {error}"
    )

    st.stop()


dashboard_view = st.segmented_control(
    "Dashboard View",
    options=[
        "Overview",
        "Data Explorer",
    ],
    default="Overview",
    selection_mode="single",
    key="dashboard_main_view",
    label_visibility="collapsed",
    width="stretch",
)

if dashboard_view == "Data Explorer":
    render_data_explorer(dataframe)

else:
    render_overview(dataframe)

st.divider()

st.caption(
    "UODA DiaML · Educational Diabetes Prediction System. "
    "Dataset statistics and model predictions are not "
    "medical diagnoses or clinical recommendations."
)
