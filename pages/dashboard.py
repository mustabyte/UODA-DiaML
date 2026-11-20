
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


# Locate the diabetes dataset.
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

FEATURE_DESCRIPTIONS = {
    "Pregnancies": "Number of pregnancies",
    "Glucose": "Plasma glucose concentration",
    "BloodPressure": "Diastolic blood pressure (mm Hg)",
    "SkinThickness": "Triceps skinfold thickness (mm)",
    "Insulin": "2-hour serum insulin (mu U/mL)",
    "BMI": "Body mass index",
    "DiabetesPedigreeFunction": "Diabetes pedigree function",
    "Age": "Age in years",
    "Outcome": "Diabetes outcome label (0 or 1)",
}


# Load and validate the headerless dataset.
@st.cache_data
def load_dashboard_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            "The diabetes dataset could not be found."
        )

    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=REQUIRED_COLUMNS,
    )

    if df.empty:
        raise ValueError("The diabetes dataset is empty.")

    if df.shape[1] != len(REQUIRED_COLUMNS):
        raise ValueError(
            "The dataset has an unexpected number of columns."
        )

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(
            "The dataset contains missing numeric values."
        )

    if not df["Outcome"].isin([0, 1]).all():
        raise ValueError(
            "Outcome must contain only 0 and 1."
        )

    return df


# Stop the dashboard gracefully if loading fails.
try:
    df = load_dashboard_data()
except (
    FileNotFoundError,
    ValueError,
    pd.errors.ParserError,
) as error:
    st.error(f"Unable to load dashboard data: {error}")
    st.stop()


# Calculate statistics from the complete dataset.
total_records = len(df)
positive_cases = int((df["Outcome"] == 1).sum())
negative_cases = int((df["Outcome"] == 0).sum())

positive_percentage = positive_cases / total_records * 100
negative_percentage = negative_cases / total_records * 100

cleaned_df = df.copy()

# Exclude missing measurement placeholders from averages.
for column in MISSING_ZERO_FEATURES:
    cleaned_df[column] = cleaned_df[column].replace(
        0,
        float("nan"),
    )

average_glucose = cleaned_df["Glucose"].mean()
average_bmi = cleaned_df["BMI"].mean()
average_age = cleaned_df["Age"].mean()


# Display the dashboard heading and navigation tabs.
st.title("Diabetes Analytics Dashboard")

st.caption(
    "Explore the Pima Indians Diabetes Dataset through "
    "interactive statistics and advanced visualizations."
)

overview_tab, explorer_tab = st.tabs(
    ["Overview", "Data Explorer"]
)


# Build the dashboard overview.
with overview_tab:
    st.subheader("Dataset Overview")

    metric_columns = st.columns(4)

    overview_metrics = [
        ("Total Records", f"{total_records:,}"),
        ("Positive Cases", f"{positive_cases:,}"),
        ("Negative Cases", f"{negative_cases:,}"),
        ("Positive Case Rate", f"{positive_percentage:.1f}%"),
    ]

    for column, (label, value) in zip(
        metric_columns,
        overview_metrics,
    ):
        with column:
            with st.container(border=True):
                st.metric(label, value)

    st.caption(
        "The positive case rate describes the proportion of "
        "positive records in this dataset, not the general population."
    )

    st.subheader("Key Health Measurements")

    health_columns = st.columns(3)

    health_metrics = [
        ("Average Glucose", f"{average_glucose:.1f} mg/dL"),
        ("Average BMI", f"{average_bmi:.1f}"),
        ("Average Age", f"{average_age:.1f} years"),
    ]

    for column, (label, value) in zip(
        health_columns,
        health_metrics,
    ):
        with column:
            with st.container(border=True):
                st.metric(label, value)

    st.caption(
        "Zero placeholders in glucose and BMI are excluded "
        "from these averages."
    )

    chart_column, insight_column = st.columns(
        [1.4, 1],
        gap="medium",
    )

    with chart_column:
        with st.container(border=True):
            st.markdown("#### Diabetes Outcome Distribution")

            outcome_fig = create_outcome_chart(df)

            st.plotly_chart(
                outcome_fig,
                use_container_width=True,
                key="overview_outcome_chart",
            )

    with insight_column:
        with st.container(border=True):
            st.markdown("#### Dataset Insights")

            st.metric(
                "Negative Class",
                f"{negative_percentage:.1f}%",
            )

            st.metric(
                "Positive Class",
                f"{positive_percentage:.1f}%",
            )

            st.markdown("**Class Distribution**")

            st.progress(positive_percentage / 100)

            st.caption(
                f"{positive_cases:,} positive records out of "
                f"{total_records:,} total records."
            )

            st.markdown("**Class Distribution Insight**")

            st.caption(
                "Negative cases outnumber positive cases in this "
                "dataset. This imbalance matters when evaluating "
                "classification models."
            )

    st.subheader("Data Quality Overview")

    quality_columns = st.columns(3)

    quality_metrics = [
        ("Missing Glucose", int((df["Glucose"] == 0).sum())),
        ("Missing BMI", int((df["BMI"] == 0).sum())),
        ("Missing Insulin", int((df["Insulin"] == 0).sum())),
    ]

    for column, (label, value) in zip(
        quality_columns,
        quality_metrics,
    ):
        with column:
            with st.container(border=True):
                st.metric(label, f"{value:,}")

    st.caption(
        "These counts represent zero placeholders in the "
        "original dataset, not missing records."
    )

    with st.container(border=True):
        st.markdown("#### About the Dataset")

        st.write(
            "UODA DiaML uses the Pima Indians Diabetes Database, "
            "containing 768 records, eight input features, "
            "and one binary diabetes outcome label."
        )

        st.markdown("**Dataset Limitation**")

        st.caption(
            "This dataset represents a specific study population. "
            "Its statistics and model predictions should not "
            "be generalized to everyone."
        )


# Build the interactive Data Explorer.
with explorer_tab:
    st.subheader("Explore the Dataset")

    st.write(
        "Investigate distributions, statistical relationships, "
        "and patterns across diabetes outcome groups."
    )

    filter_column, bin_column = st.columns(2)

    with filter_column:
        outcome_filter = st.selectbox(
            "Diabetes outcome",
            options=[
                "All Records",
                "Negative Only",
                "Positive Only",
            ],
            key="explorer_outcome_filter",
        )

    with bin_column:
        bin_count = st.slider(
            "Histogram bins",
            min_value=10,
            max_value=50,
            value=25,
            step=5,
            key="explorer_histogram_bins",
        )

    # Apply the outcome filter across Data Explorer charts.
    if outcome_filter == "Negative Only":
        filtered_df = df[df["Outcome"] == 0].copy()
    elif outcome_filter == "Positive Only":
        filtered_df = df[df["Outcome"] == 1].copy()
    else:
        filtered_df = df.copy()

    st.caption(
        f"Showing {len(filtered_df):,} of "
        f"{len(df):,} records."
    )

    st.subheader("Health Measurement Distributions")

    glucose_column, bmi_column = st.columns(2)

    with glucose_column:
        with st.container(border=True):
            glucose_fig = create_distribution_chart(
                filtered_df,
                feature="Glucose",
                title="Glucose Distribution",
                x_label="Glucose (mg/dL)",
                bins=bin_count,
            )

            st.plotly_chart(
                glucose_fig,
                use_container_width=True,
                key="explorer_glucose_chart",
            )

    with bmi_column:
        with st.container(border=True):
            bmi_fig = create_distribution_chart(
                filtered_df,
                feature="BMI",
                title="BMI Distribution",
                x_label="BMI",
                bins=bin_count,
            )

            st.plotly_chart(
                bmi_fig,
                use_container_width=True,
                key="explorer_bmi_chart",
            )

    with st.container(border=True):
        age_fig = create_distribution_chart(
            filtered_df,
            feature="Age",
            title="Age Distribution",
            x_label="Age (years)",
            bins=bin_count,
        )

        st.plotly_chart(
            age_fig,
            use_container_width=True,
            key="explorer_age_chart",
        )

    st.caption(
        "Glucose and BMI values of zero are treated as "
        "missing measurements and excluded from their charts."
    )

    st.subheader("Advanced Relationship Analysis")

    st.write(
        "Explore how the eight input features relate to one another."
    )

    with st.container(border=True):
        st.markdown("#### Feature Correlation Heatmap")

        heatmap_fig = create_correlation_heatmap(filtered_df)

        st.plotly_chart(
            heatmap_fig,
            use_container_width=True,
            key="explorer_correlation_heatmap",
        )

        st.caption(
            "Pearson correlation measures linear relationships. "
            "Missing zero placeholders are excluded pairwise. "
            "Correlation does not establish causation."
        )

    with st.container(border=True):
        st.markdown("#### Interactive Feature Relationship Explorer")

        x_column, y_column = st.columns(2)

        with x_column:
            x_feature = st.selectbox(
                "X-axis measurement",
                EXPLORER_FEATURES,
                index=1,
                key="explorer_scatter_x",
            )

        with y_column:
            y_feature = st.selectbox(
                "Y-axis measurement",
                EXPLORER_FEATURES,
                index=5,
                key="explorer_scatter_y",
            )

        if x_feature == y_feature:
            st.warning(
                "Choose two different measurements "
                "to explore their relationship."
            )
        else:
            scatter_fig = create_scatter_chart(
                filtered_df,
                x_feature=x_feature,
                y_feature=y_feature,
            )

            st.plotly_chart(
                scatter_fig,
                use_container_width=True,
                key="explorer_scatter_chart",
            )

            st.caption(
                "Each dot represents one dataset record. "
                "Records missing selected measurements are excluded."
            )

    # Compare distributions between outcome groups.
    st.subheader("Statistical Comparison")

    with st.container(border=True):
        st.markdown("#### Interactive Box Plot")

        box_feature = st.selectbox(
            "Measurement to compare",
            EXPLORER_FEATURES,
            index=1,
            key="explorer_box_feature",
        )

        box_fig = create_box_plot(
            filtered_df,
            feature=box_feature,
        )

        st.plotly_chart(
            box_fig,
            use_container_width=True,
            key="explorer_box_chart",
        )

        st.caption(
            "The box shows the middle 50% of observed values. "
            "The line inside represents the median, while points "
            "outside the whiskers indicate potential outliers."
        )

    # Show observed outcome rates by age group.
    st.subheader("Age-Group Outcome Analysis")

    with st.container(border=True):
        age_stats = calculate_age_group_statistics(filtered_df)

        age_fig = create_age_group_chart(filtered_df)

        st.plotly_chart(
            age_fig,
            use_container_width=True,
            key="explorer_age_group_chart",
        )

        st.caption(
            "Each percentage is calculated from the records "
            "within that age group and the selected outcome filter. "
            "These are dataset proportions, not population risk estimates."
        )

        with st.expander("View Age-Group Statistics"):
            display_stats = age_stats.copy()

            display_stats["Positive Rate"] = (
                display_stats["Positive Rate"]
                .map(
                    lambda value: (
                        f"{value:.1f}%"
                        if pd.notna(value)
                        else "N/A"
                    )
                )
            )

            st.dataframe(
                display_stats,
                use_container_width=True,
                hide_index=True,
            )

    # Explore several features simultaneously.
    st.subheader("Multivariable Pattern Explorer")

    with st.container(border=True):
        st.markdown("#### Parallel Coordinates Analysis")

        selected_features = st.multiselect(
            "Select 3 to 6 measurements",
            options=EXPLORER_FEATURES,
            default=[
                "Glucose",
                "BMI",
                "Age",
                "Insulin",
            ],
            max_selections=6,
            key="explorer_parallel_features",
        )

        if len(selected_features) < 3:
            st.warning(
                "Select at least three measurements "
                "to display the multivariable chart."
            )
        else:
            parallel_fig = create_parallel_coordinates_chart(
                filtered_df,
                selected_features,
            )

            st.plotly_chart(
                parallel_fig,
                use_container_width=True,
                key="explorer_parallel_chart",
            )

            st.caption(
                "Each line represents a dataset record. "
                "Teal indicates a negative outcome and amber "
                "indicates a positive outcome. Records missing "
                "any selected measurement are excluded."
            )

            st.caption(
                "Tip: Drag along an axis in the chart to "
                "highlight records within a selected range."
            )

    # Keep feature descriptions and dataset preview available.
    with st.expander("Dataset Feature Descriptions"):
        feature_table = pd.DataFrame(
            [
                {
                    "Feature": name,
                    "Description": description,
                }
                for name, description in FEATURE_DESCRIPTIONS.items()
            ]
        )

        st.dataframe(
            feature_table,
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Preview Dataset"):
        st.dataframe(
            filtered_df.head(10),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Showing up to 10 records from the selected group."
        )

    st.caption(
        "All visualizations are for educational data exploration "
        "and should not be interpreted as medical advice."
    )
