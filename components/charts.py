
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Keep the dashboard's chart colors consistent.
POSITIVE_COLOR = "#F59E0B"
NEGATIVE_COLOR = "#0D9488"

CHART_COLORS = {
    "Negative": NEGATIVE_COLOR,
    "Positive": POSITIVE_COLOR,
}

EXPLORER_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

MISSING_ZERO_FEATURES = {
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
}

AGE_GROUP_LABELS = [
    "21–30",
    "31–40",
    "41–50",
    "51–60",
    "61+",
]


# Apply consistent styling to Plotly figures.
def style_chart(fig, height=350):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=25, r=25, t=55, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#CBD5E1",
        ),
        hoverlabel=dict(
            bgcolor="#1E293B",
            font_color="#F8FAFC",
        ),
        legend_title_text="",
        xaxis=dict(
            gridcolor="rgba(148,163,184,0.12)",
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.12)",
            zeroline=False,
        ),
    )

    return fig


# Replace invalid zero placeholders without changing the original data.
def prepare_explorer_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    for feature in MISSING_ZERO_FEATURES:
        if feature in cleaned.columns:
            cleaned[feature] = cleaned[feature].replace(0, np.nan)

    return cleaned


# Display positive and negative outcome proportions.
def create_outcome_chart(df: pd.DataFrame) -> go.Figure:
    counts = df["Outcome"].value_counts().reindex(
        [0, 1],
        fill_value=0,
    )

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Negative", "Positive"],
                values=counts.values,
                hole=0.65,
                marker=dict(
                    colors=[NEGATIVE_COLOR, POSITIVE_COLOR]
                ),
                textinfo="percent",
                hovertemplate=(
                    "%{label}<br>"
                    "Records: %{value}<br>"
                    "Percentage: %{percent}"
                    "<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        annotations=[
            dict(
                text=f"{len(df):,}<br>Records",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=17, color="#F8FAFC"),
            )
        ],
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.1,
        ),
    )

    return style_chart(fig, height=350)


# Create an outcome-colored histogram.
def create_distribution_chart(
    df: pd.DataFrame,
    feature: str,
    title: str,
    x_label: str,
    bins: int = 25,
) -> go.Figure:
    if feature not in {"Glucose", "BMI", "Age"}:
        raise ValueError(f"Unsupported chart feature: {feature}")

    if not 5 <= bins <= 100:
        raise ValueError("Histogram bins must be between 5 and 100.")

    cleaned = prepare_explorer_data(df)

    chart_df = cleaned[[feature, "Outcome"]].dropna().copy()

    chart_df["Outcome Label"] = chart_df["Outcome"].map(
        {0: "Negative", 1: "Positive"}
    )

    fig = px.histogram(
        chart_df,
        x=feature,
        color="Outcome Label",
        color_discrete_map=CHART_COLORS,
        category_orders={
            "Outcome Label": ["Negative", "Positive"]
        },
        nbins=bins,
        barmode="overlay",
        opacity=0.7,
        title=title,
        labels={feature: x_label},
    )

    fig.update_traces(
        hovertemplate=(
            "Measurement: %{x}<br>"
            "Records: %{y}"
            "<extra>%{fullData.name}</extra>"
        )
    )

    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title="Number of Records",
        bargap=0.05,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.25,
        ),
    )

    return style_chart(fig, height=380)


# Show Pearson correlations between input features.
def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    cleaned = prepare_explorer_data(df)

    correlations = cleaned[EXPLORER_FEATURES].corr(
        method="pearson"
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=correlations.values,
            x=correlations.columns.tolist(),
            y=correlations.index.tolist(),
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=[
                [0.0, POSITIVE_COLOR],
                [0.5, "#172337"],
                [1.0, NEGATIVE_COLOR],
            ],
            colorbar=dict(
                title="Pearson r",
                thickness=14,
            ),
            hovertemplate=(
                "%{y} vs %{x}<br>"
                "Correlation: %{z:.3f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Feature Correlation Matrix",
        xaxis=dict(tickangle=-35),
        yaxis=dict(autorange="reversed"),
    )

    fig = style_chart(fig, height=570)

    fig.update_layout(
        margin=dict(l=125, r=30, t=55, b=125)
    )

    return fig


# Compare any two selected health measurements.
def create_scatter_chart(
    df: pd.DataFrame,
    x_feature: str = "Glucose",
    y_feature: str = "BMI",
) -> go.Figure:
    if x_feature not in EXPLORER_FEATURES:
        raise ValueError(f"Unsupported X-axis feature: {x_feature}")

    if y_feature not in EXPLORER_FEATURES:
        raise ValueError(f"Unsupported Y-axis feature: {y_feature}")

    if x_feature == y_feature:
        raise ValueError("Choose two different features.")

    cleaned = prepare_explorer_data(df)

    chart_df = cleaned.dropna(
        subset=[x_feature, y_feature, "Outcome"]
    ).copy()

    chart_df["Outcome Label"] = chart_df["Outcome"].map(
        {0: "Negative", 1: "Positive"}
    )

    extra_hover_features = [
        feature
        for feature in ["Age", "Glucose", "BMI"]
        if feature not in {x_feature, y_feature}
    ]

    fig = px.scatter(
        chart_df,
        x=x_feature,
        y=y_feature,
        color="Outcome Label",
        color_discrete_map=CHART_COLORS,
        category_orders={
            "Outcome Label": ["Negative", "Positive"]
        },
        opacity=0.7,
        hover_data=extra_hover_features,
        title=f"{x_feature} vs {y_feature}",
    )

    fig.update_traces(
        marker=dict(
            size=8,
            line=dict(
                width=0.5,
                color="rgba(255,255,255,0.25)",
            ),
        )
    )

    fig.update_layout(
        xaxis_title=x_feature,
        yaxis_title=y_feature,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.2,
        ),
    )

    return style_chart(fig, height=460)


# Compare medians, quartiles, and outliers across outcome groups.
def create_box_plot(
    df: pd.DataFrame,
    feature: str = "Glucose",
) -> go.Figure:
    if feature not in EXPLORER_FEATURES:
        raise ValueError(f"Unsupported feature: {feature}")

    cleaned = prepare_explorer_data(df)

    chart_df = cleaned[[feature, "Outcome"]].dropna().copy()

    chart_df["Outcome Label"] = chart_df["Outcome"].map(
        {0: "Negative", 1: "Positive"}
    )

    fig = px.box(
        chart_df,
        x="Outcome Label",
        y=feature,
        color="Outcome Label",
        color_discrete_map=CHART_COLORS,
        category_orders={
            "Outcome Label": ["Negative", "Positive"]
        },
        points="outliers",
        title=f"{feature} by Diabetes Outcome",
    )

    fig.update_layout(
        xaxis_title="Diabetes Outcome",
        yaxis_title=feature,
        showlegend=False,
    )

    return style_chart(fig, height=430)


# Calculate observed positive-case rates across age groups.
def calculate_age_group_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    required = {"Age", "Outcome"}

    if not required.issubset(df.columns):
        raise ValueError("Age and Outcome columns are required.")

    valid = df.loc[
        df["Age"].notna()
        & df["Outcome"].isin([0, 1])
        & (df["Age"] >= 21),
        ["Age", "Outcome"],
    ].copy()

    valid["Age Group"] = pd.cut(
        valid["Age"],
        bins=[20, 30, 40, 50, 60, np.inf],
        labels=AGE_GROUP_LABELS,
        include_lowest=True,
    )

    grouped = valid.groupby(
        "Age Group",
        observed=False,
    )["Outcome"].agg(
        Total="count",
        Positive="sum",
    )

    grouped = grouped.reset_index()

    grouped["Positive Rate"] = np.where(
        grouped["Total"] > 0,
        grouped["Positive"] / grouped["Total"] * 100,
        np.nan,
    )

    return grouped


# Display observed positive-case rates by age group.
def create_age_group_chart(df: pd.DataFrame) -> go.Figure:
    stats = calculate_age_group_statistics(df)

    chart_df = stats[stats["Total"] > 0].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=chart_df["Age Group"].astype(str),
            y=chart_df["Positive Rate"],
            marker_color=POSITIVE_COLOR,
            customdata=np.column_stack(
                [
                    chart_df["Positive"],
                    chart_df["Total"],
                ]
            ),
            text=chart_df["Positive Rate"].map(
                lambda value: f"{value:.1f}%"
            ),
            textposition="outside",
            hovertemplate=(
                "Age group: %{x}<br>"
                "Positive rate: %{y:.1f}%<br>"
                "Positive records: %{customdata[0]:.0f}<br>"
                "Total records: %{customdata[1]:.0f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Observed Positive-Case Rate by Age Group",
        xaxis_title="Age Group (years)",
        yaxis_title="Positive Records (%)",
        yaxis=dict(range=[0, 110]),
        showlegend=False,
    )

    return style_chart(fig, height=430)


# Explore several measurements together using parallel coordinates.
def create_parallel_coordinates_chart(
    df: pd.DataFrame,
    features: list[str],
) -> go.Figure:
    if not 3 <= len(features) <= 6:
        raise ValueError("Select between 3 and 6 features.")

    if len(set(features)) != len(features):
        raise ValueError("Selected features must be unique.")

    if any(feature not in EXPLORER_FEATURES for feature in features):
        raise ValueError("One or more selected features are unsupported.")

    cleaned = prepare_explorer_data(df)

    # Complete-case filtering avoids misleading missing-value coordinates.
    chart_df = cleaned[features + ["Outcome"]].dropna().copy()

    if chart_df.empty:
        return style_chart(
            go.Figure().add_annotation(
                text="No complete records for these features.",
                showarrow=False,
            ),
            height=470,
        )

    dimensions = []

    for feature in features:
        minimum = float(chart_df[feature].min())
        maximum = float(chart_df[feature].max())

        if minimum == maximum:
            minimum -= 0.5
            maximum += 0.5

        dimensions.append(
            dict(
                label=feature,
                values=chart_df[feature],
                range=[minimum, maximum],
            )
        )

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=chart_df["Outcome"],
                colorscale=[
                    [0.0, NEGATIVE_COLOR],
                    [0.499, NEGATIVE_COLOR],
                    [0.5, POSITIVE_COLOR],
                    [1.0, POSITIVE_COLOR],
                ],
                cmin=0,
                cmax=1,
                showscale=False,
            ),
            dimensions=dimensions,
            labelfont=dict(
                size=12,
                color="#E2E8F0",
            ),
            tickfont=dict(
                size=10,
                color="#CBD5E1",
            ),
        )
    )

    fig.update_layout(
        title="Multivariable Health Measurement Patterns",
    )

    return style_chart(fig, height=470)
