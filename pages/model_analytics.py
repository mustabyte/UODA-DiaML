
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    classification_report
)
from sklearn.model_selection import train_test_split

from ml.preprocessing import (
    load_dataset,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "random_forest_pipeline.joblib"
CONFIG_PATH = BASE_DIR / "models" / "model_config.joblib"
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"

TEAL = "#5EEAD4"
LAVENDER = "#C4B5FD"
BLUE = "#93C5FD"
PINK = "#F9A8D4"
YELLOW = "#FCD34D"
MUTED = "#94A3B8"
TEXT = "#E2E8F0"
CARD = "#1C2635"
TRACK = "#334155"
GRID = "rgba(148, 163, 184, 0.12)"

FEATURE_LABELS = {
    "Pregnancies": "Pregnancies",
    "Glucose": "Glucose",
    "BloodPressure": "Blood Pressure",
    "SkinThickness": "Skin Thickness",
    "Insulin": "Insulin",
    "BMI": "Body Mass Index",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
    "Age": "Age"
}


# Load and evaluate the existing trained model
@st.cache_data(show_spinner=False)
def load_model_analytics(model_modified, config_modified, data_modified):
    pipeline = joblib.load(MODEL_PATH)
    config = joblib.load(CONFIG_PATH)

    dataset = load_dataset()

    X = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    classifier = pipeline.named_steps["model"]

    positive_class = config.get("positive_class", 1)
    positive_index = list(pipeline.classes_).index(positive_class)

    probabilities = pipeline.predict_proba(X_test)[:, positive_index]

    return {
        "config": config,
        "actual": y_test.to_numpy(),
        "probabilities": probabilities,
        "training_count": len(X_train),
        "testing_count": len(X_test),
        "dataset_count": len(dataset),
        "feature_importance": classifier.feature_importances_,
        "parameters": classifier.get_params(),
        "threshold": float(config.get("threshold", 0.40))
    }


# Calculate metrics from cached test probabilities
def calculate_metrics(actual, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    matrix = confusion_matrix(
        actual,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

    return {
        "accuracy": accuracy_score(actual, predictions),
        "precision": precision_score(
            actual, predictions, zero_division=0
        ),
        "recall": recall_score(
            actual, predictions, zero_division=0
        ),
        "f1": f1_score(
            actual, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(actual, probabilities),
        "specificity": tn / (tn + fp) if tn + fp else 0,
        "matrix": matrix,
        "predictions": predictions,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


# Apply page-specific visual styling
def apply_analytics_styles():
    st.html("""
        <style>
            .analytics-stat {
                padding: 18px;
                border-radius: 16px;
                background: #1C2635;
                border: 1px solid rgba(148,163,184,0.12);
                margin-bottom: 10px;
            }

            .analytics-stat-label {
                color: #94A3B8;
                font-size: 12px;
                margin-bottom: 6px;
            }

            .analytics-stat-value {
                font-size: 27px;
                font-weight: 700;
                letter-spacing: -0.8px;
            }

            .analytics-stat-description {
                color: #94A3B8;
                font-size: 11px;
                margin-top: 5px;
                line-height: 1.5;
            }

            .analytics-note {
                background: rgba(196,181,253,0.055);
                border: 1px solid rgba(196,181,253,0.15);
                border-left: 3px solid #C4B5FD;
                padding: 18px 20px;
                border-radius: 12px;
                margin: 12px 0;
                color: #CBD5E1;
                font-size: 13px;
                line-height: 1.8;
            }

            .analytics-note-title {
                color: #C4B5FD;
                font-weight: 650;
                font-size: 14px;
                margin-bottom: 5px;
            }

            .analytics-step {
                padding: 16px;
                background: #1C2635;
                border: 1px solid rgba(148,163,184,0.12);
                border-radius: 14px;
                min-height: 125px;
            }

            .analytics-step-number {
                font-size: 11px;
                letter-spacing: 2px;
                color: #5EEAD4;
                margin-bottom: 10px;
            }

            .analytics-step-title {
                font-size: 15px;
                color: #E2E8F0;
                font-weight: 650;
                margin-bottom: 7px;
            }

            .analytics-step-description {
                font-size: 12px;
                color: #94A3B8;
                line-height: 1.6;
            }

            .analytics-ring-label {
                text-align: center;
                color: #E2E8F0;
                font-size: 13px;
                font-weight: 600;
                padding: 2px 0 12px;
            }
        </style>
    """)


# Render the page heading
def render_hero(analytics):
    st.title("Model Analytics")

    st.caption(
        "Explore the trained model's performance, feature importance "
        "and classification behavior."
    )

    st.divider()


# Render a compact statistic
def render_stat(label, value, color=TEAL, description=""):
    st.html(f"""
        <div class="analytics-stat">
            <div class="analytics-stat-label">{label}</div>
            <div class="analytics-stat-value"
                 style="color:{color};">
                {value}
            </div>
            <div class="analytics-stat-description">
                {description}
            </div>
        </div>
    """)


# Render an analytical insight
def render_note(title, description):
    st.html(f"""
        <div class="analytics-note">
            <div class="analytics-note-title">
                {title}
            </div>
            {description}
        </div>
    """)


# Apply consistent Plotly styling
def style_chart(fig, height=350):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Arial, sans-serif",
            color=TEXT,
            size=12
        ),
        margin=dict(l=20, r=20, t=25, b=35),
        hoverlabel=dict(
            bgcolor="#243244",
            font_color=TEXT,
            bordercolor=TRACK
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h"
        )
    )

    fig.update_xaxes(
        gridcolor=GRID,
        zerolinecolor=GRID
    )

    fig.update_yaxes(
        gridcolor=GRID,
        zerolinecolor=GRID
    )

    return fig


def display_chart(fig, key):
    st.plotly_chart(
        fig,
        width="stretch",
        key=key,
        config={
            "displayModeBar": False,
            "scrollZoom": False
        }
    )


# Create a circular performance indicator
def create_performance_ring(label, value, color):
    percentage = float(value) * 100

    fig = go.Figure(
        go.Pie(
            values=[percentage, 100 - percentage],
            hole=0.83,
            sort=False,
            direction="clockwise",
            rotation=0,
            marker=dict(
                colors=[color, TRACK],
                line=dict(width=0)
            ),
            textinfo="none",
            hoverinfo="skip",
            showlegend=False
        )
    )

    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text=f"<b>{percentage:.2f}%</b>",
        showarrow=False,
        xanchor="center",
        yanchor="middle",
        font=dict(
            size=21,
            color=color
        )
    )

    fig.update_layout(
        height=165,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(
            l=12,
            r=12,
            t=10,
            b=10
        )
    )

    return fig


# Render the performance ring and its separate label
def render_performance_ring(label, value, color, key):
    display_chart(
        create_performance_ring(label, value, color),
        key=key
    )

    st.html(f"""
        <div class="analytics-ring-label">
            {label}
        </div>
    """)


# Create the confusion matrix visualization
def create_confusion_matrix(matrix):
    labels = [
        ["True Negative", "False Positive"],
        ["False Negative", "True Positive"]
    ]

    color_values = [
        [0, 1],
        [2, 0]
    ]

    categories = ["Negative", "Positive"]

    fig = go.Figure(
        go.Heatmap(
            z=color_values,
            x=categories,
            y=categories,
            customdata=matrix,
            colorscale=[
                [0.0, "#174C49"],
                [0.33, "#174C49"],
                [0.34, "#665027"],
                [0.66, "#665027"],
                [0.67, "#63354D"],
                [1.0, "#63354D"]
            ],
            zmin=0,
            zmax=2,
            showscale=False,
            xgap=7,
            ygap=7,
            hovertemplate=(
                "Actual: %{y}<br>"
                "Predicted: %{x}<br>"
                "Records: %{customdata}"
                "<extra></extra>"
            )
        )
    )

    for row in range(2):
        for column in range(2):
            fig.add_annotation(
                x=categories[column],
                y=categories[row],
                text=(
                    f"<b>{matrix[row][column]}</b>"
                    f"<br><span style='font-size:12px'>"
                    f"{labels[row][column]}</span>"
                ),
                showarrow=False,
                font=dict(size=26, color=TEXT)
            )

    style_chart(fig, height=340)

    fig.update_xaxes(
        title="Predicted Classification",
        side="bottom",
        showgrid=False
    )

    fig.update_yaxes(
        title="Actual Classification",
        autorange="reversed",
        showgrid=False
    )

    return fig


# Create the ROC curve
def create_roc_chart(actual, probabilities):
    fpr, tpr, _ = roc_curve(actual, probabilities)
    auc = roc_auc_score(actual, probabilities)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"Random Forest · AUC {auc:.3f}",
            line=dict(color=TEAL, width=3),
            fill="tozeroy",
            fillcolor="rgba(94,234,212,0.07)",
            hovertemplate=(
                "False positive rate: %{x:.2%}<br>"
                "True positive rate: %{y:.2%}"
                "<extra></extra>"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random baseline",
            line=dict(
                color=MUTED,
                width=1.5,
                dash="dash"
            ),
            hoverinfo="skip"
        )
    )

    style_chart(fig, height=340)

    fig.update_xaxes(
        title="False Positive Rate",
        range=[0, 1],
        tickformat=".0%"
    )

    fig.update_yaxes(
        title="True Positive Rate",
        range=[0, 1.03],
        tickformat=".0%"
    )

    return fig


# Create the precision-recall curve
def create_precision_recall_chart(actual, probabilities):
    precision, recall, _ = precision_recall_curve(
        actual,
        probabilities
    )

    average_precision = average_precision_score(
        actual,
        probabilities
    )

    baseline = float(np.mean(actual))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"Average Precision · {average_precision:.3f}",
            line=dict(color=LAVENDER, width=3),
            fill="tozeroy",
            fillcolor="rgba(196,181,253,0.07)",
            hovertemplate=(
                "Recall: %{x:.2%}<br>"
                "Precision: %{y:.2%}"
                "<extra></extra>"
            )
        )
    )

    fig.add_hline(
        y=baseline,
        line_dash="dash",
        line_color=MUTED,
        annotation_text="Positive-class baseline",
        annotation_font_color=MUTED
    )

    style_chart(fig, height=340)

    fig.update_xaxes(
        title="Recall",
        range=[0, 1],
        tickformat=".0%"
    )

    fig.update_yaxes(
        title="Precision",
        range=[0, 1.03],
        tickformat=".0%"
    )

    return fig


# Create the feature importance chart
def create_feature_importance_chart(data):
    chart_data = data.sort_values("Importance")

    fig = go.Figure(
        go.Bar(
            x=chart_data["Importance"],
            y=chart_data["Feature"],
            orientation="h",
            marker=dict(
                color=chart_data["Importance"],
                colorscale=[
                    [0, "#64748B"],
                    [0.5, LAVENDER],
                    [1, TEAL]
                ],
                cornerradius=8
            ),
            text=[
                f"{value:.2%}"
                for value in chart_data["Importance"]
            ],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{y}<br>"
                "Importance: %{x:.2%}"
                "<extra></extra>"
            )
        )
    )

    style_chart(fig, height=410)

    fig.update_xaxes(
        title="Relative Feature Importance",
        tickformat=".0%",
        range=[
            0,
            float(chart_data["Importance"].max()) * 1.25
        ]
    )

    fig.update_yaxes(showgrid=False)

    return fig


# Create a threshold performance comparison
def create_threshold_chart(actual, probabilities, threshold):
    thresholds = np.linspace(0, 1, 101)

    accuracy_values = []
    precision_values = []
    recall_values = []
    f1_values = []

    for value in thresholds:
        predictions = (probabilities >= value).astype(int)

        accuracy_values.append(
            accuracy_score(actual, predictions)
        )

        precision_values.append(
            precision_score(
                actual, predictions, zero_division=0
            )
        )

        recall_values.append(
            recall_score(
                actual, predictions, zero_division=0
            )
        )

        f1_values.append(
            f1_score(
                actual, predictions, zero_division=0
            )
        )

    fig = go.Figure()

    series = [
        ("Accuracy", accuracy_values, TEAL),
        ("Precision", precision_values, BLUE),
        ("Recall", recall_values, LAVENDER),
        ("F1 Score", f1_values, PINK)
    ]

    for name, values, color in series:
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=values,
                mode="lines",
                name=name,
                line=dict(color=color, width=2.5),
                hovertemplate=(
                    f"{name}: "
                    "%{y:.2%}<br>"
                    "Threshold: %{x:.0%}"
                    "<extra></extra>"
                )
            )
        )

    fig.add_vline(
        x=threshold,
        line_color=YELLOW,
        line_dash="dash",
        line_width=2
    )

    style_chart(fig, height=370)

    fig.update_layout(
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.12
        )
    )

    fig.update_xaxes(
        title="Decision Threshold",
        range=[0, 1],
        tickformat=".0%"
    )

    fig.update_yaxes(
        title="Performance Score",
        range=[0, 1.03],
        tickformat=".0%"
    )

    return fig


# Render the model overview
def render_overview(analytics, metrics):
    st.subheader("Model at a Glance")

    st.caption(
        "Evaluation performed on the reproduced held-out test split."
    )

    columns = st.columns(4)

    statistics = [
        (
            "Dataset Records",
            analytics["dataset_count"],
            TEAL,
            "Original dataset"
        ),
        (
            "Training Records",
            analytics["training_count"],
            BLUE,
            "80% training partition"
        ),
        (
            "Testing Records",
            analytics["testing_count"],
            LAVENDER,
            "20% testing partition"
        ),
        (
            "Decision Threshold",
            f'{analytics["threshold"]:.0%}',
            PINK,
            "Active model setting"
        )
    ]

    for column, statistic in zip(columns, statistics):
        with column:
            render_stat(*statistic)

    st.markdown("### Performance Signature")

    st.caption(
        "Five complementary measurements of classification performance."
    )

    performance = [
        ("Accuracy", metrics["accuracy"], TEAL),
        ("Precision", metrics["precision"], BLUE),
        ("Recall", metrics["recall"], LAVENDER),
        ("F1 Score", metrics["f1"], PINK),
        ("ROC-AUC", metrics["roc_auc"], YELLOW)
    ]

    with st.container(border=True):
        columns = st.columns(5)

        for index, (column, metric) in enumerate(
            zip(columns, performance)
        ):
            with column:
                render_performance_ring(
                    *metric,
                    key=f"analytics_ring_{index}"
                )

    render_note(
        "Understanding the Performance",
        (
            f'The model correctly identifies '
            f'{metrics["recall"]:.2%} of actual positive cases '
            f'at the selected decision threshold. '
            f'Its precision is {metrics["precision"]:.2%}, '
            f'which means some positive predictions are false positives. '
            f'These metrics describe test-set performance, '
            f'not clinical diagnostic reliability.'
        )
    )

    st.markdown("### How the Model Works")

    steps = [
        (
            "01",
            "Health Measurements",
            "Eight numerical measurements are provided to the pipeline."
        ),
        (
            "02",
            "Preprocessing",
            "Invalid zero measurements are handled using median imputation."
        ),
        (
            "03",
            "Random Forest",
            "Multiple decision trees contribute to the predicted probability."
        ),
        (
            "04",
            "Classification",
            "The probability is compared with the decision threshold."
        )
    ]

    columns = st.columns(4)

    for column, (number, title, description) in zip(
        columns, steps
    ):
        with column:
            st.html(f"""
                <div class="analytics-step">
                    <div class="analytics-step-number">
                        STEP {number}
                    </div>
                    <div class="analytics-step-title">
                        {title}
                    </div>
                    <div class="analytics-step-description">
                        {description}
                    </div>
                </div>
            """)


# Render the evaluation charts and classification report
def render_performance(analytics, metrics):
    st.subheader("Performance Studio")

    st.caption(
        "Explore correct predictions, classification errors "
        "and probability-based evaluation."
    )

    left, right = st.columns([1.2, 1])

    with left:
        with st.container(border=True):
            st.markdown("**Confusion Matrix**")

            display_chart(
                create_confusion_matrix(metrics["matrix"]),
                key="analytics_confusion_matrix"
            )

    with right:
        with st.container(border=True):
            st.markdown("**Prediction Breakdown**")

            render_stat(
                "True Positives",
                metrics["tp"],
                TEAL,
                "Correctly identified positive cases"
            )

            render_stat(
                "True Negatives",
                metrics["tn"],
                BLUE,
                "Correctly identified negative cases"
            )

            render_stat(
                "False Positives",
                metrics["fp"],
                YELLOW,
                "Negative cases predicted as positive"
            )

            render_stat(
                "False Negatives",
                metrics["fn"],
                PINK,
                "Positive cases predicted as negative"
            )

    st.markdown("### Evaluation Curves")

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown("**ROC Curve**")

            st.caption(
                "Measures the ability to distinguish "
                "between positive and negative cases."
            )

            display_chart(
                create_roc_chart(
                    analytics["actual"],
                    analytics["probabilities"]
                ),
                key="analytics_roc_curve"
            )

    with right:
        with st.container(border=True):
            st.markdown("**Precision-Recall Curve**")

            st.caption(
                "Shows the trade-off between identifying "
                "positive cases and prediction precision."
            )

            display_chart(
                create_precision_recall_chart(
                    analytics["actual"],
                    analytics["probabilities"]
                ),
                key="analytics_precision_recall"
            )

    st.markdown("### Classification Report")

    report = classification_report(
        analytics["actual"],
        metrics["predictions"],
        labels=[0, 1],
        target_names=["Negative", "Positive"],
        output_dict=True,
        zero_division=0
    )

    rows = []

    for label in [
        "Negative",
        "Positive",
        "macro avg",
        "weighted avg"
    ]:
        values = report[label]

        rows.append({
            "Classification": label.title(),
            "Precision": values["precision"],
            "Recall": values["recall"],
            "F1 Score": values["f1-score"],
            "Support": int(values["support"])
        })

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Precision": st.column_config.NumberColumn(
                format="%.4f"
            ),
            "Recall": st.column_config.NumberColumn(
                format="%.4f"
            ),
            "F1 Score": st.column_config.NumberColumn(
                format="%.4f"
            ),
            "Support": st.column_config.NumberColumn(
                format="%d"
            )
        }
    )

    render_note(
        "Reading the Classification Report",
        "Precision describes how often predictions of a class "
        "are correct. Recall describes how many actual members "
        "of that class are identified. F1 balances precision "
        "and recall, while support represents the number "
        "of actual records belonging to each class."
    )


# Render feature importance analysis
def render_features(analytics):
    st.subheader("Feature Intelligence")

    st.caption(
        "Discover which measurements contribute most "
        "to the Random Forest's decision-making."
    )

    importance = pd.DataFrame({
        "Feature": [
            FEATURE_LABELS[feature]
            for feature in FEATURE_COLUMNS
        ],
        "Importance": analytics["feature_importance"]
    }).sort_values(
        "Importance",
        ascending=False
    ).reset_index(drop=True)

    columns = st.columns(3)

    colors = [TEAL, LAVENDER, BLUE]

    for index, column in enumerate(columns):
        feature = importance.iloc[index]

        with column:
            render_stat(
                f"Rank #{index + 1}",
                f'{feature["Importance"]:.2%}',
                colors[index],
                feature["Feature"]
            )

    with st.container(border=True):
        st.markdown("**Feature Importance Ranking**")

        display_chart(
            create_feature_importance_chart(importance),
            key="analytics_feature_importance"
        )

    with st.expander("View detailed feature rankings"):
        ranking = importance.copy()

        ranking.insert(
            0,
            "Rank",
            range(1, len(ranking) + 1)
        )

        ranking["Importance (%)"] = (
            ranking["Importance"] * 100
        )

        ranking = ranking.drop(columns=["Importance"])

        st.dataframe(
            ranking,
            hide_index=True,
            width="stretch",
            column_config={
                "Importance (%)": st.column_config.NumberColumn(
                    format="%.2f%%"
                )
            }
        )

    render_note(
        "What Feature Importance Means",
        "These scores represent impurity-based feature importance "
        "calculated from the trained Random Forest. Higher scores "
        "indicate greater contribution to the model's tree splits. "
        "They do not establish causation, medical significance "
        "or the direction of a feature's effect on an individual "
        "prediction."
    )


# Reset the experimental threshold
def reset_threshold(default_threshold):
    st.session_state["analytics_threshold"] = round(
        default_threshold * 100
    )


# Render the interactive threshold laboratory
def render_threshold_lab(analytics, baseline):
    st.subheader("Threshold Laboratory")

    st.caption(
        "Experiment with classification decisions "
        "without changing the deployed model."
    )

    default_threshold = analytics["threshold"]

    if "analytics_threshold" not in st.session_state:
        st.session_state["analytics_threshold"] = round(
            default_threshold * 100
        )

    with st.container(border=True):
        left, right = st.columns([3, 1])

        with left:
            selected = st.slider(
                "Experimental Decision Threshold",
                min_value=0,
                max_value=100,
                step=1,
                format="%d%%",
                key="analytics_threshold"
            )

        with right:
            st.button(
                "Reset Threshold",
                icon=":material/restart_alt:",
                width="stretch",
                on_click=reset_threshold,
                args=(default_threshold,)
            )

        threshold = selected / 100

        st.caption(
            f"Deployed threshold: {default_threshold:.0%} · "
            f"Experimental threshold: {threshold:.0%}"
        )

    experimental = calculate_metrics(
        analytics["actual"],
        analytics["probabilities"],
        threshold
    )

    st.markdown("### Live Performance")

    columns = st.columns(4)

    metric_definitions = [
        ("Accuracy", "accuracy", TEAL),
        ("Precision", "precision", BLUE),
        ("Recall", "recall", LAVENDER),
        ("F1 Score", "f1", PINK)
    ]

    for column, (label, key, color) in zip(
        columns,
        metric_definitions
    ):
        difference = (
            experimental[key] - baseline[key]
        ) * 100

        with column:
            render_stat(
                label,
                f"{experimental[key]:.2%}",
                color,
                f"{difference:+.2f} pp vs. deployed threshold"
            )

    with st.container(border=True):
        st.markdown("**Performance Across Thresholds**")

        display_chart(
            create_threshold_chart(
                analytics["actual"],
                analytics["probabilities"],
                threshold
            ),
            key="analytics_threshold_curve"
        )

    st.markdown("### Classification Comparison")

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown(
                f"**Deployed Threshold · "
                f"{default_threshold:.0%}**"
            )

            display_chart(
                create_confusion_matrix(
                    baseline["matrix"]
                ),
                key="analytics_baseline_matrix"
            )

    with right:
        with st.container(border=True):
            st.markdown(
                f"**Experimental Threshold · "
                f"{threshold:.0%}**"
            )

            display_chart(
                create_confusion_matrix(
                    experimental["matrix"]
                ),
                key="analytics_experimental_matrix"
            )

    comparison = pd.DataFrame([
        {
            "Metric": label,
            "Deployed": f"{baseline[key]:.2%}",
            "Experimental": f"{experimental[key]:.2%}",
            "Change (pp)": (
                experimental[key] - baseline[key]
            ) * 100
        }
        for label, key, _ in metric_definitions
    ])

    with st.expander("View numerical comparison"):
        st.dataframe(
            comparison,
            hide_index=True,
            width="stretch",
            column_config={
                "Change (pp)": st.column_config.NumberColumn(
                    format="%+.2f"
                )
            }
        )

    render_note(
        "Understanding the Trade-off",
        "Lower thresholds generally identify more positive cases "
        "but may increase false positives. Higher thresholds "
        "generally produce fewer positive classifications "
        "but may miss actual positive cases. "
        "This laboratory is exploratory and must not be used "
        "to select a clinical decision threshold. "
        "No changes are made to the saved model or database."
    )


# Render the model documentation
def render_model_card(analytics):
    st.subheader("Model Card")

    st.caption(
        "Transparent documentation of the model's "
        "architecture, training configuration and limitations."
    )

    parameters = analytics["parameters"]

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown("**Model Identity**")

            identity = pd.DataFrame([
                {
                    "Property": "Algorithm",
                    "Value": analytics["config"]["model_name"]
                },
                {
                    "Property": "Task",
                    "Value": "Binary Classification"
                },
                {
                    "Property": "Model Version",
                    "Value": "v1.0"
                },
                {
                    "Property": "Input Features",
                    "Value": str(len(FEATURE_COLUMNS))
                },
                {
                    "Property": "Positive Class",
                    "Value": "1 — Positive"
                },
                {
                    "Property": "Decision Threshold",
                    "Value": f'{analytics["threshold"]:.0%}'
                }
            ])

            st.dataframe(
                identity,
                hide_index=True,
                width="stretch"
            )

    with right:
        with st.container(border=True):
            st.markdown("**Hyperparameters**")

            parameter_labels = {
                "n_estimators": "Number of Trees",
                "max_depth": "Maximum Depth",
                "min_samples_split": "Minimum Samples to Split",
                "min_samples_leaf": "Minimum Samples per Leaf",
                "class_weight": "Class Weight",
                "random_state": "Random State",
                "criterion": "Split Criterion",
                "max_features": "Maximum Features"
            }

            parameter_table = pd.DataFrame([
                {
                    "Parameter": label,
                    "Value": str(parameters.get(key))
                }
                for key, label in parameter_labels.items()
            ])

            st.dataframe(
                parameter_table,
                hide_index=True,
                width="stretch"
            )

    st.markdown("### Evaluation Methodology")

    with st.container(border=True):
        st.markdown("""
- **Dataset:** Pima Indians Diabetes dataset.
- **Training partition:** 80% of the original dataset.
- **Testing partition:** 20% of the original dataset.
- **Split strategy:** Stratified by the outcome variable.
- **Random state:** 42.
- **Preprocessing:** Invalid-zero replacement and median imputation.
- **Feature scaling:** Not applied to the Random Forest pipeline.
- **Class balancing:** Balanced class weights.
- **Evaluation:** Predictions generated from the existing saved pipeline.
        """)

    st.markdown("### Model Limitations")

    render_note(
        "Dataset Representation",
        "The dataset represents a specific study population. "
        "Performance may not generalize to other populations, "
        "demographic groups or clinical environments."
    )

    render_note(
        "Probability Interpretation",
        "Predicted probabilities are model outputs and have "
        "not been established as clinically calibrated "
        "estimates of diabetes risk."
    )

    render_note(
        "Evaluation Limitations",
        "The reported results use a reproduced test partition. "
        "They are not evidence of independent external "
        "validation or clinical effectiveness. "
        "Repeated experimentation on this partition should "
        "not be treated as independent model validation."
    )

    render_note(
        "Educational Use Only",
        "UODA DiaML is an educational machine learning project. "
        "It is not a medical diagnostic system and must not "
        "be used to make clinical decisions."
    )


# Initialize the Model Analytics page
apply_analytics_styles()

try:
    with st.spinner("Loading model intelligence..."):
        analytics = load_model_analytics(
            MODEL_PATH.stat().st_mtime_ns,
            CONFIG_PATH.stat().st_mtime_ns,
            DATA_PATH.stat().st_mtime_ns
        )

except Exception as error:
    st.error("Unable to load model analytics.")

    with st.expander("Technical details"):
        st.code(str(error))

    st.stop()


baseline_metrics = calculate_metrics(
    analytics["actual"],
    analytics["probabilities"],
    analytics["threshold"]
)

render_hero(analytics)

sections = [
    "Overview",
    "Performance",
    "Features",
    "Threshold Lab",
    "Model Card"
]

section = st.segmented_control(
    "Analytics Navigation",
    options=sections,
    default="Overview",
    key="model_analytics_navigation",
    selection_mode="single",
    label_visibility="collapsed",
    width="stretch"
)

st.divider()

if section == "Performance":
    render_performance(
        analytics,
        baseline_metrics
    )

elif section == "Features":
    render_features(analytics)

elif section == "Threshold Lab":
    render_threshold_lab(
        analytics,
        baseline_metrics
    )

elif section == "Model Card":
    render_model_card(analytics)

else:
    render_overview(
        analytics,
        baseline_metrics
    )

st.divider()

st.caption(
    "UODA DiaML · Model Analytics · "
    "Educational machine learning analytics. "
    "Not intended for medical diagnosis."
)
