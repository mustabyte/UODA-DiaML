import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    balanced_accuracy_score,
    precision_recall_curve,
    average_precision_score
)


# Validate the classification threshold
def validate_threshold(threshold):
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        raise ValueError("Threshold must be numeric.")

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    return threshold


# Generate predictions using the selected classification threshold
def get_predictions(model, X, threshold=0.40):
    threshold = validate_threshold(threshold)

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return predictions, probabilities


# Calculate specificity from the confusion matrix
def calculate_specificity(y, predictions):
    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    ).ravel()

    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


# Calculate classification performance metrics
def evaluate_model(model, X, y, threshold=0.40):
    predictions, probabilities = get_predictions(
        model,
        X,
        threshold
    )

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "specificity": calculate_specificity(y, predictions),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, predictions)
        ),
        "average_precision": float(
            average_precision_score(y, probabilities)
        )
    }


# Calculate the confusion matrix for visualization
def get_confusion_matrix(model, X, y, threshold=0.40):
    predictions, _ = get_predictions(model, X, threshold)

    return confusion_matrix(y, predictions, labels=[0, 1])


# Generate ROC curve coordinates for plotting
def get_roc_curve(model, X, y):
    _, probabilities = get_predictions(model, X)

    fpr, tpr, thresholds = roc_curve(y, probabilities)

    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist()
    }


# Generate Precision-Recall curve coordinates
def get_precision_recall_curve(model, X, y):
    _, probabilities = get_predictions(model, X)

    precision, recall, thresholds = precision_recall_curve(
        y,
        probabilities
    )

    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
        "average_precision": float(
            average_precision_score(y, probabilities)
        )
    }