
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)


# Generate predictions using the selected classification threshold
def get_predictions(model, X, threshold=0.40):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return predictions, probabilities


# Calculate the main classification performance metrics
def evaluate_model(model, X, y, threshold=0.40):
    predictions, probabilities = get_predictions(model, X, threshold)

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1_score": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities))
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
