# ================================================
# evaluation.py
# ================================================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, confusion_matrix, classification_report,
    RocCurveDisplay, PrecisionRecallDisplay
)
import numpy as np
from pathlib import Path


def evaluate_model(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    roc = roc_auc_score(y_test, proba)
    pr = average_precision_score(y_test, proba)
    acc = accuracy_score(y_test, preds)

    print(f"ROC-AUC: {roc:.3f}")
    print(f"PR-AUC: {pr:.3f}")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, preds, digits=3))

    return proba, preds


def save_confusion_matrix(y_test, preds, outdir):
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    plt.title("Confusion Matrix — Tuned XGBoost")
    plt.savefig(outdir / "confusion_matrix.png", dpi=300)
    plt.close()

    cm_percent = cm.astype(float) / cm.sum(axis=1)[:, None]
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    plt.title("Confusion Matrix (Percentages)")
    plt.savefig(outdir / "confusion_matrix_percent.png", dpi=300)
    plt.close()


def save_roc_pr_curves(y_test, proba, outdir):
    fig, ax = plt.subplots(figsize=(7,6))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax)
    plt.title("ROC Curve — Tuned XGBoost")
    plt.savefig(outdir / "roc_curve.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(7,6))
    PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax)
    plt.title("PR Curve — Tuned XGBoost")
    plt.savefig(outdir / "pr_curve.png", dpi=300)
    plt.close()