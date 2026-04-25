"""
train_model.py
==============
Trains a Random Forest Classifier on the hospital readmission dataset.
Saves the trained model + scaler + encoders as a single joblib bundle.

Usage:
    python train_model.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless – no GUI needed
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, ConfusionMatrixDisplay,
)

# Project modules
from data_loader import generate_synthetic_data, preprocess_and_smote

SEED = 42
MODEL_PATH = "readmission_model.joblib"
METRICS_PATH = "metrics.json"
PLOTS_DIR = "static/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD & PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    csv_path = "patient_data.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"[train] Loaded existing dataset → {df.shape}")
    else:
        df = generate_synthetic_data(2000)
        df.to_csv(csv_path, index=False)
        print(f"[train] Generated new dataset → {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train():
    df = load_data()
    X, y, encoders, feature_cols = preprocess_and_smote(df)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / test split (80 / 20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=SEED, stratify=y
    )
    print(f"[train] Train: {X_train.shape}  |  Test: {X_test.shape}\n")

    # ── Random Forest ────────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    )
    rf.fit(X_train, y_train)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    print("=" * 50)
    print("  MODEL EVALUATION — Random Forest")
    print("=" * 50)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print("=" * 50)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["No Readmit", "Readmit"]))

    # ── Cross-validation ─────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring="f1")
    print(f"[train] 5-Fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_confusion_matrix(y_test, y_pred)
    _plot_roc_curve(y_test, y_prob, auc)
    _plot_feature_importance(rf, feature_cols)

    # ── Save metrics ─────────────────────────────────────────────────────────
    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "cv_f1_mean": round(float(cv_scores.mean()), 4),
        "cv_f1_std": round(float(cv_scores.std()), 4),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] Metrics saved → {METRICS_PATH}")

    # ── Save model bundle ────────────────────────────────────────────────────
    bundle = {
        "model": rf,
        "scaler": scaler,
        "encoders": encoders,
        "feature_cols": feature_cols,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"[train] Model bundle saved → {MODEL_PATH}")

    return bundle, metrics


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Readmit", "Readmit"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[train] Plot saved → {path}")


def _plot_roc_curve(y_true, y_prob, auc_score):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"AUC = {auc_score:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/roc_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[train] Plot saved → {path}")


def _plot_feature_importance(model, feature_cols):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2563eb"] * len(feature_cols)
    ax.bar(range(len(feature_cols)), importances[indices], color=colors, edgecolor="white")
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels([feature_cols[i] for i in indices], rotation=40, ha="right", fontsize=9)
    ax.set_title("Feature Importances", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/feature_importance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[train] Plot saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
