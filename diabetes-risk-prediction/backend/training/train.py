"""
Training script for Diabetes Risk Prediction models.

This script:
1. Loads and analyzes the diabetes dataset
2. Builds preprocessing pipelines
3. Trains Logistic Regression and Random Forest models
4. Evaluates both models using Stratified 5-Fold Cross-Validation
5. Generates comparison metrics, visualizations, and reports
6. Saves trained pipelines for production inference
7. Generates metadata for the backend API

Usage:
    python train.py

All outputs are saved to:
    - backend/models/ (serialized models and metadata)
    - ml/visualizations/ (charts and plots)
    - ml/reports/ (evaluation report)
"""

import os
import sys
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.preprocessing import (
    build_preprocessor,
    load_dataset,
    get_feature_names,
    ALL_FEATURES,
    TARGET_COLUMN,
    BINARY_FEATURES,
    GENDER_FEATURE,
    NUMERICAL_FEATURES,
    RANDOM_SEED,
)
from training.evaluate import (
    evaluate_model_cv,
    get_feature_importance,
    format_metrics_table,
)

warnings.filterwarnings("ignore")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes_data_upload.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
VIZ_DIR = os.path.join(os.path.dirname(BASE_DIR), "ml", "visualizations")
REPORTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "ml", "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def analyze_dataset(df, X, y):
    """Perform comprehensive dataset analysis and print report."""
    print("=" * 70)
    print("DATASET ANALYSIS REPORT")
    print("=" * 70)

    print(f"\n1. BASIC INFORMATION")
    print(f"   Number of rows: {len(df)}")
    print(f"   Number of columns: {len(df.columns)}")
    print(f"   Column names: {list(df.columns)}")

    print(f"\n2. DATA TYPES")
    for col in df.columns:
        print(f"   {col}: {df[col].dtype}")

    print(f"\n3. MISSING VALUES")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   No missing values found.")
    else:
        for col in df.columns:
            if missing[col] > 0:
                print(f"   {col}: {missing[col]} ({missing[col]/len(df)*100:.1f}%)")

    print(f"\n4. DUPLICATE ROWS")
    dupes = df.duplicated().sum()
    print(f"   Number of duplicate rows: {dupes}")

    print(f"\n5. UNIQUE VALUES PER COLUMN")
    for col in df.columns:
        unique = df[col].unique()
        print(f"   {col}: {len(unique)} unique values -> {sorted(unique)[:10]}{'...' if len(unique) > 10 else ''}")

    print(f"\n6. FEATURE CLASSIFICATION")
    print(f"   Numerical features: {NUMERICAL_FEATURES}")
    print(f"   Gender feature: {GENDER_FEATURE}")
    print(f"   Binary (Yes/No) features: {BINARY_FEATURES}")
    print(f"   Target column: {TARGET_COLUMN}")

    print(f"\n7. TARGET DISTRIBUTION")
    target_counts = df[TARGET_COLUMN].value_counts()
    for label, count in target_counts.items():
        print(f"   {label}: {count} ({count/len(df)*100:.1f}%)")
    
    ratio = target_counts.max() / target_counts.min()
    print(f"   Class ratio: {ratio:.2f}:1")
    if ratio > 1.5:
        print("   Note: Moderate class imbalance detected.")
    else:
        print("   Note: Classes are relatively balanced.")

    print(f"\n8. AGE STATISTICS")
    print(f"   Min: {df['Age'].min()}")
    print(f"   Max: {df['Age'].max()}")
    print(f"   Mean: {df['Age'].mean():.1f}")
    print(f"   Median: {df['Age'].median():.1f}")
    print(f"   Std: {df['Age'].std():.1f}")

    print(f"\n9. GENDER DISTRIBUTION")
    gender_counts = df["Gender"].value_counts()
    for label, count in gender_counts.items():
        print(f"   {label}: {count} ({count/len(df)*100:.1f}%)")

    print("=" * 70)

    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_values": missing.to_dict(),
        "duplicates": int(dupes),
        "target_distribution": target_counts.to_dict(),
        "age_stats": {
            "min": int(df["Age"].min()),
            "max": int(df["Age"].max()),
            "mean": float(df["Age"].mean()),
            "median": float(df["Age"].median()),
            "std": float(df["Age"].std()),
        },
        "gender_distribution": gender_counts.to_dict(),
    }


def generate_visualizations(df, lr_results, rf_results, feature_importance):
    """Generate all visualization charts."""
    sns.set_style("whitegrid")
    
    # 1. Target distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71", "#e74c3c"]
    target_counts = df[TARGET_COLUMN].value_counts()
    bars = ax.bar(target_counts.index, target_counts.values, color=colors, edgecolor="white", linewidth=1.5)
    for bar, count in zip(bars, target_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3,
                str(count), ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax.set_title("Target Distribution (class)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "target_distribution.png"), dpi=150)
    plt.close()

    # 2. Age distribution by class
    fig, ax = plt.subplots(figsize=(10, 5))
    for cls, color in zip(["Positive", "Negative"], ["#e74c3c", "#2ecc71"]):
        subset = df[df[TARGET_COLUMN] == cls]
        ax.hist(subset["Age"], bins=20, alpha=0.6, label=cls, color=color, edgecolor="white")
    ax.set_title("Age Distribution by Class", fontsize=14, fontweight="bold")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "age_distribution.png"), dpi=150)
    plt.close()

    # 3. Symptom prevalence
    fig, ax = plt.subplots(figsize=(12, 6))
    symptom_yes = {}
    for feat in BINARY_FEATURES:
        symptom_yes[feat] = (df[feat] == "Yes").sum()
    sorted_symptoms = sorted(symptom_yes.items(), key=lambda x: x[1], reverse=True)
    names = [s[0] for s in sorted_symptoms]
    values = [s[1] for s in sorted_symptoms]
    bars = ax.barh(names, values, color="#3498db", edgecolor="white")
    ax.set_title("Symptom Prevalence in Dataset", fontsize=14, fontweight="bold")
    ax.set_xlabel("Count of 'Yes'")
    ax.invert_yaxis()
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "symptom_prevalence.png"), dpi=150)
    plt.close()

    # 4. Model comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    x = np.arange(len(metrics))
    width = 0.35

    lr_means = [lr_results["metrics"][m]["mean"] for m in metrics]
    lr_stds = [lr_results["metrics"][m]["std"] for m in metrics]
    rf_means = [rf_results["metrics"][m]["mean"] for m in metrics]
    rf_stds = [rf_results["metrics"][m]["std"] for m in metrics]

    bars1 = ax.bar(x - width/2, lr_means, width, yerr=lr_stds, label="Logistic Regression",
                   color="#3498db", capsize=5, edgecolor="white")
    bars2 = ax.bar(x + width/2, rf_means, width, yerr=rf_stds, label="Random Forest",
                   color="#e67e22", capsize=5, edgecolor="white")

    ax.set_ylim(0.5, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title("Model Comparison: Cross-Validation Metrics", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.legend(fontsize=11)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "model_comparison.png"), dpi=150)
    plt.close()

    # 5. ROC curves
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(lr_results["roc_curve"]["fpr"], lr_results["roc_curve"]["tpr"],
            label=f'Logistic Regression (AUC={lr_results["metrics"]["roc_auc"]["mean"]:.3f})',
            color="#3498db", linewidth=2)
    ax.plot(rf_results["roc_curve"]["fpr"], rf_results["roc_curve"]["tpr"],
            label=f'Random Forest (AUC={rf_results["metrics"]["roc_auc"]["mean"]:.3f})',
            color="#e67e22", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "roc_curves.png"), dpi=150)
    plt.close()

    # 6. Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, results, title, cmap in [
        (axes[0], lr_results, "Logistic Regression", "Blues"),
        (axes[1], rf_results, "Random Forest", "Oranges"),
    ]:
        cm = np.array(results["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                    xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"],
                    cbar=False, annot_kws={"size": 16})
        ax.set_title(f"Confusion Matrix: {title}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "confusion_matrices.png"), dpi=150)
    plt.close()

    # 7. Feature importance chart
    if feature_importance:
        fig, ax = plt.subplots(figsize=(10, 7))
        names = list(feature_importance.keys())
        values = list(feature_importance.values())
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(names)))
        bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], edgecolor="white")
        ax.set_title("Random Forest: Feature Importance", fontsize=14, fontweight="bold")
        ax.set_xlabel("Importance Score")
        for bar, val in zip(bars, values[::-1]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, "feature_importance.png"), dpi=150)
        plt.close()

    # 8. CV score distribution (box plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_for_box = ["accuracy", "f1_score", "roc_auc"]
    labels_for_box = ["Accuracy", "F1 Score", "ROC-AUC"]
    data_lr = [lr_results["fold_metrics"][m] for m in metrics_for_box]
    data_rf = [rf_results["fold_metrics"][m] for m in metrics_for_box]

    positions_lr = [1, 4, 7]
    positions_rf = [2, 5, 8]

    bp1 = ax.boxplot(data_lr, positions=positions_lr, widths=0.6,
                     patch_artist=True, boxprops=dict(facecolor="#3498db", alpha=0.6))
    bp2 = ax.boxplot(data_rf, positions=positions_rf, widths=0.6,
                     patch_artist=True, boxprops=dict(facecolor="#e67e22", alpha=0.6))

    ax.set_xticks([1.5, 4.5, 7.5])
    ax.set_xticklabels(labels_for_box, fontsize=11)
    ax.set_title("Cross-Validation Score Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]],
              ["Logistic Regression", "Random Forest"], fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "cv_score_distribution.png"), dpi=150)
    plt.close()

    print(f"\n   Visualizations saved to: {VIZ_DIR}")


def generate_report(dataset_info, lr_results, rf_results, feature_importance, selected_model):
    """Generate the academic evaluation report."""
    report = f"""# Model Evaluation Report

## Diabetes Risk Prediction Using Logistic Regression and Random Forest with Cross-Validation

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Dataset Description

| Property | Value |
|----------|-------|
| Total Samples | {dataset_info['n_rows']} |
| Total Features | {dataset_info['n_columns'] - 1} (excluding target) |
| Numerical Features | {len(NUMERICAL_FEATURES)} (Age) |
| Binary Features | {len(BINARY_FEATURES)} (Yes/No symptoms) |
| Gender Feature | 1 (Male/Female) |
| Target Column | `class` (Positive/Negative) |
| Missing Values | None |
| Duplicate Rows | {dataset_info['duplicates']} |

### Target Distribution

| Class | Count | Percentage |
|-------|-------|------------|
"""
    for cls, count in dataset_info['target_distribution'].items():
        pct = count / dataset_info['n_rows'] * 100
        report += f"| {cls} | {count} | {pct:.1f}% |\n"

    report += f"""
### Age Statistics

| Statistic | Value |
|-----------|-------|
| Min | {dataset_info['age_stats']['min']} |
| Max | {dataset_info['age_stats']['max']} |
| Mean | {dataset_info['age_stats']['mean']:.1f} |
| Median | {dataset_info['age_stats']['median']:.1f} |
| Std Dev | {dataset_info['age_stats']['std']:.1f} |

## 2. Preprocessing Methodology

The preprocessing pipeline was built using scikit-learn's `ColumnTransformer` to ensure
consistent transformations and prevent data leakage:

1. **Age (Numerical):** Standardized using `StandardScaler` — fitted only on each training fold.
2. **Gender:** Encoded as binary (Male=1, Female=0).
3. **Symptom Features (9 binary):** Encoded as binary (Yes=1, No=0).

**Data Leakage Prevention:** The entire preprocessing pipeline is included in the sklearn
`Pipeline`, so the StandardScaler is fit only on training data during each cross-validation fold.

## 3. Model Methodology

### Model 1: Logistic Regression

- **Solver:** `lbfgs` (efficient for small-to-medium datasets)
- **Regularization:** `C=1.0` (default L2 regularization)
- **Max Iterations:** 1000 (ensures convergence)
- **Class Weight:** `balanced` (compensates for any class imbalance)
- **Random State:** 42 (reproducibility)

**Justification:** Logistic Regression serves as a strong linear baseline.
L2 regularization prevents overfitting, and balanced class weights address
the moderate class imbalance in the dataset.

### Model 2: Random Forest

- **Number of Trees:** 200 (sufficient for stable predictions)
- **Max Depth:** 10 (limits tree depth to prevent overfitting)
- **Min Samples Split:** 5 (prevents splits on very small groups)
- **Min Samples Leaf:** 2 (ensures leaf nodes have sufficient samples)
- **Class Weight:** `balanced` (compensates for class imbalance)
- **Random State:** 42 (reproducibility)

**Justification:** Random Forest captures non-linear feature interactions
while the depth and sample constraints prevent overfitting on this
moderately-sized dataset.

## 4. Cross-Validation Methodology

- **Strategy:** Stratified 5-Fold Cross-Validation
- **Stratification:** Maintains the target class distribution in each fold
- **Implementation:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`

The entire pipeline (preprocessing + model) is fit on each training fold independently,
ensuring that no information from the validation fold leaks into the training process.

## 5. Results

### Model Comparison

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
"""

    for metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        label = metric.replace("_", " ").title()
        lr_m = lr_results["metrics"][metric]
        rf_m = rf_results["metrics"][metric]
        report += f"| {label} | {lr_m['mean']:.4f} (±{lr_m['std']:.4f}) | {rf_m['mean']:.4f} (±{rf_m['std']:.4f}) |\n"

    report += f"""
### Confusion Matrices

**Logistic Regression:**

|  | Predicted Negative | Predicted Positive |
|--|-------------------|--------------------|
| Actual Negative | {lr_results['confusion_matrix'][0][0]} | {lr_results['confusion_matrix'][0][1]} |
| Actual Positive | {lr_results['confusion_matrix'][1][0]} | {lr_results['confusion_matrix'][1][1]} |

**Random Forest:**

|  | Predicted Negative | Predicted Positive |
|--|-------------------|--------------------|
| Actual Negative | {rf_results['confusion_matrix'][0][0]} | {rf_results['confusion_matrix'][0][1]} |
| Actual Positive | {rf_results['confusion_matrix'][1][0]} | {rf_results['confusion_matrix'][1][1]} |

### Feature Importance (Random Forest)

| Rank | Feature | Importance |
|------|---------|------------|
"""
    for rank, (feat, imp) in enumerate(feature_importance.items(), 1):
        report += f"| {rank} | {feat} | {imp:.4f} |\n"

    report += f"""
## 6. Model Selection

**Selected Model:** {selected_model}

The model was selected based on the highest mean F1 Score across 5-fold cross-validation.
F1 Score was chosen as the selection criterion because it balances precision and recall,
which is important in a medical risk prediction context where both false positives
and false negatives have significant implications.

## 7. Limitations

1. **Dataset Size:** {dataset_info['n_rows']} samples is relatively small for a production ML system.
2. **Feature Set:** Only {len(ALL_FEATURES)} features are available; clinical datasets typically include
   blood glucose levels, HbA1c, BMI, blood pressure, and other biomarkers.
3. **Binary Features:** Most features are binary (Yes/No), limiting the granularity of predictions.
4. **Generalizability:** The model was trained on a specific dataset and may not generalize to
   different populations or clinical settings.
5. **Not a Clinical Tool:** This is an ML-based risk assessment, NOT a medical diagnosis.

## 8. Conclusion

Both models achieve strong cross-validation performance on this dataset. The {selected_model}
was selected as the primary prediction model based on its F1 Score. However, both models
are retained in the system for comparison purposes.

The moderate class imbalance was addressed through balanced class weights in both models.
The Stratified K-Fold cross-validation ensures reliable performance estimates that account
for the class distribution.

---

> **Disclaimer:** This system provides ML-based risk predictions and is NOT a substitute
> for professional medical diagnosis. Always consult a qualified healthcare professional
> for medical advice.
"""

    report_path = os.path.join(REPORTS_DIR, "model_evaluation.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n   Report saved to: {report_path}")


def main():
    """Main training pipeline."""
    print("\n" + "=" * 70)
    print("  DIABETES RISK PREDICTION — MODEL TRAINING PIPELINE")
    print("=" * 70)

    # ── Step 1: Load and analyze dataset ──────────────────────────────────
    print("\n[STEP 1] Loading dataset...")
    X, y, df = load_dataset(DATA_PATH)
    print(f"   Loaded {len(X)} samples with {len(X.columns)} features")

    print("\n[STEP 2] Analyzing dataset...")
    dataset_info = analyze_dataset(df, X, y)

    # ── Step 2: Build pipelines ───────────────────────────────────────────
    print("\n[STEP 3] Building model pipelines...")

    # Logistic Regression pipeline
    lr_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(
            solver="lbfgs",
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )),
    ])

    # Random Forest pipeline
    rf_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])

    # ── Step 3: Cross-validation evaluation ───────────────────────────────
    print("\n[STEP 4] Evaluating Logistic Regression (5-Fold CV)...")
    lr_results = evaluate_model_cv(lr_pipeline, X, y, "Logistic Regression")

    print("\n[STEP 5] Evaluating Random Forest (5-Fold CV)...")
    rf_results = evaluate_model_cv(rf_pipeline, X, y, "Random Forest")

    # ── Step 4: Print comparison table ────────────────────────────────────
    print("\n[STEP 6] Model Comparison")
    print(format_metrics_table([lr_results, rf_results]))

    # ── Step 5: Select best model based on F1 Score ──────────────────────
    lr_f1 = lr_results["metrics"]["f1_score"]["mean"]
    rf_f1 = rf_results["metrics"]["f1_score"]["mean"]

    if rf_f1 >= lr_f1:
        selected_model = "Random Forest"
        print(f"\n   Selected model: Random Forest (F1={rf_f1:.4f} >= LR F1={lr_f1:.4f})")
    else:
        selected_model = "Logistic Regression"
        print(f"\n   Selected model: Logistic Regression (F1={lr_f1:.4f} > RF F1={rf_f1:.4f})")

    # ── Step 6: Train final models on full dataset ────────────────────────
    print("\n[STEP 7] Training final models on full dataset...")

    # Rebuild fresh pipelines for final training
    lr_final = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(
            solver="lbfgs", C=1.0, max_iter=1000,
            class_weight="balanced", random_state=RANDOM_SEED,
        )),
    ])

    rf_final = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1,
        )),
    ])

    lr_final.fit(X, y)
    rf_final.fit(X, y)
    print("   Both models trained on full dataset.")

    # ── Step 7: Extract feature importance ────────────────────────────────
    print("\n[STEP 8] Extracting feature importance...")
    feature_names = get_feature_names()
    feature_importance = get_feature_importance(rf_final, feature_names)
    print("   Top features:")
    for feat, imp in list(feature_importance.items())[:5]:
        bar = "#" * int(imp * 50)
        print(f"   {feat:<25} {bar} ({imp:.4f})")

    # ── Step 8: Save models ───────────────────────────────────────────────
    print("\n[STEP 9] Saving trained models...")
    lr_path = os.path.join(MODELS_DIR, "logistic_regression.pkl")
    rf_path = os.path.join(MODELS_DIR, "random_forest.pkl")

    joblib.dump(lr_final, lr_path)
    joblib.dump(rf_final, rf_path)
    print(f"   Logistic Regression saved: {lr_path}")
    print(f"   Random Forest saved: {rf_path}")

    # ── Step 9: Save metadata ─────────────────────────────────────────────
    print("\n[STEP 10] Saving metadata...")
    metadata = {
        "project": "Diabetes Risk Prediction",
        "version": "1.0.0",
        "training_date": datetime.now().isoformat(),
        "dataset": {
            "name": "diabetes_data_upload.csv",
            "n_samples": int(len(X)),
            "n_features": len(ALL_FEATURES),
            "features": ALL_FEATURES,
            "target": TARGET_COLUMN,
            "target_labels": ["Negative", "Positive"],
        },
        "cross_validation": {
            "method": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": RANDOM_SEED,
        },
        "models": {
            "logistic_regression": {
                "name": "Logistic Regression",
                "file": "logistic_regression.pkl",
                "metrics": {k: v for k, v in lr_results["metrics"].items()},
                "confusion_matrix": lr_results["confusion_matrix"],
                "roc_curve": lr_results["roc_curve"],
                "classification_report": lr_results["classification_report"],
            },
            "random_forest": {
                "name": "Random Forest",
                "file": "random_forest.pkl",
                "metrics": {k: v for k, v in rf_results["metrics"].items()},
                "confusion_matrix": rf_results["confusion_matrix"],
                "roc_curve": rf_results["roc_curve"],
                "feature_importance": feature_importance,
                "classification_report": rf_results["classification_report"],
            },
        },
        "selected_model": selected_model,
        "selection_criterion": "Highest mean F1 Score across 5-fold CV",
        "risk_thresholds": {
            "low": {"min": 0.0, "max": 0.35},
            "moderate": {"min": 0.35, "max": 0.65},
            "high": {"min": 0.65, "max": 1.0},
        },
        "disclaimer": "Risk level is an ML application classification and is not a clinical diagnosis.",
    }

    metadata_path = os.path.join(MODELS_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"   Metadata saved: {metadata_path}")

    # ── Step 10: Generate visualizations ──────────────────────────────────
    print("\n[STEP 11] Generating visualizations...")
    generate_visualizations(df, lr_results, rf_results, feature_importance)

    # ── Step 11: Generate report ──────────────────────────────────────────
    print("\n[STEP 12] Generating evaluation report...")
    generate_report(dataset_info, lr_results, rf_results, feature_importance, selected_model)

    # ── Final Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE — SUMMARY")
    print("=" * 70)
    print(f"\n  Selected Model: {selected_model}")
    print(f"  LR F1 Score: {lr_f1:.4f}")
    print(f"  RF F1 Score: {rf_f1:.4f}")
    print(f"  Models saved to: {MODELS_DIR}")
    print(f"  Visualizations saved to: {VIZ_DIR}")
    print(f"  Report saved to: {REPORTS_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
