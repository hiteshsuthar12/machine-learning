"""
Evaluation module for model assessment.

Provides functions for cross-validation evaluation, metric computation,
confusion matrix generation, ROC curve data, and visualization generation.
"""

import numpy as np
import json
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)


RANDOM_SEED = 42


def get_cv_strategy():
    """Return the cross-validation strategy.

    Uses StratifiedKFold to maintain class distribution across folds.
    """
    return StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED,
    )


def evaluate_model_cv(pipeline, X, y, model_name="Model"):
    """Evaluate a model pipeline using Stratified 5-Fold Cross-Validation.

    This function performs cross-validation where the entire pipeline
    (preprocessing + model) is fit on each training fold, ensuring
    no data leakage from validation data into preprocessing.

    Args:
        pipeline: sklearn Pipeline with preprocessor and estimator.
        X: Feature DataFrame.
        y: Encoded target array.
        model_name: Name for reporting.

    Returns:
        Dictionary with all evaluation metrics and data for visualization.
    """
    cv = get_cv_strategy()

    # Collect per-fold metrics
    fold_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "roc_auc": [],
    }

    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fit entire pipeline on training fold only
        pipeline.fit(X_train, y_train)

        # Predict on validation fold
        y_pred = pipeline.predict(X_val)

        # Get probability if available
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_val)[:, 1]
        else:
            y_proba = pipeline.decision_function(X_val)

        # Compute fold metrics
        fold_metrics["accuracy"].append(accuracy_score(y_val, y_pred))
        fold_metrics["precision"].append(
            precision_score(y_val, y_pred, zero_division=0)
        )
        fold_metrics["recall"].append(
            recall_score(y_val, y_pred, zero_division=0)
        )
        fold_metrics["f1_score"].append(
            f1_score(y_val, y_pred, zero_division=0)
        )
        fold_metrics["roc_auc"].append(roc_auc_score(y_val, y_proba))

        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_proba.extend(y_proba.tolist())

        print(f"  Fold {fold_idx + 1}: Accuracy={fold_metrics['accuracy'][-1]:.4f}, "
              f"F1={fold_metrics['f1_score'][-1]:.4f}, "
              f"ROC-AUC={fold_metrics['roc_auc'][-1]:.4f}")

    # Compute aggregate metrics
    results = {
        "model_name": model_name,
        "cv_folds": 5,
        "metrics": {},
        "fold_metrics": fold_metrics,
    }

    for metric_name, values in fold_metrics.items():
        results["metrics"][metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "per_fold": [float(v) for v in values],
        }

    # Compute overall confusion matrix from all folds
    cm = confusion_matrix(all_y_true, all_y_pred)
    results["confusion_matrix"] = cm.tolist()

    # Compute ROC curve data from all folds
    fpr, tpr, thresholds = roc_curve(all_y_true, all_y_proba)
    results["roc_curve"] = {
        "fpr": [float(x) for x in fpr],
        "tpr": [float(x) for x in tpr],
    }

    # Classification report
    report = classification_report(
        all_y_true, all_y_pred,
        target_names=["Negative", "Positive"],
        output_dict=True,
    )
    results["classification_report"] = report

    return results


def get_feature_importance(pipeline, feature_names):
    """Extract feature importance from a trained Random Forest pipeline.

    Args:
        pipeline: Trained sklearn Pipeline with a RandomForest estimator.
        feature_names: List of feature names matching the preprocessor output.

    Returns:
        Dictionary mapping feature names to importance scores, sorted
        in descending order of importance.
    """
    # The model is the last step in the pipeline
    model = pipeline.named_steps.get("classifier") or pipeline.steps[-1][1]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_dict = {}
        for name, imp in zip(feature_names, importances):
            importance_dict[name] = float(imp)

        # Sort by importance descending
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        return importance_dict

    return {}


def format_metrics_table(results_list):
    """Format a comparison table of multiple model results.

    Args:
        results_list: List of result dictionaries from evaluate_model_cv.

    Returns:
        Formatted string table for display.
    """
    header = f"{'Metric':<20}"
    for r in results_list:
        header += f"{'  ' + r['model_name']:<30}"
    
    lines = [header, "-" * len(header)]

    metric_labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1 Score",
        "roc_auc": "ROC-AUC",
    }

    for metric_key, label in metric_labels.items():
        row = f"{label:<20}"
        for r in results_list:
            m = r["metrics"][metric_key]
            row += f"  {m['mean']:.4f} (±{m['std']:.4f})        "
        lines.append(row)

    return "\n".join(lines)
