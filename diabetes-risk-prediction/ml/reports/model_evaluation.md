# Model Evaluation Report

## Diabetes Risk Prediction Using Logistic Regression and Random Forest with Cross-Validation

**Generated:** 2026-10-14 11:19:25

---

## 1. Dataset Description

| Property | Value |
|----------|-------|
| Total Samples | 520 |
| Total Features | 11 (excluding target) |
| Numerical Features | 1 (Age) |
| Binary Features | 9 (Yes/No symptoms) |
| Gender Feature | 1 (Male/Female) |
| Target Column | `class` (Positive/Negative) |
| Missing Values | None |
| Duplicate Rows | 275 |

### Target Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Positive | 320 | 61.5% |
| Negative | 200 | 38.5% |

### Age Statistics

| Statistic | Value |
|-----------|-------|
| Min | 16 |
| Max | 90 |
| Mean | 48.0 |
| Median | 47.5 |
| Std Dev | 12.2 |

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
| Accuracy | 0.8904 (±0.0207) | 0.9558 (±0.0178) |
| Precision | 0.9557 (±0.0221) | 0.9842 (±0.0169) |
| Recall | 0.8625 (±0.0319) | 0.9437 (±0.0306) |
| F1 Score | 0.9062 (±0.0186) | 0.9631 (±0.0153) |
| Roc Auc | 0.9576 (±0.0155) | 0.9950 (±0.0028) |

### Confusion Matrices

**Logistic Regression:**

|  | Predicted Negative | Predicted Positive |
|--|-------------------|--------------------|
| Actual Negative | 187 | 13 |
| Actual Positive | 44 | 276 |

**Random Forest:**

|  | Predicted Negative | Predicted Positive |
|--|-------------------|--------------------|
| Actual Negative | 195 | 5 |
| Actual Positive | 18 | 302 |

### Feature Importance (Random Forest)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Polyuria | 0.2962 |
| 2 | Polydipsia | 0.2377 |
| 3 | Age | 0.1104 |
| 4 | Gender | 0.1037 |
| 5 | sudden weight loss | 0.0682 |
| 6 | Polyphagia | 0.0403 |
| 7 | delayed healing | 0.0363 |
| 8 | visual blurring | 0.0319 |
| 9 | muscle stiffness | 0.0310 |
| 10 | weakness | 0.0234 |
| 11 | Genital thrush | 0.0208 |

## 6. Model Selection

**Selected Model:** Random Forest

The model was selected based on the highest mean F1 Score across 5-fold cross-validation.
F1 Score was chosen as the selection criterion because it balances precision and recall,
which is important in a medical risk prediction context where both false positives
and false negatives have significant implications.

## 7. Limitations

1. **Dataset Size:** 520 samples is relatively small for a production ML system.
2. **Feature Set:** Only 11 features are available; clinical datasets typically include
   blood glucose levels, HbA1c, BMI, blood pressure, and other biomarkers.
3. **Binary Features:** Most features are binary (Yes/No), limiting the granularity of predictions.
4. **Generalizability:** The model was trained on a specific dataset and may not generalize to
   different populations or clinical settings.
5. **Not a Clinical Tool:** This is an ML-based risk assessment, NOT a medical diagnosis.

## 8. Conclusion

Both models achieve strong cross-validation performance on this dataset. The Random Forest
was selected as the primary prediction model based on its F1 Score. However, both models
are retained in the system for comparison purposes.

The moderate class imbalance was addressed through balanced class weights in both models.
The Stratified K-Fold cross-validation ensures reliable performance estimates that account
for the class distribution.

---

> **Disclaimer:** This system provides ML-based risk predictions and is NOT a substitute
> for professional medical diagnosis. Always consult a qualified healthcare professional
> for medical advice.
