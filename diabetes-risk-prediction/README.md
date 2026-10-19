# Diabetes Risk Prediction

## Machine Learning Based Risk Assessment Using Logistic Regression and Random Forest with Cross-Validation

A complete, production-quality full-stack web application that uses machine learning to predict diabetes risk based on patient demographics and symptom data.

> **Disclaimer:** This system provides ML-based risk predictions and is NOT a substitute for professional medical diagnosis. Always consult a qualified healthcare professional for medical advice.

---

## Features

- **Dual ML Models** — Logistic Regression and Random Forest classifiers
- **5-Fold Stratified Cross-Validation** — Reliable performance estimates
- **FastAPI Backend** — Production-ready REST API with input validation
- **React Frontend** — Modern dashboard with interactive prediction form
- **Model Explainability** — Feature importance visualization
- **Risk Classification** — Low / Moderate / High risk categories
- **Model Comparison** — Side-by-side performance metrics
- **Docker Support** — Containerized deployment
- **Comprehensive Tests** — 32 passing tests covering all endpoints

---

## Architecture

```
Browser → React (Vite) → FastAPI → sklearn Pipeline → Random Forest → Risk Prediction
```

```
diabetes-risk-prediction/
├── backend/
│   ├── app/                  # FastAPI application
│   │   ├── main.py           # API endpoints
│   │   ├── schemas.py        # Pydantic models
│   │   ├── predictor.py      # Prediction logic
│   │   ├── model_loader.py   # Model management
│   │   ├── config.py         # Configuration
│   │   └── utils.py          # Utilities
│   ├── models/               # Serialized ML pipelines
│   │   ├── logistic_regression.pkl
│   │   ├── random_forest.pkl
│   │   └── metadata.json
│   ├── data/                 # Dataset
│   ├── training/             # ML pipeline
│   │   ├── train.py          # Training script
│   │   ├── evaluate.py       # Evaluation module
│   │   └── preprocessing.py  # Feature engineering
│   ├── tests/                # pytest test suite
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── pages/            # Page views
│   │   ├── services/         # API client
│   │   ├── App.jsx
│   │   └── index.css
│   └── package.json
├── ml/
│   ├── reports/              # Evaluation report
│   └── visualizations/       # Generated charts
├── docker/
│   ├── backend.Dockerfile
│   └── frontend.Dockerfile
└── docker-compose.yml
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | `diabetes_data_upload.csv` |
| Samples | 520 |
| Features | 11 (1 numerical, 1 gender, 9 binary symptoms) |
| Target | `class` (Positive / Negative) |
| Positive | 320 (61.5%) |
| Negative | 200 (38.5%) |
| Missing Values | None |

### Features

| Feature | Type | Values |
|---------|------|--------|
| Age | Numerical | 16–90 |
| Gender | Categorical | Male / Female |
| Polyuria | Binary | Yes / No |
| Polydipsia | Binary | Yes / No |
| Sudden Weight Loss | Binary | Yes / No |
| Weakness | Binary | Yes / No |
| Polyphagia | Binary | Yes / No |
| Genital Thrush | Binary | Yes / No |
| Visual Blurring | Binary | Yes / No |
| Delayed Healing | Binary | Yes / No |
| Muscle Stiffness | Binary | Yes / No |

---

## Data Preprocessing

Built using scikit-learn `Pipeline` + `ColumnTransformer` to prevent data leakage:

1. **Age** — `StandardScaler` (fitted only on training folds)
2. **Gender** — Binary encoding (Male=1, Female=0)
3. **Symptoms** — Binary encoding (Yes=1, No=0)
4. **Class weights** — `balanced` to handle moderate class imbalance (1.6:1)

---

## Machine Learning Models

### Logistic Regression
- Solver: `lbfgs`
- Regularization: L2 (`C=1.0`)
- Max iterations: 1000
- Class weight: `balanced`

### Random Forest
- Trees: 200
- Max depth: 10
- Min samples split: 5
- Min samples leaf: 2
- Class weight: `balanced`

---

## Cross-Validation

- **Method:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- The entire pipeline (preprocessing + model) is fit on each training fold independently

---

## Evaluation Metrics

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| Accuracy | 89.04% (±2.07) | 95.58% (±1.78) |
| Precision | 95.57% (±2.21) | 98.42% (±1.69) |
| Recall | 86.25% (±3.19) | 94.37% (±3.06) |
| F1 Score | 90.62% (±1.86) | 96.31% (±1.53) |
| ROC-AUC | 95.76% (±1.55) | 99.50% (±0.28) |

**Selected Model:** Random Forest (highest mean F1 Score)

### Feature Importance (Random Forest)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Polyuria | 29.62% |
| 2 | Polydipsia | 23.77% |
| 3 | Age | 11.04% |
| 4 | Gender | 10.37% |
| 5 | Sudden Weight Loss | 6.82% |

---

## Model Comparison

Both models are retained. The Random Forest was selected based on its higher F1 Score across cross-validation.

---

## API Documentation

### `GET /health`
```json
{ "status": "healthy", "model": "Random Forest", "version": "1.0.0" }
```

### `GET /model-info`
Returns model metadata, metrics, feature names, and configuration.

### `POST /predict`
```json
{
  "age": 45,
  "gender": "Male",
  "polyuria": "Yes",
  "polydipsia": "Yes",
  "sudden_weight_loss": "No",
  "weakness": "Yes",
  "polyphagia": "No",
  "genital_thrush": "No",
  "visual_blurring": "No",
  "delayed_healing": "Yes",
  "muscle_stiffness": "No"
}
```

**Response:**
```json
{
  "prediction": "Positive",
  "probability": 0.9634,
  "risk_level": "High",
  "model": "Random Forest",
  "feature_importance": { ... },
  "input_summary": { ... },
  "disclaimer": "Risk level is an ML application classification..."
}
```

Interactive docs: `http://localhost:8000/docs`

---

## Frontend

- **Home** — Project overview, tech stack, architecture
- **Assessment** — Structured prediction form with toggle switches
- **Dashboard** — Model metrics, comparison charts, confusion matrices, feature importance

---

## Installation

### Prerequisites
- Python 3.11+
- Node.js 18+
- pip, npm

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
python training/train.py      # Train models
uvicorn app.main:app --reload # Start API on :8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev                   # Start dev server on :5173
```

---

## Running Locally

1. **Train models:** `cd backend && python training/train.py`
2. **Start backend:** `cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000`
3. **Start frontend:** `cd frontend && npm run dev`
4. **Open:** `http://localhost:5173`

---

## Docker Setup

```bash
docker compose up --build
```

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:3000`

---

## Testing

```bash
cd backend
python -m pytest tests/ -v
```

**32 tests** covering:
- Health check endpoint
- Model info endpoint
- Valid predictions
- Invalid inputs (missing fields, wrong types, out-of-range values)
- Risk classification logic
- Input preparation

---

## Risk Levels

| Level | Probability Range | Description |
|-------|-------------------|-------------|
| Low | 0% – 35% | Low predicted probability |
| Moderate | 35% – 65% | Moderate predicted probability |
| High | 65% – 100% | High predicted probability |

> Risk level is an ML application classification and is NOT a clinical diagnosis.

---

## Screenshots

*(Run the application to see the live UI)*

- **Home** — Hero section with project description
- **Assessment** — Patient info + symptom toggle form
- **Result** — Gauge, prediction, feature importance
- **Dashboard** — Metrics cards, comparison charts, confusion matrices

---

## Limitations

1. **Dataset size** — 520 samples is relatively small for production ML
2. **Feature set** — Only 11 features; clinical datasets typically include blood glucose, HbA1c, BMI, blood pressure
3. **Binary features** — Most features are Yes/No, limiting prediction granularity
4. **Generalizability** — Trained on a specific dataset; may not generalize to different populations
5. **Not a clinical tool** — This is an educational ML application

---

## Future Enhancements

- Additional ML models (XGBoost, SVM, Neural Network)
- SHAP-based individual prediction explanations
- Batch prediction support
- User authentication
- Prediction history and analytics
- Additional clinical features (blood glucose, BMI, etc.)
- Model versioning and A/B testing
- Automated retraining pipeline

---

## Disclaimer

This application is developed for educational and research purposes only. It uses machine learning algorithms to estimate diabetes risk based on a limited set of symptoms and is NOT intended to replace professional medical diagnosis, treatment, or advice. Always consult a qualified healthcare professional for any health concerns.
