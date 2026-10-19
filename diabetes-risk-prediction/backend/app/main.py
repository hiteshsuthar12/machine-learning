"""
FastAPI main application module.

Diabetes Risk Prediction API
- GET  /health     — Health check
- GET  /model-info — Model metadata and metrics
- POST /predict    — Generate diabetes risk prediction
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS, LOG_LEVEL
from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse,
)
from app.model_loader import model_manager
from app.predictor import predict
from app.utils import setup_logging

# Setup logging
setup_logging(LOG_LEVEL)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler — loads models on startup."""
    logger.info("Starting Diabetes Risk Prediction API...")
    try:
        model_manager.load_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.error("Run the training pipeline first: python training/train.py")
    yield
    logger.info("Shutting down API...")


app = FastAPI(
    title="Diabetes Risk Prediction API",
    description=(
        "Machine Learning based diabetes risk prediction using Logistic Regression "
        "and Random Forest with 5-Fold Stratified Cross-Validation.\n\n"
        "**Disclaimer:** This system provides ML-based risk predictions and is NOT "
        "a substitute for professional medical diagnosis."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check endpoint",
)
async def health_check():
    """Check API health status and loaded model information."""
    try:
        selected_model = model_manager.get_selected_model_name()
        metadata = model_manager.get_metadata()
        return HealthResponse(
            status="healthy",
            model=selected_model,
            version=metadata.get("version", "1.0.0"),
        )
    except Exception:
        return HealthResponse(
            status="healthy",
            model="Not loaded",
            version="1.0.0",
        )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Get model information and metrics",
)
async def get_model_info():
    """Return detailed model information including metrics, features, and configuration."""
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models are not loaded. Run the training pipeline first.",
        )

    metadata = model_manager.get_metadata()

    return ModelInfoResponse(
        project=metadata.get("project", "Diabetes Risk Prediction"),
        version=metadata.get("version", "1.0.0"),
        training_date=metadata.get("training_date", "Unknown"),
        selected_model=metadata.get("selected_model", "Random Forest"),
        selection_criterion=metadata.get("selection_criterion", ""),
        cross_validation=metadata.get("cross_validation", {}),
        dataset=metadata.get("dataset", {}),
        models=metadata.get("models", {}),
        risk_thresholds=metadata.get("risk_thresholds", {}),
        feature_names=metadata.get("dataset", {}).get("features", []),
        disclaimer=metadata.get("disclaimer", ""),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Generate diabetes risk prediction",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input data"},
        500: {"model": ErrorResponse, "description": "Prediction error"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def predict_diabetes(request: PredictionRequest):
    """Generate a diabetes risk prediction based on patient symptoms.
    
    Accepts patient demographic and symptom information, processes it through
    the trained ML pipeline, and returns a prediction with probability,
    risk level, and feature importance.
    
    **Note:** This is an ML-based risk assessment, not a clinical diagnosis.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models are not loaded. The server needs to be restarted with trained models.",
        )

    try:
        result = predict(request)
        logger.info(
            f"Prediction: {result.prediction} | "
            f"Probability: {result.probability:.4f} | "
            f"Risk: {result.risk_level.value} | "
            f"Model: {result.model}"
        )
        return result
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred during prediction. Please try again.",
        )


@app.get("/", tags=["System"])
async def root():
    """API root endpoint."""
    return {
        "message": "Diabetes Risk Prediction API",
        "docs": "/docs",
        "health": "/health",
    }
