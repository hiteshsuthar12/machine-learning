"""
Prediction logic module.

Converts user input into the format expected by the trained sklearn pipeline
and generates risk predictions with probability and risk level classification.
"""

import logging
from typing import Dict, Any

import pandas as pd
import numpy as np

from app.model_loader import model_manager
from app.schemas import PredictionRequest, PredictionResponse, RiskLevel
from app.config import RISK_THRESHOLD_LOW, RISK_THRESHOLD_HIGH

logger = logging.getLogger(__name__)


def classify_risk(probability: float) -> RiskLevel:
    """Classify the prediction probability into a risk level.
    
    Thresholds are configurable via environment variables:
    - Low: probability < RISK_THRESHOLD_LOW (default: 0.35)
    - Moderate: RISK_THRESHOLD_LOW <= probability < RISK_THRESHOLD_HIGH (default: 0.65)
    - High: probability >= RISK_THRESHOLD_HIGH
    
    Note: These are ML application classifications, NOT clinical diagnoses.
    
    Args:
        probability: The model's predicted probability for the positive class.
        
    Returns:
        RiskLevel enum value.
    """
    if probability < RISK_THRESHOLD_LOW:
        return RiskLevel.LOW
    elif probability < RISK_THRESHOLD_HIGH:
        return RiskLevel.MODERATE
    else:
        return RiskLevel.HIGH


def prepare_input(request: PredictionRequest) -> pd.DataFrame:
    """Convert a PredictionRequest into a DataFrame matching the training format.
    
    The column names must match exactly what the trained pipeline expects.
    
    Args:
        request: Validated prediction request.
        
    Returns:
        Single-row DataFrame with columns matching the training features.
    """
    data = {
        "Age": [request.age],
        "Gender": [request.gender.value],
        "Polyuria": [request.polyuria.value],
        "Polydipsia": [request.polydipsia.value],
        "sudden weight loss": [request.sudden_weight_loss.value],
        "weakness": [request.weakness.value],
        "Polyphagia": [request.polyphagia.value],
        "Genital thrush": [request.genital_thrush.value],
        "visual blurring": [request.visual_blurring.value],
        "delayed healing": [request.delayed_healing.value],
        "muscle stiffness": [request.muscle_stiffness.value],
    }
    return pd.DataFrame(data)


def predict(request: PredictionRequest, model_name: str = None) -> PredictionResponse:
    """Generate a diabetes risk prediction.
    
    This function:
    1. Converts the request into the pipeline's expected format
    2. Loads the specified model (or default)
    3. Generates prediction and probability
    4. Classifies the risk level
    5. Retrieves feature importance (for Random Forest)
    6. Returns a structured response
    
    Args:
        request: Validated prediction input.
        model_name: Optional model to use. Defaults to the selected best model.
        
    Returns:
        PredictionResponse with prediction, probability, risk level, and metadata.
        
    Raises:
        ValueError: If the model is not available.
        RuntimeError: If prediction fails.
    """
    # Determine which model to use
    if model_name is None:
        metadata = model_manager.get_metadata()
        selected = metadata.get("selected_model", "Random Forest")
        model_key = "random_forest" if "Random Forest" in selected else "logistic_regression"
    else:
        model_key = model_name

    model_display_name = "Random Forest" if model_key == "random_forest" else "Logistic Regression"

    # Prepare input
    input_df = prepare_input(request)
    logger.info(f"Predicting with model: {model_display_name}")

    # Get model pipeline
    pipeline = model_manager.get_model(model_key)

    # Generate prediction
    prediction_encoded = pipeline.predict(input_df)[0]
    prediction_label = "Positive" if prediction_encoded == 1 else "Negative"

    # Generate probability
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(input_df)[0]
        positive_probability = float(probabilities[1])
    else:
        positive_probability = float(prediction_encoded)

    # Classify risk
    risk_level = classify_risk(positive_probability)

    # Get feature importance (Random Forest only)
    feature_importance = None
    if model_key == "random_forest":
        feature_importance = model_manager.get_feature_importance()

    # Build input summary
    input_summary = {
        "Age": request.age,
        "Gender": request.gender.value,
        "Polyuria": request.polyuria.value,
        "Polydipsia": request.polydipsia.value,
        "Sudden Weight Loss": request.sudden_weight_loss.value,
        "Weakness": request.weakness.value,
        "Polyphagia": request.polyphagia.value,
        "Genital Thrush": request.genital_thrush.value,
        "Visual Blurring": request.visual_blurring.value,
        "Delayed Healing": request.delayed_healing.value,
        "Muscle Stiffness": request.muscle_stiffness.value,
    }

    return PredictionResponse(
        prediction=prediction_label,
        probability=round(positive_probability, 4),
        risk_level=risk_level,
        model=model_display_name,
        feature_importance=feature_importance,
        input_summary=input_summary,
    )
