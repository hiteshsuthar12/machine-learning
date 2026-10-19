"""
Pydantic schemas for request/response validation.

All schemas are derived from the actual dataset columns.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal, Dict, List, Any
from enum import Enum


class GenderEnum(str, Enum):
    """Valid gender values from the dataset."""
    MALE = "Male"
    FEMALE = "Female"


class BinaryChoice(str, Enum):
    """Valid Yes/No choices for symptom features."""
    YES = "Yes"
    NO = "No"


class PredictionRequest(BaseModel):
    """Input schema for diabetes risk prediction.
    
    All fields map directly to the dataset columns.
    Age is the only numerical input; all others are categorical.
    """
    age: int = Field(
        ...,
        ge=1,
        le=120,
        description="Patient age in years (1-120)",
        examples=[45],
    )
    gender: GenderEnum = Field(
        ...,
        description="Patient gender (Male/Female)",
        examples=["Male"],
    )
    polyuria: BinaryChoice = Field(
        ...,
        description="Excessive urination (Yes/No)",
        examples=["Yes"],
    )
    polydipsia: BinaryChoice = Field(
        ...,
        description="Excessive thirst (Yes/No)",
        examples=["Yes"],
    )
    sudden_weight_loss: BinaryChoice = Field(
        ...,
        description="Sudden weight loss (Yes/No)",
        examples=["No"],
    )
    weakness: BinaryChoice = Field(
        ...,
        description="General weakness (Yes/No)",
        examples=["Yes"],
    )
    polyphagia: BinaryChoice = Field(
        ...,
        description="Excessive hunger (Yes/No)",
        examples=["No"],
    )
    genital_thrush: BinaryChoice = Field(
        ...,
        description="Genital thrush (Yes/No)",
        examples=["No"],
    )
    visual_blurring: BinaryChoice = Field(
        ...,
        description="Visual blurring (Yes/No)",
        examples=["No"],
    )
    delayed_healing: BinaryChoice = Field(
        ...,
        description="Delayed wound healing (Yes/No)",
        examples=["Yes"],
    )
    muscle_stiffness: BinaryChoice = Field(
        ...,
        description="Muscle stiffness (Yes/No)",
        examples=["No"],
    )

    @field_validator("age")
    @classmethod
    def validate_age(cls, v):
        if v < 1 or v > 120:
            raise ValueError("Age must be between 1 and 120")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
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
                    "muscle_stiffness": "No",
                }
            ]
        }
    }


class RiskLevel(str, Enum):
    """Risk classification categories.
    
    These are ML application classifications, NOT clinical diagnoses.
    """
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"


class PredictionResponse(BaseModel):
    """Response schema for prediction results."""
    prediction: str = Field(
        ...,
        description="Predicted class label (Positive/Negative)",
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model probability for the positive class",
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Risk category based on probability threshold",
    )
    model: str = Field(
        ...,
        description="Name of the model used for prediction",
    )
    feature_importance: Optional[Dict[str, float]] = Field(
        default=None,
        description="Feature importance scores (Random Forest only)",
    )
    input_summary: Dict[str, Any] = Field(
        ...,
        description="Summary of the input data used for prediction",
    )
    disclaimer: str = Field(
        default="Risk level is an ML application classification and is not a clinical diagnosis. Always consult a qualified healthcare professional.",
        description="Medical disclaimer",
    )


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str
    model: str
    version: str


class ModelInfoResponse(BaseModel):
    """Response schema for model information endpoint."""
    project: str
    version: str
    training_date: str
    selected_model: str
    selection_criterion: str
    cross_validation: Dict[str, Any]
    dataset: Dict[str, Any]
    models: Dict[str, Any]
    risk_thresholds: Dict[str, Any]
    feature_names: List[str]
    disclaimer: str


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    detail: str
    status_code: int
