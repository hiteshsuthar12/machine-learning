"""
API endpoint tests for the Diabetes Risk Prediction backend.

Tests cover:
- Health check endpoint
- Model info endpoint
- Valid prediction requests
- Invalid prediction requests (missing fields, wrong types, out-of-range values)
- Model loading
"""

import sys
import os
import pytest
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app


client = TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_has_model(self):
        response = client.get("/health")
        data = response.json()
        assert "model" in data

    def test_health_has_version(self):
        response = client.get("/health")
        data = response.json()
        assert "version" in data


class TestModelInfoEndpoint:
    """Tests for GET /model-info."""

    def test_model_info_returns_200(self):
        response = client.get("/model-info")
        assert response.status_code == 200

    def test_model_info_has_required_fields(self):
        response = client.get("/model-info")
        data = response.json()
        required_fields = [
            "project", "version", "training_date", "selected_model",
            "cross_validation", "dataset", "models", "risk_thresholds",
            "feature_names", "disclaimer",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_model_info_has_correct_cv_config(self):
        response = client.get("/model-info")
        data = response.json()
        cv = data["cross_validation"]
        assert cv["method"] == "StratifiedKFold"
        assert cv["n_splits"] == 5

    def test_model_info_has_both_models(self):
        response = client.get("/model-info")
        data = response.json()
        assert "logistic_regression" in data["models"]
        assert "random_forest" in data["models"]


class TestPredictionEndpoint:
    """Tests for POST /predict."""

    VALID_PAYLOAD = {
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

    def test_valid_prediction_returns_200(self):
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        assert response.status_code == 200

    def test_valid_prediction_has_required_fields(self):
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "risk_level" in data
        assert "model" in data
        assert "input_summary" in data
        assert "disclaimer" in data

    def test_prediction_is_valid_label(self):
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        data = response.json()
        assert data["prediction"] in ["Positive", "Negative"]

    def test_probability_is_valid_range(self):
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        data = response.json()
        assert 0.0 <= data["probability"] <= 1.0

    def test_risk_level_is_valid(self):
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        data = response.json()
        assert data["risk_level"] in ["Low", "Moderate", "High"]

    def test_feature_importance_included(self):
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        data = response.json()
        if data["model"] == "Random Forest":
            assert data["feature_importance"] is not None
            assert len(data["feature_importance"]) > 0

    def test_all_no_symptoms(self):
        """Test with all symptoms set to No — should predict Negative."""
        payload = {
            "age": 30,
            "gender": "Male",
            "polyuria": "No",
            "polydipsia": "No",
            "sudden_weight_loss": "No",
            "weakness": "No",
            "polyphagia": "No",
            "genital_thrush": "No",
            "visual_blurring": "No",
            "delayed_healing": "No",
            "muscle_stiffness": "No",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "Negative"

    def test_all_yes_symptoms(self):
        """Test with all symptoms set to Yes — should predict Positive."""
        payload = {
            "age": 55,
            "gender": "Female",
            "polyuria": "Yes",
            "polydipsia": "Yes",
            "sudden_weight_loss": "Yes",
            "weakness": "Yes",
            "polyphagia": "Yes",
            "genital_thrush": "Yes",
            "visual_blurring": "Yes",
            "delayed_healing": "Yes",
            "muscle_stiffness": "Yes",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "Positive"


class TestInvalidPredictions:
    """Tests for invalid prediction requests."""

    def test_missing_age(self):
        payload = {
            "gender": "Male",
            "polyuria": "Yes",
            "polydipsia": "No",
            "sudden_weight_loss": "No",
            "weakness": "No",
            "polyphagia": "No",
            "genital_thrush": "No",
            "visual_blurring": "No",
            "delayed_healing": "No",
            "muscle_stiffness": "No",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_age_negative(self):
        payload = {
            "age": -5,
            "gender": "Male",
            "polyuria": "Yes",
            "polydipsia": "No",
            "sudden_weight_loss": "No",
            "weakness": "No",
            "polyphagia": "No",
            "genital_thrush": "No",
            "visual_blurring": "No",
            "delayed_healing": "No",
            "muscle_stiffness": "No",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_age_too_high(self):
        payload = {
            "age": 200,
            "gender": "Male",
            "polyuria": "Yes",
            "polydipsia": "No",
            "sudden_weight_loss": "No",
            "weakness": "No",
            "polyphagia": "No",
            "genital_thrush": "No",
            "visual_blurring": "No",
            "delayed_healing": "No",
            "muscle_stiffness": "No",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_gender(self):
        payload = {
            "age": 45,
            "gender": "Other",
            "polyuria": "Yes",
            "polydipsia": "No",
            "sudden_weight_loss": "No",
            "weakness": "No",
            "polyphagia": "No",
            "genital_thrush": "No",
            "visual_blurring": "No",
            "delayed_healing": "No",
            "muscle_stiffness": "No",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_symptom_value(self):
        payload = {
            "age": 45,
            "gender": "Male",
            "polyuria": "Maybe",
            "polydipsia": "No",
            "sudden_weight_loss": "No",
            "weakness": "No",
            "polyphagia": "No",
            "genital_thrush": "No",
            "visual_blurring": "No",
            "delayed_healing": "No",
            "muscle_stiffness": "No",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_empty_body(self):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_string_age(self):
        payload = {
            "age": "forty-five",
            "gender": "Male",
            "polyuria": "Yes",
            "polydipsia": "No",
            "sudden_weight_loss": "No",
            "weakness": "No",
            "polyphagia": "No",
            "genital_thrush": "No",
            "visual_blurring": "No",
            "delayed_healing": "No",
            "muscle_stiffness": "No",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_missing_multiple_fields(self):
        payload = {
            "age": 45,
            "gender": "Male",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


class TestRootEndpoint:
    """Tests for GET /."""

    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_message(self):
        response = client.get("/")
        data = response.json()
        assert "message" in data
