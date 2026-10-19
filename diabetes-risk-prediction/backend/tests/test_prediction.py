"""
Tests for the prediction logic module.

Tests the preprocessing, input conversion, and prediction flow.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas import PredictionRequest, RiskLevel
from app.predictor import classify_risk, prepare_input


class TestRiskClassification:
    """Tests for risk level classification."""

    def test_low_risk(self):
        assert classify_risk(0.10) == RiskLevel.LOW
        assert classify_risk(0.0) == RiskLevel.LOW
        assert classify_risk(0.34) == RiskLevel.LOW

    def test_moderate_risk(self):
        assert classify_risk(0.35) == RiskLevel.MODERATE
        assert classify_risk(0.50) == RiskLevel.MODERATE
        assert classify_risk(0.64) == RiskLevel.MODERATE

    def test_high_risk(self):
        assert classify_risk(0.65) == RiskLevel.HIGH
        assert classify_risk(0.90) == RiskLevel.HIGH
        assert classify_risk(1.0) == RiskLevel.HIGH


class TestInputPreparation:
    """Tests for input DataFrame preparation."""

    def test_prepare_input_returns_dataframe(self):
        request = PredictionRequest(
            age=45, gender="Male", polyuria="Yes", polydipsia="No",
            sudden_weight_loss="No", weakness="Yes", polyphagia="No",
            genital_thrush="No", visual_blurring="No",
            delayed_healing="Yes", muscle_stiffness="No",
        )
        df = prepare_input(request)
        assert len(df) == 1
        assert "Age" in df.columns
        assert "Gender" in df.columns
        assert "Polyuria" in df.columns

    def test_prepare_input_correct_values(self):
        request = PredictionRequest(
            age=30, gender="Female", polyuria="No", polydipsia="Yes",
            sudden_weight_loss="Yes", weakness="No", polyphagia="Yes",
            genital_thrush="Yes", visual_blurring="No",
            delayed_healing="No", muscle_stiffness="Yes",
        )
        df = prepare_input(request)
        assert df.iloc[0]["Age"] == 30
        assert df.iloc[0]["Gender"] == "Female"
        assert df.iloc[0]["Polyuria"] == "No"
        assert df.iloc[0]["Polydipsia"] == "Yes"

    def test_prepare_input_column_count(self):
        request = PredictionRequest(
            age=50, gender="Male", polyuria="Yes", polydipsia="Yes",
            sudden_weight_loss="Yes", weakness="Yes", polyphagia="Yes",
            genital_thrush="Yes", visual_blurring="Yes",
            delayed_healing="Yes", muscle_stiffness="Yes",
        )
        df = prepare_input(request)
        # 11 features: Age, Gender, and 9 binary symptoms
        assert len(df.columns) == 11
