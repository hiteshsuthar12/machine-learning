"""
Model loader module.

Handles loading serialized ML pipelines and metadata at startup.
The models are loaded once and cached for efficient prediction.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import joblib

from app.config import MODELS_DIR, DEFAULT_MODEL

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages loading and accessing trained ML models.
    
    Models are loaded once during initialization and cached in memory.
    This ensures prediction requests never trigger model retraining.
    """

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._metadata: Optional[Dict[str, Any]] = None
        self._loaded = False

    def load_models(self):
        """Load all trained models and metadata from disk.
        
        Raises:
            FileNotFoundError: If model files or metadata are missing.
            RuntimeError: If model loading fails.
        """
        if self._loaded:
            logger.info("Models already loaded, skipping reload")
            return

        models_dir = Path(MODELS_DIR)

        # Load metadata
        metadata_path = models_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Model metadata not found at {metadata_path}. "
                "Run the training pipeline first: python training/train.py"
            )

        with open(metadata_path, "r") as f:
            self._metadata = json.load(f)
        logger.info("Loaded model metadata")

        # Load Logistic Regression
        lr_path = models_dir / "logistic_regression.pkl"
        if lr_path.exists():
            self._models["logistic_regression"] = joblib.load(lr_path)
            logger.info(f"Loaded Logistic Regression from {lr_path}")
        else:
            logger.warning(f"Logistic Regression model not found at {lr_path}")

        # Load Random Forest
        rf_path = models_dir / "random_forest.pkl"
        if rf_path.exists():
            self._models["random_forest"] = joblib.load(rf_path)
            logger.info(f"Loaded Random Forest from {rf_path}")
        else:
            logger.warning(f"Random Forest model not found at {rf_path}")

        if not self._models:
            raise RuntimeError("No models were loaded. Run the training pipeline first.")

        self._loaded = True
        logger.info(f"Model loading complete. Available models: {list(self._models.keys())}")

    def get_model(self, model_name: Optional[str] = None):
        """Get a loaded model pipeline by name.
        
        Args:
            model_name: Model identifier. Defaults to the configured default model.
            
        Returns:
            The sklearn Pipeline for the specified model.
            
        Raises:
            ValueError: If the specified model is not available.
        """
        if not self._loaded:
            self.load_models()

        name = model_name or DEFAULT_MODEL
        if name not in self._models:
            available = list(self._models.keys())
            raise ValueError(
                f"Model '{name}' not available. Available models: {available}"
            )
        return self._models[name]

    def get_metadata(self) -> Dict[str, Any]:
        """Get the model metadata dictionary."""
        if not self._loaded:
            self.load_models()
        return self._metadata or {}

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the Random Forest model metadata."""
        metadata = self.get_metadata()
        rf_info = metadata.get("models", {}).get("random_forest", {})
        return rf_info.get("feature_importance", {})

    def get_selected_model_name(self) -> str:
        """Get the name of the selected best model."""
        metadata = self.get_metadata()
        return metadata.get("selected_model", "Random Forest")

    @property
    def is_loaded(self) -> bool:
        """Check if models have been loaded."""
        return self._loaded


# Singleton instance
model_manager = ModelManager()
