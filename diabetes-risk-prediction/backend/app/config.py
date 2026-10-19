"""
Application configuration.

Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# CORS
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
).split(",")

# Model
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "random_forest")

# Risk thresholds (configurable via environment)
RISK_THRESHOLD_LOW = float(os.getenv("RISK_THRESHOLD_LOW", "0.35"))
RISK_THRESHOLD_HIGH = float(os.getenv("RISK_THRESHOLD_HIGH", "0.65"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
