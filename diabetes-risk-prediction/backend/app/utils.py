"""
Utility functions for the backend application.
"""

import logging
import sys


def setup_logging(level: str = "INFO"):
    """Configure structured logging for the application.
    
    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
    """
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Suppress noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def format_probability(probability: float) -> str:
    """Format probability as a percentage string.
    
    Args:
        probability: Float between 0 and 1.
        
    Returns:
        Formatted percentage string (e.g., '94.2%').
    """
    return f"{probability * 100:.1f}%"
