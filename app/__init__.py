"""
Customer Churn Prediction Package
Contains modules for predicting customer churn using machine learning models.
"""

from .main import CustomerAnalyzer, PredictionService, build_flask_application, launch_application

__version__ = "2.0.0"
__author__ = "Mihir Suhanda"

# Export main classes and functions
__all__ = [
    "CustomerAnalyzer",
    "PredictionService", 
    "build_flask_application",
    "launch_application"
]
