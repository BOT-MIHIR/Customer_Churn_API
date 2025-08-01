"""
Utility functions for customer churn prediction API
Contains helper functions for data processing, validation, and logging.
"""

import os
import json
import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime


class DataValidator:
    """Validates input data for churn prediction."""
    
    @staticmethod
    def validate_customer_data(customer_info: Dict[str, Any]) -> bool:
        """Validate if customer data contains required fields."""
        if not isinstance(customer_info, dict):
            return False
        
        # Add specific validation rules here based on your model requirements
        required_fields = []  # Define based on your model's expected features
        
        for field in required_fields:
            if field not in customer_info:
                return False
        
        return True
    
    @staticmethod
    def sanitize_input_data(customer_info: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and sanitize input data."""
        sanitized = {}
        for key, value in customer_info.items():
            # Remove null values and clean data
            if value is not None and value != "":
                sanitized[key] = value
        return sanitized


class ResponseFormatter:
    """Formats API responses consistently."""
    
    @staticmethod
    def create_success_response(probability: float, prediction: str) -> Dict[str, Any]:
        """Create a standardized success response."""
        return {
            "churn_probability": round(probability, 2),
            "churn_prediction": prediction,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_error_response(error_message: str, error_code: str = "PREDICTION_ERROR") -> Dict[str, Any]:
        """Create a standardized error response."""
        return {
            "error": error_message,
            "error_code": error_code,
            "status": "failed",
            "timestamp": datetime.now().isoformat()
        }


class LoggingManager:
    """Manages application logging configuration."""
    
    @staticmethod
    def setup_advanced_logging(log_level: str = "INFO", log_file: Optional[str] = None):
        """Configure advanced logging with file and console handlers."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create formatters
        formatter = logging.Formatter(log_format)
        
        # Get root logger
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    @staticmethod
    def log_prediction_request(customer_data: Dict[str, Any], result: Dict[str, Any]):
        """Log prediction requests for audit purposes."""
        logging.info(f"Prediction Request - Input: {json.dumps(customer_data)}, Output: {json.dumps(result)}")


class ConfigurationManager:
    """Manages application configuration settings."""
    
    DEFAULT_CONFIG = {
        "server": {
            "host": "localhost",
            "port": 8000,
            "debug": False
        },
        "model": {
            "threshold": 0.5,
            "model_path": "app/model.pkl",
            "transformer_path": "app/transformer.pkl"
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(levelname)s - %(message)s"
        }
    }
    
    @classmethod
    def load_configuration(cls, config_file: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file or return defaults."""
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load config file {config_file}: {e}")
        
        return cls.DEFAULT_CONFIG
    
    @staticmethod
    def save_configuration(config: Dict[str, Any], config_file: str):
        """Save configuration to file."""
        try:
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save config file {config_file}: {e}")


class BatchProcessingUtilities:
    """Utility functions for batch processing operations."""
    
    @staticmethod
    def prepare_batch_data(csv_file: str) -> pd.DataFrame:
        """Load and prepare data for batch processing."""
        try:
            df = pd.read_csv(csv_file)
            # Remove rows with too many null values
            df = df.dropna(thresh=len(df.columns) * 0.5)
            return df
        except Exception as e:
            logging.error(f"Failed to load batch data from {csv_file}: {e}")
            raise
    
    @staticmethod
    def convert_row_to_request_format(row: pd.Series) -> Dict[str, Any]:
        """Convert DataFrame row to API request format."""
        return {"customer": row.dropna().to_dict()}
    
    @staticmethod
    def calculate_batch_statistics(probabilities: List[float]) -> Dict[str, float]:
        """Calculate statistics for batch processing results."""
        if not probabilities:
            return {}
        
        return {
            "count": len(probabilities),
            "average_probability": sum(probabilities) / len(probabilities),
            "min_probability": min(probabilities),
            "max_probability": max(probabilities),
            "high_risk_count": sum(1 for p in probabilities if p >= 0.7),
            "medium_risk_count": sum(1 for p in probabilities if 0.3 <= p < 0.7),
            "low_risk_count": sum(1 for p in probabilities if p < 0.3)
        }


def initialize_application_environment():
    """Initialize the application environment with necessary directories and settings."""
    # Create necessary directories
    directories = ["logs", "data", "config", "output"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Setup logging
    LoggingManager.setup_advanced_logging(log_file="logs/application.log")
    
    logging.info("Application environment initialized successfully")


def health_check() -> Dict[str, Any]:
    """Perform a basic health check of the application."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "checks": {}
    }
    
    # Check if model files exist
    model_files = ["app/model.pkl", "app/transformer.pkl"]
    for file_path in model_files:
        health_status["checks"][file_path] = os.path.exists(file_path)
    
    # Check if logs directory exists
    health_status["checks"]["logs_directory"] = os.path.exists("logs")
    
    # Overall status
    all_checks_passed = all(health_status["checks"].values())
    health_status["status"] = "healthy" if all_checks_passed else "unhealthy"
    
    return health_status
