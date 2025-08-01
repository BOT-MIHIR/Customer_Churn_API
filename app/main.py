"""
Customer Churn Prediction API
A Flask-based web service for predicting customer churn using machine learning models.
"""

import os
import logging
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
from flask import Flask, request, jsonify


class CustomerAnalyzer:
    """Handles customer churn prediction logic."""
    
    def __init__(self, model_path: str, transformer_path: str):
        """Initialize the predictor with model and transformer paths."""
        self.model = self._initialize_ml_model(model_path)
        self.transformer = self._initialize_data_processor(transformer_path)
        self.churn_threshold = 0.5
        
    def _initialize_ml_model(self, path: str):
        """Load the trained machine learning model."""
        try:
            return joblib.load(path)
        except Exception as e:
            logging.error(f"Failed to load model from {path}: {e}")
            raise
            
    def _initialize_data_processor(self, path: str):
        """Load the data transformer/preprocessor."""
        try:
            return joblib.load(path)
        except Exception as e:
            logging.error(f"Failed to load transformer from {path}: {e}")
            raise
    
    def convert_to_dataframe(self, customer_info: Dict[str, Any]) -> pd.DataFrame:
        """Convert customer data to DataFrame format."""
        return pd.DataFrame([customer_info])
    
    def compute_probability_score(self, processed_data: pd.DataFrame) -> float:
        """Calculate the probability of customer churn."""
        transformed_features = self.transformer.transform(processed_data)
        churn_prob = self.model.predict_proba(transformed_features)[0][1]
        return churn_prob
    
    def determine_churn_status(self, churn_prob: float) -> str:
        """Make binary churn prediction based on probability threshold."""
        return "Yes" if churn_prob >= self.churn_threshold else "No"
    
    def analyze_customer_retention(self, customer_data: Dict[str, Any]) -> Tuple[float, str]:
        """Complete churn prediction pipeline."""
        df = self.convert_to_dataframe(customer_data)
        probability = self.compute_probability_score(df)
        prediction = self.determine_churn_status(probability)
        return probability, prediction


class PredictionService:
    """Flask API wrapper for churn prediction service."""
    
    def __init__(self):
        """Initialize the Flask application and churn predictor."""
        self.app = Flask(__name__)
        self.analyzer = CustomerAnalyzer("app/model.pkl", "app/transformer.pkl")
        self._initialize_api_routes()
        self._setup_application_logging()
    
    def _setup_application_logging(self):
        """Configure application logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def _initialize_api_routes(self):
        """Configure API routes."""
        self.app.add_url_rule('/predict', 'predict', self._process_prediction_request, methods=['POST'])
    
    def _process_prediction_request(self):
        """Handle churn prediction API requests."""
        try:
            request_data = request.get_json()
            
            if not request_data or 'customer' not in request_data:
                return jsonify({"error": "Missing 'customer' data in request"}), 400
            
            customer_info = request_data['customer']
            probability, prediction = self.analyzer.analyze_customer_retention(customer_info)
            
            response_data = {
                "churn_probability": round(probability, 2),
                "churn_prediction": prediction
            }
            
            logging.info(f"Prediction made: {response_data}")
            return jsonify(response_data)
            
        except KeyError as e:
            error_msg = f"Missing required field: {str(e)}"
            logging.error(error_msg)
            return jsonify({"error": error_msg}), 400
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            logging.error(error_msg)
            return jsonify({"error": error_msg}), 500
    
    def start_web_server(self, host: str = 'localhost', port: int = 8000):
        """Start the Flask development server."""
        logging.info(f"Starting churn prediction API on {host}:{port}")
        self.app.run(host=host, port=port)


def build_flask_application():
    """Factory function to create and configure the Flask application."""
    service = PredictionService()
    return service.app


def launch_application():
    """Main application entry point."""
    try:
        prediction_service = PredictionService()
        prediction_service.start_web_server()
    except Exception as e:
        logging.error(f"Failed to start application: {e}")
        raise


if __name__ == "__main__":
    launch_application()
