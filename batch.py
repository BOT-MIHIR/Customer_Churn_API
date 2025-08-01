"""
Batch Customer Churn Prediction Processing
Processes multiple customers for churn prediction using the updated API structure.
"""

import pandas as pd
import requests
import json
import argparse
import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple
from app.utils import BatchProcessingUtilities, LoggingManager, ResponseFormatter


class BatchChurnProcessor:
    """Handles batch processing of customer churn predictions."""
    
    def __init__(self, api_url: str = "http://localhost:8000/predict"):
        """Initialize the batch processor with API configuration."""
        self.api_endpoint = api_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._setup_logging_environment()
    
    def _setup_logging_environment(self):
        """Configure logging for batch processing."""
        log_directory = "logs"
        os.makedirs(log_directory, exist_ok=True)
        
        log_filename = os.path.join(log_directory, "batch_processing.log")
        LoggingManager.setup_advanced_logging(log_file=log_filename)
        
        logging.info("Batch processing environment initialized")
    
    def convert_customer_row_to_request(self, customer_row: pd.Series) -> Dict[str, Any]:
        """Convert DataFrame row to API request format."""
        return BatchProcessingUtilities.convert_row_to_request_format(customer_row)
    
    def make_prediction_request(self, customer_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Make a single prediction request to the API."""
        try:
            response = self.session.post(
                url=self.api_endpoint,
                data=json.dumps(customer_data),
                timeout=30
            )
            
            if response.status_code == 200:
                return True, response.json()
            else:
                error_response = ResponseFormatter.create_error_response(
                    f"API returned status {response.status_code}",
                    "HTTP_ERROR"
                )
                logging.warning(f"API request failed with status {response.status_code}")
                return False, error_response
                
        except requests.exceptions.RequestException as e:
            error_response = ResponseFormatter.create_error_response(
                f"Request failed: {str(e)}",
                "REQUEST_ERROR"
            )
            logging.error(f"Request exception: {str(e)}")
            return False, error_response
    
    def process_single_customer(self, customer_row: pd.Series, row_index: int) -> Tuple[pd.Series, bool]:
        """Process a single customer and return updated row with prediction."""
        request_data = self.convert_customer_row_to_request(customer_row)
        success, response = self.make_prediction_request(request_data)
        
        if success:
            customer_row["churn_probability"] = response["churn_probability"]
            customer_row["churn_prediction"] = response["churn_prediction"]
            customer_row["prediction_status"] = "success"
            logging.info(f"Successfully processed customer at row {row_index}")
        else:
            customer_row["churn_probability"] = None
            customer_row["churn_prediction"] = None
            customer_row["prediction_status"] = "failed"
            customer_row["error_message"] = response.get("error", "Unknown error")
            logging.error(f"Failed to process customer at row {row_index}: {response}")
        
        return customer_row, success
    
    def execute_batch_scoring(self, input_csv_file: str, output_csv_file: str = "batch_predictions.csv") -> Dict[str, Any]:
        """Execute batch scoring on the input CSV file."""
        logging.info(f"Starting batch processing for file: {input_csv_file}")
        
        # Load and prepare data
        customer_dataframe = BatchProcessingUtilities.prepare_batch_data(input_csv_file)
        total_customers = len(customer_dataframe)
        processed_results = []
        successful_predictions = 0
        probability_scores = []
        
        logging.info(f"Processing {total_customers} customers")
        
        # Process each customer
        for row_idx, customer_row in customer_dataframe.iterrows():
            processed_row, was_successful = self.process_single_customer(customer_row, row_idx)
            processed_results.append(processed_row)
            
            if was_successful:
                successful_predictions += 1
                probability_scores.append(processed_row["churn_probability"])
        
        # Save results to CSV
        results_dataframe = pd.DataFrame(processed_results)
        results_dataframe.to_csv(output_csv_file, index=False)
        
        # Calculate summary statistics
        batch_stats = BatchProcessingUtilities.calculate_batch_statistics(probability_scores)
        processing_summary = {
            "input_file": input_csv_file,
            "output_file": output_csv_file,
            "total_customers": total_customers,
            "successful_predictions": successful_predictions,
            "failed_predictions": total_customers - successful_predictions,
            "success_rate": (successful_predictions / total_customers) * 100 if total_customers > 0 else 0,
            "statistics": batch_stats,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        # Log summary
        logging.info(f"Batch processing completed: {processing_summary}")
        
        # Print summary to console
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Input file: {input_csv_file}")
        print(f"Output file: {output_csv_file}")
        print(f"Total customers: {total_customers}")
        print(f"Successful predictions: {successful_predictions}")
        print(f"Failed predictions: {total_customers - successful_predictions}")
        print(f"Success rate: {processing_summary['success_rate']:.2f}%")
        
        if probability_scores:
            print(f"\nPREDICTION STATISTICS:")
            print(f"Average churn probability: {batch_stats.get('average_probability', 0):.4f}")
            print(f"High risk customers (>=70%): {batch_stats.get('high_risk_count', 0)}")
            print(f"Medium risk customers (30-70%): {batch_stats.get('medium_risk_count', 0)}")
            print(f"Low risk customers (<30%): {batch_stats.get('low_risk_count', 0)}")
        
        print(f"{'='*60}\n")
        
        return processing_summary


def execute_batch_prediction_workflow(input_file_path: str, output_file_path: str = None):
    """Main workflow function for batch prediction processing."""
    if output_file_path is None:
        base_name = os.path.splitext(os.path.basename(input_file_path))[0]
        output_file_path = f"scored_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    processor = BatchChurnProcessor()
    summary = processor.execute_batch_scoring(input_file_path, output_file_path)
    
    # Save summary to JSON
    summary_file = f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Processing summary saved to: {summary_file}")
    return summary


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Batch Customer Churn Prediction Processing")
    argument_parser.add_argument("--input", type=str, required=True, 
                               help="Path to input CSV file containing customer data")
    argument_parser.add_argument("--output", type=str, default=None,
                               help="Path to output CSV file for results (optional)")
    argument_parser.add_argument("--api-url", type=str, default="http://localhost:8000/predict",
                               help="API endpoint URL for predictions")
    
    parsed_args = argument_parser.parse_args()
    
    try:
        execute_batch_prediction_workflow(parsed_args.input, parsed_args.output)
    except Exception as e:
        logging.error(f"Batch processing failed: {str(e)}")
        print(f"Error: {str(e)}")
        exit(1)