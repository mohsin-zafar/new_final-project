"""
Manufacturing Equipment Output Prediction - Prediction Module
==============================================================
This module handles model loading and predictions.
"""

import joblib
import numpy as np
import os
from typing import Dict, Any

from preprocess import preprocess_input, validate_input

# Get the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')


class ManufacturingPredictor:
    """
    Predictor class for manufacturing output prediction.
    Handles model loading and prediction operations.
    """
    
    def __init__(self):
        """Initialize the predictor by loading the model."""
        self.model = None
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load the trained Linear Regression model."""
        model_path = os.path.join(MODEL_DIR, 'model.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                "Please ensure model.pkl is in the model folder."
            )
        
        try:
            self.model = joblib.load(model_path)
            self.model_loaded = True
            print(f"✓ Model loaded successfully from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction based on input features.
        
        Args:
            input_data: Dictionary containing all required features
        
        Returns:
            Dictionary containing prediction results
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Please call load_model() first.")
        
        try:
            # Validate input
            validated_data = validate_input(input_data)
            
            # Preprocess input
            preprocessed_features = preprocess_input(validated_data)
            
            # Make prediction
            prediction = self.model.predict(preprocessed_features)
            
            # Round to 2 decimal places
            predicted_value = round(float(prediction[0]), 2)
            
            return {
                "success": True,
                "predicted_parts_per_hour": predicted_value,
                "message": "Prediction successful"
            }
            
        except Exception as e:
            return {
                "success": False,
                "predicted_parts_per_hour": None,
                "message": f"Prediction failed: {str(e)}"
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model_loaded:
            return {"message": "Model not loaded"}
        
        return {
            "model_type": type(self.model).__name__,
            "model_loaded": self.model_loaded,
            "n_features": len(self.model.coef_),
            "intercept": float(self.model.intercept_),
            "coefficients_count": len(self.model.coef_)
        }


# Create a singleton instance
predictor = None


def get_predictor() -> ManufacturingPredictor:
    """Get or create the predictor instance."""
    global predictor
    if predictor is None:
        predictor = ManufacturingPredictor()
    return predictor


def make_prediction(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to make a prediction.
    
    Args:
        input_data: Dictionary containing all required features
    
    Returns:
        Dictionary containing prediction results
    """
    pred = get_predictor()
    return pred.predict(input_data)
