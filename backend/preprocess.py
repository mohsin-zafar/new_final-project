"""
Manufacturing Equipment Output Prediction - Preprocessing Module
================================================================
This module handles data preprocessing for the prediction pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
import joblib
import os

# Get the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')


def load_scaler():
    """Load the StandardScaler from saved file."""
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    return joblib.load(scaler_path)


def load_label_encoders():
    """Load the label encoders from saved file."""
    encoders_path = os.path.join(MODEL_DIR, 'label_encoders.pkl')
    if not os.path.exists(encoders_path):
        raise FileNotFoundError(f"Label encoders file not found at {encoders_path}")
    return joblib.load(encoders_path)


def load_feature_names():
    """Load the feature names from saved file."""
    features_path = os.path.join(MODEL_DIR, 'feature_names.pkl')
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature names file not found at {features_path}")
    return joblib.load(features_path)


def encode_categorical_features(data: Dict[str, Any], label_encoders: Dict) -> Dict[str, Any]:
    """
    Encode categorical features using saved label encoders.
    
    Args:
        data: Dictionary containing input features
        label_encoders: Dictionary of label encoders for each categorical column
    
    Returns:
        Dictionary with encoded categorical features
    """
    encoded_data = data.copy()
    
    # Categorical columns that need encoding
    categorical_cols = ['Shift', 'Machine_Type', 'Material_Grade', 'Day_of_Week']
    
    for col in categorical_cols:
        if col in encoded_data and col in label_encoders:
            le = label_encoders[col]
            value = str(encoded_data[col])
            
            # Check if value is in the encoder's classes
            if value in le.classes_:
                encoded_data[col] = le.transform([value])[0]
            else:
                # Handle unknown values - use the first class
                print(f"Warning: Unknown value '{value}' for {col}. Using default.")
                encoded_data[col] = 0
    
    return encoded_data


def preprocess_input(input_data: Dict[str, Any]) -> np.ndarray:
    """
    Preprocess input data for model prediction.
    
    Args:
        input_data: Dictionary containing all input features
    
    Returns:
        Numpy array of preprocessed and scaled features
    """
    # Load necessary components
    scaler = load_scaler()
    label_encoders = load_label_encoders()
    feature_names = load_feature_names()
    
    # Encode categorical features
    encoded_data = encode_categorical_features(input_data, label_encoders)
    
    # Create feature array in correct order
    features = []
    for feature in feature_names:
        if feature in encoded_data:
            features.append(encoded_data[feature])
        else:
            raise ValueError(f"Missing feature: {feature}")
    
    # Convert to numpy array and reshape
    features_array = np.array(features).reshape(1, -1)
    
    # Scale features
    scaled_features = scaler.transform(features_array)
    
    return scaled_features


def validate_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate input data ranges and types.
    
    Args:
        input_data: Dictionary containing input features
    
    Returns:
        Validated and cleaned input data
    """
    validated_data = {}
    
    # Define expected feature ranges
    feature_ranges = {
        'Injection_Temperature': (190, 260),
        'Injection_Pressure': (80, 160),
        'Cycle_Time': (15, 50),
        'Cooling_Time': (5, 20),
        'Material_Viscosity': (100, 500),
        'Ambient_Temperature': (15, 35),
        'Machine_Age': (0, 15),
        'Operator_Experience': (0, 100),
        'Maintenance_Hours': (30, 80),
        'Temperature_Pressure_Ratio': (1.0, 3.0),
        'Total_Cycle_Time': (20, 70),
        'Efficiency_Score': (0, 1),
        'Machine_Utilization': (0.1, 0.8)
    }
    
    # Categorical value options
    categorical_options = {
        'Shift': ['Day', 'Evening', 'Night'],
        'Machine_Type': ['Type_A', 'Type_B', 'Type_C'],
        'Material_Grade': ['Economy', 'Standard', 'Premium'],
        'Day_of_Week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    }
    
    for key, value in input_data.items():
        validated_data[key] = value
    
    return validated_data


# Feature descriptions for documentation
FEATURE_DESCRIPTIONS = {
    'Injection_Temperature': 'Temperature during injection process (°C)',
    'Injection_Pressure': 'Pressure during injection (bar)',
    'Cycle_Time': 'Total cycle time (seconds)',
    'Cooling_Time': 'Cooling phase duration (seconds)',
    'Material_Viscosity': 'Viscosity of material used',
    'Ambient_Temperature': 'Surrounding temperature (°C)',
    'Machine_Age': 'Age of the machine (years)',
    'Operator_Experience': 'Operator experience level (years)',
    'Maintenance_Hours': 'Hours since last maintenance',
    'Shift': 'Work shift (Day/Evening/Night)',
    'Machine_Type': 'Type of machine (Type_A/Type_B)',
    'Material_Grade': 'Grade of material (Economy/Standard/Premium)',
    'Day_of_Week': 'Day of the week',
    'Temperature_Pressure_Ratio': 'Ratio of temperature to pressure',
    'Total_Cycle_Time': 'Total time for complete cycle',
    'Efficiency_Score': 'Efficiency score (0-1)',
    'Machine_Utilization': 'Machine utilization rate (0-1)'
}
