"""
Manufacturing Equipment Output Prediction - FastAPI Backend
============================================================
This is the main entry point for the FastAPI backend server.
It provides REST API endpoints for manufacturing output predictions.

API Endpoints:
- GET  /          : Health check
- GET  /health    : API health status
- POST /predict   : Make prediction
- GET  /model/info: Get model information
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

from predict import get_predictor, make_prediction

# ============================================
# FASTAPI APP INITIALIZATION
# ============================================
app = FastAPI(
    title="Manufacturing Output Prediction API",
    description="API for predicting Parts Per Hour based on manufacturing parameters",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================
# CORS MIDDLEWARE
# ============================================
# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# PYDANTIC MODELS (REQUEST/RESPONSE SCHEMAS)
# ============================================

class PredictionInput(BaseModel):
    """Input schema for prediction endpoint."""
    Injection_Temperature: float = Field(..., ge=150, le=300, description="Injection temperature in °C")
    Injection_Pressure: float = Field(..., ge=50, le=200, description="Injection pressure in bar")
    Cycle_Time: float = Field(..., ge=10, le=60, description="Cycle time in seconds")
    Cooling_Time: float = Field(..., ge=3, le=30, description="Cooling time in seconds")
    Material_Viscosity: float = Field(..., ge=50, le=600, description="Material viscosity")
    Ambient_Temperature: float = Field(..., ge=10, le=45, description="Ambient temperature in °C")
    Machine_Age: float = Field(..., ge=0, le=20, description="Machine age in years")
    Operator_Experience: float = Field(..., ge=0, le=100, description="Operator experience in years")
    Maintenance_Hours: float = Field(..., ge=0, le=100, description="Hours since maintenance")
    Shift: str = Field(..., description="Work shift: Day, Evening, or Night")
    Machine_Type: str = Field(..., description="Machine type: Type_A, Type_B, or Type_C")
    Material_Grade: str = Field(..., description="Material grade: Economy, Standard, or Premium")
    Day_of_Week: str = Field(..., description="Day of the week")
    Temperature_Pressure_Ratio: float = Field(..., ge=0.5, le=5, description="Temperature to pressure ratio")
    Total_Cycle_Time: float = Field(..., ge=15, le=100, description="Total cycle time in seconds")
    Efficiency_Score: float = Field(..., ge=0, le=1, description="Efficiency score (0-1)")
    Machine_Utilization: float = Field(..., ge=0, le=1, description="Machine utilization rate (0-1)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "Injection_Temperature": 220.0,
                "Injection_Pressure": 130.0,
                "Cycle_Time": 30.0,
                "Cooling_Time": 12.0,
                "Material_Viscosity": 300.0,
                "Ambient_Temperature": 25.0,
                "Machine_Age": 5.0,
                "Operator_Experience": 10.0,
                "Maintenance_Hours": 50.0,
                "Shift": "Day",
                "Machine_Type": "Type_A",
                "Material_Grade": "Standard",
                "Day_of_Week": "Monday",
                "Temperature_Pressure_Ratio": 1.7,
                "Total_Cycle_Time": 42.0,
                "Efficiency_Score": 0.5,
                "Machine_Utilization": 0.45
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    success: bool
    predicted_parts_per_hour: Optional[float]
    message: str


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    message: str
    version: str


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""
    model_type: str
    model_loaded: bool
    n_features: int
    intercept: float
    coefficients_count: int


# ============================================
# API ENDPOINTS
# ============================================

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """
    Root endpoint - Health check.
    Returns basic API status information.
    """
    return {
        "status": "healthy",
        "message": "Manufacturing Output Prediction API is running",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Verifies that the API and model are operational.
    """
    try:
        # Try to load the predictor to verify model is available
        predictor = get_predictor()
        return {
            "status": "healthy",
            "message": "API is running and model is loaded",
            "version": "1.0.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Model loading issue: {str(e)}",
            "version": "1.0.0"
        }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(input_data: PredictionInput):
    """
    Make a prediction for Parts Per Hour.
    
    Takes manufacturing parameters as input and returns
    the predicted parts per hour output.
    """
    try:
        # Convert Pydantic model to dictionary
        input_dict = input_data.model_dump()
        
        # Make prediction
        result = make_prediction(input_dict)
        
        if result["success"]:
            return {
                "success": True,
                "predicted_parts_per_hour": result["predicted_parts_per_hour"],
                "message": "Prediction successful"
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=result["message"]
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """
    Get model information.
    Returns details about the loaded model.
    """
    try:
        predictor = get_predictor()
        info = predictor.get_model_info()
        return info
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )


# ============================================
# STARTUP EVENT
# ============================================

@app.on_event("startup")
async def startup_event():
    """
    Startup event handler.
    Pre-loads the model when the server starts.
    """
    print("=" * 50)
    print("Starting Manufacturing Output Prediction API...")
    print("=" * 50)
    try:
        predictor = get_predictor()
        print("✓ Model pre-loaded successfully")
    except Exception as e:
        print(f"⚠ Warning: Could not pre-load model: {e}")
    print("=" * 50)


# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
