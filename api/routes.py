from fastapi import APIRouter, HTTPException, Depends
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import os
from dotenv import load_dotenv
from src.optimization.hybrid_optimizer import optimize_energy
from api.schemas import PredictionRequest, PredictionResponse
from src.monitoring.model_monitor import PredictionLogger

load_dotenv()

router = APIRouter()
logger = logging.getLogger(__name__)

# Use absolute path or environment variable for production
ARTIFACT_PATH = Path(os.getenv("MODEL_PATH", "artifacts")).resolve()

# Initialize prediction logger
predictor_logger = PredictionLogger()

# Load models with error handling
solar_model = None
wind_model = None

try:
    solar_model = joblib.load(ARTIFACT_PATH / "solar_model.pkl")
    logger.info("Solar model loaded successfully")
except FileNotFoundError as e:
    logger.error(f"Solar model not found at {ARTIFACT_PATH / 'solar_model.pkl'}: {e}")
except Exception as e:
    logger.error(f"Error loading solar model: {e}")

try:
    wind_model = joblib.load(ARTIFACT_PATH / "wind_model.pkl")
    logger.info("Wind model loaded successfully")
except FileNotFoundError as e:
    logger.error(f"Wind model not found at {ARTIFACT_PATH / 'wind_model.pkl'}: {e}")
except Exception as e:
    logger.error(f"Error loading wind model: {e}")


@router.get("/health")
def health_check():
    """Check if models are loaded and API is healthy"""
    return {
        "status": "healthy" if solar_model and wind_model else "unhealthy",
        "models_loaded": {
            "solar": solar_model is not None,
            "wind": wind_model is not None
        }
    }


@router.get("/predict", response_model=PredictionResponse)
def predict_energy(request: PredictionRequest = Depends()):
    """
    Predict energy output from solar and wind sources.
    
    Pydantic automatically validates all input ranges.
    """
    
    # Validate model availability
    if solar_model is None or wind_model is None:
        logger.error("Models not loaded")
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. API is unavailable."
        )
    
    try:
        # Create DataFrames to maintain feature names (required for some XGBoost/SKLearn versions)
        solar_features = pd.DataFrame([{
            "IRRADIATION": request.irradiation,
            "AMBIENT_TEMPERATURE": request.temperature,
            "MODULE_TEMPERATURE": request.module,
            "hour": request.hour,
            "day": request.day,
            "month": request.month
        }])
        
        wind_features = pd.DataFrame([{
            "Wind Speed (m/s)": request.wind_speed,
            "Wind Direction (°)": request.direction,
            "Theoretical_Power_Curve (KWh)": request.theoretical
        }])

        # Get predictions
        solar = float(solar_model.predict(solar_features)[0])
        wind = float(wind_model.predict(wind_features)[0])

        # Ensure predictions are valid numbers
        if np.isnan(solar) or np.isnan(wind):
            logger.error(f"Model returned NaN: solar={solar}, wind={wind}")
            raise HTTPException(
                status_code=500,
                detail="Model returned invalid prediction (NaN)"
            )
        
        if np.isinf(solar) or np.isinf(wind):
            logger.error(f"Model returned Inf: solar={solar}, wind={wind}")
            raise HTTPException(
                status_code=500,
                detail="Model returned invalid prediction (Inf)"
            )

        # Get optimization result
        result = optimize_energy(solar, wind)
        
        # Log prediction for monitoring
        predictor_logger.log_prediction(
            inputs={
                "irradiation": request.irradiation,
                "temperature": request.temperature,
                "module": request.module,
                "hour": request.hour,
                "day": request.day,
                "month": request.month,
                "wind_speed": request.wind_speed,
                "direction": request.direction,
                "theoretical": request.theoretical
            },
            solar=solar,
            wind=wind,
            recommendation=result["recommended_source"]
        )
        
        logger.info(f"Prediction: solar={solar:.2f}, wind={wind:.2f}, recommendation={result['recommended_source']}")
        
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal error"
        )