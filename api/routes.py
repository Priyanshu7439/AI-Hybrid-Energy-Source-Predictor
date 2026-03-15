from fastapi import APIRouter
import joblib
import numpy as np
from pathlib import Path
from src.optimization.hybrid_optimizer import optimize_energy

router = APIRouter()

ARTIFACT_PATH = Path("artifacts")

solar_model = joblib.load(ARTIFACT_PATH / "solar_model.pkl")
wind_model = joblib.load(ARTIFACT_PATH / "wind_model.pkl")


@router.get("/predict")
def predict_energy():

    solar_features = np.array([[800, 32, 40, 12, 15, 6]])
    wind_features = np.array([[6.5, 250, 700]])

    solar = float(solar_model.predict(solar_features)[0])
    wind = float(wind_model.predict(wind_features)[0])

    result = optimize_energy(solar, wind)

    return result