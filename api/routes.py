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
@router.get("/predict")
def predict_energy(
    irradiation: float,
    temperature: float,
    module: float,
    hour: int,
    day: int,
    month: int,
    wind_speed: float,
    direction: float,
    theoretical: float
):

    solar_features = np.array([[irradiation, temperature, module, hour, day, month]])
    wind_features = np.array([[wind_speed, direction, theoretical]])

    solar = float(solar_model.predict(solar_features)[0])
    wind = float(wind_model.predict(wind_features)[0])

    result = optimize_energy(solar, wind)

    return result