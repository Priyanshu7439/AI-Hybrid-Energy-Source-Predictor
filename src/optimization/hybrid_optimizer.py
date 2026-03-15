import joblib
import numpy as np
from pathlib import Path

ARTIFACT_PATH = Path("artifacts")

solar_model = joblib.load(ARTIFACT_PATH / "solar_model.pkl")
wind_model = joblib.load(ARTIFACT_PATH / "wind_model.pkl")


def optimize_energy(solar, wind):

    total = solar + wind

    if solar > wind:
        best = "Solar"
    else:
        best = "Wind"

    return {
    "solar_power": float(solar),
    "wind_power": float(wind),
    "total_energy": float(total),
    "recommended_source": best
}


if __name__ == "__main__":

    solar_features = [800, 32, 40, 12, 15, 6]
    wind_features = [6.5, 250, 700]

    result = optimize_energy(solar_features, wind_features)

    print(result)