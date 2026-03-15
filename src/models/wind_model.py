from xgboost import XGBRegressor

def get_wind_model():
    model = XGBRegressor(
        n_estimators = 200,
        learning_rates = 0.05,
        max_depth = 6,
        random_state = 42
    )

    return model