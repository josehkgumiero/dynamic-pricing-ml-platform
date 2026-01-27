from xgboost import XGBRegressor

def build_model():
    return XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        objective="reg:squarederror"
    )