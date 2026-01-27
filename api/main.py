from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

from feature_store.build_features import build_features

app = FastAPI(title="Dynamic Pricing API")

model = joblib.load("model.joblib")

class PricingRequest(BaseModel):
    price: float
    freight_value: float

@app.post("/predict")
def predict(data: PricingRequest):
    df = pd.DataFrame([data.dict()])
    features = build_features(df)
    prediction = model.predict(features)[0]

    return {"predicted_dynamic_price": float(prediction)}