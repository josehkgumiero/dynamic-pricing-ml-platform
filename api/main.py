"""
API Main Module
===============
API de inferência para o modelo de precificação dinâmica.
"""

from pathlib import Path
import logging

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from feature_store import build_features


# ------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Path Resolution
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"


# ------------------------------------------------------------------
# App Initialization
# ------------------------------------------------------------------
app = FastAPI(title="Dynamic Pricing API")


# ------------------------------------------------------------------
# Model Loading
# ------------------------------------------------------------------
def load_model():
    """
    Carrega o modelo treinado do disco.
    """
    logger.info("Carregando modelo de %s", MODEL_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em: {MODEL_PATH}. "
            "Execute o pipeline de treinamento antes de iniciar a API."
        )

    return joblib.load(MODEL_PATH)


model = load_model()


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.post("/predict")
def predict(payload: dict):
    """
    Realiza predição de preço com base nos dados de entrada.
    """
    try:
        df = pd.DataFrame([payload])
        features = build_features(df)

        prediction = model.predict(features)[0]

        return {"predicted_price": float(prediction)}

    except Exception as error:
        logger.error("Erro durante predição", exc_info=True)
        raise HTTPException(status_code=400, detail=str(error))
