"""
Batch Training Pipeline
=======================
Pipeline responsável pelo treinamento batch do modelo
de precificação dinâmica.
"""

from pathlib import Path
import logging

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from feature_store import build_features
from models import build_model


# ------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Constants & Paths
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "brazilian-ecommerce" / "olist_order_items_dataset.csv"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "model.joblib"

TEST_SIZE = 0.2
RANDOM_STATE = 42
FREIGHT_PREMIUM = 0.05


# ------------------------------------------------------------------
# Core Functions
# ------------------------------------------------------------------
def load_dataset(path: Path) -> pd.DataFrame:
    """
    Carrega o dataset a partir do caminho informado.

    Parameters
    ----------
    path : Path
        Caminho do arquivo CSV.

    Returns
    -------
    pd.DataFrame
        Dataset carregado.
    """
    logger.info("Carregando dataset de %s", path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado em: {path}")

    return pd.read_csv(path)


def create_target(df: pd.DataFrame) -> pd.Series:
    """
    Cria a variável alvo de precificação dinâmica.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset original.

    Returns
    -------
    pd.Series
        Série com o target.
    """
    logger.info("Criando variável alvo")

    freight_median = df["freight_value"].median()
    target = df["price"] * (
        1 + (df["freight_value"] > freight_median) * FREIGHT_PREMIUM
    )

    return target


def train_model(X: pd.DataFrame, y: pd.Series):
    """
    Treina o modelo de Machine Learning.

    Parameters
    ----------
    X : pd.DataFrame
        Features de entrada.
    y : pd.Series
        Variável alvo.

    Returns
    -------
    object
        Modelo treinado.
    float
        MSE no conjunto de teste.
    """
    logger.info("Dividindo dados em treino e teste")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    logger.info("Inicializando modelo")
    model = build_model()

    logger.info("Treinando modelo")
    model.fit(X_train, y_train)

    logger.info("Avaliando modelo")
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    return model, mse


def save_model(model, path: Path) -> None:
    """
    Salva o modelo treinado em disco.

    Parameters
    ----------
    model : object
        Modelo treinado.
    path : Path
        Caminho de saída.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

    logger.info("Modelo salvo em %s", path)


# ------------------------------------------------------------------
# Pipeline Orchestration
# ------------------------------------------------------------------
def run_training_pipeline() -> None:
    """
    Executa o pipeline completo de treinamento.
    """
    logger.info("Iniciando pipeline de treinamento")

    df = load_dataset(DATA_PATH)

    y = create_target(df)
    X = build_features(df)

    model, mse = train_model(X, y)

    logger.info("MSE do modelo: %.4f", mse)

    save_model(model, MODEL_OUTPUT_PATH)

    logger.info("Pipeline finalizado com sucesso")


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------
if __name__ == "__main__":
    run_training_pipeline()
