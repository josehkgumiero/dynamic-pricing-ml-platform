"""
Gradient Boosting Model Factory
===============================
Define a fábrica de modelos baseados em XGBoost
para tarefas de regressão.
"""

from typing import Optional, Dict, Any
import logging

from xgboost import XGBRegressor


# ------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Default Hyperparameters
# ------------------------------------------------------------------
DEFAULT_PARAMS: Dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}


# ------------------------------------------------------------------
# Model Factory
# ------------------------------------------------------------------
def build_model(
    params: Optional[Dict[str, Any]] = None
) -> XGBRegressor:
    """
    Cria e retorna um modelo XGBRegressor configurado.

    Parameters
    ----------
    params : dict, optional
        Dicionário de hiperparâmetros para sobrescrever
        os valores padrão.

    Returns
    -------
    XGBRegressor
        Modelo de regressão configurado.
    """
    model_params = DEFAULT_PARAMS.copy()

    if params:
        model_params.update(params)

    logger.info(
        "Inicializando XGBRegressor com parâmetros: %s",
        model_params
    )

    return XGBRegressor(**model_params)
