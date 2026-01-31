"""
Feature Engineering Module
==========================
Responsável pela criação e padronização de features
utilizadas nos pipelines de treino e inferência.
"""

from typing import List
import logging

import pandas as pd


# ------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
REQUIRED_COLUMNS: List[str] = [
    "price",
    "freight_value",
]


FEATURE_COLUMNS: List[str] = [
    "price",
    "freight_value",
    "total_value",
    "high_freight",
]


# ------------------------------------------------------------------
# Core Function
# ------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features derivadas a partir do dataset bruto.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset contendo as colunas necessárias.

    Returns
    -------
    pd.DataFrame
        DataFrame contendo apenas as features utilizadas
        pelo modelo.

    Raises
    ------
    ValueError
        Caso colunas obrigatórias estejam ausentes.
    """
    logger.info("Iniciando processo de feature engineering")

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Colunas obrigatórias ausentes no dataset: {missing_columns}"
        )

    features_df = df.copy()

    logger.debug("Criando feature total_value")
    features_df["total_value"] = (
        features_df["price"] + features_df["freight_value"]
    )

    logger.debug("Criando feature high_freight")
    freight_median = features_df["freight_value"].median()
    features_df["high_freight"] = (
        features_df["freight_value"] > freight_median
    ).astype(int)

    logger.info("Feature engineering finalizado")

    return features_df[FEATURE_COLUMNS]
