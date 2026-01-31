"""
Feature Store Package
=====================
Responsável pela criação e padronização de features
utilizadas nos pipelines de treino e inferência.
"""

from .build_features import build_features

__all__ = ["build_features"]
