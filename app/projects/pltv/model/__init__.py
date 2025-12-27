"""
Model Module
============

XGBoost-based prediction service for PLTV.

Classes:
    ModelService: Runs multi-step predictions across partitions (promo/non-promo).
                  Each step predicts avg_net_billings for a time horizon (30-730 days).

Usage:
    from projects.pltv import Level, ModelService
    
    service = ModelService(level, df)
    service.run()
    results = service.results  # List[ModelStepMetadata]
"""

from projects.pltv.model.model_service import ModelService

__all__ = ["ModelService"]
