"""
Simple Neural Network wrapper.
"""

from typing import Dict, Any
from sklearn.neural_network import MLPClassifier
from .base_model import BaseModel
from ..utils import load_config

mlp_config = load_config("config/models/mlp_config.yaml")

class MLPWrapper(BaseModel):
    """Simple Neural Network wrapper."""
    
    def _create_model(self, **params) -> MLPClassifier:
        params['random_state'] = mlp_config.get('random_state', 42)
        params['max_iter'] = mlp_config.get('max_iter', 200)
        params['early_stopping'] = mlp_config.get('early_stopping', False)
        params['validation_fraction'] = mlp_config.get('validation_fraction', 0.1)
        params['n_iter_no_change'] = mlp_config.get('n_iter_no_change', 10)    
        return MLPClassifier(**params)
    
