"""
Simple Random Forest wrapper.
"""

from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel
from utils import load_config

rf_config = load_config("config/models/rf_config.yaml")

class RandomForestWrapper(BaseModel):
    """Simple Random Forest wrapper."""
    
    def _create_model(self, **params) -> RandomForestClassifier:
        params['random_state'] = rf_config.get('random_state', 42)
        params['class_weight'] = rf_config.get('class_weight', 'balanced')
        params['n_jobs'] = rf_config.get('n_jobs', 1)
        return RandomForestClassifier(**params)
    
