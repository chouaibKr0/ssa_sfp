"""
Simple SVM wrapper.
"""

from typing import Dict, Any
from sklearn.svm import SVC
from .base_model import BaseModel
from utils import load_config
svm_config = load_config("config/svm_config.yaml")
class SVMWrapper(BaseModel):
    """Simple SVM wrapper."""
    
    def _create_model(self, **params) -> SVC:
        # Always enable probability
        params['probability'] = svm_config.get('probability', True)
        params['random_state'] = svm_config.get('random_state', 42)
        params['class_weight'] = svm_config.get('class_weight', 'balanced')
        return SVC(**params)
    
    
