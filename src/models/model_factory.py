"""
Simple model factory.
"""

from typing import Dict, Any
from .rf import RandomForestWrapper
from .svm import SVMWrapper
from .mlp import MLPWrapper

_MODELS = {
    'rf': RandomForestWrapper,
    'svm': SVMWrapper,
    'mlp': MLPWrapper,
}

def create_model(model_type: str, random_state: int = 42, **params):
    """
    Create a model instance.
    
    Args:
        model_type: Type of model ('random_forest', 'svm', 'neural_network')
        random_state: Random seed
        **params: Model parameters
        
    Returns:
        Model instance
    """
    model_type = model_type.lower()
    
    if model_type not in _MODELS:
        available = list(_MODELS.keys())
        raise ValueError(f"Unknown model: {model_type}. Available: {available}")
    
    model_class = _MODELS[model_type]
    model = model_class(random_state=random_state)
    
  
    model.set_params(**params)

        
    return model

def get_available_models():
    """Get list of available model types."""
    return list(_MODELS.keys())
