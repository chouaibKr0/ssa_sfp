"""
Simple base model class for HPO research.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from evaluation.metrics import get_primary_metric, get_secondary_metrics

class BaseModel(ABC):
    """Simple base class for model wrappers."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        
    @abstractmethod
    def _create_model(self, **params) -> BaseEstimator:
        """Create the sklearn model."""
        pass
    
    
    def set_params(self, **params) -> 'BaseModel':
        """Set model parameters."""
        # Add random_state if not present
        if 'random_state' not in params:
            params['random_state'] = self.random_state
            
        self.model = self._create_model(**params)
        return self
    
    def get_params(self) -> Dict[str, Any]:
        """Get current parameters."""
        if self.model is None:
            return self.get_default_params()
        return self.model.get_params()
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray):
        """Fit the model."""
        if self.model is None:
            self.set_params()
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        self.model.fit(X, y)
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict probabilities."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)

    def evaluate_cv(self, X, y, cv_config, scoring=get_primary_metric()):
        """Evaluate using cross-validation."""
        from ..evaluation.cross_validation import evaluate_model_cv
        
        if self.model is None:
            self.set_params()
            
        return evaluate_model_cv(self.model, X, y, cv_config, scoring)
    
    def evaluate_cv_full(self, X, y, cv_config, scoring=get_secondary_metrics()):
        """Evaluate using cross-validation."""
        from ..evaluation.cross_validation import evaluate_model_cv
        
        if self.model is None:
            self.set_params()
            
        return evaluate_model_cv(self.model, X, y, cv_config, scoring)
    
    def clone(self):
        """Create a copy of the model."""
        new_model = self.__class__(random_state=self.random_state)
        if self.model is not None:
            new_model.set_params(**self.get_params())
        return new_model
