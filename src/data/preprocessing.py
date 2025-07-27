"""
Data preprocessing utilities for cleaning and preparing datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from utils import load_config


preprocessing_config = load_config('config/data/preprocessing_config.yaml')

feature_selection_config = preprocessing_config.get('feature_selection', {})
scaling_config = preprocessing_config.get('scaler', {})
missing_value_config = preprocessing_config.get('missing_values', {})

class DataPreprocessor:
    """Main class for data preprocessing operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Preprocessing configuration dictionary
        """
        self.config = config or {}
        self.fitted_transformers = {}
        self.label_encoders = {}
        
    def reduce_dimensionality(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce dimensionality of features based on configuration.
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with reduced features
        """
        selected_features = feature_selection_config.get('selected_features', [])
        if not selected_features:
            return X
        
        # Select only specified features
        X_reduced = X[selected_features]
        
        return X_reduced

    def handle_missing_values(self, 
                            X: pd.DataFrame,
                            categorical_strategy: str = 'most_frequent') -> pd.DataFrame:
        """
        Handle missing values in features.
        
        Args:
            X: Feature matrix
            categorical_strategy: Strategy for categorical features
            
        Returns:
            DataFrame with missing values handled
        """
        strategy = missing_value_config.get('strategy', 'mean')
        if X.isnull().sum().sum() == 0:
            return X
        
        
        X_imputed = X.copy()
        
        # Separate numeric and categorical columns
        numeric_cols = X_imputed.select_dtypes(include=[np.number]).columns
        categorical_cols = X_imputed.select_dtypes(exclude=[np.number]).columns
        
        # Handle numeric missing values
        if len(numeric_cols) > 0 and X_imputed[numeric_cols].isnull().any().any():
            if 'numeric_imputer' not in self.fitted_transformers:
                self.fitted_transformers['numeric_imputer'] = SimpleImputer(strategy=strategy)
                X_imputed[numeric_cols] = self.fitted_transformers['numeric_imputer'].fit_transform(X_imputed[numeric_cols])
            else:
                X_imputed[numeric_cols] = self.fitted_transformers['numeric_imputer'].transform(X_imputed[numeric_cols])
        
        # Handle categorical missing values
        if len(categorical_cols) > 0 and X_imputed[categorical_cols].isnull().any().any():
            if 'categorical_imputer' not in self.fitted_transformers:
                self.fitted_transformers['categorical_imputer'] = SimpleImputer(strategy=categorical_strategy)
                X_imputed[categorical_cols] = self.fitted_transformers['categorical_imputer'].fit_transform(X_imputed[categorical_cols])
            else:
                X_imputed[categorical_cols] = self.fitted_transformers['categorical_imputer'].transform(X_imputed[categorical_cols])
        
        return X_imputed
    
    def encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using label encoding.
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with encoded categorical features
        """
        X_encoded = X.copy()
        categorical_cols = X_encoded.select_dtypes(exclude=[np.number]).columns
        
        if len(categorical_cols) == 0:
            return X_encoded
        
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X_encoded[col] = self.label_encoders[col].fit_transform(X_encoded[col].astype(str))
            else:
                X_encoded[col] = self.label_encoders[col].transform(X_encoded[col].astype(str))
        
        return X_encoded
    def encode_label(self, y: pd.Series) -> pd.Series:
        """
        Encode categorical features using label encoding.
        
        Args:
            y: Target vector
        Returns:
            Series with encoded labels
        """
        y_encoded = y.copy()
        if y_encoded.dtype == 'object':
            y_encoded = y_encoded.astype(str)
            y_encoded = self.label_encoders['target'].fit_transform(y_encoded)
        return y_encoded
        
    def scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with scaled features
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return X
        
        
        X_scaled = X.copy()
        method = scaling_config.get('method', 'standard')
        # Choose scaler
        scaler_key = f'{method}_scaler'
        if scaler_key not in self.fitted_transformers:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            self.fitted_transformers[scaler_key] = scaler
            X_scaled[numeric_cols] = scaler.fit_transform(X_scaled[numeric_cols])
        else:
            X_scaled[numeric_cols] = self.fitted_transformers[scaler_key].transform(X_scaled[numeric_cols])
        
        return X_scaled
    
    def preprocess_pipeline(self, 
                           X: pd.DataFrame, 
                           y: pd.Series = None,
                           fit: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete preprocessing pipeline.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            fit: Whether to fit transformers (True for training, False for test)
            
        Returns:
            Preprocessed (X, y) data
        """
        

        # Step 1: Dimensionality reduction
        if feature_selection_config.get('enabled', True):
            X_processed = self.reduce_dimensionality(X)

        # Step 2: Handle missing values
        X_processed = self.handle_missing_values(X_processed)

        # Step 3: Encode categorical features
        X_processed = self.encode_categorical_features(X_processed)

        # Step 4: Encode target if provided
        if y is not None:
            y_processed = self.encode_label(y)

        # Step 5: Scale features
        scaling_method = self.config.get('scaling_method', 'standard')
        X_processed = self.scale_features(X_processed, scaling_method)

        if y is not None:
            y_processed = self.encode_label(y)
            return X_processed, y_processed

        return X_processed, y
