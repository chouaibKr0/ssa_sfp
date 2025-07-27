"""
Simple cross-validation utilities for model evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from sklearn.model_selection import (
    KFold, StratifiedKFold, RepeatedStratifiedKFold,
    cross_validate
)
from sklearn.base import BaseEstimator


def get_cv_splitter(cv_config: Dict[str, Any]):
    """
    Create a CV splitter from config.
    
    Args:
        cv_config: CV configuration from YAML
        
    Returns:
        CV splitter object
    """
    method = cv_config.get('method', 'stratified_k_fold').lower()
    n_splits = cv_config.get('n_splits', 5)
    shuffle = cv_config.get('shuffle', True)
    random_state = cv_config.get('random_state', 42)
    
    if method == 'stratified_k_fold' or method == 'stratified':
        return StratifiedKFold(
            n_splits=n_splits, 
            shuffle=shuffle, 
            random_state=random_state
        )
    
    elif method == 'k_fold':
        return KFold(
            n_splits=n_splits, 
            shuffle=shuffle, 
            random_state=random_state
        )
    
    elif method == 'repeated_stratified_k_fold' or method == 'repeated_stratified':
        n_repeats = cv_config.get('n_repeats', 10)
        return RepeatedStratifiedKFold(
            n_splits=n_splits, 
            n_repeats=n_repeats,
            random_state=random_state
        )
    
    else:
        return StratifiedKFold(
            n_splits=n_splits, 
            shuffle=shuffle, 
            random_state=random_state
        )

def evaluate_model_cv(model: BaseEstimator,
                     X: Union[np.ndarray, pd.DataFrame],
                     y: np.ndarray,
                     cv_config: Dict[str, Any],
                     scoring: Union[str, list] = 'roc_auc') -> Dict[str, Any]:
    """
    Evaluate model using cross-validation.
    
    Args:
        model: Model to evaluate
        X: Features
        y: Target
        cv_config: CV configuration
        scoring: Scoring metrics
        
    Returns:
        CV results dictionary
    """
    # Convert DataFrame to numpy if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    # Get CV splitter
    cv_splitter = get_cv_splitter(cv_config)
    
    # Run CV
    cv_results = cross_validate(
        estimator=model,
        X=X, y=y,
        cv=cv_splitter,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Process results
    processed_results = {}
    
    # Handle single or multiple metrics
    if isinstance(scoring, str):
        metrics = [scoring]
    else:
        metrics = scoring
    
    for metric in metrics:
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        
        processed_results[f'{metric}_test_mean'] = float(np.mean(test_scores))
        processed_results[f'{metric}_test_std'] = float(np.std(test_scores))
        processed_results[f'{metric}_train_mean'] = float(np.mean(train_scores))
        processed_results[f'{metric}_train_std'] = float(np.std(train_scores))
        processed_results[f'{metric}_test_scores'] = test_scores.tolist()
    
    # Add timing info
    processed_results['fit_time_mean'] = float(np.mean(cv_results['fit_time']))
    processed_results['n_splits'] = len(cv_results['fit_time'])
    
    return processed_results

def compare_models_cv(models: Dict[str, BaseEstimator],
                     X: Union[np.ndarray, pd.DataFrame],
                     y: np.ndarray,
                     cv_config: Dict[str, Any],
                     scoring: str = 'roc_auc') -> pd.DataFrame:
    """
    Compare multiple models using CV.
    
    Args:
        models: Dictionary of models to compare
        X: Features
        y: Target
        cv_config: CV configuration
        scoring: Primary scoring metric
        
    Returns:
        Comparison results as DataFrame
    """
    results = []
    
    for model_name, model in models.items():
        
        cv_results = evaluate_model_cv(model, X, y, cv_config, scoring)
        
        results.append({
            'model': model_name,
            f'{scoring}_mean': cv_results[f'{scoring}_test_mean'],
            f'{scoring}_std': cv_results[f'{scoring}_test_std'],
            'fit_time': cv_results['fit_time_mean'],
            'n_splits': cv_results['n_splits']
        })
    
    df = pd.DataFrame(results)
    
    # Sort by primary metric (descending)
    df = df.sort_values(f'{scoring}_mean', ascending=False)
    
    return df
