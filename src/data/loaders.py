
"""
Data loading utilities for different data formats and sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List, Union
from ..utils import get_project_root

class DatasetLoader:
    """Main class for loading datasets with consistent interface."""
    
    def __init__(self, data_dir: Union[str, Path] = "data/PROMISE/interim"):
        self.data_dir = get_project_root()/data_dir

    def load_csv_dataset(self, 
                        file_name: Union[str, Path], 
                        target_column: Optional[str] = None,
                        **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load CSV dataset and separate features from target.
        
        Args:
            filepath: Path to CSV file
            target_column: Name of target column (if None, assumes last column)
            **kwargs: Additional arguments for pd.read_csv()
            
        Returns:
            Tuple of (X, y) where X is features DataFrame and y is target Series
        """
        filepath = Path(self.data_dir / file_name)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")
            
        
        # Load CSV with default arguments that work well for most cases
        default_kwargs = {
            'index_col': None,
            'header': 0,
            'encoding': 'utf-8'
        }
        default_kwargs.update(kwargs)
        
        try:
            df = pd.read_csv(filepath, **default_kwargs)
        except Exception as e:
            raise ValueError(f"Error loading CSV file {filepath}: {e}")
        
        # Separate features and target
        if target_column is None:
            # Assume last column is target
            X = df.iloc[:, :-1].copy()
            y = df.iloc[:, -1].copy()
            target_column = df.columns[-1]
        else:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")
            X = df.drop(columns=[target_column]).copy()
            y = df[target_column].copy()
        
        return X, y
    
    def load_multiple_datasets(self, 
                              dataset_configs: List[Dict[str, Any]]) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Load multiple datasets for comparison studies.
        
        Args:
            dataset_configs: List of dictionaries with dataset configurations
                            Each dict should have 'name', 'filepath', and optionally 'target_column'
                            
        Returns:
            Dictionary mapping dataset names to (X, y) tuples
            
        Example:
            configs = [
                {'name': 'iris', 'filepath': 'data/iris.csv', 'target_column': 'species'},
                {'name': 'wine', 'filepath': 'data/wine.csv'}  # assumes last column is target
            ]
        """
        datasets = {}
        
        for config in dataset_configs:
            name = config['name']
            filepath = config['filepath']
            target_column = config.get('target_column', None)
            
            try:
                X, y = self.load_csv_dataset(filepath, target_column)
                datasets[name] = (X, y)
            except Exception as e:
                continue
                
        return datasets
    
    def get_sample_data(self, dataset_name: str = "sample") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load sample dataset for testing and development.
        
        Args:
            dataset_name: Name of sample dataset
            
        Returns:
            Sample (X, y) data
        """
        sample_path = self.data_dir / "sample" / "processed" / f"{dataset_name}.csv"
        
        if not sample_path.exists():
            # Create a simple synthetic dataset if no sample exists
            return self._create_synthetic_data()
        
        return self.load_csv_dataset(sample_path)
    
    def _create_synthetic_data(self, n_samples: int = 1000, n_features: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
        """Create synthetic dataset for testing."""
        from sklearn.datasets import make_classification
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=3,
            n_informative=n_features//2,
            n_redundant=n_features//4,
            random_state=42
        )
        
        # Convert to DataFrame with meaningful column names
        feature_names = [f'feature_{i+1}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        return X_df, y_series

# Convenience functions for backward compatibility with your utils.py
def load_csv_dataset(filepath: Union[str, Path], 
                    target_column: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Convenience function that matches your utils.py interface."""
    loader = DatasetLoader()
    return loader.load_csv_dataset(filepath, target_column)

def load_multiple_datasets(dataset_configs: List[Dict[str, Any]]) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """Convenience function for loading multiple datasets."""
    loader = DatasetLoader()
    return loader.load_multiple_datasets(dataset_configs)
