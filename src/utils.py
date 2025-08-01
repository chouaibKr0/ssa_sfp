"""
Utility functions for ML hyperparameter optimization and statistical comparison project.
Focuses on simplicity and academic research needs.
"""

import os
import yaml
import json
import logging
import random
import numpy as np 
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import warnings
# Suppress common warnings to keep output clean
warnings.filterwarnings('ignore', category=FutureWarning)

def get_project_root() -> Path:
    """Get the root directory of the project."""
    return Path(__file__).parent.parent


def setup_logging(log_level: str = "INFO", log_dir: str = "experiments/logs") -> logging.Logger:
    """
    Setup simple logging for experiments.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory to save log files
    
    Returns:
        Configured logger
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('ml_experiment')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter - simple and readable
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = get_project_root() / Path(log_dir) / f"experiment_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

def load_config(config_relative_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file from the project root, regardless of current working directory.

    Args:
        config_relative_path: Path to YAML file, relative to the project root.

    Returns:
        Configuration dictionary.
    """
    project_root = get_project_root()  # This is a Path object
    config_path = project_root / config_relative_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {config_path}: {e}")

def merge_configs(*config_paths: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and merge multiple config files in order.
    Later configs override earlier ones.
    
    Args:
        *config_paths: Paths to config files to merge
        
    Returns:
        Merged configuration dictionary
        
    Example:
        config = merge_configs(
            'config/base_config.yaml',
            'config/models/svm_config.yaml', 
            'config/experiments/comparison_config.yaml'
        )
    """
    merged_config = {}
    
    for config_path in config_paths:
        if config_path and Path(config_path).exists():
            config = load_config(config_path)
            merged_config.update(config)
            
    return merged_config

def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Set sklearn random state if available
    try:
        from sklearn.utils import check_random_state
        check_random_state(seed)
    except ImportError:
        pass


def create_experiment_directories(experiment_name: str, base_dir: str = "results") -> Dict[str, Path]:
    """
    Create directory structure for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        base_dir: Base directory for results
        
    Returns:
        Dictionary with paths to created directories
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_group = load_config("config/base_config.yaml").get("current_experiment", "unspecified")
    experiment_dir = get_project_root() / Path(base_dir) / f"{experiment_group}" / f"{experiment_name}_{timestamp}"

    # Create subdirectories
    directories = ['raw_results', 'plots', 'models', 'logs']
    paths = {}
    
    for dir_name in directories:
        dir_path = experiment_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        paths[dir_name] = dir_path
    
    paths['experiment_dir'] = experiment_dir
    return paths

def save_results(results: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save experiment results to JSON file.
    
    Args:
        results: Results dictionary to save
        filepath: Path to save the results
    """
    filepath = get_project_root() / Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Convert all numpy types in the results
    json_results = json.loads(json.dumps(results, default=convert_numpy))
    
    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=2)

def load_results(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load experiment results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def load_csv_data(filepath: Union[str, Path], target_column: str = None) -> tuple:
    """
    Load CSV data and split into features and target.
    
    Args:
        filepath: Path to CSV file
        target_column: Name of target column. If None, assumes last column.
        
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    df = pd.read_csv(filepath)
    
    if target_column is None:
        # Assume last column is target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    else:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    
    return X, y

def get_dataset_info(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Get basic information about the dataset.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        Dictionary with dataset information
    """
    from collections import Counter
    
    info = {
        'n_samples': len(X),
        'n_features': X.shape[1],
        'feature_names': X.columns.tolist() if hasattr(X, 'columns') else None,
        'target_classes': sorted(list(set(y))),
        'class_distribution': dict(Counter(y)),
        'missing_values': X.isnull().sum().sum() if hasattr(X, 'isnull') else 0
    }
    
    return info

def create_experiment_id() -> str:
    """Create unique experiment ID based on timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds

def log_experiment_start(experiment_name: str, config: Dict[str, Any], logger: logging.Logger) -> str:
    """
    Log the start of an experiment with configuration details.
    
    Args:
        experiment_name: Name of the experiment
        config: Experiment configuration
        logger: Logger instance
        
    Returns:
        Experiment ID
    """
    experiment_id = create_experiment_id()
    
    logger.info("=" * 60)
    logger.info(f"STARTING EXPERIMENT: {experiment_name}")
    logger.info(f"Experiment ID: {experiment_id}")
    logger.info(f"Timestamp: {datetime.now()}")
    logger.info("=" * 60)
    
    # Log key configuration parameters
    logger.info("Configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")
    
    logger.info("-" * 60)
    
    return experiment_id

def log_experiment_end(experiment_id: str, results: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Log the end of an experiment with results summary.
    
    Args:
        experiment_id: Experiment ID
        results: Experiment results
        logger: Logger instance
    """
    logger.info("-" * 60)
    logger.info(f"EXPERIMENT COMPLETED: {experiment_id}")
    logger.info(f"Timestamp: {datetime.now()}")
    
    # Log key results
    logger.info("Results Summary:")
    for key, value in results.items():
        if isinstance(value, (int, float, str)):
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    
    logger.info("=" * 60)

def setup_mlflow_experiment(experiment_name: str, tracking_uri: str = "mlruns") -> None:
    """
    Setup MLflow experiment tracking (optional - only if MLflow is installed).
    
    Args:
        experiment_name: Name of the MLflow experiment
        tracking_uri: MLflow tracking URI
    """
    try:
        import mlflow
        
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(experiment_name)
        print(f"MLflow experiment '{experiment_name}' ready. Tracking URI: {tracking_uri}")
        
    except ImportError:
        print("MLflow not installed. Skipping MLflow setup.")

def validate_config_keys(config: Dict[str, Any], required_keys: List[str]) -> None:
    """
    Validate that config contains required keys.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        
    Raises:
        ValueError: If required keys are missing
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.2f} minutes"
    else:
        return f"{seconds/3600:.2f} hours"

# Convenience function to setup everything for an experiment
def setup_experiment(experiment_name: str, config_files: List[str]) -> tuple:
    """
    One-stop function to setup everything for an experiment.
    
    Args:
        experiment_name: Name of the experiment
        config_files: List of config file paths to merge
        
    Returns:
        Tuple of (config, logger, directories, experiment_id)
        
    Example:
        config, logger, dirs, exp_id = setup_experiment(
            "svm_metaheuristics_comparison",
            ["config/base_config.yaml", "config/models/svm_config.yaml", 
             "config/experiments/comparison_config.yaml"]
        )
    """
    # Load and merge configurations
    config = merge_configs(*config_files)
    
    # Set random seeds for reproducibility
    seed = config.get('random_state', 42)
    set_random_seeds(seed)
    
    # Setup logging
    log_level = config.get('log_level', 'INFO')
    logger = setup_logging(log_level)
    
    # Create experiment directories
    directories = create_experiment_directories(experiment_name)
    
    # Log experiment start
    experiment_id = log_experiment_start(experiment_name, config, logger)
    
    # Setup MLflow if config specifies it
    if config.get('use_mlflow', True):
        setup_mlflow_experiment(experiment_name, config.get('mlflow_tracking_uri', 'mlruns'))
    
    return config, logger, directories, experiment_id

def save_experiment(experiment_id: str, model_name: str, hpo_name: str, config: Dict[str, Any], results: Dict[str, Any], 
                   directories: Dict[str, Path], logger: logging.Logger) -> Dict[str, Path]:

    """
    Save complete experiment package: experiment_id, config, and results.

    Args:
        experiment_id: Unique experiment identifier
        config: Complete experiment configuration
        results: Experiment results dictionary
        directories: Dictionary of experiment directories from create_experiment_directories
        logger: Logger instance

    Returns:
        Dictionary with paths to saved files
    """
    
    # Create complete experiment package
    experiment_package = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().isoformat(),
        'experiment_name': directories['experiment_dir'].name,
        'dataset': config.get('dataset', 'unknown'),
        'model_name': model_name,
        'hpo_name': hpo_name,
        'config': config,
        'results': results
    }
    
    # Save to local files
    saved_files = {}
    
    # 1. Save complete package as single JSON
    package_path = directories['raw_results'] / f"experiment_package_{experiment_id}.json"
    save_results(experiment_package, package_path)
    saved_files['package'] = package_path
    
    # 2. Save individual components (for easier access)
    config_path = directories['raw_results'] / f"config_{experiment_id}.json"
    results_path = directories['raw_results'] / f"results_{experiment_id}.json"
    
    save_results(config, config_path)
    save_results(results, results_path)
    saved_files['config'] = config_path
    saved_files['results'] = results_path
    
    # 3. MLflow integration (if enabled)
    if load_config("config/base_config.yaml").get('use_mlflow', True):
        try:
            import mlflow
            
            with mlflow.start_run():
                # Log parameters from config
                if 'model_params' in config:
                    mlflow.log_params(config['model_params'])
                if 'experiment_params' in config:
                    mlflow.log_params(config['experiment_params'])
                
                # Log metrics from results
                metrics_to_log = {k: v for k, v in results.items() 
                                if isinstance(v, (int, float))}
                if metrics_to_log:
                    mlflow.log_metrics(metrics_to_log)
                
                # Log artifacts (the JSON files)
                mlflow.log_artifacts(str(directories['raw_results']))
                
        except ImportError:
            logger.warning("MLflow not available, skipping MLflow logging")
    
    # Log the save operation
    logger.info(f"Experiment saved: {experiment_id}")
    logger.info(f"Files saved: {list(saved_files.keys())}")
    
    return saved_files
    