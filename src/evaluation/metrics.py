
from src.utils import load_config


base_config = load_config("config/base_config.yaml")
def get_primary_metric() -> str:
    """
    Get the primary evaluation metric from the configuration.
    
    Returns:
        Primary metric name as a string
    """
    return base_config.get('metrics', {}).get('primary', 'roc_auc')

def get_secondary_metrics() -> list:
    """
    Get the secondary evaluation metrics from the configuration.
    
    Returns:
        List of secondary metric names
    """
    return base_config.get('metrics', {}).get('secondary', ['matthews_corrcoef', 'f1_macro', 'precision_macro', 'recall_macro'])