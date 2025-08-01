from ...utils import load_config
from typing import Dict, Any
_MODELS = {
    'rf',
    'svm',
    'mlp'
}


def get_svm_search_space():
    svm_config = load_config('config/models/svm_config.yaml')

    return svm_config.get('search_space', {})

def get_rf_search_space(trial):
    rf_config = load_config('config/models/rf_config.yaml')

    return rf_config.get('search_space', {})


def get_mlp_search_space(trial):
    mlp_config = load_config('config/models/mlp_config.yaml')

    return mlp_config.get('search_space', {})

def get_search_space(model: str) -> Dict[str, Any]:
    """
    Returns the search space for the specified model.
    
    Args:
        model (str): The model type ('rf', 'svm', 'mlp').
    
    Returns:
        Dict[str, Any]: The search space configuration.
    
    Raises:
        ValueError: If the model is not supported.
    """
    if model == 'svm':
        return get_svm_search_space()
    elif model == 'rf':
        return get_rf_search_space()
    elif model == 'mlp':
        return get_mlp_search_space()
    else:
        raise ValueError(f"Unsupported model: {model}. Supported models are: {_MODELS}")