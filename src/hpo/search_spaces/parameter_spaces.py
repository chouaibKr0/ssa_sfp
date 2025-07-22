import optuna 

def get_rf_search_space(trial):
    """Generate RandomForest continuous hyperparameter search space"""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 10, 1000, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 50) if trial.suggest_float('max_depth_choice', 0, 1) > 0.2 else None,
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.2),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 1000) if trial.suggest_float('max_leaf_nodes_choice', 0, 1) > 0.7 else None
    }

def get_svm_search_space(trial):
    """Generate SVM continuous hyperparameter search space"""
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
    
    # Base parameters
    params = {
        'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
        'kernel': kernel
    }
    
    # Kernel-specific parameters
    if kernel in ['rbf', 'poly', 'sigmoid']:
        params['gamma'] = trial.suggest_float('gamma', 1e-5, 1e2, log=True)
    
    if kernel in ['poly', 'sigmoid']:
        params['coef0'] = trial.suggest_float('coef0', -1.0, 1.0)
    
    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 6)
    
    return params

def get_mlp_search_space(trial):
    """Generate MLP continuous hyperparameter search space"""
    return {
        'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [
            (50,), (100,), (200,), (500,),
            (50, 25), (100, 50), (200, 100), (500, 250),
            (100, 50, 25), (200, 100, 50), (500, 250, 125)
        ]),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-5, 1e-1, log=True),
        'alpha': trial.suggest_float('alpha', 1e-6, 1e-1, log=True),
        'batch_size': trial.suggest_int('batch_size', 16, 512, log=True),
        'beta_1': trial.suggest_float('beta_1', 0.8, 0.99),
        'beta_2': trial.suggest_float('beta_2', 0.9, 0.9999),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
        'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs', 'sgd']),
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive'])
    }

# Alternative: If you prefer dictionary-based approach for compatibility
def get_rf_param_distributions():
    """Get RandomForest parameter distributions for RandomizedSearchCV"""
    from scipy.stats import uniform, loguniform, randint
    
    return {
        'n_estimators': loguniform(10, 1000),
        'max_depth': [None] + list(range(3, 51)),  # Mix of None and integers
        'min_samples_split': randint(2, 21),
        'min_samples_leaf': randint(1, 21),
        'max_features': uniform(0.1, 0.9),
        'bootstrap': [True, False],
        'min_impurity_decrease': uniform(0.0, 0.2)
    }

def get_svm_param_distributions():
    """Get SVM parameter distributions for RandomizedSearchCV"""
    from scipy.stats import uniform, loguniform
    
    return {
        'C': loguniform(1e-3, 1e3),
        'gamma': loguniform(1e-5, 1e2),
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'degree': [2, 3, 4, 5, 6],
        'coef0': uniform(-1.0, 2.0)
    }

def get_mlp_param_distributions():
    """Get MLP parameter distributions for RandomizedSearchCV"""
    from scipy.stats import uniform, loguniform
    
    return {
        'hidden_layer_sizes': [
            (50,), (100,), (200,), (500,),
            (50, 25), (100, 50), (200, 100), (500, 250),
            (100, 50, 25), (200, 100, 50), (500, 250, 125)
        ],
        'learning_rate_init': loguniform(1e-5, 1e-1),
        'alpha': loguniform(1e-6, 1e-1),
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'lbfgs', 'sgd'],
        'learning_rate': ['constant', 'invscaling', 'adaptive']
    }
