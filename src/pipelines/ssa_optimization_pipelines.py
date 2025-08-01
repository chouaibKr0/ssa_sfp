from requests import get
from ..utils import load_config
from ..data.loaders import DatasetLoader
from ..data.preprocessing import DataPreprocessor
import pandas as pd
from ..hpo.optimizers.metaheuristics import SalpSwarmOptimizer
from ..evaluation.cross_validation import evaluate_model_cv
from ..evaluation.metrics import get_primary_metric, get_secondary_metrics

class SSA_SVM_Optimization_Pipeline:
    """
    Pipeline for SSA optimization.
    """
    
    def __init__(self):
        self.base_config = load_config("config/base_config.yaml")
        self.preprocessing_config = load_config("config/data/preprocessing_config.yaml")
        self.hpo_config = load_config("config/hpo_config.yaml")
        self.cv_config = self.base_config.get("cross_validation", {})
        self.metrics = get_secondary_metrics()

    def run(self, dataset: str):
        """
        Run the SSA optimization pipeline for SVM on a given dataset.
        Args:
            dataset: Name of the dataset to process 
        Returns:
            results{
            "dataset": dataset,
            "model_name": model_name,
            "hpo_name": hpo_name,
            "config": {
                "base_config": self.base_config,
                "preprocessing_config": self.preprocessing_config,
                "hpo_config": self.hpo_config,
                "cv_config": self.cv_config
            },
            "results": {
                "best_params": best_params,
                "evaluation_results": evaluation_results
            }
        }
        """
        X, y = self.data_pipeline(dataset)
        best_params = self.optimization_pipeline(X, y)
        evaluation_results = self.evaluation_pipeline(X, y, best_params)
        return self.exporting_results_pipeline(dataset, "svm", "ssa", best_params, evaluation_results)

    def data_pipeline(self, dataset:str):
        """
        Load and preprocess the dataset.
        Args:
            dataset: Name of the dataset to load
        Returns:
            X: Processed and reduced feature set
            y: Processed labels
        """
        loader = DatasetLoader(self.base_config.get("data_dir", "data"))
        X, y = loader.load_csv_dataset(dataset)
        dataProcessor = DataPreprocessor(self.preprocessing_config)
        X = dataProcessor.reduce_dimensionality(X)
        X = dataProcessor.handle_missing_values(X)
        X = dataProcessor.scale_features(X)
        y = dataProcessor.encode_label(y)
        return X, y

    def optimization_pipeline(self, X: pd.DataFrame, y: pd.Series):
        """
        Optimize hyperparameters using SSA.
        Args:
            X: Features
            y: Target labels
        Returns:
            best_params: Best SVM hyperparameters found by SSA
        """
        ssa_config = self.hpo_config.get("salp_swarm", {})

        model_name = self.hpo_config.get("default_model", "svm")

        ssa = SalpSwarmOptimizer(model_name,
                                ssa_config.get("num_salps", 30),
                                ssa_config.get("max_iter", 80),
                                ssa_config.get("strategy", "basic"),
                                ssa_config.get("transformation_function", "baseline"))
        
        ssa.optimize(X, y, ssa.objective_function)
        best_params = ssa.get_best_params()

        return best_params

    def evaluation_pipeline(self, X: pd.DataFrame, y: pd.Series, best_params: dict):
        """
        Evaluate the model using cross-validation.
        Args:
            X: Features
            y: Target labels
            best_params: Best hyperparameters from SSA
        Returns:
            results: Evaluation results from cross-validation
        """
        return evaluate_model_cv(self.model, X, y, self.cv_config, self.metrics, best_params)

    def exporting_results_pipeline(self, dataset: str, model_name: str, hpo_name: str, best_params: dict, evaluation_results: dict):
        """
        Get the results of the optimization pipeline.
        Args:
            dataset: Name of the dataset
            model_name: Name of the model
            hpo_name: Name of the hyperparameter optimization method
            results: best params + Evaluation results from cross-validation
        """
        return {
            "dataset": dataset,
            "model_name": model_name,
            "hpo_name": hpo_name,
            "config": {
                "base_config": self.base_config,
                "preprocessing_config": self.preprocessing_config,
                "hpo_config": self.hpo_config,
                "cv_config": self.cv_config
            },
            "results": {
                "best_params": best_params,
                "evaluation_results": evaluation_results
            }
        }