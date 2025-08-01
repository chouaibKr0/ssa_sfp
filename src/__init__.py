from .utils import *

base_config = load_config("config/base_config.yaml")

__version__ = base_config.get("version", "1.0.0")
__author__ = base_config.get("author", "chouaib")
__description__ = base_config.get("project_description", "")

from .data import *
from .pipelines.ssa_optimization_pipelines import SSA_SVM_Optimization_Pipeline

config_paths =  {"config/base_config.yaml","config/hpo_config.yaml","config/data/preprocessing_config.yaml","config/models/svm_config.yaml"}
config, logger, directories, experiment_id = setup_experiment("ssa_svm", config_paths)
pipeline = SSA_SVM_Optimization_Pipeline()
results = pipeline.run("ant-1.3.csv")
save_experiment(experiment_id, results['model_name'], results['hpo_name'], results['config'], results['results'], directories, logger)
