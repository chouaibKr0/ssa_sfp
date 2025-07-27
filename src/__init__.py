from utils import *

base_config = load_config("config/base_config.yaml")

__version__ = base_config.get("version", "1.0.0")
__author__ = base_config.get("author", "chouaib")
__description__ = base_config.get("project_description", "")


from . import analysis
from . import data
from . import evaluation
from . import features
from . import hpo
from . import models
from . import pipelines
from . import tracking
from . import visualization
