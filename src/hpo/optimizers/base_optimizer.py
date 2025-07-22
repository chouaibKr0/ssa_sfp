from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple,Callable
import numpy as np
import random


class BaseOptimizer(ABC):
    def __init__(self, search_space: Dict[str, Any], **kwargs):
        self.search_space = search_space
        self.history = []
    
    @abstractmethod
    def optimize(self, objective_function, n_trials: int) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_best_params(self) -> Dict[str, Any]:
        pass




"""
discrete_optimizer.py
---------------------

A tiny “wrapper” optimizer that converts any *continuous* (or integer–range)
hyper-parameter description found in the input ``search_space`` into a **small
discrete grid**.  After conversion the class behaves like a very light-weight
random-search optimiser, so it satisfies the two abstract methods declared in
``BaseOptimizer`` and can be dropped into any code that expects a normal
optimizer instance (e.g. GridSearch, Optuna, …).

Key points
----------
1.  Inherits from the user-supplied ``BaseOptimizer``.
2.  Accepts the same YAML–style dictionaries we used earlier  
    (``type: log_uniform | uniform | int_uniform | int_log_uniform |
    categorical``).
3.  Every continuous range is turned into *`grid_points`* evenly spaced values
    (log-spaced if the prior is log-uniform).
4.  The original ``self.search_space`` is **replaced** by the discrete grid so
    the rest of the pipeline (GridSearchCV, etc.) will no longer crash.
"""




class DiscreteBaseOptimizer(BaseOptimizer):
    """
    Convert any continuous / integer range in *search_space* to a discrete list
    of values, then perform a very simple random search.
    """

    def __init__(
        self,
        search_space: Dict[str, Any],
        grid_points: int = 7,                # how many values per continuous dim
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(search_space, **kwargs)
        self.grid_points = max(grid_points, 2)
        self.rng = np.random.default_rng(random_state)

        # The public attribute `search_space` is overwritten with its
        # discretised counterpart so that *other* code (GridSearch, etc.) can
        # use it directly.
        self.search_space: Dict[str, List[Any]] = self._discretise_space(
            self.search_space
        )

        self._best_params: Dict[str, Any] | None = None
        self._best_score: float = float("inf")

    # --------------------------------------------------------------------- #
    # Abstract-method implementations
    # --------------------------------------------------------------------- #
    def optimize(
        self,
        objective_function: Callable[[Dict[str, Any]], float],
        n_trials: int = 50,
    ) -> Dict[str, Any]:
        """
        Very small random search: sample *n_trials* parameter sets from the
        newly-discrete grid, evaluate ``objective_function`` and keep the best.
        The objective is assumed to be **minimised**.
        """
        for _ in range(n_trials):
            params = {k: random.choice(v) for k, v in self.search_space.items()}
            score = objective_function(params)

            self.history.append((params, score))

            if score < self._best_score:
                self._best_score = score
                self._best_params = params

        return self._best_params

    def get_best_params(self) -> Dict[str, Any]:
        return self._best_params

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _discretise_space(self, raw_space: Dict[str, Any]) -> Dict[str, List[Any]]:
        """
        Convert every entry of *raw_space* to a discrete list.

        Supported continuous descriptors (same as in the YAML files):
            type: log_uniform   ->  Real or Int, log-spaced
            type: uniform       ->  Real, linear spaced
            type: int_uniform   ->  Integer, linear spaced
            type: int_log_uniform-> Integer, log-spaced
            type: categorical   ->  Already discrete
        Everything that is already a list is returned unmodified.
        """

        def _linspace(a: float, b: float, n: int) -> List[float]:
            return np.linspace(a, b, n).tolist()

        def _logspace(a: float, b: float, n: int) -> List[float]:
            return np.logspace(np.log10(a), np.log10(b), n).tolist()

        discrete: Dict[str, List[Any]] = {}

        for name, cfg in raw_space.items():
            # ----------------------------------------------------------------
            # Case 1 – already discrete (list) or a single constant
            # ----------------------------------------------------------------
            if isinstance(cfg, list):
                discrete[name] = cfg
                continue
            if not isinstance(cfg, dict):
                discrete[name] = [cfg]
                continue

            # ----------------------------------------------------------------
            # Case 2 – YAML-style dictionary
            # ----------------------------------------------------------------
            p_type = cfg.get("type", "").lower()

            # Log-uniform real
            if p_type == "log_uniform":
                vals = _logspace(cfg["min_value"], cfg["max_value"], self.grid_points)

            # Uniform real
            elif p_type == "uniform":
                vals = _linspace(cfg["min_value"], cfg["max_value"], self.grid_points)

            # Integer uniform
            elif p_type == "int_uniform":
                low, high = cfg["min_value"], cfg["max_value"]
                step = max((high - low) // (self.grid_points - 1), 1)
                vals = list(range(low, high + 1, step))

            # Integer log-uniform
            elif p_type == "int_log_uniform":
                raw = _logspace(cfg["min_value"], cfg["max_value"], self.grid_points)
                vals = [int(round(x)) for x in raw]

            # Categorical
            elif p_type == "categorical":
                vals = list(cfg["choices"])

            # Fallback – treat as single constant
            else:
                vals = [cfg]

            discrete[name] = vals

        return discrete
