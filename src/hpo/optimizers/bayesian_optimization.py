from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from .base_optimizer import BaseOptimizer


class BayesianOptimizer(BaseOptimizer):
    def __init__(self, search_space, n_calls=50, **kwargs):
        super().__init__(search_space, **kwargs)
        self.n_calls = n_calls

    def optimize(self, estimator, X, y, cv, scoring='f1'):
        skopt_search_space = self._convert_generic_space()
        
        bayes_search = BayesSearchCV(
            estimator=estimator,
            search_spaces=skopt_search_space,
            n_iter=self.n_calls,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=42
        )
        bayes_search.fit(X, y)
        self._best_params = bayes_search.best_params_
        return self._best_params

    def get_best_params(self):
        return self._best_params

    def _convert_generic_space(self):
        """Convert YAML-style search space to skopt-compatible space."""
        skopt_space = {}
        
        for param_name, param_config in self.search_space.items():
            # Already discrete: list
            if isinstance(param_config, list):
                skopt_space[param_name] = Categorical(param_config)
                continue

            # Single value
            if not isinstance(param_config, dict):
                skopt_space[param_name] = Categorical([param_config])
                continue

            ptype = param_config.get("type", "")
            if ptype == "log_uniform":
                skopt_space[param_name] = Real(
                    param_config["min_value"],
                    param_config["max_value"],
                    prior="log-uniform"
                )
            elif ptype == "uniform":
                skopt_space[param_name] = Real(
                    param_config["min_value"],
                    param_config["max_value"]
                )
            elif ptype == "int_uniform":
                skopt_space[param_name] = Integer(
                    param_config["min_value"],
                    param_config["max_value"]
                )
            elif ptype == "int_log_uniform":
                skopt_space[param_name] = Integer(
                    param_config["min_value"],
                    param_config["max_value"],
                    prior="log-uniform"
                )
            elif ptype == "categorical":
                skopt_space[param_name] = Categorical(param_config["choices"])

            else:  # fallback
                skopt_space[param_name] = Categorical([param_config])

        return skopt_space
