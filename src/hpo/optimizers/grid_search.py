from sklearn.model_selection import GridSearchCV
from .base_optimizer import DiscreteBaseOptimizer  # your safe parent class

class GridSearchOptimizer(DiscreteBaseOptimizer):
    def __init__(self, search_space, grid_points=5, **kwargs):
        """
        GridSearch Optimizer — based on the discrete optimizer parent.
        Converts any continuous-based YAML config into discrete values for GridSearchCV.
        """
        super().__init__(search_space, grid_points=grid_points, **kwargs)
        self._best_params = None
        self._best_score = None

    def optimize(self, estimator, X, y, cv, scoring='f1'):
        """
        Run GridSearchCV on the model with the discretized search space.
        """
        print(f"Running GridSearchCV on model: {estimator.__class__.__name__}")
        print(f"Search space (discretized):\n{self.search_space}\n")

        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=self.search_space,  # safe from parent
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X, y)

        self._best_params = grid_search.best_params_
        self._best_score = grid_search.best_score_
        self.history.append((self._best_params, self._best_score))

        return self._best_params

    def get_best_params(self):
        return self._best_params
