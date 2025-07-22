import numpy as np
import random
import math
from func_timeout import func_timeout, FunctionTimedOut
from .base_optimizer import BaseOptimizer

class SalpSwarmOptimizer(BaseOptimizer):
    """
    Salp Swarm Optimizer compatible with your BaseOptimizer
    and YAML hyperparameter space.

    - Uses the search_space: Dict[str, Any] from YAML (with continuous/categorical).
    - Handles mapping between real-valued salp positions and hyperparameter values.
    - Minimize: lower objective score is better.
    """
    def __init__(self, search_space, num_salps=20, max_iter=30, strategy='basic', **kwargs):
        super().__init__(search_space, **kwargs)
        self.num_salps = num_salps
        self.max_iter = max_iter
        self.strategy = strategy
        self.param_info = self._parse_search_space(search_space)
        self.dim = len(self.param_info)
        # Scalar bounds
        self.lb = np.array([info["lb"] for info in self.param_info], dtype=float)
        self.ub = np.array([info["ub"] for info in self.param_info], dtype=float)
        # Salp population
        self.positions = np.random.uniform(self.lb, self.ub, (self.num_salps, self.dim))
        self.food_position = np.zeros(self.dim)
        self.food_fitness = float("inf")
        self._best_params = None

    def _parse_search_space(self, search_space):
        """
        Converts the search space dict into a flat config list of
        dicts. Each item describes:
        - name: parameter name
        - type: (log, linear, int, cat)
        - lb, ub: lower, upper bounds
        - choices: for categorical
        """
        param_info = []
        for k, v in search_space.items():
            if isinstance(v, dict):
                t = v.get("type", "")
                if t == "log_uniform":
                    param_info.append({"name": k, "type": "log", "lb": math.log10(v["min_value"]), "ub": math.log10(v["max_value"])})
                elif t == "uniform":
                    param_info.append({"name": k, "type": "linear", "lb": v["min_value"], "ub": v["max_value"]})
                elif t == "int_uniform" or t == "int_log_uniform":
                    # integer (but will round in decode)
                    is_log = "log" in t
                    if is_log:
                        param_info.append({"name": k, "type": "int_log", "lb": math.log10(v["min_value"]), "ub": math.log10(v["max_value"])})
                    else:
                        param_info.append({"name": k, "type": "int", "lb": v["min_value"], "ub": v["max_value"]})
                elif t == "categorical":
                    param_info.append({"name": k, "type": "cat", "lb": 0, "ub": len(v["choices"]) - 1, "choices": v["choices"]})
            elif isinstance(v, list):
                # List: treat as categorical
                param_info.append({"name": k, "type": "cat", "lb": 0, "ub": len(v) - 1, "choices": v})
            else:
                # Single fixed value: degenerate interval
                param_info.append({"name": k, "type": "constant", "lb": 0, "ub": 0, "choices": [v]})
        return param_info

    def _decode_position(self, pos):
        """
        Map optimizer's real-valued position to actual hyperparameter values to try.
        """
        hp = {}
        for i, info in enumerate(self.param_info):
            if info["type"] == "log":
                hp[info["name"]] = 10 ** float(pos[i])
            elif info["type"] == "int_log":
                hp[info["name"]] = int(round(10 ** float(pos[i])))
            elif info["type"] == "int":
                hp[info["name"]] = int(round(pos[i]))
            elif info["type"] == "linear":
                hp[info["name"]] = float(pos[i])
            elif info["type"] == "cat":
                idx = int(round(pos[i]))
                idx = max(0, min(idx, len(info["choices"]) - 1))
                hp[info["name"]] = info["choices"][idx]
            elif info["type"] == "constant":
                hp[info["name"]] = info["choices"][0]
        return hp

    def optimize(self, objective_function, n_trials=None):
        """
        objective_function: expects param dict, returns scalar to minimize
        n_trials is ignored; sweeps self.max_iter * num_salps
        Returns: best_params: dict
        """
        def _objective_fn(position):
            try:
                params = self._decode_position(position)
                score = objective_function(params)
                return score
            except Exception as e:
                print(f"[SSA Warning] Failed to evaluate params due to: {e}")
                return float("inf")
        
        for t in range(self.max_iter):
            c1 = 2 * math.exp(-((4 * t / self.max_iter) ** 2))
            for i in range(self.num_salps):
                if i == 0:
                    # Leader
                    self.positions[i] = self._update_leader(c1, t)
                else:
                    # Follower
                    self.positions[i] = self._update_follower(self.positions[i], self.positions[i-1])
                # Evaluate
                try:
                    fitness = _objective_fn(self.positions[i])
                except Exception:
                    fitness = float("inf")
                if fitness < self.food_fitness:
                    self.food_fitness = fitness
                    self.food_position = self.positions[i].copy()
            # Local search (optional)
            if self.strategy == "hybrid" and t >= self.max_iter / 2:
                self._brownian_local_search(_objective_fn)
        # Save best
        self._best_params = self._decode_position(self.food_position)
        return self._best_params

    def get_best_params(self):
        return self._best_params

    # Core SSA methods
    def levy_flight(self, beta=1.5):
        sigma_u = (
            math.gamma(1 + beta)
            * math.sin(math.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = np.random.normal(0, sigma_u, self.dim)
        v = np.random.normal(0, 1, self.dim)
        return u / (np.abs(v) ** (1 / beta))

    def _update_leader(self, c1, t):
        # Uses your code structure
        leader = np.empty(self.dim)
        if self.strategy == "basic":
            for j in range(self.dim):
                c2, c3 = np.random.rand(), np.random.rand()
                step = (self.ub[j] - self.lb[j]) * c2 + self.lb[j]
                if c3 < 0.5:
                    leader[j] = self.food_position[j] + c1 * step
                else:
                    leader[j] = self.food_position[j] - c1 * step
        elif self.strategy == "levy":
            if t < self.max_iter / 2 and np.random.rand() < 0.2:
                leader = self.food_position + 0.01 * self.levy_flight()
            else:
                for j in range(self.dim):
                    c2, c3 = np.random.rand(), np.random.rand()
                    step = (self.ub[j] - self.lb[j]) * c2 + self.lb[j]
                    if c3 < 0.5:
                        leader[j] = self.food_position[j] + c1 * step
                    else:
                        leader[j] = self.food_position[j] - c1 * step
        elif self.strategy == "hybrid":
            if t < self.max_iter / 2 and random.random() < 0.3:
                new_pos = self.food_position + 0.01 * self.levy_flight()
            else:
                new_pos = np.zeros(self.dim)
                for j in range(self.dim):
                    c2, c3 = random.random(), random.random()
                    step = (self.ub[j] - self.lb[j]) * c2 + self.lb[j]
                    if c3 < 0.5:
                        new_pos[j] = self.food_position[j] + c1 * step
                    else:
                        new_pos[j] = self.food_position[j] - c1 * step
                # Crossover
                partner_idx = random.randint(0, self.num_salps - 1)
                partner = self.positions[partner_idx]
                mask = np.random.rand(self.dim) < 0.5
                new_pos[mask] = partner[mask]
                leader = new_pos
        else:
            raise ValueError(f"Unknown SSA strategy: {self.strategy}")
        return np.clip(leader, self.lb, self.ub)

    def _update_follower(self, curr, prev):
        return (curr + prev) / 2

    def _brownian_local_search(self, _objective_fn):
        brownian = np.random.normal(0, 1, self.dim)
        candidate = self.food_position + brownian * (self.ub - self.lb) * 0.01
        candidate = np.clip(candidate, self.lb, self.ub)
        try:
            fit = _objective_fn(candidate)
            if fit < self.food_fitness:
                self.food_fitness = fit
                self.food_position = candidate.copy()
        except Exception:
            pass
