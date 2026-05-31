"""
retrieval_optimizer.py — Bayesian Optimization for Retrieval Parameters.

Optimizes continuous search parameters using a simple Gaussian Process (GP)
surrogate model with Upper Confidence Bound (UCB) acquisition.

Parameters optimized:
  - alpha:              BM25 vs embedding weight (0.0 = pure BM25, 1.0 = pure embedding)
  - top_k:              Number of chunks to retrieve (3 - 15)
  - rrf_k:              RRF fusion constant (20 - 100)
  - reranker_top_n:     How many chunks the reranker keeps per sub-query (1 - 5)

How Bayesian Optimization works:
  1. Start with a few random parameter samples
  2. Fit a Gaussian Process (GP) to observed (params → reward) pairs
  3. Use UCB acquisition to pick the next params to try:
       UCB(x) = mu(x) + kappa * sigma(x)
     where mu = predicted reward, sigma = uncertainty, kappa = exploration weight
  4. Run the pipeline with those params, observe reward
  5. Add to observations, repeat

This is more sample-efficient than grid search or random search because
the GP models the reward surface and intelligently picks where to explore next.
"""

import json
import numpy as np
from pathlib import Path


class RetrievalOptimizer:
    """
    Bayesian optimization for retrieval hyperparameters.

    Uses a simple RBF-kernel GP approximation that runs on CPU without
    external dependencies like scikit-optimize or GPyTorch.
    """

    # Parameter bounds: (name, min, max, type)
    PARAM_SPACE = [
        ("alpha",          0.2,  0.9,  "float"),   # BM25 vs embedding weight
        ("top_k",          3,    15,   "int"),      # chunks to retrieve
        ("rrf_k",          20,   100,  "int"),      # RRF constant
        ("reranker_top_n", 1,    5,    "int"),      # reranker keeps per sub-query
    ]

    def __init__(self, state_path: str = "rl/retrieval_state.json"):
        self.state_path = Path(state_path)
        self.n_params = len(self.PARAM_SPACE)
        self.kappa = 2.0  # UCB exploration weight

        if self.state_path.exists():
            self._load_state()
        else:
            self.X = []  # observed parameter vectors (normalized to [0,1])
            self.Y = []  # observed rewards
            self.history = []
            self.best_params = self._get_default_params()
            self.best_reward = 0.0

    def _get_default_params(self) -> dict:
        """Return default (current) parameter values."""
        return {
            "alpha": 0.5,
            "top_k": 8,
            "rrf_k": 60,
            "reranker_top_n": 2,
        }

    def suggest_params(self) -> dict:
        """
        Suggest the next set of parameters to try.

        Uses random sampling for the first 5 observations,
        then switches to UCB acquisition on the GP surrogate.
        """
        if len(self.X) < 5:
            # Random exploration phase
            params = self._random_sample()
        else:
            # GP-based acquisition
            params = self._ucb_acquisition()

        return params

    def report_reward(self, params: dict, reward: float):
        """
        Record an observation: these params produced this reward.

        Args:
            params: The parameter dict that was used
            reward: The scalar reward obtained
        """
        x_norm = self._normalize(params)
        self.X.append(x_norm)
        self.Y.append(reward)

        if reward > self.best_reward:
            self.best_reward = reward
            self.best_params = params.copy()

        self.history.append({
            "params": params,
            "reward": round(reward, 4),
            "is_best": reward >= self.best_reward,
        })

        self._save_state()

    def get_best_params(self) -> dict:
        """Return the parameters that produced the highest reward so far."""
        return self.best_params.copy()

    def get_stats(self) -> dict:
        """Return optimization statistics."""
        return {
            "n_observations": len(self.X),
            "best_reward": round(self.best_reward, 4),
            "best_params": self.best_params,
            "recent_rewards": [round(y, 4) for y in self.Y[-10:]],
        }

    # ── Internal Methods ──────────────────────────────────────────────────

    def _normalize(self, params: dict) -> list:
        """Normalize parameters to [0, 1] range."""
        x = []
        for name, lo, hi, _ in self.PARAM_SPACE:
            val = params[name]
            x.append((val - lo) / (hi - lo))
        return x

    def _denormalize(self, x: list) -> dict:
        """Convert [0, 1] normalized vector back to parameter dict."""
        params = {}
        for i, (name, lo, hi, ptype) in enumerate(self.PARAM_SPACE):
            val = lo + x[i] * (hi - lo)
            if ptype == "int":
                val = int(round(val))
            else:
                val = round(val, 3)
            params[name] = val
        return params

    def _random_sample(self) -> dict:
        """Sample random parameters uniformly from the space."""
        x = [np.random.uniform(0, 1) for _ in range(self.n_params)]
        return self._denormalize(x)

    def _rbf_kernel(self, x1, x2, length_scale=0.3):
        """RBF (Gaussian) kernel between two points."""
        x1, x2 = np.array(x1), np.array(x2)
        return np.exp(-0.5 * np.sum((x1 - x2) ** 2) / length_scale ** 2)

    def _gp_predict(self, x_new):
        """
        Simple GP prediction at a new point.

        Returns (mu, sigma) — predicted mean and standard deviation.
        Uses the kernel matrix inverse (exact GP, feasible for <100 observations).
        """
        X = np.array(self.X)
        Y = np.array(self.Y)
        n = len(X)

        # Kernel matrix K(X, X) + noise
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self._rbf_kernel(X[i], X[j])
        K += 1e-4 * np.eye(n)  # noise for numerical stability

        # Kernel vector k(X, x_new)
        k_new = np.array([self._rbf_kernel(X[i], x_new) for i in range(n)])

        # GP predictive mean and variance
        try:
            K_inv = np.linalg.inv(K)
            mu = k_new @ K_inv @ Y
            sigma2 = 1.0 - k_new @ K_inv @ k_new
            sigma = np.sqrt(max(0, sigma2))
        except np.linalg.LinAlgError:
            mu = np.mean(Y)
            sigma = 1.0

        return mu, sigma

    def _ucb_acquisition(self, n_candidates=200) -> dict:
        """
        Upper Confidence Bound acquisition.

        UCB(x) = mu(x) + kappa * sigma(x)

        Generates random candidate points, evaluates UCB on each,
        and returns the parameters with the highest UCB score.
        """
        best_ucb = -np.inf
        best_x = None

        for _ in range(n_candidates):
            x = [np.random.uniform(0, 1) for _ in range(self.n_params)]
            mu, sigma = self._gp_predict(x)
            ucb = mu + self.kappa * sigma

            if ucb > best_ucb:
                best_ucb = ucb
                best_x = x

        return self._denormalize(best_x)

    def _save_state(self):
        """Persist optimizer state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "X": self.X,
            "Y": self.Y,
            "best_params": self.best_params,
            "best_reward": self.best_reward,
            "history": self.history[-100:],
        }
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load optimizer state from disk."""
        with open(self.state_path) as f:
            state = json.load(f)
        self.X = state["X"]
        self.Y = state["Y"]
        self.best_params = state["best_params"]
        self.best_reward = state["best_reward"]
        self.history = state.get("history", [])
