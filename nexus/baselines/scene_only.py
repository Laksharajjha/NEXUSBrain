"""SceneOnly shortcut baseline.

Predicts truth value from scene features alone (object counts, feature counts,
relation counts) using L2-regularized logistic regression implemented with NumPy.
It completely ignores sentence tokens.
"""

from __future__ import annotations

import numpy as np

from nexus.eval.learner_api import Prediction
from nexus.world.example import Correction, Observation
from nexus.world.scene import Scene


class SceneOnly:
    """Logistic regression model predicting truth from scene features alone."""

    name: str = "scene_only"

    def __init__(self, seed: int = 0, lr: float = 0.05, reg: float = 0.01, steps: int = 400) -> None:
        self.seed = seed
        self.lr = lr
        self.reg = reg
        self.steps = steps
        self.feature_vocab: list[str] = []
        self.feat2idx: dict[str, int] = {}
        self.rel_names: list[str] = []
        self.rel2idx: dict[str, int] = {}
        self.observations: dict[str, tuple[Scene, bool]] = {}
        self.weights: np.ndarray | None = None
        self._dirty: bool = True

    def _ensure_scene_vocab(self, scene: Scene) -> None:
        for obj in scene.objects:
            for feat in obj.features():
                if feat not in self.feat2idx:
                    self.feat2idx[feat] = len(self.feature_vocab)
                    self.feature_vocab.append(feat)
                    self._dirty = True
        for r in scene.relations:
            if r.name not in self.rel2idx:
                self.rel2idx[r.name] = len(self.rel_names)
                self.rel_names.append(r.name)
                self._dirty = True

    def observe(self, observation: Observation) -> None:
        """Learn from one labelled observation."""
        self._ensure_scene_vocab(observation.scene)
        self.observations[observation.example_id] = (observation.scene, observation.label)
        self._dirty = True

    def correct(self, correction: Correction) -> None:
        """Update label on corrective feedback."""
        if correction.target_example_id in self.observations:
            sc, _ = self.observations[correction.target_example_id]
            self.observations[correction.target_example_id] = (sc, correction.corrected_label)
            self._dirty = True

    def _extract_features(self, scene: Scene) -> np.ndarray:
        """Extract object counts, predicate feature counts, and relation counts."""
        nf = len(self.feature_vocab)
        nr = len(self.rel_names)
        # Features: [n_objs (1), feature_counts (nf), n_rels (1), rel_counts (nr), bias (1)]
        f = np.zeros(1 + nf + 1 + nr + 1, dtype=np.float64)
        f[0] = float(len(scene.objects))
        for obj in scene.objects:
            for feat in obj.features():
                if feat in self.feat2idx:
                    f[1 + self.feat2idx[feat]] += 1.0
        f[1 + nf] = float(len(scene.relations))
        for r in scene.relations:
            if r.name in self.rel2idx:
                f[1 + nf + 1 + self.rel2idx[r.name]] += 1.0
        f[-1] = 1.0  # bias
        return f

    def _fit(self) -> None:
        """Fit weights via gradient descent with L2 regularization."""
        if not self.observations:
            self.weights = np.zeros(1)
            self._dirty = False
            return

        X = np.array([self._extract_features(sc) for sc, _ in self.observations.values()])
        y = np.array([float(label) for _, label in self.observations.values()])

        rng = np.random.RandomState(self.seed % (2**32))
        w = rng.normal(0, 0.01, size=X.shape[1])

        n = len(y)
        for _ in range(self.steps):
            logits = np.clip(X @ w, -15.0, 15.0)
            p = 1.0 / (1.0 + np.exp(-logits))
            grad = (X.T @ (p - y)) / n + self.reg * w
            w -= self.lr * grad

        self.weights = w
        self._dirty = False

    def predict(self, scene: Scene, tokens: tuple[str, ...]) -> Prediction:
        """Predict truth from scene alone."""
        if not self.observations:
            return Prediction(p_true=0.5)
        self._ensure_scene_vocab(scene)
        expected_dim = 1 + len(self.feature_vocab) + 1 + len(self.rel_names) + 1
        if self._dirty or self.weights is None or len(self.weights) != expected_dim:
            self._fit()

        x = self._extract_features(scene)
        logit = float(x @ self.weights)
        p = float(1.0 / (1.0 + np.exp(-np.clip(logit, -15.0, 15.0))))
        return Prediction(p_true=p)

    def model_size(self) -> int:
        """Number of linear model weights."""
        return len(self.weights) if self.weights is not None else 0
