"""SentenceOnly shortcut baseline.

Predicts truth value from sentence features alone (bag of words, length,
position-specific token indicators) using L2-regularized logistic regression
implemented with NumPy. It completely ignores the perceptual scene.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from nexus.eval.learner_api import Prediction
from nexus.world.example import Correction, Observation
from nexus.world.scene import Scene


class SentenceOnly:
    """Logistic regression model predicting truth from sentence tokens alone."""

    name: str = "sentence_only"

    def __init__(self, seed: int = 0, lr: float = 0.05, reg: float = 0.01, steps: int = 400) -> None:
        self.seed = seed
        self.lr = lr
        self.reg = reg
        self.steps = steps
        self.vocab: list[str] = []
        self.word2idx: dict[str, int] = {}
        self.max_len: int = 12
        self.observations: dict[str, tuple[tuple[str, ...], bool]] = {}
        self.weights: np.ndarray | None = None
        self._dirty: bool = True

    def _ensure_vocab(self, tokens: tuple[str, ...]) -> None:
        for t in tokens:
            if t not in self.word2idx:
                self.word2idx[t] = len(self.vocab)
                self.vocab.append(t)
                self._dirty = True

    def observe(self, observation: Observation) -> None:
        """Learn from one labelled observation."""
        self._ensure_vocab(observation.tokens)
        self.observations[observation.example_id] = (observation.tokens, observation.label)
        self._dirty = True

    def correct(self, correction: Correction) -> None:
        """Update label on corrective feedback."""
        if correction.target_example_id in self.observations:
            toks, _ = self.observations[correction.target_example_id]
            self.observations[correction.target_example_id] = (toks, correction.corrected_label)
            self._dirty = True

    def _extract_features(self, tokens: tuple[str, ...]) -> np.ndarray:
        """Extract Bag-of-Words, length, and position-specific token indicators."""
        v = len(self.vocab)
        # Features: [BoW (v), length (1), pos_features (v * max_len), bias (1)]
        f = np.zeros(v + 1 + v * self.max_len + 1, dtype=np.float64)
        for pos, t in enumerate(tokens):
            if t in self.word2idx:
                idx = self.word2idx[t]
                f[idx] += 1.0
                if pos < self.max_len:
                    f[v + 1 + pos * v + idx] = 1.0
        f[v] = float(len(tokens))
        f[-1] = 1.0  # bias
        return f

    def _fit(self) -> None:
        """Fit weights via gradient descent with L2 regularization."""
        if not self.observations:
            self.weights = np.zeros(1)
            self._dirty = False
            return

        X = np.array([self._extract_features(toks) for toks, _ in self.observations.values()])
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
        """Predict truth from tokens alone."""
        if not self.observations:
            return Prediction(p_true=0.5)
        self._ensure_vocab(tokens)
        if self._dirty or self.weights is None:
            self._fit()

        x = self._extract_features(tokens)
        if len(self.weights) != len(x):
            self._fit()
            x = self._extract_features(tokens)
        logit = float(x @ self.weights)
        p = float(1.0 / (1.0 + np.exp(-np.clip(logit, -15.0, 15.0))))
        return Prediction(p_true=p)

    def model_size(self) -> int:
        """Number of linear model weights."""
        return len(self.weights) if self.weights is not None else 0
