"""Baseline A: the memorizer.

A lookup table with no generalization mechanism beyond label frequencies.
Prediction backs off through three levels, each a Laplace-smoothed
frequency ``(n_true + 1) / (n + 2)``:

1. exact (sentence, canonical scene) seen before,
2. otherwise the same sentence seen with any scene,
3. otherwise the global label frequency.

Corrections overwrite the stored label of the referenced observation.
``model_size`` = number of exact entries + number of sentence entries.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from nexus.eval.learner_api import Prediction
from nexus.world.example import Correction, Observation
from nexus.world.scene import Scene


class UnknownExampleError(KeyError):
    """Raised when a correction references an observation never seen."""


@dataclass
class _Counts:
    n_true: int = 0
    n: int = 0

    def add(self, label: bool, sign: int = 1) -> None:
        self.n_true += sign * int(label)
        self.n += sign
        if self.n < 0 or self.n_true < 0:
            raise ValueError("negative counts")

    def p_true(self) -> float:
        return (self.n_true + 1) / (self.n + 2)


@dataclass
class Memorizer:
    """Exact-match lookup table with frequency back-off."""

    seed: int = 0
    name: str = "memorizer"
    exact: dict[tuple, _Counts] = field(default_factory=lambda: defaultdict(_Counts))
    by_sentence: dict[tuple, _Counts] = field(default_factory=lambda: defaultdict(_Counts))
    overall: _Counts = field(default_factory=_Counts)
    _seen: dict[str, tuple[tuple, tuple, bool]] = field(default_factory=dict)

    def observe(self, observation: Observation) -> None:
        """Store the label under the exact key and the sentence key."""
        if observation.example_id in self._seen:
            raise ValueError(f"duplicate observation id {observation.example_id}")
        exact_key = (observation.tokens, observation.scene.canonical_key())
        self._apply(exact_key, observation.tokens, observation.label, +1)
        self._seen[observation.example_id] = (exact_key, observation.tokens, observation.label)

    def correct(self, correction: Correction) -> None:
        """Replace the stored label of the referenced observation."""
        if correction.target_example_id not in self._seen:
            raise UnknownExampleError(correction.target_example_id)
        exact_key, tokens, old = self._seen[correction.target_example_id]
        self._apply(exact_key, tokens, old, -1)
        self._apply(exact_key, tokens, correction.corrected_label, +1)
        self._seen[correction.target_example_id] = (exact_key, tokens, correction.corrected_label)

    def _apply(self, exact_key: tuple, tokens: tuple, label: bool, sign: int) -> None:
        self.exact[exact_key].add(label, sign)
        self.by_sentence[tokens].add(label, sign)
        self.overall.add(label, sign)

    def predict(self, scene: Scene, tokens: tuple[str, ...]) -> Prediction:
        """Back off from exact match to sentence match to global prior."""
        exact_key = (tokens, scene.canonical_key())
        if exact_key in self.exact and self.exact[exact_key].n > 0:
            return Prediction(self.exact[exact_key].p_true())
        if tokens in self.by_sentence and self.by_sentence[tokens].n > 0:
            return Prediction(self.by_sentence[tokens].p_true())
        return Prediction(self.overall.p_true())

    def model_size(self) -> int:
        """Number of exact entries plus sentence entries."""
        return sum(1 for c in self.exact.values() if c.n) + sum(1 for c in self.by_sentence.values() if c.n)
