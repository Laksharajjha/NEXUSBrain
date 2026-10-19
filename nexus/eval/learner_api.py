"""The interface every learner (NEXUS and baselines) implements.

Learners only ever receive :class:`~nexus.world.example.Observation` and
:class:`~nexus.world.example.Correction` objects, never logical forms or
the grammar. The task is truth judgment: given a scene and a sentence,
estimate the probability that the sentence is true of the scene (for a
question, that the answer is "yes").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from nexus.world.example import Correction, Observation
from nexus.world.logic import LogicalForm
from nexus.world.scene import Scene


@dataclass(frozen=True)
class Prediction:
    """A learner's answer for one (scene, sentence) item."""

    p_true: float
    lf: LogicalForm | None = None  # optional parse, scored when provided

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_true <= 1.0:
            raise ValueError(f"p_true must be in [0, 1], got {self.p_true}")

    @property
    def truth(self) -> bool:
        """Predicted truth value (ties at 0.5 resolve to True, deterministically)."""
        return self.p_true >= 0.5

    @property
    def confidence(self) -> float:
        """Probability assigned to the predicted truth value."""
        return max(self.p_true, 1.0 - self.p_true)


@runtime_checkable
class Learner(Protocol):
    """Protocol for all learners run through the harness."""

    name: str

    def observe(self, observation: Observation) -> None:
        """Learn from one labelled observation."""

    def correct(self, correction: Correction) -> None:
        """Apply corrective feedback about an earlier observation."""

    def predict(self, scene: Scene, tokens: tuple[str, ...]) -> Prediction:
        """Judge whether ``tokens`` is true of ``scene``. Must not learn."""

    def model_size(self) -> int:
        """Number of stored parameters / entries (learner-defined, documented)."""
