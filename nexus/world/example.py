"""Data records exchanged between the world, the harness, and learners.

:class:`Example` is the full ground-truth record. :class:`Observation` is the
learner-visible view: scene, tokens, and a (possibly erroneous) label. Logical
forms are deliberately absent from observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nexus.world.logic import LogicalForm
from nexus.world.scene import Scene


@dataclass(frozen=True)
class Example:
    """A ground-truth example: (scene, sentence, logical form, truth value)."""

    id: str
    scene: Scene
    tokens: tuple[str, ...]
    lf: LogicalForm
    truth: bool
    construction: str
    contrast_group: str | None = None
    tags: tuple[str, ...] = ()

    def observation(self, label: bool | None = None, source: str = "teacher") -> "Observation":
        """Learner-visible view. ``label`` defaults to the true truth value."""
        return Observation(
            example_id=self.id,
            scene=self.scene,
            tokens=self.tokens,
            label=self.truth if label is None else label,
            source=source,
        )

    def item_key(self) -> tuple[tuple[str, ...], str]:
        """Identity of an item for leakage checks: (tokens, canonical scene)."""
        return (self.tokens, self.scene.canonical_key())

    def to_json(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "id": self.id,
            "scene": self.scene.to_json(),
            "tokens": list(self.tokens),
            "lf": self.lf.to_json(),
            "lf_str": self.lf.canonical(),
            "truth": self.truth,
            "construction": self.construction,
            "contrast_group": self.contrast_group,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class Observation:
    """What a learner sees when it learns from an example.

    ``label`` is the truth value claimed by ``source`` (for a question, the
    claimed answer). It can be wrong in the contradiction stream.
    """

    example_id: str
    scene: Scene
    tokens: tuple[str, ...]
    label: bool
    source: str


@dataclass(frozen=True)
class Correction:
    """Corrective feedback: the label given for ``target_example_id`` was wrong."""

    target_example_id: str
    corrected_label: bool
    source: str
    reason: str


@dataclass(frozen=True)
class StreamEvent:
    """One event in a support stream: an observation or a correction.

    ``ground_truth`` is the underlying example (harness-only, never given to
    the learner). ``mislabelled`` marks observations whose label is wrong.
    """

    kind: str  # "observe" | "correct"
    observation: Observation | None = None
    correction: Correction | None = None
    ground_truth: Example | None = None
    mislabelled: bool = False

    def __post_init__(self) -> None:
        if self.kind == "observe" and (self.observation is None or self.correction is not None):
            raise ValueError("observe event needs exactly an observation")
        if self.kind == "correct" and (self.correction is None or self.observation is not None):
            raise ValueError("correct event needs exactly a correction")
        if self.kind not in {"observe", "correct"}:
            raise ValueError(f"unknown event kind {self.kind!r}")
