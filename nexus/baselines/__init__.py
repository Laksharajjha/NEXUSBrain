"""Baselines for NEXUS-0: Memorizer (A), CrossSituational (B), NoRevision (C), and Shortcut Baselines."""

from nexus.baselines.cross_situational import CrossSituationalLearner
from nexus.baselines.memorizer import Memorizer
from nexus.baselines.no_revision import NoRevisionLearner
from nexus.baselines.scene_only import SceneOnly
from nexus.baselines.sentence_only import SentenceOnly

__all__ = [
    "CrossSituationalLearner",
    "Memorizer",
    "NoRevisionLearner",
    "SceneOnly",
    "SentenceOnly",
]
