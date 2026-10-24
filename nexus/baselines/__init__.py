"""Baselines for NEXUS-0: Memorizer (A), CrossSituational (B), NoRevision (C), PositionAwareCS (D), Neural (E), and Shortcut Baselines."""

from nexus.baselines.cross_situational import CrossSituationalLearner
from nexus.baselines.memorizer import Memorizer
from nexus.baselines.neural_ref import FromScratchNeuralBaseline
from nexus.baselines.no_revision import NoRevisionLearner
from nexus.baselines.position_aware_cs import PositionAwareCrossSituationalLearner
from nexus.baselines.scene_only import SceneOnly
from nexus.baselines.sentence_only import SentenceOnly

__all__ = [
    "CrossSituationalLearner",
    "FromScratchNeuralBaseline",
    "Memorizer",
    "NoRevisionLearner",
    "PositionAwareCrossSituationalLearner",
    "SceneOnly",
    "SentenceOnly",
]
