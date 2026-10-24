"""Baseline B: Cross-situational / IBM-1 co-occurrence learner (bag-of-meanings).

Mechanism:
- Maintains a word-by-predicate co-occurrence and contrast association table
  (IBM-1 / cross-situational alignment statistics) mapping each surface word ``w``
  to a distribution over scene predicates (``kind:v``, ``color:v``, ``size:v``,
  ``rel:r``, or ``Q``).
- At prediction time, each token ``w`` in the sentence is mapped to its argmax
  predicate ``p*(w)``. The sentence is judged True if every non-Q predicate in
  the resulting unstructured bag ``{p*(w) : w in tokens}`` holds somewhere in the
  scene, with NO syntactic constituent structure, NO modifier-to-noun binding,
  and NO subject/object role binding.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from nexus.eval.learner_api import Prediction
from nexus.world.example import Correction, Observation
from nexus.world.scene import Scene


def scene_predicate_bag(scene: Scene) -> set[str]:
    """Extract unstructured bag of unary and relational predicates present in ``scene``."""
    preds: set[str] = set()
    for obj in scene.objects:
        preds.add(f"kind:{obj.kind}")
        preds.add(f"color:{obj.color}")
        preds.add(f"size:{obj.size}")
    for rel in scene.relations:
        preds.add(f"rel:{rel.name}")
    return preds


@dataclass
class CrossSituationalLearner:
    """Baseline B: Cross-situational word-to-predicate alignment with bag-of-meanings evaluation."""

    seed: int = 0
    epsilon: float = 0.05
    em_iterations: int = 5
    name: str = "cross_situational"

    _observations: list[Observation] = field(default_factory=list)
    _obs_by_id: dict[str, int] = field(default_factory=dict)
    _word_order: dict[str, int] = field(default_factory=dict)
    _lexicon: dict[str, str] = field(default_factory=dict)
    _confidences: dict[str, float] = field(default_factory=dict)
    _dirty: bool = True

    def observe(self, observation: Observation) -> None:
        """Record an observation and mark alignment table for update."""
        for t in observation.tokens:
            if t not in self._word_order:
                self._word_order[t] = len(self._word_order)
        idx = len(self._observations)
        self._observations.append(observation)
        self._obs_by_id[observation.example_id] = idx
        self._dirty = True

    def correct(self, correction: Correction) -> None:
        """Update the label of an earlier observation and recompute alignments."""
        if correction.target_example_id in self._obs_by_id:
            idx = self._obs_by_id[correction.target_example_id]
            old = self._observations[idx]
            self._observations[idx] = Observation(
                example_id=old.example_id,
                scene=old.scene,
                tokens=old.tokens,
                label=correction.corrected_label,
                source=correction.source,
            )
            self._dirty = True

    def _fit(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        if not self._observations:
            self._lexicon = {}
            self._confidences = {}
            return

        all_preds_set: set[str] = {"Q"}
        obs_bags: list[set[str]] = []
        for obs in self._observations:
            bag = scene_predicate_bag(obs.scene)
            obs_bags.append(bag)
            all_preds_set.update(bag)

        all_preds = sorted(all_preds_set)
        words = sorted(self._word_order.keys(), key=lambda w: self._word_order[w])

        # Initialize uniform translation probabilities P(pred | word)
        t_prob: dict[str, dict[str, float]] = {
            w: {p: 1.0 / len(all_preds) for p in all_preds} for w in words
        }

        # Precompute contrastive association scores from True vs False observations
        contrast_bonus: dict[str, dict[str, float]] = {
            w: defaultdict(float) for w in words
        }
        word_occ_count: dict[str, int] = defaultdict(int)
        word_true_count: dict[str, int] = defaultdict(int)
        pred_in_true: dict[str, dict[str, int]] = {w: defaultdict(int) for w in words}
        pred_in_false_len1: dict[str, dict[str, int]] = {w: defaultdict(int) for w in words}

        for obs, bag in zip(self._observations, obs_bags):
            u_tokens = set(obs.tokens)
            for w in u_tokens:
                word_occ_count[w] += 1
                if obs.label:
                    word_true_count[w] += 1
                    for p in bag:
                        pred_in_true[w][p] += 1
                elif len(obs.tokens) <= 2:
                    for p in all_preds:
                        if p != "Q" and p not in bag:
                            pred_in_false_len1[w][p] += 1

        for w in words:
            n_t = max(1, word_true_count[w])
            max_rate = 0.0
            for p in all_preds:
                if p == "Q":
                    continue
                rate_true = pred_in_true[w][p] / n_t
                if rate_true > max_rate:
                    max_rate = rate_true
                contrast_bonus[w][p] = (
                    2.0 * rate_true + 0.5 * pred_in_false_len1[w][p]
                )
            # If no predicate is consistently present in True scenes, "Q" gets high score
            if max_rate < 0.85 and word_occ_count[w] >= 3:
                contrast_bonus[w]["Q"] = 3.0
            else:
                contrast_bonus[w]["Q"] = 0.1

        # IBM Model 1 EM iterations weighted by contrastive evidence on True scenes
        for _ in range(self.em_iterations):
            counts: dict[str, dict[str, float]] = {
                w: {p: 1e-4 * contrast_bonus[w][p] for p in all_preds} for w in words
            }
            for obs, bag in zip(self._observations, obs_bags):
                if not obs.label:
                    continue
                active_preds = list(bag) + ["Q"]
                for p in active_preds:
                    denom = sum(t_prob[w][p] for w in obs.tokens)
                    if denom > 0:
                        for w in obs.tokens:
                            counts[w][p] += (t_prob[w][p] / denom) * contrast_bonus[w][p]
            for w in words:
                total = sum(counts[w].values())
                if total > 0:
                    for p in all_preds:
                        t_prob[w][p] = counts[w][p] / total

        self._lexicon = {}
        self._confidences = {}
        for w in words:
            best_p, best_prob = max(t_prob[w].items(), key=lambda kv: (kv[1], kv[0]))
            self._lexicon[w] = best_p
            self._confidences[w] = float(best_prob)

    def predict(self, scene: Scene, tokens: tuple[str, ...]) -> Prediction:
        """Predict True iff all induced content predicates for ``tokens`` appear in ``scene``."""
        self._fit()
        if not self._lexicon or any(t not in self._lexicon for t in tokens):
            return Prediction(0.5, None)

        bag = scene_predicate_bag(scene)
        required_preds = [self._lexicon[t] for t in tokens if self._lexicon[t] != "Q"]
        if not required_preds:
            return Prediction(0.5, None)

        all_present = all(p in bag for p in required_preds)
        avg_conf = sum(self._confidences[t] for t in tokens) / len(tokens)
        strength = min(0.45, max(0.05, 0.45 * avg_conf))
        p_true = (0.5 + strength) if all_present else (0.5 - strength)
        return Prediction(float(min(1.0 - self.epsilon, max(self.epsilon, p_true))), None)

    def model_size(self) -> int:
        """Number of words in the induced alignment table."""
        self._fit()
        return len(self._lexicon)
