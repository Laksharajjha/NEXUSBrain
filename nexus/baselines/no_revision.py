"""Baseline C: Greedy top-1 (K=1) no-revision categorial learner.

Mechanism:
- Shares the exact same categorial grammar hypothesis space as :class:`NEXUSLearner`
  (``N``, ``N/N``, ``N\\N``, ``S/S``, ``S\\S``, ``(S\\NP)\\NP``, ``(S\\NP)/NP``, ``(S/NP)/NP``).
- Maintains a single hypothesis (``K = 1``) of committed lexical entries.
- When a word ``w`` is first encountered in an observation, commits greedily to the
  first syntactic category and scene predicate consistent with that observation, and
  NEVER revises a committed word on subsequent observations or ``correct()`` events.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nexus.eval.learner_api import Prediction
from nexus.learner.categorial import (
    LexicalEntry,
    evaluate_parsed_lf,
    parse_tokens,
)
from nexus.world.example import Correction, Observation
from nexus.world.scene import Scene


@dataclass
class NoRevisionLearner:
    """Baseline C: Greedy K=1 learner that commits to the first consistent entry per word."""

    seed: int = 0
    epsilon: float = 0.02
    name: str = "no_revision"

    _committed: dict[str, LexicalEntry] = field(default_factory=dict)
    _word_order: dict[str, int] = field(default_factory=dict)

    def observe(self, observation: Observation) -> None:
        """Greedily commit any new words in ``observation.tokens`` to the first consistent entry."""
        for t in observation.tokens:
            if t not in self._word_order:
                self._word_order[t] = len(self._word_order)

        uncommitted = [t for t in observation.tokens if t not in self._committed]
        if not uncommitted:
            return

        scene = observation.scene
        kinds = sorted({f"kind:{o.kind}" for o in scene.objects})
        mods = sorted(
            {f"color:{o.color}" for o in scene.objects}
            | {f"size:{o.size}" for o in scene.objects}
        )
        rels = sorted({r.name for r in scene.relations})

        tokens = observation.tokens
        n = len(tokens)

        # Case 1: 1-token sentence -> must be category "N"
        if n == 1:
            w = tokens[0]
            for kp in kinds:
                entry = LexicalEntry(w, "N", kp, (observation.example_id,))
                trial = {**self._committed, w: entry}
                parsed = parse_tokens(tokens, trial)
                if parsed is not None:
                    _, lf, is_sym = parsed
                    if evaluate_parsed_lf(lf, scene, is_sym) == observation.label:
                        self._committed[w] = entry
                        return
            if kinds:
                self._committed[w] = LexicalEntry(w, "N", kinds[0], (observation.example_id,))
            return

        # Case 2: Try to find a joint assignment for the uncommitted words in this observation
        # that parses tokens and matches observation.label
        cand_lists: list[list[LexicalEntry]] = []
        for w in uncommitted:
            idx = tokens.index(w)
            w_cands: list[LexicalEntry] = []
            # If at boundary of a >=2 token sentence, include Q candidate after content candidates
            for kp in kinds:
                w_cands.append(LexicalEntry(w, "N", kp, (observation.example_id,)))
            for mp in mods:
                if idx == 0 or idx < n - 1:
                    w_cands.append(LexicalEntry(w, "N/N", mp, (observation.example_id,)))
                if idx > 0:
                    w_cands.append(LexicalEntry(w, "N\\N", mp, (observation.example_id,)))
            if n >= 3:
                for r_name in rels:
                    for v_cat in ("(S\\NP)\\NP", "(S\\NP)/NP", "(S/NP)/NP"):
                        w_cands.append(
                            LexicalEntry(w, v_cat, f"rel:{r_name}(x0,x1)", (observation.example_id,))
                        )
                        w_cands.append(
                            LexicalEntry(w, v_cat, f"rel:{r_name}:sym(x0,x1)", (observation.example_id,))
                        )
            if idx == 0:
                w_cands.append(LexicalEntry(w, "S/S", "Q", (observation.example_id,)))
            if idx == n - 1:
                w_cands.append(LexicalEntry(w, "S\\S", "Q", (observation.example_id,)))
            cand_lists.append(w_cands)

        # Backtracking search for the first consistent assignment (bounded to 250 steps)
        steps = 0

        def search(pos: int, current: dict[str, LexicalEntry]) -> bool:
            nonlocal steps
            if pos == len(uncommitted):
                steps += 1
                parsed = parse_tokens(tokens, current)
                if parsed is None:
                    return False
                _, lf, is_sym = parsed
                if evaluate_parsed_lf(lf, scene, is_sym) == observation.label:
                    self._committed.update(current)
                    return True
                return False

            for cand in cand_lists[pos]:
                if steps > 250:
                    return False
                nxt = dict(current)
                nxt[uncommitted[pos]] = cand
                if search(pos + 1, nxt):
                    return True
            return False

        if search(0, dict(self._committed)):
            return

        # Fallback: commit each uncommitted word to its first candidate
        for w, w_cands in zip(uncommitted, cand_lists):
            if w_cands:
                self._committed[w] = w_cands[0]

    def correct(self, correction: Correction) -> None:
        """No-revision learner never revises already committed lexical entries."""
        return None

    def predict(self, scene: Scene, tokens: tuple[str, ...]) -> Prediction:
        """Evaluate ``tokens`` on ``scene`` using the single committed lexicon."""
        if any(t not in self._committed for t in tokens):
            return Prediction(0.5, None)
        parsed = parse_tokens(tokens, self._committed)
        if parsed is None:
            return Prediction(0.5, None)
        _, lf, is_sym = parsed
        truth = evaluate_parsed_lf(lf, scene, is_sym)
        p_true = (1.0 - self.epsilon) if truth else self.epsilon
        return Prediction(p_true, lf)

    def model_size(self) -> int:
        """Number of committed lexical entries."""
        return len(self._committed)
