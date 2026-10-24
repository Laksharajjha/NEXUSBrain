"""Baseline C: Greedy top-1 (K=1) no-revision categorial learner.

Mechanism:
- Shares the exact same categorial grammar hypothesis space as :class:`NEXUSLearner`
  (``N``, ``N/N``, ``N\\N``, ``S/S``, ``S\\S``, ``(S\\NP)\\NP``, ``(S\\NP)/NP``, ``(S/NP)/NP``).
- Maintains a single hypothesis (``K = 1``) of committed lexical entries.
- Why the original Phase 1 implementation scored ``0.505`` on ``train``:
  1. It tried to commit words on their very first observation (``n_obs = 1``), including
     ``False`` observations where any absent predicate trivially evaluates to ``False``.
  2. Even on a ``True`` 1-token observation, a scene has 2-4 objects, so picking the
     alphabetically first kind in a single scene chooses a distractor object ~65% of the time.
  3. Whenever backtracking failed on a multi-word sentence, the fallback
     ``self._committed[w] = w_cands[0]`` unconditionally assigned category ``"N"`` to
     modifiers, verbs, and the question particle, making all multi-word sentences unparseable.
- Fixed mechanism (``legacy_buggy_fallback=False``):
  - Accumulates observations and commits each word ``w`` greedily to its single top-scoring
    ``(category, predicate)`` candidate once ``w`` has been observed ``commit_after_n`` times
    (or on demand at ``predict()`` for few-shot words with ``< commit_after_n`` exposures).
  - Once ``w`` is committed in ``self._committed[w]``, it is **permanently frozen** and is
    NEVER revised on subsequent observations or ``correct()`` events.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from nexus.eval.learner_api import Prediction
from nexus.learner.categorial import (
    LexicalEntry,
    evaluate_parsed_lf,
    parse_tokens,
)
from nexus.learner.nexus_learner import NEXUSLearner, _ObsEntry
from nexus.world.example import Correction, Observation
from nexus.world.scene import Scene


@dataclass
class NoRevisionLearner:
    """Baseline C: Greedy K=1 learner that permanently commits each word and never revises."""

    seed: int = 0
    epsilon: float = 0.02
    commit_after_n: int = 12
    legacy_buggy_fallback: bool = False
    name: str = "no_revision"

    _committed: dict[str, LexicalEntry] = field(default_factory=dict)
    _word_order: dict[str, int] = field(default_factory=dict)
    _word_counts: Counter[str] = field(default_factory=Counter)
    _observations: list[_ObsEntry] = field(default_factory=list)
    _proposer: NEXUSLearner | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._proposer = NEXUSLearner(seed=self.seed, beam_width=1, epsilon=self.epsilon)

    def observe(self, observation: Observation) -> None:
        """Record observation and greedily freeze any word reaching ``commit_after_n`` exposures."""
        for t in observation.tokens:
            if t not in self._word_order:
                self._word_order[t] = len(self._word_order)

        if self.legacy_buggy_fallback:
            self._observe_legacy_buggy(observation)
            return

        self._observations.append(
            _ObsEntry(
                example_id=observation.example_id,
                scene=observation.scene,
                tokens=observation.tokens,
                label=observation.label,
                source=observation.source,
                corrected=False,
            )
        )
        for t in set(observation.tokens):
            self._word_counts[t] += 1

        # If any uncommitted word in this observation just reached commit_after_n,
        # greedily commit its top-1 candidate now and lock it forever.
        newly_ready = [
            t
            for t in set(observation.tokens)
            if t not in self._committed and self._word_counts[t] >= self.commit_after_n
        ]
        if newly_ready:
            assert self._proposer is not None
            cands = self._proposer._propose_word_candidates(self._observations)
            used_preds = {
                e.predicate.split("(", 1)[0].replace(":sym", "")
                for e in self._committed.values()
                if e.predicate != "Q"
            }
            for w in sorted(newly_ready, key=lambda x: self._word_order[x]):
                if w in cands and cands[w]:
                    # Pick highest-scoring candidate (preferring unused predicate if tied/close)
                    best_entry = cands[w][0][0]
                    best_score = -1e18
                    for entry, sc in cands[w]:
                        base_p = entry.predicate.split("(", 1)[0].replace(":sym", "")
                        adj_sc = sc - (2.5 if (base_p != "Q" and base_p in used_preds) else 0.0)
                        if adj_sc > best_score:
                            best_score = adj_sc
                            best_entry = entry
                    self._committed[w] = best_entry
                    bp = best_entry.predicate.split("(", 1)[0].replace(":sym", "")
                    if bp != "Q":
                        used_preds.add(bp)

    def _observe_legacy_buggy(self, observation: Observation) -> None:
        """Original buggy implementation that committed on n=1/False and fell back to w_cands[0]='N'."""
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

        cand_lists: list[list[LexicalEntry]] = []
        for w in uncommitted:
            idx = tokens.index(w)
            w_cands: list[LexicalEntry] = []
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

        for w, w_cands in zip(uncommitted, cand_lists):
            if w_cands:
                self._committed[w] = w_cands[0]

    def correct(self, correction: Correction) -> None:
        """No-revision learner NEVER revises already committed lexical entries."""
        return None

    def _active_lexicon(self, tokens: tuple[str, ...]) -> dict[str, LexicalEntry]:
        """Return committed lexicon, committing any low-count word on first query."""
        if self.legacy_buggy_fallback or all(t in self._committed for t in tokens):
            return self._committed
        unseen_or_low = [t for t in tokens if t not in self._committed and self._word_counts[t] > 0]
        if unseen_or_low and self._proposer is not None:
            cands = self._proposer._propose_word_candidates(self._observations)
            used_preds = {
                e.predicate.split("(", 1)[0].replace(":sym", "")
                for e in self._committed.values()
                if e.predicate != "Q"
            }
            for w in unseen_or_low:
                if w in cands and cands[w]:
                    best_entry = cands[w][0][0]
                    best_score = -1e18
                    for entry, sc in cands[w]:
                        base_p = entry.predicate.split("(", 1)[0].replace(":sym", "")
                        adj_sc = sc - (2.5 if (base_p != "Q" and base_p in used_preds) else 0.0)
                        if adj_sc > best_score:
                            best_score = adj_sc
                            best_entry = entry
                    self._committed[w] = best_entry
                    bp = best_entry.predicate.split("(", 1)[0].replace(":sym", "")
                    if bp != "Q":
                        used_preds.add(bp)
        return self._committed

    def predict(self, scene: Scene, tokens: tuple[str, ...]) -> Prediction:
        """Evaluate ``tokens`` on ``scene`` using the single committed lexicon."""
        lex = self._active_lexicon(tokens)
        if any(t not in lex for t in tokens):
            return Prediction(0.5, None)
        parsed = parse_tokens(tokens, lex)
        if parsed is None:
            return Prediction(0.5, None)
        _, lf, is_sym = parsed
        truth = evaluate_parsed_lf(lf, scene, is_sym)
        p_true = (1.0 - self.epsilon) if truth else self.epsilon
        return Prediction(p_true, lf)

    def model_size(self) -> int:
        """Number of committed lexical entries."""
        return len(self._committed)
