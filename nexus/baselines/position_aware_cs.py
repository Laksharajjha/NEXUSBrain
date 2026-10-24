"""Baseline D: Position-aware, per-object cross-situational learner (no grammar search).

Mechanism:
- Induces a word-to-predicate contrast score table ``C(w, p)`` from True vs. False
  scene co-occurrences, plus positional statistics (token index, adjacency to kind
  words, and linear-order argument role statistics).
- At prediction time (NO categorial grammar or CKY parse search):
  1. Classifies each token via its top-scoring predicate into ``kind:*`` (object anchor),
     ``color:*`` / ``size:*`` (attribute), ``rel:*`` (relation), or ``Q`` (peripheral
     question marker with low scene contrast).
  2. **Per-object adjacency bundling**: attaches each attribute token to the nearest
     ``kind:*`` token in linear string position ``|i_attr - i_kind|``, requiring a single
     :class:`SceneObject` to satisfy the entire bundle ``(kind, attrs)`` simultaneously.
  3. **Positional relational linking**: for two object bundles ``(B_1, B_2)`` in linear
     order and a relation token ``w_rel``, evaluates ``r(o_1, o_2)`` (or learned
     role-reversed / symmetric orientation from linear-order co-occurrence counts).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from nexus.eval.learner_api import Prediction
from nexus.world.example import Correction, Observation
from nexus.world.scene import Scene


@dataclass
class _PosObs:
    example_id: str
    scene: Scene
    tokens: tuple[str, ...]
    label: bool


@dataclass
class PositionAwareCrossSituationalLearner:
    """Baseline D: Position-aware, per-object cross-situational learner without grammar search."""

    seed: int = 0
    epsilon: float = 0.05
    name: str = "position_aware_cs"

    _observations: list[_PosObs] = field(default_factory=list)
    _obs_by_id: dict[str, int] = field(default_factory=dict)
    _dirty: bool = True

    _word_pred: dict[str, str] = field(default_factory=dict)
    _rel_mode: dict[str, str] = field(default_factory=dict)  # w -> "fwd" | "rev" | "sym"

    @staticmethod
    def _scene_predicates(scene: Scene) -> set[str]:
        preds: set[str] = set()
        for o in scene.objects:
            preds.add(f"kind:{o.kind}")
            preds.add(f"color:{o.color}")
            preds.add(f"size:{o.size}")
        for r in scene.relations:
            preds.add(f"rel:{r.name}")
        return preds

    def observe(self, observation: Observation) -> None:
        idx = len(self._observations)
        self._observations.append(
            _PosObs(
                example_id=observation.example_id,
                scene=observation.scene,
                tokens=observation.tokens,
                label=observation.label,
            )
        )
        self._obs_by_id[observation.example_id] = idx
        self._dirty = True

    def correct(self, correction: Correction) -> None:
        if correction.target_example_id in self._obs_by_id:
            idx = self._obs_by_id[correction.target_example_id]
            self._observations[idx].label = correction.corrected_label
            self._dirty = True

    def _fit(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        self._word_pred.clear()
        self._rel_mode.clear()

        if not self._observations:
            return

        all_kinds: set[str] = set()
        all_mods: set[str] = set()
        all_rels: set[str] = set()
        word_occs: dict[str, list[_PosObs]] = defaultdict(list)

        for obs in self._observations:
            for o in obs.scene.objects:
                all_kinds.add(f"kind:{o.kind}")
                all_mods.add(f"color:{o.color}")
                all_mods.add(f"size:{o.size}")
            for r in obs.scene.relations:
                all_rels.add(f"rel:{r.name}")
            for w in set(obs.tokens):
                word_occs[w].append(obs)

        all_preds = sorted(all_kinds | all_mods | all_rels)

        # Step 1: Identify peripheral question markers (always initial or final, len >= 2, low invariance)
        q_words: set[str] = set()
        for w, occs in word_occs.items():
            if any(len(o.tokens) == 1 for o in occs):
                continue
            if not (all(o.tokens[0] == w for o in occs) or all(o.tokens[-1] == w for o in occs)):
                continue
            true_occs = [o for o in occs if o.label]
            if not true_occs:
                continue
            max_rate = 0.0
            for p in all_preds:
                rate = sum(p in self._scene_predicates(o.scene) for o in true_occs) / len(true_occs)
                if rate > max_rate:
                    max_rate = rate
            if max_rate < 0.85:
                q_words.add(w)
                self._word_pred[w] = "Q"

        def strip_q(tokens: tuple[str, ...]) -> tuple[str, ...]:
            if len(tokens) >= 2 and tokens[0] in q_words:
                return tokens[1:]
            if len(tokens) >= 2 and tokens[-1] in q_words:
                return tokens[:-1]
            return tokens

        # Step 2: Score each word by positional + cross-situational contrast
        for w, occs in word_occs.items():
            if w in q_words:
                continue
            core_lengths = [len(strip_q(o.tokens)) for o in occs]
            in_len1 = any(L == 1 for L in core_lengths)
            in_len2 = any(L == 2 for L in core_lengths)
            min_len = min(core_lengths) if core_lengths else 1

            # Determine candidate predicate pool from positional sentence length statistics
            if in_len1:
                cand_pool = sorted(all_kinds)
            elif in_len2 and min_len == 2:
                cand_pool = sorted(all_mods | all_kinds)
            elif min_len >= 3:
                cand_pool = sorted(all_rels | all_mods | all_kinds)
            else:
                cand_pool = all_preds

            best_p = cand_pool[0] if cand_pool else "Q"
            best_score = -1e9
            for p in cand_pool:
                score = 0.0
                for o in occs:
                    core = strip_q(o.tokens)
                    present = p in self._scene_predicates(o.scene)
                    weight = 2.0 if len(core) <= 2 else 1.0
                    if o.label:
                        score += weight * (1.0 if present else -2.0)
                    else:
                        score += weight * (1.2 if not present else -0.2)
                if score > best_score:
                    best_score = score
                    best_p = p
            self._word_pred[w] = best_p

        # Step 3: Refine modifier scores using per-object co-occurrence with adjacent kind words
        for w, occs in word_occs.items():
            if w in q_words:
                continue
            cur_p = self._word_pred.get(w, "")
            if cur_p.startswith("kind:") and any(len(strip_q(o.tokens)) == 1 for o in occs):
                continue
            if min(len(strip_q(o.tokens)) for o in occs) >= 3 and cur_p.startswith("rel:"):
                continue

            # Evaluate modifier candidates against adjacent noun kind on the SAME object
            best_mp = cur_p
            best_ms = -1e9
            for mp in sorted(all_mods):
                fam, val = mp.split(":", 1)
                s = 0.0
                for o in occs:
                    core = strip_q(o.tokens)
                    if w not in core:
                        continue
                    w_idx = core.index(w)
                    # Find nearest noun token in core
                    noun_tokens = [
                        (abs(i - w_idx), t)
                        for i, t in enumerate(core)
                        if i != w_idx and self._word_pred.get(t, "").startswith("kind:")
                    ]
                    if noun_tokens:
                        noun_tokens.sort(key=lambda item: item[0])
                        near_k = self._word_pred[noun_tokens[0][1]].split(":", 1)[1]
                        holds = any(
                            obj.kind == near_k and obj.value(fam) == val
                            for obj in o.scene.objects
                        )
                    else:
                        holds = any(obj.value(fam) == val for obj in o.scene.objects)

                    if len(core) == 2:
                        s += 2.0 if (holds == o.label) else -2.0
                    else:
                        if o.label:
                            s += 1.0 if holds else -2.0
                        else:
                            s += 0.8 if not holds else -0.1
                if s > best_ms:
                    best_ms = s
                    best_mp = mp
            if best_ms > 0.0 and not any(len(strip_q(o.tokens)) == 1 for o in occs):
                self._word_pred[w] = best_mp

        # Step 4: Estimate linear-order relation orientation ("fwd", "rev", "sym") for relation words
        for w, occs in word_occs.items():
            p = self._word_pred.get(w, "")
            if not p.startswith("rel:"):
                continue
            # Also re-score which relation name fits best with per-object bundles
            best_rel_name = p.split(":", 1)[1]
            best_mode = "fwd"
            best_rel_score = -1e9
            for r_pred in sorted(all_rels):
                r_name = r_pred.split(":", 1)[1]
                for mode in ("fwd", "sym", "rev"):
                    tie = 0.02 if mode == "fwd" else (0.01 if mode == "sym" else 0.0)
                    sc = tie
                    for o in occs:
                        core = strip_q(o.tokens)
                        bundles, _ = self._extract_bundles(core)
                        if len(bundles) < 2:
                            continue
                        (k1, attrs1), (k2, attrs2) = bundles[0], bundles[1]
                        holds = self._check_relation(
                            o.scene, r_name, mode, k1, attrs1, k2, attrs2
                        )
                        sc += 2.0 if (holds == o.label) else -2.0
                    if sc > best_rel_score:
                        best_rel_score = sc
                        best_rel_name = r_name
                        best_mode = mode
            self._word_pred[w] = f"rel:{best_rel_name}"
            self._rel_mode[w] = best_mode

    def _extract_bundles(
        self, tokens: tuple[str, ...]
    ) -> tuple[list[tuple[str, list[tuple[str, str]]]], list[str]]:
        """Group tokens into per-object (kind, [(fam, val), ...]) bundles by nearest adjacency."""
        noun_positions: list[tuple[int, str]] = []
        mod_positions: list[tuple[int, str, str]] = []
        rel_words: list[str] = []

        for idx, t in enumerate(tokens):
            p = self._word_pred.get(t)
            if p is None or p == "Q":
                continue
            if p.startswith("kind:"):
                noun_positions.append((idx, p.split(":", 1)[1]))
            elif p.startswith("color:") or p.startswith("size:"):
                fam, val = p.split(":", 1)
                mod_positions.append((idx, fam, val))
            elif p.startswith("rel:"):
                rel_words.append(t)

        if not noun_positions:
            return [], rel_words

        bundles: list[tuple[str, list[tuple[str, str]]]] = [
            (k, []) for _, k in noun_positions
        ]
        for m_idx, fam, val in mod_positions:
            # Bind modifier to nearest noun token in linear distance
            best_b = min(
                range(len(noun_positions)),
                key=lambda b_i: abs(m_idx - noun_positions[b_i][0]),
            )
            bundles[best_b][1].append((fam, val))

        return bundles, rel_words

    @staticmethod
    def _check_relation(
        scene: Scene,
        r_name: str,
        mode: str,
        k1: str,
        attrs1: list[tuple[str, str]],
        k2: str,
        attrs2: list[tuple[str, str]],
    ) -> bool:
        objs = scene.objects
        m1 = [
            i
            for i, o in enumerate(objs)
            if o.kind == k1 and all(o.value(f) == v for f, v in attrs1)
        ]
        m2 = [
            j
            for j, o in enumerate(objs)
            if o.kind == k2 and all(o.value(f) == v for f, v in attrs2)
        ]
        for i in m1:
            for j in m2:
                if i == j:
                    continue
                if mode == "fwd" and scene.holds(r_name, i, j, False):
                    return True
                if mode == "rev" and scene.holds(r_name, j, i, False):
                    return True
                if mode == "sym" and scene.holds(r_name, i, j, True):
                    return True
        return False

    def predict(self, scene: Scene, tokens: tuple[str, ...]) -> Prediction:
        self._fit()
        if any(t not in self._word_pred for t in tokens):
            return Prediction(0.5, None)

        bundles, rel_words = self._extract_bundles(tokens)
        if not bundles:
            return Prediction(0.5, None)

        # Check sortal exclusivity within each bundle (cannot have 2 distinct values of same family)
        for _, attrs in bundles:
            seen_fams: dict[str, str] = {}
            for fam, val in attrs:
                if fam in seen_fams and seen_fams[fam] != val:
                    return Prediction(0.5, None)
                seen_fams[fam] = val

        # Multi-relation sentences are not supported by single-relation positional linking
        if len(rel_words) > 1:
            return Prediction(0.5, None)

        if len(bundles) == 1 and not rel_words:
            k1, attrs1 = bundles[0]
            holds = any(
                o.kind == k1 and all(o.value(f) == v for f, v in attrs1)
                for o in scene.objects
            )
            p_true = (1.0 - self.epsilon) if holds else self.epsilon
            return Prediction(p_true, None)

        if len(bundles) == 2 and len(rel_words) == 1:
            w_rel = rel_words[0]
            r_name = self._word_pred[w_rel].split(":", 1)[1]
            mode = self._rel_mode.get(w_rel, "fwd")
            (k1, attrs1), (k2, attrs2) = bundles[0], bundles[1]
            holds = self._check_relation(scene, r_name, mode, k1, attrs1, k2, attrs2)
            p_true = (1.0 - self.epsilon) if holds else self.epsilon
            return Prediction(p_true, None)

        return Prediction(0.5, None)

    def model_size(self) -> int:
        self._fit()
        return len(self._word_pred)
