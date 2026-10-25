"""NEXUS-0 Phase 1 Learner: Categorial Grammar & Lexicon Induction via MDL Beam Search.

Mechanism summary:
- Syntactic hypothesis space: Categorial grammar over ``{N, NP, S}`` with slash depth <= 2
  (``N``, ``N/N``, ``N\\N``, ``S/S``, ``S\\S``, ``(S\\NP)\\NP``, ``(S\\NP)/NP``, ``(S/NP)/NP``),
  using Forward Application (``>``) and Backward Application (``<``) only.
- Semantic hypothesis space:
  - Unary predicates: ``kind:<v>``, ``color:<v>``, ``size:<v>``
  - Binary relations: ``rel:<r>(x0,x1)``, ``rel:<r>(x1,x0)``, ``rel:<r>:sym(x0,x1)``
  - Speech-act marker: ``Q``
- Scoring:
  - MDL prior: ``-alpha * (|lexicon| + sum(category_depths) + sum(atom_counts))``
  - Log-likelihood with noise ``epsilon = 0.02``:
    ``log(1 - epsilon)`` if ``eval(LF, scene) == label`` else ``log(epsilon)``
- Search & Prediction:
  - Beam of up to ``K = 200`` joint lexicon/grammar hypotheses.
  - Proposals generated online from True-vs-False scene contrasts and distributional spans.
  - Predictions computed by Bayesian model averaging (BMA) over the beam.
  - Full JSONL experience log stored so ``correct()`` replays and updates beliefs cleanly.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from nexus.eval.learner_api import Prediction
from nexus.learner.categorial import (
    CATEGORY_DEPTHS,
    LexicalEntry,
    ParseNode,
    category_depth,
    evaluate_parsed_lf,
    find_witness_or_failure,
    parse_all_tokens,
    parse_tokens,
)
from nexus.learner.explain import ChangeRecord, Explanation
from nexus.world.example import Correction, Observation
from nexus.world.logic import LogicalForm
from nexus.world.scene import Scene


@dataclass
class _ObsEntry:
    """Internal mutable observation record in the experience log."""

    example_id: str
    scene: Scene
    tokens: tuple[str, ...]
    label: bool
    source: str
    corrected: bool = False


@dataclass(frozen=True)
class Hypothesis:
    """A joint lexicon-and-syntax hypothesis in the beam."""

    entries: Mapping[str, LexicalEntry]
    prior_score: float
    likelihood_score: float
    total_score: float
    signature: tuple[tuple[int, str, str], ...]

    def short_id(self) -> str:
        """Compact deterministic identifier for provenance tracking."""
        items: list[str] = []
        for w, e in self.entries.items():
            if e.secondary_predicate is not None:
                items.append(
                    f"{w}:{e.category}={e.predicate}|{e.secondary_category}={e.secondary_predicate}"
                )
            else:
                items.append(f"{w}:{e.category}={e.predicate}")
        return "{" + ", ".join(items) + "}"

    def to_json(self) -> dict[str, Any]:
        return {
            "entries": {w: e.to_json() for w, e in self.entries.items()},
            "prior_score": self.prior_score,
            "likelihood_score": self.likelihood_score,
            "total_score": self.total_score,
        }


def compute_mdl_prior(entries: Mapping[str, LexicalEntry], alpha: float) -> float:
    """Compute exact MDL prior ``-alpha * (|lexicon| + sum(depths) + sum(atom_counts))``."""
    n_entries = sum(e.entry_count for e in entries.values() if e.predicate != "NULL")
    sum_depths = sum(e.depth for e in entries.values() if e.predicate != "NULL")
    sum_atoms = sum(e.atom_count for e in entries.values())
    return -alpha * float(n_entries + sum_depths + sum_atoms)


def compute_bit_cost_prior(
    entries: Mapping[str, LexicalEntry],
    num_categories: int = 8,
    num_predicates: int = 23,
    in_nats: bool = True,
) -> float:
    """Compute parameter-free information-theoretic MDL prior from inventory sizes.

    Each lexical entry (or sense) encodes:
    - syntactic category from ``num_categories``: ``log2(|C|) + 2 * depth(e)`` bits
    - semantic predicate atoms from ``num_predicates``: ``atom_count(e) * log2(|P|)`` bits
    Null/filler entries (``predicate == "NULL"``) encode 0 content atoms and 1 bit of syntactic
    passthru overhead. Converted to nats via ``ln(2)`` so it is on the same natural-log scale as
    the Bernoulli log-likelihood ``sum log P(y_i | e_i, h)``.
    """
    if not entries:
        return 0.0
    cat_bits = math.log2(max(2, num_categories))
    pred_bits = math.log2(max(2, num_predicates))
    total_bits = 0.0
    for e in entries.values():
        if e.predicate == "NULL":
            total_bits += 1.0
        else:
            total_bits += (
                float(e.entry_count) * cat_bits
                + 2.0 * float(e.depth)
                + float(e.atom_count) * pred_bits
            )
    scale = math.log(2.0) if in_nats else 1.0
    return -scale * total_bits


class NEXUSLearner:
    """Inspectable categorial grammar and lexicon learner with MDL beam search."""

    name: str = "nexus_phase1"

    def __init__(
        self,
        seed: int = 0,
        beam_width: int = 200,
        alpha: float = 0.05,
        epsilon: float = 0.02,
        max_slash_depth: int = 2,
        given_lexicon: Mapping[str, str] | None = None,
        given_syntax: Mapping[str, str] | None = None,
        max_candidates_per_word: int = 5,
        prior_mode: str = "linear",
        evidence_channel: str = "legacy",
        cooc_weight: float = 0.25,
        allow_multi_sense: bool = False,
        allow_null_entries: bool = False,
        allow_synonyms: bool = False,
    ) -> None:
        self.seed = seed
        self.beam_width = int(beam_width)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.max_slash_depth = int(max_slash_depth)
        self.given_lexicon = dict(given_lexicon) if given_lexicon is not None else None
        self.given_syntax = dict(given_syntax) if given_syntax is not None else None
        self.max_candidates_per_word = int(max_candidates_per_word)
        self.prior_mode = str(prior_mode)
        self.evidence_channel = str(evidence_channel)
        self.cooc_weight = float(cooc_weight)
        self.allow_multi_sense = bool(allow_multi_sense)
        self.allow_null_entries = bool(allow_null_entries)
        self.allow_synonyms = bool(allow_synonyms)

        self._observations: list[_ObsEntry] = []
        self._obs_by_id: dict[str, int] = {}
        self.experience_log_jsonl: list[str] = []

        # Encounter-order index for strict lexicon-permutation invariance
        self._word_order: dict[str, int] = {}
        self._beam: list[Hypothesis] = []
        self._beam_weights: list[float] = []
        self._beam_dirty: bool = True

        # Provenance change records indexed by example_id
        self._change_records: dict[str, ChangeRecord] = {}

        # Caches for fast parsing and evaluation
        self._parse_cache: dict[
            tuple[tuple[str, ...], tuple[tuple[str, str, str, str | None, str | None], ...]],
            tuple[ParseNode, LogicalForm, bool] | None,
        ] = {}
        self._eval_cache: dict[
            tuple[str, tuple[tuple[str, str, str, str | None, str | None], ...]],
            bool | None,
        ] = {}

    # ------------------------------------------------------------------
    # Learner Protocol
    # ------------------------------------------------------------------

    def _register_tokens(self, tokens: Sequence[str]) -> None:
        for tok in tokens:
            if tok not in self._word_order:
                self._word_order[tok] = len(self._word_order)

    def observe(self, observation: Observation) -> None:
        """Record one observation in the JSONL experience log and mark beam for update."""
        self._register_tokens(observation.tokens)
        entry = _ObsEntry(
            example_id=observation.example_id,
            scene=observation.scene,
            tokens=observation.tokens,
            label=observation.label,
            source=observation.source,
            corrected=False,
        )
        idx = len(self._observations)
        self._observations.append(entry)
        self._obs_by_id[observation.example_id] = idx
        self.experience_log_jsonl.append(
            json.dumps(
                {
                    "event": "observe",
                    "seq": len(self.experience_log_jsonl),
                    "example_id": observation.example_id,
                    "tokens": list(observation.tokens),
                    "label": observation.label,
                    "source": observation.source,
                    "scene": observation.scene.to_json(),
                },
                sort_keys=True,
            )
        )
        self._beam_dirty = True

    def correct(self, correction: Correction) -> None:
        """Apply corrective feedback by updating the experience log and marking beam for replay."""
        self.experience_log_jsonl.append(
            json.dumps(
                {
                    "event": "correct",
                    "seq": len(self.experience_log_jsonl),
                    "target_example_id": correction.target_example_id,
                    "corrected_label": correction.corrected_label,
                    "source": correction.source,
                    "reason": correction.reason,
                },
                sort_keys=True,
            )
        )

        if correction.target_example_id in self._obs_by_id:
            idx = self._obs_by_id[correction.target_example_id]
            self._observations[idx].label = correction.corrected_label
            self._observations[idx].corrected = True
            self._change_records.pop(correction.target_example_id, None)
            self._beam_dirty = True

    def predict(self, scene: Scene, tokens: tuple[str, ...]) -> Prediction:
        """Predict truth probability via Bayesian model averaging over the beam."""
        self._ensure_beam()
        if not self._beam:
            return Prediction(0.5, None)

        p_total = 0.0
        w_total = 0.0
        top_lf: LogicalForm | None = None
        lf_truth_cache: dict[tuple[LogicalForm, bool], bool] = {}

        for idx, (hyp, weight) in enumerate(zip(self._beam, self._beam_weights)):
            if idx > 0 and weight < 1e-5:
                break
            w_total += weight
            if self.allow_multi_sense and any(
                t in hyp.entries and hyp.entries[t].secondary_predicate is not None for t in tokens
            ):
                all_p = parse_all_tokens(tokens, hyp.entries)
                if not all_p:
                    p_hyp = 0.5
                else:
                    if top_lf is None:
                        top_lf = all_p[0][1]
                    truth = any(evaluate_parsed_lf(lf_i, scene, sym_i) for _, lf_i, sym_i in all_p)
                    p_hyp = (1.0 - self.epsilon) if truth else self.epsilon
            else:
                parsed = self._parse_with_hypothesis(tokens, hyp.entries)
                if parsed is None:
                    p_hyp = 0.5
                else:
                    _, lf, is_sym = parsed
                    if top_lf is None:
                        top_lf = lf
                    lf_key = (lf, is_sym)
                    if lf_key not in lf_truth_cache:
                        lf_truth_cache[lf_key] = evaluate_parsed_lf(lf, scene, is_sym)
                    truth = lf_truth_cache[lf_key]
                    p_hyp = (1.0 - self.epsilon) if truth else self.epsilon
            p_total += weight * p_hyp

        if w_total > 0.0:
            p_total /= w_total
        p_clamped = min(1.0, max(0.0, p_total))
        return Prediction(p_clamped, top_lf)

    def model_size(self) -> int:
        """Number of non-null lexical senses in the highest-posterior hypothesis."""
        self._ensure_beam()
        if not self._beam:
            return 0
        return sum(
            e.entry_count
            for e in self._beam[0].entries.values()
            if e.predicate != "NULL"
        )

    # ------------------------------------------------------------------
    # Explainability & Provenance
    # ------------------------------------------------------------------

    def explain(
        self,
        sentence_or_scene: Sequence[str] | str | Scene,
        scene_or_sentence: Scene | Sequence[str] | str,
    ) -> Explanation:
        """Return parse tree, lexical entries with supporting IDs, LF, witness/failure, and weight."""
        if isinstance(sentence_or_scene, Scene):
            scene = sentence_or_scene
            raw_sent = scene_or_sentence
        else:
            raw_sent = sentence_or_scene
            assert isinstance(scene_or_sentence, Scene)
            scene = scene_or_sentence

        if isinstance(raw_sent, str):
            tokens = tuple(raw_sent.split())
        else:
            tokens = tuple(raw_sent)

        self._ensure_beam()
        pred = self.predict(scene, tokens)

        if not self._beam:
            return Explanation(
                tokens=tokens,
                parse_tree=None,
                lexical_entries=(),
                derived_lf=None,
                is_rel_symmetric=False,
                truth=pred.truth,
                p_true=pred.p_true,
                posterior_weight=0.0,
                witness=None,
                failure_reason="No hypotheses in beam.",
            )

        # Find highest-posterior hypothesis that parses the sentence (or fallback to top hypothesis)
        chosen_hyp = self._beam[0]
        chosen_weight = self._beam_weights[0]
        parsed = self._parse_with_hypothesis(tokens, chosen_hyp.entries)
        if parsed is None:
            for hyp, w in zip(self._beam, self._beam_weights):
                cand = self._parse_with_hypothesis(tokens, hyp.entries)
                if cand is not None:
                    chosen_hyp = hyp
                    chosen_weight = w
                    parsed = cand
                    break

        lex_entries = tuple(
            chosen_hyp.entries[t] for t in tokens if t in chosen_hyp.entries
        )

        if parsed is None:
            missing = [t for t in tokens if t not in chosen_hyp.entries]
            reason = (
                f"Unknown words: {missing}"
                if missing
                else "No valid categorial derivation (> or <) spans the sentence."
            )
            return Explanation(
                tokens=tokens,
                parse_tree=None,
                lexical_entries=lex_entries,
                derived_lf=None,
                is_rel_symmetric=False,
                truth=pred.truth,
                p_true=pred.p_true,
                posterior_weight=chosen_weight,
                witness=None,
                failure_reason=reason,
            )

        root_node, lf, is_sym = parsed
        witness, reason = find_witness_or_failure(lf, scene, is_sym)
        return Explanation(
            tokens=tokens,
            parse_tree=root_node.to_tree_dict(),
            lexical_entries=lex_entries,
            derived_lf=lf,
            is_rel_symmetric=is_sym,
            truth=witness is not None,
            p_true=pred.p_true,
            posterior_weight=chosen_weight,
            witness=witness,
            failure_reason=reason,
        )

    def what_changed(self, example_id: str) -> ChangeRecord:
        """Return hypotheses promoted, demoted, or created when ``example_id`` was processed."""
        if example_id in self._change_records:
            return self._change_records[example_id]

        if example_id not in self._obs_by_id:
            return ChangeRecord(
                example_id=example_id,
                event_kind="unknown",
                promoted=(),
                demoted=(),
                created=(),
            )

        idx = self._obs_by_id[example_id]
        obs_entry = self._observations[idx]
        if obs_entry.corrected:
            self._ensure_beam()
            beam_after = list(self._beam)
            weights_after = list(self._beam_weights)
            # Reconstruct pre-correction observation list
            obs_pre = list(self._observations)
            obs_pre[idx] = _ObsEntry(
                example_id=obs_entry.example_id,
                scene=obs_entry.scene,
                tokens=obs_entry.tokens,
                label=not obs_entry.label,
                source=obs_entry.source,
                corrected=False,
            )
            beam_before, weights_before = self._build_beam_for_observations(obs_pre)
            rec = self._diff_beams(
                example_id,
                "correct",
                beam_before,
                weights_before,
                beam_after,
                weights_after,
            )
            self._change_records[example_id] = rec
            return rec

        obs_before = self._observations[:idx]
        obs_after = self._observations[: idx + 1]
        beam_before, weights_before = self._build_beam_for_observations(obs_before)
        beam_after, weights_after = self._build_beam_for_observations(obs_after)
        rec = self._diff_beams(
            example_id,
            "observe",
            beam_before,
            weights_before,
            beam_after,
            weights_after,
        )
        self._change_records[example_id] = rec
        return rec

    def top_hypothesis(self) -> Hypothesis | None:
        """Return the highest-posterior hypothesis in the beam."""
        self._ensure_beam()
        return self._beam[0] if self._beam else None

    def beam(self) -> list[Hypothesis]:
        """Return current beam of hypotheses ordered from highest to lowest posterior."""
        self._ensure_beam()
        return list(self._beam)

    # ------------------------------------------------------------------
    # Internal Beam Construction & Contrast Proposal Engine
    # ------------------------------------------------------------------

    def _ensure_beam(self) -> None:
        if not self._beam_dirty:
            return
        self._beam, self._beam_weights = self._build_beam_for_observations(self._observations)
        self._beam_dirty = False

    def _diff_beams(
        self,
        example_id: str,
        event_kind: str,
        before_beam: Sequence[Hypothesis],
        before_weights: Sequence[float],
        after_beam: Sequence[Hypothesis],
        after_weights: Sequence[float],
    ) -> ChangeRecord:
        before_map = {h.signature: (rank, w, h) for rank, (h, w) in enumerate(zip(before_beam, before_weights))}
        promoted: list[str] = []
        demoted: list[str] = []
        created: list[str] = []

        for after_rank, (h, after_w) in enumerate(zip(after_beam[:20], after_weights[:20])):
            if h.signature not in before_map:
                created.append(h.short_id())
            else:
                before_rank, before_w, _ = before_map[h.signature]
                if after_rank < before_rank or after_w > before_w + 1e-4:
                    promoted.append(h.short_id())
                elif after_rank > before_rank or after_w < before_w - 1e-4:
                    demoted.append(h.short_id())

        after_sigs = {h.signature for h in after_beam[:20]}
        for before_rank, (h, _) in enumerate(zip(before_beam[:10], before_weights[:10])):
            if h.signature not in after_sigs:
                demoted.append(h.short_id())

        return ChangeRecord(
            example_id=example_id,
            event_kind=event_kind,
            promoted=tuple(promoted[:10]),
            demoted=tuple(demoted[:10]),
            created=tuple(created[:10]),
        )

    def _allowed_category(self, word: str, cat: str) -> bool:
        if category_depth(cat) > self.max_slash_depth:
            return False
        if self.given_syntax is not None and word in self.given_syntax:
            return self.given_syntax[word] == cat
        return True

    def _allowed_predicate(self, word: str, pred: str) -> bool:
        if self.given_lexicon is not None and word in self.given_lexicon:
            given = self.given_lexicon[word]
            if pred == given:
                return True
            # If given lexicon specifies base relation "rel:r", allow directional variants
            if given.startswith("rel:") and "(" not in given:
                base = pred.split("(", 1)[0].replace(":sym", "")
                return base == given
            return False
        return True

    def _propose_word_candidates_channel(
        self, observations: Sequence[_ObsEntry]
    ) -> dict[str, list[tuple[LexicalEntry, float]]]:
        """Induce candidate LexicalEntry items per word for 'truth', 'cooc', or 'both' channels.

        - 'truth': Strictly label-centered correlation between predicate truth in scene
          and observation label ``y_i - y_bar(w)``. Zero expected score under random labels.
        - 'cooc': IBM Model 1 / Cross-Situational Referential Alignment Model (pointwise mutual
          information between word w in U and predicate p in F(S)), ignoring ``obs.label``.
        - 'both': ``S_truth + cooc_weight * S_cooc``.
        """
        log_hit = math.log(2.0 * (1.0 - self.epsilon))
        use_truth = self.evidence_channel in ("truth", "both")
        use_cooc = self.evidence_channel in ("cooc", "both")
        w_cooc = 1.0 if self.evidence_channel == "cooc" else self.cooc_weight

        words_seen: list[str] = []
        seen_set: set[str] = set()
        for obs in observations:
            for t in obs.tokens:
                if t not in seen_set:
                    seen_set.add(t)
                    words_seen.append(t)

        if not words_seen:
            return {}

        n_total_obs = max(1, len(observations))

        # Discover perceptual inventory and background scene frequencies P(p in F(S))
        all_kinds: set[str] = set()
        all_colors: set[str] = set()
        all_sizes: set[str] = set()
        all_rels: set[str] = set()
        bg_pred_counts: dict[str, int] = defaultdict(int)
        for obs in observations:
            scene_preds: set[str] = set()
            for o in obs.scene.objects:
                kp = f"kind:{o.kind}"
                cp = f"color:{o.color}"
                sp = f"size:{o.size}"
                all_kinds.add(kp)
                all_colors.add(cp)
                all_sizes.add(sp)
                scene_preds.add(kp)
                scene_preds.add(cp)
                scene_preds.add(sp)
            for r in obs.scene.relations:
                all_rels.add(r.name)
                scene_preds.add(f"rel:{r.name}")
            for sp_name in scene_preds:
                bg_pred_counts[sp_name] += 1

        kinds_sorted = sorted(all_kinds)
        mods_sorted = sorted(all_colors | all_sizes)
        rels_sorted = sorted(all_rels)
        bg_prob: dict[str, float] = {
            k: (v + 0.5) / (n_total_obs + 1.0) for k, v in bg_pred_counts.items()
        }

        word_occ: dict[str, list[_ObsEntry]] = defaultdict(list)
        for obs in observations:
            for t in set(obs.tokens):
                word_occ[t].append(obs)

        # Empirical label mean per word for exact zero-mean centering in 'truth' mode
        word_y_bar: dict[str, float] = {}
        for w in words_seen:
            occs = word_occ[w]
            if not occs:
                word_y_bar[w] = 0.5
            elif len(occs) >= 8:
                word_y_bar[w] = sum(1.0 if o.label else 0.0 for o in occs) / len(occs)
            else:
                word_y_bar[w] = 0.5

        def y_centered(w: str, obs: _ObsEntry) -> float:
            y_val = 1.0 if obs.label else 0.0
            # Scale by 2.0 so balanced 0.5 mean gives +1.0 / -1.0
            return 2.0 * (y_val - word_y_bar[w])

        # ------------------------------------------------------------------
        # Phase 1: Identify peripheral speech-act marker candidates ("Q")
        # ------------------------------------------------------------------
        q_candidates: dict[str, tuple[str, float, tuple[str, ...]]] = {}
        for w in words_seen:
            occs = word_occ[w]
            if not occs or any(len(o.tokens) == 1 for o in occs):
                continue
            all_initial = all(o.tokens[0] == w for o in occs)
            all_final = all(o.tokens[-1] == w for o in occs)
            if not (all_initial or all_final):
                continue
            lengths = {len(o.tokens) for o in occs}
            # Distributional invariance across all occurrences of w (unlabeled)
            max_inv_rate = 0.0
            for p in kinds_sorted + mods_sorted:
                fam, val = p.split(":", 1)
                rate = sum(any(obj.value(fam) == val for obj in o.scene.objects) for o in occs) / len(occs)
                if rate > max_inv_rate:
                    max_inv_rate = rate
            for r_name in rels_sorted:
                rate = sum(any(rel.name == r_name for rel in o.scene.relations) for o in occs) / len(occs)
                if rate > max_inv_rate:
                    max_inv_rate = rate

            cat = "S/S" if all_initial else "S\\S"
            if self._allowed_category(w, cat) and self._allowed_predicate(w, "Q"):
                is_given_q = self.given_lexicon is not None and self.given_lexicon.get(w) == "Q"
                if is_given_q or (min(lengths) == 2 and max(lengths) >= 4 and max_inv_rate < 0.85):
                    supp = tuple(o.example_id for o in occs[:10])
                    bonus = 10.0 * len(occs) * log_hit if len(occs) >= 20 else 0.0
                    q_candidates[w] = (cat, bonus, supp)

        primary_q_words = {w for w, (_, bonus, _) in q_candidates.items() if bonus > 0.0}

        def strip_q(tokens: tuple[str, ...]) -> tuple[str, ...]:
            if len(tokens) >= 2 and tokens[0] in primary_q_words:
                return tokens[1:]
            if len(tokens) >= 2 and tokens[-1] in primary_q_words:
                return tokens[:-1]
            return tokens

        # ------------------------------------------------------------------
        # Phase 2: Induce Noun ("N") candidates
        # ------------------------------------------------------------------
        noun_scores: dict[str, dict[str, float]] = {w: {} for w in words_seen}
        noun_support: dict[str, dict[str, list[str]]] = {w: defaultdict(list) for w in words_seen}
        core_len1_counts: dict[str, int] = defaultdict(int)

        for obs in observations:
            core = strip_q(obs.tokens)
            if len(core) == 1:
                core_len1_counts[core[0]] += 1

        for w in words_seen:
            if w in primary_q_words or not self._allowed_category(w, "N"):
                continue
            occs = word_occ[w]
            if not occs:
                continue
            for k_pred in kinds_sorted:
                if not self._allowed_predicate(w, k_pred):
                    continue
                _, k_val = k_pred.split(":", 1)
                s_truth = 0.0
                s_cooc = 0.0
                supp: list[str] = []
                p_bg = bg_prob.get(k_pred, 0.25)
                cooc_hits = 0
                for obs in occs:
                    core = strip_q(obs.tokens)
                    present = any(o.kind == k_val for o in obs.scene.objects)
                    h_sign = 1.0 if present else -1.0
                    yc = y_centered(w, obs)
                    w_len = 2.0 if (len(core) == 1 and core[0] == w) else 0.75
                    s_truth += w_len * yc * h_sign * log_hit
                    if present:
                        cooc_hits += 1
                    if (use_truth and (present == obs.label)) or (not use_truth and present):
                        supp.append(obs.example_id)

                p_emp = (cooc_hits + 0.5) / (len(occs) + 1.0)
                pmi = math.log(p_emp / p_bg)
                s_cooc = float(len(occs)) * pmi * log_hit

                total_s = (s_truth if use_truth else 0.0) + (w_cooc * s_cooc if use_cooc else 0.0)
                noun_scores[w][k_pred] = total_s
                noun_support[w][k_pred] = supp

        best_noun_pred: dict[str, tuple[str, float]] = {}
        likely_nouns: set[str] = set()
        for w in words_seen:
            if noun_scores[w]:
                top_k, top_s = max(noun_scores[w].items(), key=lambda kv: kv[1])
                best_noun_pred[w] = (top_k, top_s)
                if core_len1_counts[w] > 0 or (
                    self.given_syntax is not None and self.given_syntax.get(w) == "N"
                ):
                    likely_nouns.add(w)

        # Fallback noun detection if a noun never appeared in a 1-token sentence
        for w in words_seen:
            if w in likely_nouns or w in primary_q_words:
                continue
            occs = word_occ[w]
            if not occs:
                continue
            if use_cooc:
                true_occs = [o for o in occs if o.label] if use_truth else occs
                if not true_occs:
                    continue
                max_kind_rate = max(
                    (
                        sum(any(obj.kind == kp.split(":", 1)[1] for obj in o.scene.objects) for o in true_occs)
                        / len(true_occs)
                        for kp in kinds_sorted
                    ),
                    default=0.0,
                )
                max_mod_rate = max(
                    (
                        sum(
                            any(obj.value(mp.split(":", 1)[0]) == mp.split(":", 1)[1] for obj in o.scene.objects)
                            for o in true_occs
                        )
                        / len(true_occs)
                        for mp in mods_sorted
                    ),
                    default=0.0,
                )
                max_rel_rate = max(
                    (
                        sum(any(rel.name == r_name for rel in o.scene.relations) for o in true_occs)
                        / len(true_occs)
                        for r_name in rels_sorted
                    ),
                    default=0.0,
                )
                if max_kind_rate >= 0.95 and max_mod_rate < 0.85 and max_rel_rate < 0.85:
                    likely_nouns.add(w)
            elif use_truth and w in best_noun_pred and best_noun_pred[w][1] > 1.5 * log_hit:
                # Only promote to noun if its best kind contrast exceeds its best modifier contrast
                max_mod_truth = 0.0
                for mp in mods_sorted:
                    fam, val = mp.split(":", 1)
                    st = sum(
                        y_centered(w, o) * (1.0 if any(obj.value(fam) == val for obj in o.scene.objects) else -1.0)
                        for o in occs
                    ) * log_hit
                    if st > max_mod_truth:
                        max_mod_truth = st
                if best_noun_pred[w][1] > max_mod_truth + 1.0 * log_hit:
                    likely_nouns.add(w)

        # ------------------------------------------------------------------
        # Phase 3: Induce Modifier ("N/N", "N\N") candidates
        # ------------------------------------------------------------------
        mod_scores: dict[str, dict[tuple[str, str], float]] = {w: {} for w in words_seen}
        mod_support: dict[str, dict[tuple[str, str], list[str]]] = {
            w: defaultdict(list) for w in words_seen
        }

        for w in words_seen:
            if w in primary_q_words or w in likely_nouns:
                continue
            occs = word_occ[w]
            if not occs:
                continue
            pre_count = 0
            post_count = 0
            for obs in occs:
                core = strip_q(obs.tokens)
                if len(core) == 2 and w in core:
                    other = core[1] if core[0] == w else core[0]
                    if other in likely_nouns:
                        if core[0] == w:
                            pre_count += 1
                        else:
                            post_count += 1
                elif len(core) >= 3 and w in core:
                    idx = core.index(w)
                    if idx + 1 < len(core) and core[idx + 1] in likely_nouns:
                        pre_count += 1
                    if idx - 1 >= 0 and core[idx - 1] in likely_nouns:
                        post_count += 1

            allowed_cats: list[str] = []
            if pre_count > 0 or post_count == 0:
                if self._allowed_category(w, "N/N"):
                    allowed_cats.append("N/N")
            if post_count > 0 or pre_count == 0:
                if self._allowed_category(w, "N\\N"):
                    allowed_cats.append("N\\N")

            for cat in allowed_cats:
                dir_diff = (pre_count - post_count) if cat == "N/N" else (post_count - pre_count)
                if dir_diff < 0:
                    continue
                for m_pred in mods_sorted:
                    if not self._allowed_predicate(w, m_pred):
                        continue
                    fam, val = m_pred.split(":", 1)
                    s_truth = 0.0
                    s_cooc = 0.0
                    supp: list[str] = []
                    p_bg = bg_prob.get(m_pred, 0.30)
                    cooc_hits = 0
                    for obs in occs:
                        core = strip_q(obs.tokens)
                        partner_kind: str | None = None
                        if len(core) == 2 and w in core:
                            other = core[1] if core[0] == w else core[0]
                            if other in likely_nouns and other in best_noun_pred:
                                partner_kind = best_noun_pred[other][0].split(":", 1)[1]
                        elif len(core) >= 3 and w in core:
                            idx = core.index(w)
                            if cat == "N/N" and idx + 1 < len(core) and core[idx + 1] in likely_nouns:
                                partner_kind = best_noun_pred[core[idx + 1]][0].split(":", 1)[1]
                            elif cat == "N\\N" and idx - 1 >= 0 and core[idx - 1] in likely_nouns:
                                partner_kind = best_noun_pred[core[idx - 1]][0].split(":", 1)[1]

                        if partner_kind is not None:
                            holds = any(
                                o.kind == partner_kind and o.value(fam) == val
                                for o in obs.scene.objects
                            )
                            w_len = 2.0 if len(core) == 2 else 1.0
                        else:
                            holds = any(o.value(fam) == val for o in obs.scene.objects)
                            w_len = 0.5

                        h_sign = 1.0 if holds else -1.0
                        yc = y_centered(w, obs)
                        s_truth += w_len * yc * h_sign * log_hit
                        if holds:
                            cooc_hits += 1
                        if (use_truth and (holds == obs.label)) or (not use_truth and holds):
                            supp.append(obs.example_id)

                    p_emp = (cooc_hits + 0.5) / (len(occs) + 1.0)
                    # Joint object+attribute background is lower than attribute-alone
                    p_bg_eff = max(0.08, 0.5 * p_bg)
                    pmi = math.log(p_emp / p_bg_eff)
                    s_cooc = (float(len(occs)) * pmi + 0.2 * dir_diff) * log_hit
                    if use_truth and not use_cooc:
                        # Small tie-breaker for syntactic direction only when truth signal is positive
                        if s_truth > 0:
                            s_truth += 0.05 * dir_diff * log_hit

                    total_s = (s_truth if use_truth else 0.0) + (w_cooc * s_cooc if use_cooc else 0.0)
                    mod_scores[w][(cat, m_pred)] = total_s
                    mod_support[w][(cat, m_pred)] = supp

        likely_mods: dict[str, tuple[str, str, float]] = {}
        for w in words_seen:
            if w in likely_nouns or w in primary_q_words:
                continue
            if mod_scores[w]:
                (best_cat, best_mp), best_ms = max(mod_scores[w].items(), key=lambda kv: kv[1])
                in_len2 = any(len(strip_q(o.tokens)) == 2 for o in word_occ[w])
                if in_len2 or (
                    self.given_syntax is not None and self.given_syntax.get(w) in ("N/N", "N\\N")
                ) or best_ms > 0.75 * len(word_occ[w]) * log_hit:
                    likely_mods[w] = (best_cat, best_mp, best_ms)

        # ------------------------------------------------------------------
        # Phase 4: Induce Transitive Verb candidates
        # ------------------------------------------------------------------
        rel_scores: dict[str, dict[tuple[str, str], float]] = {w: {} for w in words_seen}
        rel_support: dict[str, dict[tuple[str, str], list[str]]] = {
            w: defaultdict(list) for w in words_seen
        }

        verb_cats = ["(S\\NP)\\NP", "(S\\NP)/NP", "(S/NP)/NP"]
        for w in words_seen:
            if w in primary_q_words or w in likely_nouns or w in likely_mods:
                continue
            occs = word_occ[w]
            if not any(len(strip_q(o.tokens)) >= 3 for o in occs):
                continue

            after_both = 0
            between_both = 0
            before_both = 0
            parsed_frames: list[
                tuple[_ObsEntry, str, list[tuple[str, str]], str, list[tuple[str, str]]]
            ] = []

            for obs in occs:
                core = strip_q(obs.tokens)
                if len(core) < 3 or w not in core:
                    continue
                w_idx = core.index(w)
                noun_positions = [
                    (i, t) for i, t in enumerate(core) if i != w_idx and t in likely_nouns and t in best_noun_pred
                ]
                if len(noun_positions) != 2:
                    continue
                (n1_idx, n1_tok), (n2_idx, n2_tok) = noun_positions[0], noun_positions[1]
                if w_idx > n2_idx:
                    after_both += 1
                elif n1_idx < w_idx < n2_idx:
                    between_both += 1
                elif w_idx < n1_idx:
                    before_both += 1

                k1 = best_noun_pred[n1_tok][0].split(":", 1)[1]
                k2 = best_noun_pred[n2_tok][0].split(":", 1)[1]
                adjs1: list[tuple[str, str]] = []
                adjs2: list[tuple[str, str]] = []
                for i, t in enumerate(core):
                    if i in (w_idx, n1_idx, n2_idx):
                        continue
                    if t in likely_mods:
                        m_cat, m_pred, _ = likely_mods[t]
                        mfam, mval = m_pred.split(":", 1)
                        if m_cat == "N/N":
                            if i < n1_idx:
                                adjs1.append((mfam, mval))
                            else:
                                adjs2.append((mfam, mval))
                        else:
                            if i > n2_idx or (i > n1_idx and abs(i - n2_idx) < abs(i - n1_idx)):
                                adjs2.append((mfam, mval))
                            else:
                                adjs1.append((mfam, mval))
                parsed_frames.append((obs, k1, adjs1, k2, adjs2))

            for v_cat in verb_cats:
                if not self._allowed_category(w, v_cat):
                    continue
                if v_cat == "(S\\NP)\\NP" and (between_both > 0 or before_both > 0) and after_both == 0:
                    continue
                if v_cat == "(S\\NP)/NP" and (after_both > 0 or before_both > 0) and between_both == 0:
                    continue
                if v_cat == "(S/NP)/NP" and (after_both > 0 or between_both > 0) and before_both == 0:
                    continue

                for r_name in rels_sorted:
                    variants = [
                        (f"rel:{r_name}(x0,x1)", False, False),
                        (f"rel:{r_name}(x1,x0)", False, True),
                        (f"rel:{r_name}:sym(x0,x1)", True, False),
                    ]
                    p_bg_rel = max(0.05, 0.4 * bg_prob.get(f"rel:{r_name}", 0.25))
                    for r_pred, is_sym, is_rev in variants:
                        if not self._allowed_predicate(w, r_pred):
                            continue
                        tie_pref = 0.02 if (not is_sym and not is_rev) else (0.01 if is_sym else 0.0)
                        s_truth = tie_pref
                        s_cooc = tie_pref
                        supp: list[str] = []
                        cooc_hits = 0
                        for obs, k1, adjs1, k2, adjs2 in parsed_frames:
                            holds = self._eval_transitive_frame(
                                obs.scene, r_name, is_sym, is_rev, k1, adjs1, k2, adjs2
                            )
                            h_sign = 1.0 if holds else -1.0
                            yc = y_centered(w, obs)
                            s_truth += 2.0 * yc * h_sign * log_hit
                            if holds:
                                cooc_hits += 1
                            if (use_truth and (holds == obs.label)) or (not use_truth and holds):
                                supp.append(obs.example_id)

                        n_pf = max(1, len(parsed_frames))
                        p_emp = (cooc_hits + 0.5) / (n_pf + 1.0)
                        pmi = math.log(p_emp / p_bg_rel)
                        s_cooc += float(n_pf) * pmi * log_hit
                        total_s = (s_truth if use_truth else 0.0) + (w_cooc * s_cooc if use_cooc else 0.0)
                        rel_scores[w][(v_cat, r_pred)] = total_s
                        rel_support[w][(v_cat, r_pred)] = supp

        likely_verbs: set[str] = {w for w in words_seen if rel_scores[w]}

        for w in words_seen:
            if w not in primary_q_words and w not in likely_mods and w not in likely_verbs:
                likely_nouns.add(w)

        # ------------------------------------------------------------------
        # Assemble top candidate LexicalEntry list per word
        # ------------------------------------------------------------------
        candidates_by_word: dict[str, list[tuple[LexicalEntry, float]]] = {}
        for w in words_seen:
            cands: list[tuple[LexicalEntry, float]] = []

            if w in primary_q_words:
                cat, bonus, supp = q_candidates[w]
                cands.append((LexicalEntry(w, cat, "Q", supp), bonus))
                if self.allow_null_entries and self._allowed_category(w, cat):
                    cands.append((LexicalEntry(w, cat, "NULL", ()), bonus - 0.1))
            else:
                if w in likely_verbs and rel_scores[w]:
                    ranked_rels = sorted(rel_scores[w].items(), key=lambda kv: (-kv[1], kv[0]))
                    for (v_cat, r_pred), s in ranked_rels[: self.max_candidates_per_word]:
                        supp = tuple(rel_support[w][(v_cat, r_pred)][:10])
                        cands.append((LexicalEntry(w, v_cat, r_pred, supp), s))

                if w in likely_mods and mod_scores[w]:
                    ranked_mods = sorted(mod_scores[w].items(), key=lambda kv: (-kv[1], kv[0]))
                    for (m_cat, m_pred), s in ranked_mods[: self.max_candidates_per_word]:
                        supp = tuple(mod_support[w][(m_cat, m_pred)][:10])
                        cands.append((LexicalEntry(w, m_cat, m_pred, supp), s))

                if (w in likely_nouns or not cands) and noun_scores[w]:
                    ranked_nouns = sorted(noun_scores[w].items(), key=lambda kv: (-kv[1], kv[0]))
                    for k_pred, s in ranked_nouns[: self.max_candidates_per_word]:
                        supp = tuple(noun_support[w][k_pred][:10])
                        cands.append((LexicalEntry(w, "N", k_pred, supp), s))

                if w in q_candidates and not cands:
                    cat, bonus, supp = q_candidates[w]
                    cands.append((LexicalEntry(w, cat, "Q", supp), bonus))

                # Task 4: Optional multi-sense (homonym/polysemy) candidate
                if self.allow_multi_sense and len(cands) >= 2:
                    c1, s1 = cands[0]
                    for c2, s2 in cands[1:4]:
                        if c2.family != c1.family and c2.predicate not in ("Q", "NULL"):
                            multi_entry = LexicalEntry(
                                word=w,
                                category=c1.category,
                                predicate=c1.predicate,
                                supporting_examples=c1.supporting_examples,
                                secondary_category=c2.category,
                                secondary_predicate=c2.predicate,
                            )
                            cands.append((multi_entry, s1 + 0.35 * max(0.0, s2)))

                # Task 4: Optional null/filler entry candidate (0 semantic atoms)
                if self.allow_null_entries:
                    occs = word_occ[w]
                    all_init = bool(occs) and all(o.tokens[0] == w for o in occs)
                    all_fin = bool(occs) and all(o.tokens[-1] == w for o in occs)
                    if all_init:
                        if self._allowed_category(w, "S/S"):
                            cands.append((LexicalEntry(w, "S/S", "NULL", ()), 0.0))
                    elif all_fin:
                        if self._allowed_category(w, "S\\S"):
                            cands.append((LexicalEntry(w, "S\\S", "NULL", ()), 0.0))
                    else:
                        if self._allowed_category(w, "N/N"):
                            cands.append((LexicalEntry(w, "N/N", "NULL", ()), 0.0))
                        if self._allowed_category(w, "N\\N"):
                            cands.append((LexicalEntry(w, "N\\N", "NULL", ()), 0.0))

            if cands:
                cands.sort(
                    key=lambda item: (
                        -item[1],
                        item[0].category,
                        item[0].predicate,
                        item[0].secondary_predicate or "",
                    )
                )
                candidates_by_word[w] = cands[: self.max_candidates_per_word + 2]

        return candidates_by_word

    def _propose_word_candidates(
        self, observations: Sequence[_ObsEntry]
    ) -> dict[str, list[tuple[LexicalEntry, float]]]:
        """Induce candidate LexicalEntry items per word from True-vs-False scene contrasts."""
        if self.evidence_channel in ("truth", "cooc", "both") or self.allow_multi_sense or self.allow_null_entries:
            return self._propose_word_candidates_channel(observations)
        # Log-likelihood ratio relative to chance (0.5): positive for hits, negative for misses
        log_hit = math.log(2.0 * (1.0 - self.epsilon))
        log_miss = math.log(2.0 * self.epsilon)

        # Collect encountered words in encounter order
        words_seen: list[str] = []
        seen_set: set[str] = set()
        for obs in observations:
            for t in obs.tokens:
                if t not in seen_set:
                    seen_set.add(t)
                    words_seen.append(t)

        if not words_seen:
            return {}

        # Discover perceptual inventory from observed scenes
        all_kinds: set[str] = set()
        all_colors: set[str] = set()
        all_sizes: set[str] = set()
        all_rels: set[str] = set()
        for obs in observations:
            for o in obs.scene.objects:
                all_kinds.add(f"kind:{o.kind}")
                all_colors.add(f"color:{o.color}")
                all_sizes.add(f"size:{o.size}")
            for r in obs.scene.relations:
                all_rels.add(r.name)

        kinds_sorted = sorted(all_kinds)
        mods_sorted = sorted(all_colors | all_sizes)
        rels_sorted = sorted(all_rels)

        # ------------------------------------------------------------------
        # Phase 1: Identify peripheral speech-act marker candidates ("Q")
        # ------------------------------------------------------------------
        word_occ: dict[str, list[_ObsEntry]] = defaultdict(list)
        for obs in observations:
            for t in set(obs.tokens):
                word_occ[t].append(obs)

        q_candidates: dict[str, tuple[str, float, tuple[str, ...]]] = {}
        for w in words_seen:
            occs = word_occ[w]
            if not occs:
                continue
            if any(len(o.tokens) == 1 for o in occs):
                continue
            all_initial = all(o.tokens[0] == w for o in occs)
            all_final = all(o.tokens[-1] == w for o in occs)
            if not (all_initial or all_final):
                continue
            lengths = {len(o.tokens) for o in occs}
            true_occs = [o for o in occs if o.label]
            # Check if any single unary or relational predicate is invariantly present in True scenes
            max_inv_rate = 0.0
            if true_occs:
                for p in kinds_sorted + mods_sorted:
                    fam, val = p.split(":", 1)
                    rate = sum(any(obj.value(fam) == val for obj in o.scene.objects) for o in true_occs) / len(true_occs)
                    if rate > max_inv_rate:
                        max_inv_rate = rate
                for r_name in rels_sorted:
                    rate = sum(any(rel.name == r_name for rel in o.scene.relations) for o in true_occs) / len(true_occs)
                    if rate > max_inv_rate:
                        max_inv_rate = rate

            cat = "S/S" if all_initial else "S\\S"
            if self._allowed_category(w, cat) and self._allowed_predicate(w, "Q"):
                is_given_q = self.given_lexicon is not None and self.given_lexicon.get(w) == "Q"
                # A question marker appears in 2-token bare-NP questions or has no invariant content predicate
                if is_given_q or (min(lengths) == 2 and max(lengths) >= 4) or max_inv_rate < 0.85:
                    supp = tuple(o.example_id for o in occs[:10])
                    bonus = 10.0 * len(occs) * log_hit if (is_given_q or max_inv_rate < 0.85) else 0.0
                    q_candidates[w] = (cat, bonus, supp)

        primary_q_words = {w for w, (_, bonus, _) in q_candidates.items() if bonus > 0.0}

        def strip_q(tokens: tuple[str, ...]) -> tuple[str, ...]:
            if len(tokens) >= 2 and tokens[0] in primary_q_words:
                return tokens[1:]
            if len(tokens) >= 2 and tokens[-1] in primary_q_words:
                return tokens[:-1]
            return tokens

        # ------------------------------------------------------------------
        # Phase 2: Induce Noun ("N") candidates from 1-token and multi-token contrasts
        # ------------------------------------------------------------------
        noun_scores: dict[str, dict[str, float]] = {w: {} for w in words_seen}
        noun_support: dict[str, dict[str, list[str]]] = {w: defaultdict(list) for w in words_seen}
        core_len1_counts: dict[str, int] = defaultdict(int)

        for obs in observations:
            core = strip_q(obs.tokens)
            if len(core) == 1:
                core_len1_counts[core[0]] += 1

        for w in words_seen:
            if w in primary_q_words or not self._allowed_category(w, "N"):
                continue
            occs = word_occ[w]
            for k_pred in kinds_sorted:
                if not self._allowed_predicate(w, k_pred):
                    continue
                _, k_val = k_pred.split(":", 1)
                score = 0.0
                supp: list[str] = []
                for obs in occs:
                    core = strip_q(obs.tokens)
                    present = any(o.kind == k_val for o in obs.scene.objects)
                    if len(core) == 1 and core[0] == w:
                        if present == obs.label:
                            score += 2.0 * log_hit
                            supp.append(obs.example_id)
                        else:
                            score += 2.0 * log_miss
                    else:
                        if obs.label:
                            if present:
                                score += 0.5 * log_hit
                                supp.append(obs.example_id)
                            else:
                                score += 1.5 * log_miss
                noun_scores[w][k_pred] = score
                noun_support[w][k_pred] = supp

        best_noun_pred: dict[str, tuple[str, float]] = {}
        likely_nouns: set[str] = set()
        for w in words_seen:
            if noun_scores[w]:
                top_k, top_s = max(noun_scores[w].items(), key=lambda kv: kv[1])
                best_noun_pred[w] = (top_k, top_s)
                if core_len1_counts[w] > 0 or (
                    self.given_syntax is not None and self.given_syntax.get(w) == "N"
                ):
                    likely_nouns.add(w)

        # If a noun never appeared in a 1-token sentence, detect whether its True scenes have an
        # invariant kind:k but NO invariant color/size/relation
        for w in words_seen:
            if w in likely_nouns or w in primary_q_words:
                continue
            true_occs = [o for o in word_occ[w] if o.label]
            if not true_occs:
                continue
            max_kind_rate = max(
                (
                    sum(any(obj.kind == kp.split(":", 1)[1] for obj in o.scene.objects) for o in true_occs)
                    / len(true_occs)
                    for kp in kinds_sorted
                ),
                default=0.0,
            )
            max_mod_rate = max(
                (
                    sum(
                        any(obj.value(mp.split(":", 1)[0]) == mp.split(":", 1)[1] for obj in o.scene.objects)
                        for o in true_occs
                    )
                    / len(true_occs)
                    for mp in mods_sorted
                ),
                default=0.0,
            )
            max_rel_rate = max(
                (
                    sum(any(rel.name == r_name for rel in o.scene.relations) for o in true_occs)
                    / len(true_occs)
                    for r_name in rels_sorted
                ),
                default=0.0,
            )
            if max_kind_rate >= 0.95 and max_mod_rate < 0.85 and max_rel_rate < 0.85:
                likely_nouns.add(w)

        # ------------------------------------------------------------------
        # Phase 3: Induce Modifier ("N/N", "N\N") candidates from NP contrasts
        # ------------------------------------------------------------------
        mod_scores: dict[str, dict[tuple[str, str], float]] = {w: {} for w in words_seen}
        mod_support: dict[str, dict[tuple[str, str], list[str]]] = {
            w: defaultdict(list) for w in words_seen
        }

        for w in words_seen:
            if w in primary_q_words or w in likely_nouns:
                continue
            occs = word_occ[w]
            pre_count = 0
            post_count = 0
            for obs in occs:
                core = strip_q(obs.tokens)
                if len(core) == 2 and w in core:
                    other = core[1] if core[0] == w else core[0]
                    if other in likely_nouns:
                        if core[0] == w:
                            pre_count += 1
                        else:
                            post_count += 1
                elif len(core) >= 3 and w in core:
                    idx = core.index(w)
                    if idx + 1 < len(core) and core[idx + 1] in likely_nouns:
                        pre_count += 1
                    if idx - 1 >= 0 and core[idx - 1] in likely_nouns:
                        post_count += 1

            allowed_cats: list[str] = []
            if pre_count > 0 or post_count == 0:
                if self._allowed_category(w, "N/N"):
                    allowed_cats.append("N/N")
            if post_count > 0 or pre_count == 0:
                if self._allowed_category(w, "N\\N"):
                    allowed_cats.append("N\\N")

            for cat in allowed_cats:
                dir_diff = (pre_count - post_count) if cat == "N/N" else (post_count - pre_count)
                if dir_diff < 0:
                    continue
                for m_pred in mods_sorted:
                    if not self._allowed_predicate(w, m_pred):
                        continue
                    fam, val = m_pred.split(":", 1)
                    score = 0.2 * dir_diff * log_hit
                    supp: list[str] = []
                    for obs in occs:
                        core = strip_q(obs.tokens)
                        partner_kind: str | None = None
                        if len(core) == 2 and w in core:
                            other = core[1] if core[0] == w else core[0]
                            if other in likely_nouns and other in best_noun_pred:
                                partner_kind = best_noun_pred[other][0].split(":", 1)[1]
                        elif len(core) >= 3 and w in core:
                            idx = core.index(w)
                            if cat == "N/N" and idx + 1 < len(core) and core[idx + 1] in likely_nouns:
                                partner_kind = best_noun_pred[core[idx + 1]][0].split(":", 1)[1]
                            elif cat == "N\\N" and idx - 1 >= 0 and core[idx - 1] in likely_nouns:
                                partner_kind = best_noun_pred[core[idx - 1]][0].split(":", 1)[1]

                        if partner_kind is not None and len(core) == 2:
                            holds = any(
                                o.kind == partner_kind and o.value(fam) == val
                                for o in obs.scene.objects
                            )
                            if holds == obs.label:
                                score += 2.0 * log_hit
                                supp.append(obs.example_id)
                            else:
                                score += 2.0 * log_miss
                        elif partner_kind is not None and len(core) >= 3:
                            holds = any(
                                o.kind == partner_kind and o.value(fam) == val
                                for o in obs.scene.objects
                            )
                            if obs.label:
                                if holds:
                                    score += 1.0 * log_hit
                                    supp.append(obs.example_id)
                                else:
                                    score += 2.0 * log_miss
                            else:
                                if not holds:
                                    score += 0.8 * log_hit
                                    supp.append(obs.example_id)
                                else:
                                    score += 0.1 * log_miss
                        else:
                            holds_any = any(o.value(fam) == val for o in obs.scene.objects)
                            if obs.label:
                                if holds_any:
                                    score += 0.5 * log_hit
                                    supp.append(obs.example_id)
                                else:
                                    score += 1.5 * log_miss
                    mod_scores[w][(cat, m_pred)] = score
                    mod_support[w][(cat, m_pred)] = supp

        likely_mods: dict[str, tuple[str, str, float]] = {}
        for w in words_seen:
            if w in likely_nouns or w in primary_q_words:
                continue
            if mod_scores[w]:
                (best_cat, best_mp), best_ms = max(mod_scores[w].items(), key=lambda kv: kv[1])
                in_len2 = any(len(strip_q(o.tokens)) == 2 for o in word_occ[w])
                if in_len2 or best_ms > 0.0 or (
                    self.given_syntax is not None and self.given_syntax.get(w) in ("N/N", "N\\N")
                ):
                    # Verify this word is not purely a transitive verb
                    min_core_len = min(len(strip_q(o.tokens)) for o in word_occ[w])
                    if min_core_len == 2 or best_ms > 2.0 * log_hit:
                        likely_mods[w] = (best_cat, best_mp, best_ms)

        # ------------------------------------------------------------------
        # Phase 4: Induce Transitive Verb ("(S\NP)\NP", "(S\NP)/NP", "(S/NP)/NP") candidates
        # ------------------------------------------------------------------
        rel_scores: dict[str, dict[tuple[str, str], float]] = {w: {} for w in words_seen}
        rel_support: dict[str, dict[tuple[str, str], list[str]]] = {
            w: defaultdict(list) for w in words_seen
        }

        verb_cats = ["(S\\NP)\\NP", "(S\\NP)/NP", "(S/NP)/NP"]
        for w in words_seen:
            if w in primary_q_words or w in likely_nouns or w in likely_mods:
                continue
            occs = word_occ[w]
            if not any(len(strip_q(o.tokens)) >= 3 for o in occs):
                continue

            after_both = 0
            between_both = 0
            before_both = 0
            parsed_frames: list[
                tuple[_ObsEntry, str, list[tuple[str, str]], str, list[tuple[str, str]]]
            ] = []

            for obs in occs:
                core = strip_q(obs.tokens)
                if len(core) < 3 or w not in core:
                    continue
                w_idx = core.index(w)
                noun_positions = [
                    (i, t) for i, t in enumerate(core) if i != w_idx and t in likely_nouns and t in best_noun_pred
                ]
                if len(noun_positions) != 2:
                    continue
                (n1_idx, n1_tok), (n2_idx, n2_tok) = noun_positions[0], noun_positions[1]
                if w_idx > n2_idx:
                    after_both += 1
                elif n1_idx < w_idx < n2_idx:
                    between_both += 1
                elif w_idx < n1_idx:
                    before_both += 1

                k1 = best_noun_pred[n1_tok][0].split(":", 1)[1]
                k2 = best_noun_pred[n2_tok][0].split(":", 1)[1]
                adjs1: list[tuple[str, str]] = []
                adjs2: list[tuple[str, str]] = []
                for i, t in enumerate(core):
                    if i in (w_idx, n1_idx, n2_idx):
                        continue
                    if t in likely_mods:
                        m_cat, m_pred, _ = likely_mods[t]
                        mfam, mval = m_pred.split(":", 1)
                        if m_cat == "N/N":
                            if i < n1_idx:
                                adjs1.append((mfam, mval))
                            else:
                                adjs2.append((mfam, mval))
                        else:
                            if i > n2_idx or (i > n1_idx and abs(i - n2_idx) < abs(i - n1_idx)):
                                adjs2.append((mfam, mval))
                            else:
                                adjs1.append((mfam, mval))
                parsed_frames.append((obs, k1, adjs1, k2, adjs2))

            for v_cat in verb_cats:
                if not self._allowed_category(w, v_cat):
                    continue
                if v_cat == "(S\\NP)\\NP" and (between_both > 0 or before_both > 0) and after_both == 0:
                    continue
                if v_cat == "(S\\NP)/NP" and (after_both > 0 or before_both > 0) and between_both == 0:
                    continue
                if v_cat == "(S/NP)/NP" and (after_both > 0 or between_both > 0) and before_both == 0:
                    continue

                for r_name in rels_sorted:
                    variants = [
                        (f"rel:{r_name}(x0,x1)", False, False),
                        (f"rel:{r_name}(x1,x0)", False, True),
                        (f"rel:{r_name}:sym(x0,x1)", True, False),
                    ]
                    for r_pred, is_sym, is_rev in variants:
                        if not self._allowed_predicate(w, r_pred):
                            continue
                        tie_pref = 0.02 if (not is_sym and not is_rev) else (0.01 if is_sym else 0.0)
                        score = tie_pref
                        supp: list[str] = []
                        for obs, k1, adjs1, k2, adjs2 in parsed_frames:
                            holds = self._eval_transitive_frame(
                                obs.scene, r_name, is_sym, is_rev, k1, adjs1, k2, adjs2
                            )
                            if holds == obs.label:
                                score += 2.0 * log_hit
                                supp.append(obs.example_id)
                            else:
                                score += 2.0 * log_miss
                        rel_scores[w][(v_cat, r_pred)] = score
                        rel_support[w][(v_cat, r_pred)] = supp

        likely_verbs: set[str] = {w for w in words_seen if rel_scores[w]}

        # If a novel word appeared ONLY in a transitive or modified frame (never in len 1),
        # ensure it is also considered as a Noun ("N")!
        for w in words_seen:
            if w not in primary_q_words and w not in likely_mods and w not in likely_verbs:
                likely_nouns.add(w)

        # ------------------------------------------------------------------
        # Assemble top candidate LexicalEntry list per word
        # ------------------------------------------------------------------
        candidates_by_word: dict[str, list[tuple[LexicalEntry, float]]] = {}
        for w in words_seen:
            cands: list[tuple[LexicalEntry, float]] = []

            if w in primary_q_words:
                cat, bonus, supp = q_candidates[w]
                cands.append((LexicalEntry(w, cat, "Q", supp), bonus))
            else:
                if w in likely_verbs and rel_scores[w]:
                    ranked_rels = sorted(rel_scores[w].items(), key=lambda kv: (-kv[1], kv[0]))
                    for (v_cat, r_pred), s in ranked_rels[: self.max_candidates_per_word]:
                        supp = tuple(rel_support[w][(v_cat, r_pred)][:10])
                        cands.append((LexicalEntry(w, v_cat, r_pred, supp), s))

                if w in likely_mods and mod_scores[w]:
                    ranked_mods = sorted(mod_scores[w].items(), key=lambda kv: (-kv[1], kv[0]))
                    for (m_cat, m_pred), s in ranked_mods[: self.max_candidates_per_word]:
                        supp = tuple(mod_support[w][(m_cat, m_pred)][:10])
                        cands.append((LexicalEntry(w, m_cat, m_pred, supp), s))

                if (w in likely_nouns or not cands) and noun_scores[w]:
                    ranked_nouns = sorted(noun_scores[w].items(), key=lambda kv: (-kv[1], kv[0]))
                    for k_pred, s in ranked_nouns[: self.max_candidates_per_word]:
                        supp = tuple(noun_support[w][k_pred][:10])
                        cands.append((LexicalEntry(w, "N", k_pred, supp), s))

                if w in q_candidates and not cands:
                    cat, bonus, supp = q_candidates[w]
                    cands.append((LexicalEntry(w, cat, "Q", supp), bonus))

            if cands:
                cands.sort(key=lambda item: (-item[1], item[0].category, item[0].predicate))
                candidates_by_word[w] = cands[: self.max_candidates_per_word]

        return candidates_by_word

    @staticmethod
    def _eval_transitive_frame(
        scene: Scene,
        r_name: str,
        is_sym: bool,
        is_rev: bool,
        k1: str,
        adjs1: Sequence[tuple[str, str]],
        k2: str,
        adjs2: Sequence[tuple[str, str]],
    ) -> bool:
        """Fast check of a candidate relational frame on ``scene``."""
        objs = scene.objects
        n = len(objs)
        match1 = [
            i
            for i in range(n)
            if objs[i].kind == k1 and all(objs[i].value(f) == v for f, v in adjs1)
        ]
        match2 = [
            j
            for j in range(n)
            if objs[j].kind == k2 and all(objs[j].value(f) == v for f, v in adjs2)
        ]
        if not match1 or not match2:
            return False
        for i in match1:
            for j in match2:
                if i == j:
                    continue
                subj, obj = (j, i) if is_rev else (i, j)
                if scene.holds(r_name, subj, obj, is_sym):
                    return True
        return False

    def _build_beam_for_observations(
        self, observations: Sequence[_ObsEntry]
    ) -> tuple[list[Hypothesis], list[float]]:
        """Construct and score the beam of up to K joint hypotheses."""
        if not observations:
            return [], []

        candidates_by_word = self._propose_word_candidates(observations)
        if not candidates_by_word:
            return [], []

        ordered_words = sorted(candidates_by_word.keys(), key=lambda w: self._word_order[w])

        # Compute dynamic predicate inventory size for information-theoretic bit-cost prior
        num_cats = len(CATEGORY_DEPTHS)
        obs_preds: set[str] = {"Q"}
        for obs in observations:
            for o in obs.scene.objects:
                obs_preds.add(f"kind:{o.kind}")
                obs_preds.add(f"color:{o.color}")
                obs_preds.add(f"size:{o.size}")
            for r in obs.scene.relations:
                obs_preds.add(f"rel:{r.name}(x0,x1)")
                obs_preds.add(f"rel:{r.name}(x1,x0)")
                obs_preds.add(f"rel:{r.name}:sym(x0,x1)")
        num_preds = max(2, len(obs_preds))

        # Incremental beam search over words to propose top K candidate lexicons
        partial_beam: list[tuple[dict[str, LexicalEntry], set[str], float]] = [({}, set(), 0.0)]
        for w in ordered_words:
            next_beam: list[tuple[dict[str, LexicalEntry], set[str], float]] = []
            for entries_map, used_base_preds, running_score in partial_beam:
                for entry, local_score in candidates_by_word[w]:
                    base_pred = entry.predicate.split("(", 1)[0].replace(":sym", "")
                    # Prefer mutual exclusivity (one-to-one lexicon) unless allow_synonyms is enabled
                    if self.allow_synonyms or base_pred in ("Q", "NULL"):
                        dup_penalty = 0.0
                    else:
                        dup_penalty = -2.5 if base_pred in used_base_preds else 0.0
                    if self.prior_mode == "log2_bits":
                        mdl_step = compute_bit_cost_prior({w: entry}, num_cats, num_preds, in_nats=True)
                    elif self.prior_mode == "log2_raw_bits":
                        mdl_step = compute_bit_cost_prior({w: entry}, num_cats, num_preds, in_nats=False)
                    elif self.prior_mode == "flat":
                        mdl_step = 0.0
                    else:
                        mdl_step = compute_mdl_prior({w: entry}, self.alpha)
                    new_map = dict(entries_map)
                    new_map[w] = entry
                    new_used = set(used_base_preds)
                    new_used.add(base_pred)
                    next_beam.append((new_map, new_used, running_score + local_score + mdl_step + dup_penalty))

            next_beam.sort(
                key=lambda item: (
                    -item[2],
                    tuple(
                        (
                            self._word_order[kw],
                            item[0][kw].category,
                            item[0][kw].predicate,
                            item[0][kw].secondary_predicate or "",
                        )
                        for kw in sorted(item[0], key=lambda x: self._word_order[x])
                    ),
                )
            )
            partial_beam = next_beam[: self.beam_width]

        # Score every candidate joint hypothesis in partial_beam with exact MDL prior + joint log-likelihood
        log_hit = math.log(1.0 - self.epsilon)
        log_miss = math.log(self.epsilon)
        log_unparsed = math.log(0.5)
        scored_hypotheses: list[Hypothesis] = []

        obs_indices_by_word: dict[str, list[int]] = defaultdict(list)
        for idx, obs in enumerate(observations):
            for t in set(obs.tokens):
                obs_indices_by_word[t].append(idx)

        def obs_ll(truth_eval: bool | None, label: bool) -> float:
            if truth_eval is None:
                return log_unparsed
            return log_hit if (truth_eval == label) else log_miss

        ref_map = partial_beam[0][0]
        if self.evidence_channel == "cooc":
            ref_obs_scores = [
                0.0 if (self._parse_with_hypothesis(obs.tokens, ref_map) is not None) else log_unparsed
                for obs in observations
            ]
            ref_ll = sum(ref_obs_scores)
        else:
            ref_obs_scores = [
                obs_ll(self._eval_observation_cached(obs, ref_map), obs.label)
                for obs in observations
            ]
            ref_ll = sum(ref_obs_scores)

        for entries_map, _, running_score in partial_beam:
            if self.prior_mode == "log2_bits":
                prior = compute_bit_cost_prior(entries_map, num_cats, num_preds, in_nats=True)
            elif self.prior_mode == "log2_raw_bits":
                prior = compute_bit_cost_prior(entries_map, num_cats, num_preds, in_nats=False)
            elif self.prior_mode == "flat":
                prior = 0.0
            else:
                prior = compute_mdl_prior(entries_map, self.alpha)

            diff_words = [w for w in ordered_words if entries_map[w] != ref_map[w]]
            if self.evidence_channel == "cooc":
                # Pure co-occurrence channel ignores obs.label; uses generative alignment score
                # plus syntactic parseability check (unparseable sentences get log(0.5) penalty)
                if not diff_words:
                    parse_pen = ref_ll
                else:
                    affected_c: set[int] = set()
                    for dw in diff_words:
                        affected_c.update(obs_indices_by_word[dw])
                    delta_pen = 0.0
                    for idx in affected_c:
                        obs = observations[idx]
                        new_p = (
                            0.0
                            if (self._parse_with_hypothesis(obs.tokens, entries_map) is not None)
                            else log_unparsed
                        )
                        delta_pen += new_p - ref_obs_scores[idx]
                    parse_pen = ref_ll + delta_pen
                ll = (running_score - prior) + parse_pen
            else:
                if not diff_words:
                    truth_ll = ref_ll
                else:
                    affected: set[int] = set()
                    for dw in diff_words:
                        affected.update(obs_indices_by_word[dw])
                    delta_ll = 0.0
                    for idx in affected:
                        obs = observations[idx]
                        new_s = obs_ll(self._eval_observation_cached(obs, entries_map), obs.label)
                        delta_ll += new_s - ref_obs_scores[idx]
                    truth_ll = ref_ll + delta_ll
                if self.evidence_channel == "both":
                    ll = truth_ll + 0.1 * (running_score - prior)
                else:
                    ll = truth_ll

            sig = tuple(
                (
                    self._word_order[kw],
                    entries_map[kw].category,
                    entries_map[kw].predicate,
                    entries_map[kw].secondary_predicate or "",
                )
                for kw in ordered_words
            )
            scored_hypotheses.append(
                Hypothesis(
                    entries=entries_map,
                    prior_score=prior,
                    likelihood_score=ll,
                    total_score=prior + ll,
                    signature=sig,
                )
            )

        scored_hypotheses.sort(key=lambda h: (-h.total_score, h.signature))
        final_beam = scored_hypotheses[: self.beam_width]

        if not final_beam:
            return [], []

        max_score = final_beam[0].total_score
        raw_weights = [math.exp(max(-500.0, h.total_score - max_score)) for h in final_beam]
        z = sum(raw_weights)
        norm_weights = [w / z for w in raw_weights]
        return final_beam, norm_weights

    def _parse_with_hypothesis(
        self,
        tokens: tuple[str, ...],
        entries: Mapping[str, LexicalEntry],
    ) -> tuple[ParseNode, LogicalForm, bool] | None:
        if any(t not in entries for t in tokens):
            return None
        sub_sig = tuple(
            (t, entries[t].category, entries[t].predicate, entries[t].secondary_category, entries[t].secondary_predicate)
            for t in tokens
        )
        key = (tokens, sub_sig)
        if key not in self._parse_cache:
            self._parse_cache[key] = parse_tokens(tokens, entries)
        return self._parse_cache[key]

    def _eval_observation_cached(
        self,
        obs: _ObsEntry,
        entries: Mapping[str, LexicalEntry],
    ) -> bool | None:
        if any(t not in entries for t in obs.tokens):
            return None
        sub_sig = tuple(
            (t, entries[t].category, entries[t].predicate, entries[t].secondary_category, entries[t].secondary_predicate)
            for t in obs.tokens
        )
        eval_key = (obs.example_id, sub_sig)
        if eval_key not in self._eval_cache:
            if self.allow_multi_sense and any(entries[t].secondary_predicate is not None for t in obs.tokens):
                all_p = parse_all_tokens(obs.tokens, entries)
                if not all_p:
                    self._eval_cache[eval_key] = None
                else:
                    self._eval_cache[eval_key] = any(
                        evaluate_parsed_lf(lf_i, obs.scene, sym_i) for _, lf_i, sym_i in all_p
                    )
            else:
                parsed = self._parse_with_hypothesis(obs.tokens, entries)
                if parsed is None:
                    self._eval_cache[eval_key] = None
                else:
                    _, lf, is_sym = parsed
                    self._eval_cache[eval_key] = evaluate_parsed_lf(lf, obs.scene, is_sym)
        return self._eval_cache[eval_key]
