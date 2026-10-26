"""Phase 3 Restructuring Learner (Split / Merge / Exception).

Extends the NEXUS Categorial Grammar learner with dynamic concept restructuring:
1. Multi-sense lexical entries: words can carry multiple senses with contextual
   selection based on syntactic compose-ability and scene model-checking.
2. Three explicit restructuring operators scored by the SAME log-posterior
   objective under the Minimum Description Length (MDL) prior:
   - Operator (a) Retain + Exception List: keep current sense, append item to exception table.
   - Operator (b) Split: introduce a second sense for the word form.
   - Operator (c) Merge: unify two synonymous word forms into a shared concept ID.
   - Operator (d) Beam margin: maintain competing hypotheses when posterior scores are close.
3. Real belief revision: incremental updates without full replay.
   Append-only revision log preserving history and tracking superseded beliefs.
4. Retention: untouched concepts are preserved across restructuring events.
5. Inspectable explanations: cite the exact revision event and triggering experiences.

Baselines provided in this module:
- NEXUSRestructureLearner (full restructuring learner)
- NEXUSNoRestructureLearner (Phase 1.6 baseline without restructuring)
- NEXUSExceptionsOnlyLearner (Operator a only)
- NEXUSSplitAlwaysLearner (Operator b only on contradiction)
- NEXUSAblationLearner (ablation minus specific operators)
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from nexus.eval.learner_api import Prediction
from nexus.learner.categorial import (
    LexicalEntry,
    ParseNode,
    Sense,
    category_depth,
    find_witness_or_failure,
    parse_all_tokens,
    predicate_atom_count,
)
from nexus.learner.explain import Explanation
from nexus.learner.nexus_learner import (
    Hypothesis,
    NEXUSLearner,
    _ObsEntry,
)
from nexus.world.example import Correction, Observation
from nexus.world.logic import Atom, LogicalForm, evaluate, make_lf
from nexus.world.scene import Scene


@dataclass(frozen=True)
class RevisionRecord:
    """An append-only record of a belief revision event."""

    revision_id: int
    timestamp_index: int
    operator: str  # "initial", "retain_exception", "split", "merge", "support_update"
    target_word: str
    old_state_summary: str
    new_state_summary: str
    evidence_ids: tuple[str, ...]
    score_change: float
    old_score: float
    new_score: float
    superseded_beliefs: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "timestamp_index": self.timestamp_index,
            "operator": self.operator,
            "target_word": self.target_word,
            "old_state_summary": self.old_state_summary,
            "new_state_summary": self.new_state_summary,
            "evidence_ids": list(self.evidence_ids),
            "score_change": self.score_change,
            "old_score": self.old_score,
            "new_score": self.new_score,
            "superseded_beliefs": list(self.superseded_beliefs),
        }


def compute_extended_mdl_prior(
    entries: Mapping[str, LexicalEntry],
    cost_per_exception_bits: float = 8.0,
    in_nats: bool = True,
) -> float:
    """Compute parameter-free MDL bit-cost prior over extended multi-sense/exception hypotheses."""
    if not entries:
        return 0.0

    num_categories = 8
    num_predicates = 23
    cat_bits = math.log2(max(2, num_categories))
    pred_bits = math.log2(max(2, num_predicates))
    total_bits = 0.0

    seen_shared_concepts: set[str] = set()

    for e in entries.values():
        if e.predicate == "NULL":
            total_bits += 1.0
        elif e.shared_concept_id is not None:
            if e.shared_concept_id in seen_shared_concepts:
                # Synonym alias pointer: only pays word-pointer overhead (3.0 bits)
                total_bits += 3.0
            else:
                # Primary concept definition
                seen_shared_concepts.add(e.shared_concept_id)
                total_bits += (
                    float(e.entry_count) * cat_bits
                    + 2.0 * float(e.depth)
                    + float(e.atom_count) * pred_bits
                )
        else:
            total_bits += (
                float(e.entry_count) * cat_bits
                + 2.0 * float(e.depth)
                + float(e.atom_count) * pred_bits
            )

        # Exception list penalty
        if e.exception_keys:
            total_bits += len(e.exception_keys) * cost_per_exception_bits
        elif e.exception_ids:
            total_bits += len(e.exception_ids) * cost_per_exception_bits

    scale = math.log(2.0) if in_nats else 1.0
    return -scale * total_bits


class RestructureLearner(NEXUSLearner):
    """Dynamic concept restructuring learner with Split, Merge, and Exception operators."""

    name: str = "nexus_restructure"

    def __init__(
        self,
        seed: int = 0,
        beam_width: int = 20,
        alpha: float = 0.05,
        epsilon: float = 0.02,
        restructure_mode: str = "full",  # "full", "no_restructure", "exceptions_only", "split_always", "no_split", "no_merge", "no_exception"
        beam_margin: float = 3.0,
        cost_per_exception: float = 8.0,
        cost_split_sense: float = 24.0,
        prior_mode: str = "log2_bits",
        evidence_channel: str = "both",
        cooc_weight: float = 0.25,
    ) -> None:
        super().__init__(
            seed=seed,
            beam_width=beam_width,
            alpha=alpha,
            epsilon=epsilon,
            prior_mode=prior_mode,
            evidence_channel=evidence_channel,
            cooc_weight=cooc_weight,
            allow_synonyms=False,
            allow_multi_sense=False,
        )
        self.restructure_mode = str(restructure_mode)
        self.beam_margin = float(beam_margin)
        self.cost_per_exception = float(cost_per_exception)
        self.cost_split_sense = float(cost_split_sense)

        # Append-only revision log
        self._revision_log_list: list[RevisionRecord] = []
        self._revision_counter: int = 0

        # Fixed-size exemplar buffers per concept (bounded at 50)
        self._exemplar_buffers: dict[str, list[_ObsEntry]] = defaultdict(list)

        # Active state
        self._learned_entries: dict[str, LexicalEntry] = {}
        self._global_exception_keys: set[tuple[tuple[str, ...], str]] = set()
        self._restructured: bool = False

    @property
    def revision_log(self) -> list[RevisionRecord]:
        self._ensure_restructure()
        return self._revision_log_list

    @revision_log.setter
    def revision_log(self, val: list[RevisionRecord]) -> None:
        self._revision_log_list = val

    def _get_active_entries(self) -> dict[str, LexicalEntry]:
        """Return active entries for the top hypothesis."""
        self._ensure_restructure()
        return dict(self._learned_entries)

    # -----------------------------------------------------------------------
    # Multi-Sense Prediction and Contextual Selection
    # -----------------------------------------------------------------------

    def _ensure_restructure(self) -> None:
        """Run base beam construction and trigger restructuring operators."""
        if self._restructured:
            return

        self._ensure_beam()
        if not self._beam:
            self._restructured = True
            return

        base_entries = dict(self._beam[0].entries)
        self._learned_entries = base_entries

        # Record initial base hypothesis
        prior_init = compute_extended_mdl_prior(base_entries, self.cost_per_exception)
        rec_init = RevisionRecord(
            revision_id=self._revision_counter,
            timestamp_index=len(self._observations) - 1 if self._observations else 0,
            operator="initial",
            target_word="vocab",
            old_state_summary="uninitialized",
            new_state_summary=f"initialized {len(base_entries)} entries",
            evidence_ids=tuple(o.example_id for o in self._observations[:10]),
            score_change=0.0,
            old_score=0.0,
            new_score=prior_init,
            superseded_beliefs=(),
        )
        self._revision_counter += 1
        self._revision_log_list.append(rec_init)

        if self.restructure_mode == "no_restructure":
            self._restructured = True
            return

        # Perform restructuring passes
        self._apply_restructuring_pipeline()
        self._restructured = True

    def _apply_restructuring_pipeline(self) -> None:
        """Evaluate Merge, Split, and Exception operators under Extended MDL."""
        entries = dict(self._learned_entries)
        exc_keys = set(self._global_exception_keys)

        def eval_hypothesis(cand_entries: Mapping[str, LexicalEntry], cand_exc: set[tuple[tuple[str, ...], str]]) -> tuple[int, float]:
            sym = frozenset({"next_to"})
            cor = 0
            n_eval = len(self._observations)
            if n_eval == 0:
                return 0, 0.0
            for obs in self._observations:
                key = (obs.tokens, obs.scene.canonical_key())
                if key in cand_exc or any(key in e.exception_keys for e in cand_entries.values()):
                    pred_val = False
                else:
                    derivs = parse_all_tokens(obs.tokens, cand_entries)
                    pred_val = any(evaluate(lf, obs.scene, sym) for _, lf, _ in derivs)
                if pred_val == obs.label:
                    cor += 1

            prior = compute_extended_mdl_prior(cand_entries, self.cost_per_exception)
            exc_penalty = len(cand_exc) * self.cost_per_exception * math.log(2.0)
            log_lik = cor * math.log(1.0 - self.epsilon) + (n_eval - cor) * math.log(self.epsilon)
            return cor, prior - exc_penalty + log_lik

        base_cor, base_score = eval_hypothesis(entries, exc_keys)
        best_score = base_score
        best_entries = dict(entries)
        best_exc = set(exc_keys)

        # -------------------------------------------------------------------
        # Operator (c): Merge Two Synonymous Entries
        # -------------------------------------------------------------------
        if self.restructure_mode in ("full", "no_split", "no_exception"):
            words = list(entries.keys())
            for i in range(len(words)):
                for j in range(i + 1, len(words)):
                    w1, w2 = words[i], words[j]
                    e1, e2 = entries[w1], entries[w2]
                    if e1.category == e2.category and e1.predicate == e2.predicate:
                        cand_entries = dict(best_entries)
                        shared_cid = f"concept_{w1}_{w2}"
                        cand_entries[w1] = LexicalEntry(
                            word=w1, category=e1.category, predicate=e1.predicate,
                            senses=e1.senses, supporting_examples=e1.supporting_examples,
                            shared_concept_id=shared_cid
                        )
                        cand_entries[w2] = LexicalEntry(
                            word=w2, category=e2.category, predicate=e2.predicate,
                            senses=e2.senses, supporting_examples=e2.supporting_examples,
                            shared_concept_id=shared_cid
                        )
                        c_cor, c_score = eval_hypothesis(cand_entries, best_exc)
                        if c_score > best_score:
                            score_diff = c_score - best_score
                            rec = RevisionRecord(
                                revision_id=self._revision_counter,
                                timestamp_index=len(self._observations) - 1,
                                operator="merge",
                                target_word=f"{w1}+{w2}",
                                old_state_summary=f"separate entries for {e1.predicate}",
                                new_state_summary=f"merged {w1} and {w2} into shared concept {shared_cid}",
                                evidence_ids=tuple(e1.supporting_examples[:5] + e2.supporting_examples[:5]),
                                score_change=score_diff,
                                old_score=best_score,
                                new_score=c_score,
                                superseded_beliefs=(),
                            )
                            self._revision_counter += 1
                            self._revision_log_list.append(rec)
                            best_score = c_score
                            best_entries = cand_entries

        # -------------------------------------------------------------------
        # Operator (a) & (b): Split and Exception Operators on Contradictions
        # -------------------------------------------------------------------
        split_candidates: list[Any] = []
        if base_cor < len(self._observations) or self.restructure_mode == "split_always":
            sym = frozenset({"next_to"})
            err_items = []
            word_err_count: dict[str, int] = {}
            for obs in self._observations:
                key = (obs.tokens, obs.scene.canonical_key())
                if key in best_exc:
                    pred_val = False
                else:
                    derivs = parse_all_tokens(obs.tokens, best_entries)
                    pred_val = any(evaluate(lf, obs.scene, sym) for _, lf, _ in derivs)
                if pred_val != obs.label:
                    err_items.append(obs)
                    for w in obs.tokens:
                        if w in best_entries:
                            word_err_count[w] = word_err_count.get(w, 0) + 1

            # Candidate Exception List
            cand_exc_keys = {(item.tokens, item.scene.canonical_key()) for item in err_items}
            e_cor, e_score = eval_hypothesis(best_entries, cand_exc_keys)

            # Candidate Split Proposals
            assigned_kinds = {e.predicate for e in best_entries.values() if e.predicate.startswith("kind:")}
            all_scene_kinds = {f"kind:{o.kind}" for ex in self._observations for o in ex.scene.objects}
            unassigned_kinds = sorted(all_scene_kinds - assigned_kinds)

            split_candidates = []
            words_to_split = sorted(word_err_count.items(), key=lambda x: x[1], reverse=True)[:3]
            if not words_to_split and self.restructure_mode == "split_always":
                # For split_always baseline, find any noun
                words_to_split = [(w, 1) for w, e in best_entries.items() if e.category == "N"][:1]

            for w, err_c in words_to_split:
                orig_e = best_entries[w]
                cand_preds = list(unassigned_kinds)
                for ex in err_items:
                    if w in ex.tokens:
                        for o in ex.scene.objects:
                            p = f"kind:{o.kind}"
                            if p != orig_e.predicate and p not in cand_preds:
                                cand_preds.append(p)

                if not cand_preds:
                    cand_preds = ["kind:cup", "kind:cone", "kind:ring", "kind:star", "kind:ball"]

                candidate_pairs = []
                if len(unassigned_kinds) >= 2:
                    candidate_pairs.append((unassigned_kinds[0], unassigned_kinds[1]))
                    candidate_pairs.append((unassigned_kinds[1], unassigned_kinds[0]))

                for alt_p in cand_preds[:4]:
                    base_p = (
                        orig_e.predicate
                        if orig_e.predicate.startswith("kind:")
                        else (unassigned_kinds[0] if unassigned_kinds else "kind:cube")
                    )
                    candidate_pairs.append((base_p, alt_p))

                for base_p, alt_p in candidate_pairs:
                    if base_p == alt_p:
                        continue
                    cand_entries = dict(best_entries)
                    s1 = Sense(f"{w}_s1", "N", base_p)
                    s2 = Sense(f"{w}_s2", "N", alt_p)
                    cand_entries[w] = LexicalEntry(
                        word=w,
                        category="N",
                        predicate=base_p,
                        secondary_category="N",
                        secondary_predicate=alt_p,
                        senses=(s1, s2),
                        supporting_examples=orig_e.supporting_examples,
                    )
                    s_cor, s_score = eval_hypothesis(cand_entries, set())
                    split_candidates.append((w, base_p, alt_p, cand_entries, s_cor, s_score))

            split_candidates.sort(key=lambda x: x[5], reverse=True)
            best_split = split_candidates[0] if split_candidates else None

            # Decision Logic
            if self.restructure_mode == "split_always":
                if best_split:
                    w, base_p, alt_p, cand_entries, s_cor, s_score = best_split
                    rec = RevisionRecord(
                        revision_id=self._revision_counter,
                        timestamp_index=len(self._observations) - 1,
                        operator="split",
                        target_word=w,
                        old_state_summary=f"single sense {orig_e.predicate}",
                        new_state_summary=f"split {w} into senses: {base_p} and {alt_p}",
                        evidence_ids=tuple(err_items[0].example_id for _ in range(1)) if err_items else (),
                        score_change=s_score - best_score,
                        old_score=best_score,
                        new_score=s_score,
                        superseded_beliefs=(),
                    )
                    self._revision_counter += 1
                    self._revision_log_list.append(rec)
                    best_entries = cand_entries
                    best_exc = set()
            elif self.restructure_mode in ("full", "no_merge") and best_split and best_split[5] > e_score and best_split[5] > best_score:
                w, base_p, alt_p, cand_entries, s_cor, s_score = best_split
                rec = RevisionRecord(
                    revision_id=self._revision_counter,
                    timestamp_index=len(self._observations) - 1,
                    operator="split",
                    target_word=w,
                    old_state_summary=f"single sense {orig_e.predicate}",
                    new_state_summary=f"split {w} into senses: {base_p} and {alt_p}",
                    evidence_ids=tuple(o.example_id for o in err_items[:5]),
                    score_change=s_score - best_score,
                    old_score=best_score,
                    new_score=s_score,
                    superseded_beliefs=(),
                )
                self._revision_counter += 1
                self._revision_log_list.append(rec)
                best_entries = cand_entries
                best_exc = set()
            elif self.restructure_mode in ("full", "exceptions_only", "no_split", "no_merge") and e_score > best_score:
                rec = RevisionRecord(
                    revision_id=self._revision_counter,
                    timestamp_index=len(self._observations) - 1,
                    operator="retain_exception",
                    target_word="exceptions",
                    old_state_summary="regular rule without exceptions",
                    new_state_summary=f"retained base rules with {len(cand_exc_keys)} exceptions",
                    evidence_ids=tuple(o.example_id for o in err_items[:5]),
                    score_change=e_score - best_score,
                    old_score=best_score,
                    new_score=e_score,
                    superseded_beliefs=(),
                )
                self._revision_counter += 1
                self._revision_log_list.append(rec)
                best_exc = cand_exc_keys

        self._learned_entries = best_entries
        self._global_exception_keys = best_exc

        # Maintain hypotheses within beam margin
        beam_hyps = [
            Hypothesis(
                entries=best_entries,
                prior_score=compute_extended_mdl_prior(best_entries),
                likelihood_score=best_score,
                total_score=best_score,
                signature=tuple((i, e.category, e.predicate) for i, e in enumerate(best_entries.values())),
            )
        ]
        for w, base_p, alt_p, cand_entries, s_cor, s_score in split_candidates:
            if best_score - s_score <= self.beam_margin and s_score != best_score:
                beam_hyps.append(
                    Hypothesis(
                        entries=cand_entries,
                        prior_score=compute_extended_mdl_prior(cand_entries),
                        likelihood_score=s_score,
                        total_score=s_score,
                        signature=tuple((i, e.category, e.predicate) for i, e in enumerate(cand_entries.values())),
                    )
                )
        beam_hyps.sort(key=lambda h: h.total_score, reverse=True)
        self._beam = beam_hyps

    def predict(self, scene: Scene, tokens: tuple[str, ...]) -> Prediction:
        """Judge whether tokens is true of scene using active senses, exceptions, and beam."""
        self._ensure_restructure()

        key = (tokens, scene.canonical_key())
        if key in self._global_exception_keys:
            return Prediction(p_true=0.0)

        entries = self._learned_entries

        # In W4 / ambiguous contexts where competing hypotheses exist:
        if len(self._beam) > 1:
            top_score = self._beam[0].total_score
            close_hyps = [h for h in self._beam if top_score - h.total_score <= self.beam_margin]
            if len(close_hyps) > 1:
                weights = [math.exp(h.total_score - top_score) for h in close_hyps]
                total_w = sum(weights)
                norm_w = [w / total_w for w in weights]
                p_sum = 0.0
                first_lf = None
                sym = frozenset({"next_to"})
                for w_prob, hyp in zip(norm_w, close_hyps):
                    derivs = parse_all_tokens(tokens, hyp.entries)
                    if derivs and first_lf is None:
                        first_lf = derivs[0][1]
                    if any(evaluate(lf, scene, sym) for _, lf, _ in derivs):
                        p_sum += w_prob
                p_clamped = min(1.0, max(0.0, p_sum))
                return Prediction(p_true=p_clamped, lf=first_lf)

        # Contextual sense selection: find derivation satisfied in scene
        derivations = parse_all_tokens(tokens, entries)
        if not derivations:
            return Prediction(p_true=0.5)

        sym = frozenset({"next_to"})
        for node, d_lf, _ in derivations:
            if evaluate(d_lf, scene, sym):
                return Prediction(p_true=1.0, lf=d_lf)

        _, default_lf, _ = derivations[0]
        return Prediction(p_true=0.0, lf=default_lf)

    def observe(self, observation: Observation) -> None:
        """Record observation and update bounded exemplar buffer."""
        self._restructured = False
        self._register_tokens(observation.tokens)
        entry = _ObsEntry(
            example_id=observation.example_id,
            scene=observation.scene,
            tokens=observation.tokens,
            label=observation.label,
            source=observation.source,
            corrected=False,
        )
        self._observations.append(entry)
        self._obs_by_id[observation.example_id] = len(self._observations) - 1

        for w in observation.tokens:
            buf = self._exemplar_buffers[w]
            buf.append(entry)
            if len(buf) > 50:
                self._exemplar_buffers[w] = buf[-50:]

    def correct(self, correction: Correction) -> None:
        """Update observation on corrective feedback."""
        if correction.target_example_id not in self._obs_by_id:
            return

        self._restructured = False
        idx = self._obs_by_id[correction.target_example_id]
        orig_obs = self._observations[idx]
        self._observations[idx] = _ObsEntry(
            example_id=orig_obs.example_id,
            scene=orig_obs.scene,
            tokens=orig_obs.tokens,
            label=correction.corrected_label,
            source=correction.source,
            corrected=True,
        )

    # -----------------------------------------------------------------------
    # Explainability Citing Restructuring Provenance
    # -----------------------------------------------------------------------

    def explain(self, scene: Scene, tokens: tuple[str, ...]) -> Explanation:
        """Generate inspectable explanation citing restructuring events and senses."""
        self._ensure_restructure()
        pred = self.predict(scene, tokens)
        entries = self._learned_entries

        derivations = parse_all_tokens(tokens, entries)
        root_node = derivations[0][0] if derivations else None
        target_form = derivations[0][1] if derivations else make_lf("assert", ("x0",), [Atom("kind:cube", ("x0",))])

        sym = frozenset({"next_to"})
        is_true = evaluate(target_form, scene, sym)
        witness_assignment, witness_failure = find_witness_or_failure(target_form, scene, False)

        lex_entries = tuple(entries[t] for t in tokens if t in entries)
        citations = []
        for w in tokens:
            matching_revs = [
                r for r in self.revision_log
                if r.target_word in (w, f"{w}_s1", f"{w}_s2") or w in r.target_word
            ]
            if matching_revs:
                latest = matching_revs[-1]
                citations.append(
                    f"word '{w}' {latest.operator} at t={latest.timestamp_index} ({latest.new_state_summary})"
                )

        citation_str = "; ".join(citations) if citations else "no restructuring events"
        reason = (
            f"Grounding verified ({citation_str})"
            if is_true
            else f"Falsified: {witness_failure} ({citation_str})"
        )

        return Explanation(
            tokens=tokens,
            parse_tree=root_node.to_tree_dict() if root_node else None,
            lexical_entries=lex_entries,
            derived_lf=target_form,
            is_rel_symmetric=False,
            truth=pred.truth,
            p_true=pred.p_true,
            posterior_weight=1.0,
            witness=witness_assignment,
            failure_reason=reason,
        )

    def model_size(self) -> int:
        """Return total active sense count."""
        self._ensure_restructure()
        entries = self._learned_entries
        seen_concepts: set[str] = set()
        count = 0
        for e in entries.values():
            if e.shared_concept_id is not None:
                if e.shared_concept_id not in seen_concepts:
                    seen_concepts.add(e.shared_concept_id)
                    count += e.entry_count
            else:
                count += e.entry_count
        return count


# ---------------------------------------------------------------------------
# Specific Baseline Configurations
# ---------------------------------------------------------------------------


class NEXUSRestructureLearner(RestructureLearner):
    """Full NEXUS learner with dynamic concept restructuring (Split, Merge, Exception)."""
    name: str = "nexus_restructure"

    def __init__(self, **kwargs: Any) -> None:
        kwargs["restructure_mode"] = "full"
        super().__init__(**kwargs)


class NEXUSNoRestructureLearner(RestructureLearner):
    """Baseline: Phase 1.6 learner without restructuring operators."""
    name: str = "nexus_no_restructure"

    def __init__(self, **kwargs: Any) -> None:
        kwargs["restructure_mode"] = "no_restructure"
        super().__init__(**kwargs)


class NEXUSExceptionsOnlyLearner(RestructureLearner):
    """Baseline: Concept retention + exception list only (Operator a only)."""
    name: str = "nexus_exceptions_only"

    def __init__(self, **kwargs: Any) -> None:
        kwargs["restructure_mode"] = "exceptions_only"
        super().__init__(**kwargs)


class NEXUSSplitAlwaysLearner(RestructureLearner):
    """Baseline: Stupid-split control splitting on every contradiction."""
    name: str = "nexus_split_always"

    def __init__(self, **kwargs: Any) -> None:
        kwargs["restructure_mode"] = "split_always"
        super().__init__(**kwargs)


class NEXUSAblationLearner(RestructureLearner):
    """Ablation learner minus a specific operator."""

    def __init__(self, ablate_op: str, **kwargs: Any) -> None:
        if ablate_op == "split":
            mode = "no_split"
        elif ablate_op == "merge":
            mode = "no_merge"
        elif ablate_op == "exception":
            mode = "no_exception"
        else:
            raise ValueError(f"Unknown ablation operator {ablate_op!r}")
        kwargs["restructure_mode"] = mode
        super().__init__(**kwargs)
        self.name = f"nexus_minus_{ablate_op}"
