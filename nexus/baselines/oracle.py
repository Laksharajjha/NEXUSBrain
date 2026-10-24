"""Oracle ceiling baseline for harness validation.

The Oracle is NOT a Learner-protocol learner. It is explicitly allowed to read
the ground-truth grammar, ontology, and logical forms to validate that the
evaluation harness, generator, and model checker are completely sound.

Expected behavior: ~100% accuracy on train, dev, and all test splits.
"""

from __future__ import annotations

import inspect
import sys
from typing import Any, Mapping, Sequence

from nexus.eval.learner_api import Prediction
from nexus.world.config import GrammarConfig, Ontology
from nexus.world.example import Correction, Observation
from nexus.world.grammar import Meaning, NPSpec
from nexus.world.logic import Atom, LogicalForm, evaluate, make_lf
from nexus.world.scene import Scene


class OracleLeakageError(RuntimeError):
    """Raised if the Oracle is imported or accessed from within nexus.learner."""


def _check_oracle_import_guard() -> None:
    """Raise OracleLeakageError if imported from inside nexus/learner."""
    for frame_info in inspect.stack():
        fname = frame_info.filename.replace("\\", "/")
        if "/nexus/learner/" in fname:
            raise OracleLeakageError(
                f"Oracle must never be imported into learner modules! Detected caller: {fname}"
            )


_check_oracle_import_guard()


class Oracle:
    """Ground-truth ceiling baseline that parses tokens using the hidden grammar."""

    name: str = "oracle"

    def __init__(self, grammar: GrammarConfig, ontology: Ontology, seed: int = 0) -> None:
        self.grammar = grammar
        self.ontology = ontology
        self.seed = seed
        self.symmetric = ontology.symmetric_relations
        self._build_lexical_index()

    def _build_lexical_index(self) -> None:
        """Invert the grammar lexicon to map words to predicate symbols."""
        self.word_to_pred: dict[str, str] = {w: p for p, w in self.grammar.lexicon.items()}
        self.q_word = self.grammar.question_particle

    def observe(self, observation: Observation) -> None:
        """Oracle needs no training, but accepts observation events."""

    def correct(self, correction: Correction) -> None:
        """Oracle tracks truth directly; correction events are accepted."""

    def model_size(self) -> int:
        """Oracle has no learned parameters."""
        return 0

    def parse(self, tokens: tuple[str, ...]) -> LogicalForm:
        """Parse a sequence of tokens into its exact ground-truth LogicalForm.

        Works under arbitrary NP order, clause order, and Q position.
        """
        # 1. Determine speech act and strip question particle
        has_q = self.q_word in tokens
        act = "ask" if has_q else "assert"
        lex_tokens = [t for t in tokens if t != self.q_word]

        # 2. Check for relation / verb tokens
        verb_token = None
        for t in lex_tokens:
            pred = self.word_to_pred[t]
            if pred.startswith("rel:"):
                verb_token = t
                break

        if verb_token is None:
            # Single NP construction (np_assert or np_ask)
            kind = None
            adjectives: list[tuple[str, str]] = []
            for t in lex_tokens:
                pred = self.word_to_pred[t]
                prefix, val = pred.split(":", 1)
                if prefix == "kind":
                    kind = val
                else:
                    adjectives.append((prefix, val))
            if kind is None:
                raise ValueError(f"Could not find kind noun in tokens: {tokens}")
            spec = NPSpec(kind, tuple(sorted(adjectives)))
            atoms = [Atom(spec.kind_predicate, ("x0",))]
            atoms.extend(Atom(p, ("x0",)) for p in spec.adjective_predicates)
            return make_lf(act, ("x0",), atoms)

        # Transitive construction (trans_assert or trans_ask)
        rel_name = self.word_to_pred[verb_token].split(":", 1)[1]
        trans_spec = self.grammar.constructions["trans_assert"]
        
        # Locate the two NPs.
        # Each NP has exactly 1 kind word.
        kind_indices = [i for i, t in enumerate(lex_tokens) if self.word_to_pred[t].startswith("kind:")]
        if len(kind_indices) != 2:
            raise ValueError(f"Expected exactly 2 kind nouns for transitive, got {len(kind_indices)} in {tokens}")
        
        verb_idx = lex_tokens.index(verb_token)
        np_tokens_only = [t for t in lex_tokens if t != verb_token]

        # In np_tokens_only: find the two noun indices
        k_indices = [i for i, t in enumerate(np_tokens_only) if self.word_to_pred[t].startswith("kind:")]
        if len(k_indices) != 2:
            raise ValueError(f"Expected exactly 2 kind nouns in np_tokens_only, got {len(k_indices)} in {tokens}")
        k1_idx, k2_idx = k_indices[0], k_indices[1]

        # In np_order: where does noun sit?
        noun_pos_in_order = self.grammar.np_order.index("noun")
        if noun_pos_in_order == 0:
            # Noun-initial: NP1 is from k1_idx up to k2_idx, NP2 is from k2_idx onwards
            np1_tokens = np_tokens_only[:k2_idx]
            np2_tokens = np_tokens_only[k2_idx:]
        elif noun_pos_in_order == len(self.grammar.np_order) - 1:
            # Noun-final: NP1 ends at k1_idx, NP2 starts right after
            np1_tokens = np_tokens_only[:k1_idx + 1]
            np2_tokens = np_tokens_only[k1_idx + 1:]
        else:
            # Noun-medial: NP1 is up to k2_idx - adjectives before noun2
            # Each NP has at most 1 adjective of each type in ADJECTIVE_ATTRIBUTES
            # For noun-medial (e.g. size noun color):
            # Split point is between NP1's trailing adjectives and NP2's leading adjectives
            leading_attrs = self.grammar.np_order[:noun_pos_in_order]
            # Scan backwards from k2_idx for tokens matching leading_attrs
            split_idx = k2_idx
            while split_idx > k1_idx + 1:
                pred = self.word_to_pred[np_tokens_only[split_idx - 1]]
                if pred.split(":", 1)[0] in leading_attrs:
                    split_idx -= 1
                else:
                    break
            np1_tokens = np_tokens_only[:split_idx]
            np2_tokens = np_tokens_only[split_idx:]

        # Map NP1 and NP2 to subj/obj according to trans_spec.template
        # Determine whether NP_subj precedes NP_obj in template
        subj_idx_in_template = trans_spec.template.index("NP_subj")
        obj_idx_in_template = trans_spec.template.index("NP_obj")
        
        if subj_idx_in_template < obj_idx_in_template:
            subj_tokens, obj_tokens = np1_tokens, np2_tokens
        else:
            obj_tokens, subj_tokens = np1_tokens, np2_tokens

        def parse_np_tokens(np_toks: Sequence[str]) -> NPSpec:
            k = None
            adjs = []
            for t in np_toks:
                p = self.word_to_pred[t]
                prefix, val = p.split(":", 1)
                if prefix == "kind":
                    k = val
                else:
                    adjs.append((prefix, val))
            return NPSpec(k, tuple(sorted(adjs)))

        subj_spec = parse_np_tokens(subj_tokens)
        obj_spec = parse_np_tokens(obj_tokens)

        atoms = [
            Atom(subj_spec.kind_predicate, ("x0",)),
            Atom(obj_spec.kind_predicate, ("x1",)),
            Atom(f"rel:{rel_name}", ("x0", "x1")),
        ]
        atoms.extend(Atom(p, ("x0",)) for p in subj_spec.adjective_predicates)
        atoms.extend(Atom(p, ("x1",)) for p in obj_spec.adjective_predicates)

        return make_lf(act, ("x0", "x1"), atoms)

    def predict(self, scene: Scene, tokens: tuple[str, ...]) -> Prediction:
        """Parse tokens into LF and model check against the scene."""
        lf = self.parse(tokens)
        truth = evaluate(lf, scene, self.symmetric)
        return Prediction(p_true=1.0 if truth else 0.0, lf=lf)


class StubbornOracle(Oracle):
    """Oracle with one corrupted lexical entry that ignores all corrections (Stage A4).

    Maps the surface word for ``corrupted_target`` (e.g. ``color:red``) to
    ``corrupted_confusion`` (e.g. ``color:blue``) and ignores both observations
    and corrective feedback. On diagnostic contradiction probes, this must score
    clearly below 0.50 (specifically 0.00).
    """

    name: str = "stubborn_oracle"

    def __init__(
        self,
        grammar: GrammarConfig,
        ontology: Ontology,
        corrupted_target: str,
        corrupted_confusion: str,
        seed: int = 0,
    ) -> None:
        super().__init__(grammar, ontology, seed=seed)
        self.corrupted_target = corrupted_target
        self.corrupted_confusion = corrupted_confusion
        self.target_word = self.grammar.word_for(corrupted_target)
        # Corrupt the lexical entry: map target_word to corrupted_confusion
        self.word_to_pred[self.target_word] = corrupted_confusion


class CorrectedOracle(StubbornOracle):
    """Oracle starting with the same corrupted lexical entry that applies corrections (Stage A4).

    Records observations from the support stream. When ``correct()`` is called,
    updates the stored label of the corrected observation and re-solves the lexical
    entry for ``target_word`` over the candidate predicates in the ontology family,
    scoring 1.00 on diagnostic probes after corrections.
    """

    name: str = "corrected_oracle"

    def __init__(
        self,
        grammar: GrammarConfig,
        ontology: Ontology,
        corrupted_target: str,
        corrupted_confusion: str,
        seed: int = 0,
    ) -> None:
        super().__init__(grammar, ontology, corrupted_target, corrupted_confusion, seed=seed)
        self._support_obs: dict[str, tuple[Scene, tuple[str, ...], bool]] = {}
        self._corrected_ids: set[str] = set()

    def observe(self, observation: Observation) -> None:
        """Store support observation (and if mislabelled, keep the corrupted reading)."""
        self._support_obs[observation.example_id] = (
            observation.scene,
            observation.tokens,
            observation.label,
        )

    def correct(self, correction: Correction) -> None:
        """Apply corrective feedback and re-fit the lexical entry for ``target_word``."""
        if correction.target_example_id not in self._support_obs:
            raise KeyError(f"unknown example id {correction.target_example_id!r}")
        scene, tokens, _ = self._support_obs[correction.target_example_id]
        self._support_obs[correction.target_example_id] = (
            scene,
            tokens,
            correction.corrected_label,
        )
        self._corrected_ids.add(correction.target_example_id)
        self._reconcile_corrupted_word()

    def _reconcile_corrupted_word(self) -> None:
        """Select the ontology predicate in the target attribute family matching corrected examples."""
        family = self.corrupted_target.split(":", 1)[0]
        candidates = [f"{family}:{v}" for v in self.ontology.values(family)]
        relevant = [
            self._support_obs[eid]
            for eid in self._corrected_ids
            if self.target_word in self._support_obs[eid][1]
        ]
        if not relevant:
            return
        best_pred = self.word_to_pred[self.target_word]
        best_hits = -1
        for cand in candidates:
            self.word_to_pred[self.target_word] = cand
            hits = sum(
                evaluate(self.parse(toks), sc, self.symmetric) == lbl
                for sc, toks, lbl in relevant
            )
            if hits > best_hits:
                best_hits = hits
                best_pred = cand
        self.word_to_pred[self.target_word] = best_pred

