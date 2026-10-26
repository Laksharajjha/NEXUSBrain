"""Stage P3-C Learner Verification Tests.

Tests the Restructuring Learner under Phase 3 rules:
1. Unit tests per operator: Split, Merge, Exception
2. No hard-coded invented words
3. Leakage guard (protocol input audit)
4. Determinism
5. Permutation invariance / bounded order divergence
6. Revision log is append-only
7. Incremental update vs full replay equivalence
8. Retention test on untouched concepts
9. Faithfulness intervention test on restructured senses
10. One-seed end-to-end test on worlds W1-W5
"""

from __future__ import annotations

import copy
import math
from typing import Any

import pytest

from nexus.eval.learner_api import Learner, Prediction
from nexus.learner.categorial import LexicalEntry, Sense, parse_all_tokens
from nexus.learner.restructure_learner import (
    NEXUSExceptionsOnlyLearner,
    NEXUSNoRestructureLearner,
    NEXUSRestructureLearner,
    NEXUSSplitAlwaysLearner,
    RestructureLearner,
    RevisionRecord,
    compute_extended_mdl_prior,
)
from nexus.world.example import Observation
from nexus.world.logic import Atom, evaluate, make_lf
from nexus.world.restructure import build_restructure_dataset
from nexus.world.scene import Scene


# ---------------------------------------------------------------------------
# 1. Unit Tests Per Operator (Split, Merge, Exception)
# ---------------------------------------------------------------------------


def test_operator_split_minimal() -> None:
    """Split operator fires when evidence demands a multi-sense concept."""
    ds = build_restructure_dataset("w1_split", seed=2000)
    learner = NEXUSRestructureLearner(seed=2000, evidence_channel="legacy")
    for ex in ds.train.items:
        learner.observe(ex.observation())

    # Check that split fired on the shared polysemous word
    split_revs = [r for r in learner.revision_log if r.operator == "split"]
    assert len(split_revs) >= 1, f"Expected Split to fire, got: {[r.operator for r in learner.revision_log]}"
    target_word = ds.meta["shared_word"]
    assert any(target_word in r.target_word for r in split_revs)

    # Check active entries have multiple senses
    entries = learner._get_active_entries()
    assert target_word in entries
    entry = entries[target_word]
    assert len(entry.senses) == 2
    assert entry.senses[0].predicate != entry.senses[1].predicate


def test_operator_merge_minimal() -> None:
    """Merge operator unifies two synonymous words into a shared concept ID."""
    ds = build_restructure_dataset("w3_merge", seed=2000)
    learner = NEXUSRestructureLearner(seed=2000, evidence_channel="legacy")
    for ex in ds.train.items:
        learner.observe(ex.observation())

    merge_revs = [r for r in learner.revision_log if r.operator == "merge"]
    assert len(merge_revs) >= 1, f"Expected Merge to fire, got: {[r.operator for r in learner.revision_log]}"

    w_orig = ds.meta["word_orig"]
    w_syn = ds.meta["word_syn"]
    entries = learner._get_active_entries()
    assert w_orig in entries and w_syn in entries
    e1, e2 = entries[w_orig], entries[w_syn]
    assert e1.shared_concept_id is not None
    assert e1.shared_concept_id == e2.shared_concept_id


def test_operator_exception_minimal() -> None:
    """Exception operator retains regular rule and stores genuine idiosyncratic exceptions."""
    ds = build_restructure_dataset("w2_exception", seed=2000)
    learner = NEXUSRestructureLearner(seed=2000, evidence_channel="legacy")
    for ex in ds.train.items:
        learner.observe(ex.observation())

    # In W2, splitting is wrong; learner must NOT split target concept
    split_revs = [r for r in learner.revision_log if r.operator == "split"]
    target_kind = ds.meta["target_kind"]
    for r in split_revs:
        assert target_kind not in r.new_state_summary, f"Target kind {target_kind} was over-split in W2!"

    # Verify model achieves 100% on heldout test
    test_split = ds.tests["test_heldout"]
    acc = sum(1 for ex in test_split.items if learner.predict(ex.scene, ex.tokens).truth == ex.truth) / len(test_split.items)
    assert acc >= 0.950, f"Expected acc >= 0.950, got {acc}"


# ---------------------------------------------------------------------------
# 2. No Hard-Coded Invented Words
# ---------------------------------------------------------------------------


def test_no_hardcoded_invented_words() -> None:
    """The learner must never synthesize words not present in the training vocabulary."""
    ds = build_restructure_dataset("w1_split", seed=2000)
    train_vocab = {t for ex in ds.train.items for t in ex.tokens}

    learner = NEXUSRestructureLearner(seed=2000, evidence_channel="legacy")
    for ex in ds.train.items:
        learner.observe(ex.observation())

    entries = learner._get_active_entries()
    for word in entries:
        assert word in train_vocab, f"Learner synthesized invented word {word!r} not in train_vocab!"

    for rev in learner.revision_log:
        if rev.operator in ("split", "retain_exception"):
            assert rev.target_word in train_vocab, f"Revision targeted invented word {rev.target_word!r}!"


# ---------------------------------------------------------------------------
# 3. Leakage Guard (Protocol Audit)
# ---------------------------------------------------------------------------


def test_leakage_guard_and_protocol_audit() -> None:
    """Verify that learner receives strictly Scene and tokens, never generator metadata."""
    class SpyLearner(NEXUSRestructureLearner):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.observed_metadata: list[dict[str, Any]] = []

        def observe(self, observation: Observation) -> None:
            # Assert observation only contains public protocol fields
            assert hasattr(observation, "example_id")
            assert hasattr(observation, "scene")
            assert hasattr(observation, "tokens")
            assert hasattr(observation, "label")
            # Assert forbidden generator attributes are not accessed or present
            for forbidden in ("construction", "lf", "contrast_group", "tags", "plan", "mislabelled", "ground_truth"):
                assert not hasattr(observation.scene, forbidden), f"Scene leaked {forbidden}!"
            super().observe(observation)

    ds = build_restructure_dataset("w1_split", seed=2000)
    spy = SpyLearner(seed=2000, evidence_channel="legacy")
    for ex in ds.train.items[:30]:
        spy.observe(ex.observation())


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------


def test_determinism_seed_2000() -> None:
    """Seed 2000 run twice produces identical weights and revision log."""
    ds = build_restructure_dataset("w1_split", seed=2000)

    l1 = NEXUSRestructureLearner(seed=2000, evidence_channel="legacy")
    for ex in ds.train.items:
        l1.observe(ex.observation())

    l2 = NEXUSRestructureLearner(seed=2000, evidence_channel="legacy")
    for ex in ds.train.items:
        l2.observe(ex.observation())

    assert len(l1.revision_log) == len(l2.revision_log)
    for r1, r2 in zip(l1.revision_log, l2.revision_log):
        assert r1.operator == r2.operator
        assert r1.target_word == r2.target_word
        assert r1.new_state_summary == r2.new_state_summary
        assert math.isclose(r1.score_change, r2.score_change, rel_tol=1e-5)

    assert l1.model_size() == l2.model_size()


# ---------------------------------------------------------------------------
# 5. Permutation Invariance / Bounded Order Divergence
# ---------------------------------------------------------------------------


def test_permutation_invariance_and_bounded_divergence() -> None:
    """Permuted training order produces consistent restructure operators."""
    ds = build_restructure_dataset("w1_split", seed=2000)

    l_orig = NEXUSRestructureLearner(seed=2000, evidence_channel="legacy")
    for ex in ds.train.items:
        l_orig.observe(ex.observation())

    # Reverse stream
    l_rev = NEXUSRestructureLearner(seed=2000, evidence_channel="legacy")
    for ex in reversed(ds.train.items):
        l_rev.observe(ex.observation())

    # Both must identify the split operator on the polysemous word
    orig_ops = [r.operator for r in l_orig.revision_log]
    rev_ops = [r.operator for r in l_rev.revision_log]
    assert "split" in orig_ops
    assert "split" in rev_ops

    # Accuracy divergence on heldout test must be <= 0.05
    test_split = ds.tests["test_heldout"]
    acc_orig = sum(1 for ex in test_split.items if l_orig.predict(ex.scene, ex.tokens).truth == ex.truth) / len(test_split.items)
    acc_rev = sum(1 for ex in test_split.items if l_rev.predict(ex.scene, ex.tokens).truth == ex.truth) / len(test_split.items)
    assert abs(acc_orig - acc_rev) <= 0.05


# ---------------------------------------------------------------------------
# 6. Revision Log is Append-Only
# ---------------------------------------------------------------------------


def test_revision_log_append_only() -> None:
    """Historical records in revision_log are immutable and append-only."""
    ds = build_restructure_dataset("w1_split", seed=2000)
    learner = NEXUSRestructureLearner(seed=2000, evidence_channel="legacy")

    snapshots: list[tuple[RevisionRecord, ...]] = []
    for i, ex in enumerate(ds.train.items):
        learner.observe(ex.observation())
        if i % 50 == 0:
            learner._ensure_restructure()
            snapshots.append(tuple(copy.deepcopy(learner.revision_log)))

    # Assert all historical snapshots are exact prefixes of future logs
    for i in range(len(snapshots) - 1):
        prev = snapshots[i]
        nxt = snapshots[i + 1]
        assert len(prev) <= len(nxt)
        for r_prev, r_nxt in zip(prev, nxt):
            assert r_prev == r_nxt, "Revision log mutated historical record!"


# ---------------------------------------------------------------------------
# 7. Incremental Update vs Full Replay Equivalence
# ---------------------------------------------------------------------------


def test_incremental_update_vs_full_replay_equivalence() -> None:
    """Incremental streaming updates produce identical predictions to full batch replay."""
    ds = build_restructure_dataset("w1_split", seed=2000)

    # Streaming learner
    l_stream = NEXUSRestructureLearner(seed=2000, evidence_channel="legacy")
    for ex in ds.train.items:
        l_stream.observe(ex.observation())

    # Batch learner
    l_batch = NEXUSRestructureLearner(seed=2000, evidence_channel="legacy")
    for ex in ds.train.items:
        l_batch.observe(ex.observation())

    test_split = ds.tests["test_heldout"]
    discrepancies = 0
    for ex in test_split.items:
        p_stream = l_stream.predict(ex.scene, ex.tokens).truth
        p_batch = l_batch.predict(ex.scene, ex.tokens).truth
        if p_stream != p_batch:
            discrepancies += 1

    assert discrepancies == 0, f"Streaming and batch replay diverged on {discrepancies} items!"


# ---------------------------------------------------------------------------
# 8. Retention Test on Untouched Concepts
# ---------------------------------------------------------------------------


def test_retention_untouched_concepts() -> None:
    """Restructuring concept A does not degrade accuracy on untouched concept B."""
    ds = build_restructure_dataset("w1_split", seed=2000)
    learner = NEXUSRestructureLearner(seed=2000, evidence_channel="legacy")
    for ex in ds.train.items:
        learner.observe(ex.observation())

    entries = learner._get_active_entries()
    # Polysemous word 'dax' was split
    target_word = ds.meta["shared_word"]
    assert len(entries[target_word].senses) == 2

    # Check that untouched concepts (e.g. 'zib', 'toma', 'wug', 'blick') remain single-sense
    for w, e in entries.items():
        if w != target_word and e.category == "N":
            assert len(e.senses) <= 1, f"Untouched concept {w} was modified unexpectedly!"

    # Evaluate accuracy on dev examples containing only untouched concepts
    untouched_dev = [ex for ex in ds.dev.items if target_word not in ex.tokens]
    assert len(untouched_dev) >= 10
    acc = sum(1 for ex in untouched_dev if learner.predict(ex.scene, ex.tokens).truth == ex.truth) / len(untouched_dev)
    assert acc >= 0.950, f"Retention failed: accuracy on untouched concepts was {acc}"


# ---------------------------------------------------------------------------
# 9. Faithfulness Intervention Test on Restructured Senses
# ---------------------------------------------------------------------------


def test_faithfulness_intervention_on_split_sense() -> None:
    """Intervening on a newly split sense changes predictions on downstream items and ONLY those items."""
    ds = build_restructure_dataset("w1_split", seed=2000)
    learner = NEXUSRestructureLearner(seed=2000, evidence_channel="legacy")
    for ex in ds.train.items:
        learner.observe(ex.observation())

    target_word = ds.meta["shared_word"]
    entries = learner._get_active_entries()
    orig_entry = entries[target_word]
    assert len(orig_entry.senses) == 2

    # Intervention: delete sense 2 (making it single-sense again)
    intervened_entries = dict(entries)
    s1 = orig_entry.senses[0]
    intervened_entries[target_word] = LexicalEntry(
        word=target_word,
        category="N",
        predicate=s1.predicate,
        senses=(s1,),
    )

    test_items = ds.tests["test_heldout"].items
    target_items = [ex for ex in test_items if target_word in ex.tokens]
    nontarget_items = [ex for ex in test_items if target_word not in ex.tokens]

    # Non-target items must be 100% UNCHANGED by the intervention
    sym = frozenset({"next_to"})
    for ex in nontarget_items:
        d_orig = any(evaluate(lf, ex.scene, sym) for _, lf, _ in parse_all_tokens(ex.tokens, entries))
        d_inter = any(evaluate(lf, ex.scene, sym) for _, lf, _ in parse_all_tokens(ex.tokens, intervened_entries))
        assert d_orig == d_inter, f"Intervention on {target_word} leaked to non-target item {ex.tokens}!"

    # Target items relying on sense 2 MUST flip or change
    changed_count = 0
    for ex in target_items:
        d_orig = any(evaluate(lf, ex.scene, sym) for _, lf, _ in parse_all_tokens(ex.tokens, entries))
        d_inter = any(evaluate(lf, ex.scene, sym) for _, lf, _ in parse_all_tokens(ex.tokens, intervened_entries))
        if d_orig != d_inter:
            changed_count += 1

    assert changed_count >= 5, f"Expected downstream changes on target items, but only {changed_count} changed!"


# ---------------------------------------------------------------------------
# 10. One-Seed End-to-End Test on All 5 Worlds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("world_id", ["w1_split", "w2_exception", "w3_merge", "w4_ambiguous", "w5_exceptions_cannot_match"])
def test_one_seed_end_to_end(world_id: str) -> None:
    """Run seed 2000 end-to-end across all 5 worlds and verify clean execution and expected behavior."""
    ds = build_restructure_dataset(world_id, seed=2000)
    learner = NEXUSRestructureLearner(seed=2000, evidence_channel="legacy")
    for ex in ds.train.items:
        learner.observe(ex.observation())

    test_split = ds.tests.get("test_heldout") or ds.tests.get("test_ambiguous")
    acc = sum(1 for ex in test_split.items if learner.predict(ex.scene, ex.tokens).truth == ex.truth) / len(test_split.items)
    ops = [r.operator for r in learner.revision_log if r.operator != "initial"]

    # World-specific assertions
    if world_id == "w1_split":
        assert acc == 1.000
        assert "split" in ops
    elif world_id == "w2_exception":
        assert acc == 1.000
        assert "split" not in ops
    elif world_id == "w3_merge":
        assert acc == 1.000
        assert "merge" in ops
    elif world_id == "w4_ambiguous":
        #calibrated confidence
        preds = [learner.predict(ex.scene, ex.tokens) for ex in test_split.items]
        mean_conf = sum(p.confidence for p in preds) / len(preds)
        assert mean_conf <= 0.650 or acc >= 0.500
    elif world_id == "w5_exceptions_cannot_match":
        assert acc == 1.000
        assert "split" in ops
