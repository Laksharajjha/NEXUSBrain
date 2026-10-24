"""Stage B Tests: Phase 1 NEXUSLearner, Explainability, Faithfulness, Invariance, and Baselines B & C."""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from nexus.baselines.cross_situational import CrossSituationalLearner
from nexus.baselines.no_revision import NoRevisionLearner
from nexus.baselines.oracle import OracleLeakageError
from nexus.eval.learner_api import Learner
from nexus.eval.splits import build_dataset
from nexus.learner import (
    CATEGORY_DEPTHS,
    LexicalEntry,
    NEXUSLearner,
    category_depth,
    compute_mdl_prior,
    evaluate_parsed_lf,
    intervene_on_lexicon,
    intervene_on_scene,
    parse_tokens,
)
from nexus.world.config import WorldConfig, load_world_config
from nexus.world.example import Correction, Observation


def test_source_scan_no_invented_words_or_oracle_in_learner():
    """No file in nexus/learner/ may reference oracle or any of the 16 invented surface words."""
    cfg = load_world_config()
    invented_words = set(cfg.grammar.lexicon.values()) | {cfg.grammar.question_particle}
    assert len(invented_words) == 16

    learner_dir = Path(__file__).resolve().parent.parent / "learner"
    py_files = list(learner_dir.rglob("*.py"))
    assert len(py_files) >= 3

    for py_file in py_files:
        text = py_file.read_text(encoding="utf-8").lower()
        assert "oracle" not in text, f"Forbidden substring 'oracle' found in {py_file}"
        # Check tokenized identifiers/words in file against invented words
        import re

        words_in_file = set(re.findall(r"[a-z_][a-z0-9_]*", text))
        leaked = words_in_file & invented_words
        assert not leaked, f"Invented surface words {leaked} found in {py_file}"


def test_mdl_prior_and_likelihood_scoring(small_config: WorldConfig):
    """Unit test for MDL prior formula, epsilon=0.02 likelihood, and beam pruning."""
    entries = {
        "w_noun": LexicalEntry("w_noun", "N", "kind:cube", ("ex1",)),
        "w_adj": LexicalEntry("w_adj", "N/N", "color:red", ("ex1",)),
        "w_verb": LexicalEntry("w_verb", "(S\\NP)/NP", "rel:chases(x0,x1)", ("ex2",)),
        "w_q": LexicalEntry("w_q", "S\\S", "Q", ("ex3",)),
    }
    alpha = 0.05
    # |lexicon| = 4
    # depths = 0 + 1 + 2 + 1 = 4
    # atoms  = 1 + 1 + 1 + 0 = 3
    expected_prior = -alpha * (4 + 4 + 3)
    assert math.isclose(compute_mdl_prior(entries, alpha), expected_prior, rel_tol=1e-12)

    # Check beam pruning respects beam_width K
    ds = build_dataset(small_config, seed=0, world_mode="randomized")
    learner = NEXUSLearner(seed=0, beam_width=7, alpha=0.05, epsilon=0.02)
    for ex in ds.train.items[:40]:
        learner.observe(ex.observation())

    beam = learner.beam()
    assert 1 <= len(beam) <= 7
    # Scores must be sorted descending and equal to prior_score + likelihood_score
    for i, hyp in enumerate(beam):
        assert math.isclose(hyp.total_score, hyp.prior_score + hyp.likelihood_score, rel_tol=1e-12)
        if i + 1 < len(beam):
            assert hyp.total_score >= beam[i + 1].total_score - 1e-12


def test_lexicon_permutation_invariance_20_seeds(small_config: WorldConfig):
    """NEXUSLearner predictions and induced concepts must be invariant to surface word permutations across 20 seeds."""
    for seed in range(20):
        ds = build_dataset(small_config, seed=seed, world_mode="randomized")
        vocab = sorted(set(ds.grammar.lexicon.values()) | {ds.grammar.question_particle})
        rng = random.Random(seed + 5000)
        shuffled = list(vocab)
        rng.shuffle(shuffled)
        perm = dict(zip(vocab, shuffled))

        learner_orig = NEXUSLearner(seed=seed, beam_width=50)
        learner_perm = NEXUSLearner(seed=seed, beam_width=50)

        for ex in ds.train.items:
            obs = ex.observation()
            learner_orig.observe(obs)
            perm_obs = Observation(
                example_id=obs.example_id,
                scene=obs.scene,
                tokens=tuple(perm[t] for t in obs.tokens),
                label=obs.label,
                source=obs.source,
            )
            learner_perm.observe(perm_obs)

        top_orig = learner_orig.top_hypothesis()
        top_perm = learner_perm.top_hypothesis()
        assert top_orig is not None and top_perm is not None
        assert set(top_perm.entries.keys()) == {perm[w] for w in top_orig.entries.keys()}
        for w, orig_entry in top_orig.entries.items():
            perm_entry = top_perm.entries[perm[w]]
            assert orig_entry.category == perm_entry.category, f"Seed {seed}: category mismatch on {w}"
            assert orig_entry.predicate == perm_entry.predicate, f"Seed {seed}: predicate mismatch on {w}"

        for ex in ds.dev.items[:10]:
            p_orig = learner_orig.predict(ex.scene, ex.tokens)
            p_perm = learner_perm.predict(ex.scene, tuple(perm[t] for t in ex.tokens))
            assert math.isclose(p_orig.p_true, p_perm.p_true, abs_tol=1e-12)
            assert p_orig.lf == p_perm.lf


def test_explain_and_faithfulness_interventions(small_config: WorldConfig):
    """explain() returns derivation, provenance, witness/failure; intervening flips prediction."""
    ds = build_dataset(small_config, seed=3, world_mode="randomized")
    learner = NEXUSLearner(seed=3)
    for ex in ds.train.items:
        learner.observe(ex.observation())

    top = learner.top_hypothesis()
    assert top is not None

    checked_true = 0
    checked_false = 0
    for ex in ds.dev.items:
        exp = learner.explain(ex.tokens, ex.scene)
        assert exp.parse_tree is not None
        assert exp.derived_lf is not None
        assert len(exp.lexical_entries) == len(ex.tokens)
        assert exp.posterior_weight > 0.0
        # Every content entry must cite supporting training example IDs
        for entry in exp.lexical_entries:
            assert len(entry.supporting_examples) > 0

        if exp.truth:
            assert exp.witness is not None
            checked_true += 1
        else:
            assert exp.witness is None
            assert "failed" in exp.failure_reason or "does not hold" in exp.failure_reason
            checked_false += 1

        # 1. Scene intervention faithfulness: altering the witness/atoms named in exp flips prediction
        cf_scene = intervene_on_scene(ex.scene, exp)
        cf_pred = learner.predict(cf_scene, ex.tokens)
        assert cf_pred.truth != exp.truth, (
            f"Scene intervention failed to flip prediction on {ex.id} (was {exp.truth}, got {cf_pred.truth})"
        )

        # 2. Lexicon intervention faithfulness: altering the lexical entry cited in exp flips evaluation
        # (For False transitive sentences with a repeated token, check when scene has a satisfiable relation)
        has_repeated_token = len(set(ex.tokens)) < len(ex.tokens)
        if exp.truth or not has_repeated_token:
            cf_entries = intervene_on_lexicon(top.entries, exp, ex.scene)
            cf_parsed = parse_tokens(ex.tokens, cf_entries)
            assert cf_parsed is not None
            _, cf_lf, cf_sym = cf_parsed
            cf_lex_truth = evaluate_parsed_lf(cf_lf, ex.scene, cf_sym)
            assert cf_lex_truth != exp.truth, (
                f"Lexicon intervention failed to flip prediction on {ex.id} (was {exp.truth}, got {cf_lex_truth})"
            )

    assert checked_true >= 5 and checked_false >= 5


def test_what_changed_and_correction_retraction(small_config: WorldConfig):
    """what_changed() tracks created, promoted, and demoted hypotheses on observe and correct."""
    ds = build_dataset(small_config, seed=1, world_mode="randomized")
    learner = NEXUSLearner(seed=1, beam_width=50)

    # First observation creates hypotheses in the beam
    first_obs = ds.train.items[0].observation()
    learner.observe(first_obs)
    rec0 = learner.what_changed(first_obs.example_id)
    assert rec0.example_id == first_obs.example_id
    assert rec0.event_kind == "observe"
    assert len(rec0.created) > 0

    # Stream remaining training examples
    for ex in ds.train.items[1:]:
        learner.observe(ex.observation())

    # Apply contradiction support stream and verify correction flips misleading hypothesis
    contra = ds.tests["test_contradiction"]
    for ev in contra.support:
        if ev.kind == "observe":
            assert ev.observation is not None
            learner.observe(ev.observation)
        else:
            assert ev.correction is not None
            learner.correct(ev.correction)
            crec = learner.what_changed(ev.correction.target_example_id)
            assert crec.event_kind == "correct"
            assert len(crec.promoted) > 0 or len(crec.demoted) > 0

    # After corrections, accuracy on contradiction probes must be >= 0.90
    preds = [learner.predict(e.scene, e.tokens) for e in contra.items]
    acc = sum(p.truth == e.truth for p, e in zip(preds, contra.items)) / len(contra.items)
    assert acc >= 0.90


def test_baselines_b_and_c_protocol_and_hierarchy(small_config: WorldConfig):
    """Baselines B (CrossSituational) and C (NoRevision) implement Learner and NEXUS outperforms them."""
    ds = build_dataset(small_config, seed=2, world_mode="randomized")
    b_learner = CrossSituationalLearner(seed=2)
    c_learner = NoRevisionLearner(seed=2)
    nexus = NEXUSLearner(seed=2)

    assert isinstance(b_learner, Learner)
    assert isinstance(c_learner, Learner)
    assert isinstance(nexus, Learner)

    for ex in ds.train.items:
        obs = ex.observation()
        b_learner.observe(obs)
        c_learner.observe(obs)
        nexus.observe(obs)

    assert b_learner.model_size() > 0
    assert c_learner.model_size() > 0
    assert nexus.model_size() > 0

    novel_comb = ds.tests["test_novel_combination"].items
    acc_b = sum(b_learner.predict(e.scene, e.tokens).truth == e.truth for e in novel_comb) / len(novel_comb)
    acc_c = sum(c_learner.predict(e.scene, e.tokens).truth == e.truth for e in novel_comb) / len(novel_comb)
    acc_nexus = sum(nexus.predict(e.scene, e.tokens).truth == e.truth for e in novel_comb) / len(novel_comb)

    assert acc_nexus >= 0.90
    assert acc_nexus > acc_b
    assert acc_nexus > acc_c
