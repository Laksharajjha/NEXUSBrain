"""Test SQLite state persistence roundtrip for RestructureLearner (Stage P3-E)."""

import tempfile
from pathlib import Path

import pytest

from nexus.learner.restructure_learner import RestructureLearner
from nexus.storage.restructure_repository import SQLiteRestructureRepository
from nexus.world.restructure import build_restructure_dataset


def test_sqlite_persistence_roundtrip_w1():
    """Verify that saving RestructureLearner to SQLite and reloading produces bit-identical test predictions and revision log."""
    ds = build_restructure_dataset("w1_split", seed=2000)
    learner = RestructureLearner(seed=2000, evidence_channel="legacy")
    for ex in ds.train.items:
        learner.observe(ex.observation())

    # Pre-save predictions
    test_items = ds.tests["test_heldout"].items
    pre_preds = [learner.predict(ex.scene, ex.tokens) for ex in test_items]
    pre_log = list(learner.revision_log)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_restructure.db"
        repo = SQLiteRestructureRepository(db_path)

        repo.save_learner("run_w1_2000", learner)
        loaded = repo.load_learner("run_w1_2000")

        # Check predictions match exactly
        post_preds = [loaded.predict(ex.scene, ex.tokens) for ex in test_items]
        assert len(post_preds) == len(pre_preds)
        for pre, post in zip(pre_preds, post_preds):
            assert pre.truth == post.truth
            assert pytest.approx(pre.p_true, abs=1e-6) == post.p_true
            assert pytest.approx(pre.confidence, abs=1e-6) == post.confidence

        # Check revision log
        assert len(loaded.revision_log) == len(pre_log)
        for r_pre, r_post in zip(pre_log, loaded.revision_log):
            assert r_pre.revision_id == r_post.revision_id
            assert r_pre.operator == r_post.operator
            assert r_pre.target_word == r_post.target_word
            assert pytest.approx(r_pre.score_change, abs=1e-6) == r_post.score_change

        # Check explain
        sample_ex = test_items[0]
        pre_expl = learner.explain(sample_ex.scene, sample_ex.tokens)
        post_expl = loaded.explain(sample_ex.scene, sample_ex.tokens)
        assert pre_expl.truth == post_expl.truth
        assert pre_expl.failure_reason == post_expl.failure_reason


def test_sqlite_persistence_roundtrip_w2_exception():
    """Verify persistence on W2 where exceptions and parsimony are exercised."""
    ds = build_restructure_dataset("w2_exception", seed=2005)
    learner = RestructureLearner(seed=2005, evidence_channel="legacy")
    for ex in ds.train.items:
        learner.observe(ex.observation())

    test_items = ds.tests["test_heldout"].items
    pre_preds = [learner.predict(ex.scene, ex.tokens) for ex in test_items]

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_w2.db"
        repo = SQLiteRestructureRepository(db_path)

        repo.save_learner("run_w2_2005", learner)
        loaded = repo.load_learner("run_w2_2005")

        post_preds = [loaded.predict(ex.scene, ex.tokens) for ex in test_items]
        for pre, post in zip(pre_preds, post_preds):
            assert pre.truth == post.truth
            assert pytest.approx(pre.p_true, abs=1e-6) == post.p_true
