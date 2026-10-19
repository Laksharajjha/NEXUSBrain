"""Tests for dataset split construction, disjointness, and properties."""

from __future__ import annotations

import pytest

from nexus.eval.splits import build_dataset
from nexus.world.config import ADJECTIVE_ATTRIBUTES, load_world_config


def test_split_sizes_and_balance(dataset):
    """Verify expected sizes and exact 50/50 balance on test splits."""
    assert len(dataset.train.items) == 600
    assert len(dataset.dev.items) == 120  # 4 constructions * 30

    for name, split in dataset.tests.items():
        assert len(split.items) > 0
        truths = [e.truth for e in split.items]
        # Test splits must have exactly balanced truth values
        assert truths.count(True) == truths.count(False), f"{name} is not balanced"


def test_no_leakage_between_train_and_tests(dataset):
    """Hard requirement: no test item ((tokens, canonical_scene)) appears in train."""
    train_keys = {e.item_key() for e in dataset.train.items}
    train_tokens = {e.tokens for e in dataset.train.items}

    # Dev is scene-disjoint from train
    for e in dataset.dev.items:
        assert e.item_key() not in train_keys

    # test_novel_combination: token sequences must not appear in train
    for e in dataset.tests["test_novel_combination"].items:
        assert e.item_key() not in train_keys
        assert e.tokens not in train_tokens
        assert any(t.startswith("heldout_") for t in e.tags)

    # test_novel_word_in_known_frame: novel noun never in train
    novel_kind = dataset.plan.novel_kind_predicate
    for e in dataset.train.items:
        assert all(atom.predicate != novel_kind for atom in e.lf.atoms)
    for e in dataset.tests["test_novel_word_in_known_frame"].items:
        assert e.item_key() not in train_keys
        assert any(atom.predicate == novel_kind for atom in e.lf.atoms)

    # test_longer: strictly longer than max_train_length
    for e in dataset.tests["test_longer"].items:
        assert e.item_key() not in train_keys
        assert e.tokens not in train_tokens
        assert len(e.tokens) > dataset.max_train_length

    # test_contradiction: items are disjoint from train
    for e in dataset.tests["test_contradiction"].items:
        assert e.item_key() not in train_keys


def test_support_stream_structure(dataset, small_config):
    """Check support stream events and checkpoints for support-requiring splits."""
    nw = dataset.tests["test_novel_word_in_known_frame"]
    assert len(nw.support) == 2  # 1 contrast pair = 2 examples
    assert len(nw.support_checkpoints) == 2
    assert nw.support_checkpoints[0] == ("before_support", 0)
    assert nw.support_checkpoints[1] == ("after_support", 2)

    tc = dataset.tests["test_contradiction"]
    assert len(tc.support) > 0
    assert len(tc.support_checkpoints) == 3
    # Mislabelled examples must be marked
    mislabelled = [ev for ev in tc.support if ev.mislabelled]
    assert len(mislabelled) == small_config.splits.n_contradiction_misleading
    # Corrections must target the misleading examples
    corrections = [ev for ev in tc.support if ev.kind == "correct"]
    assert len(corrections) == len(mislabelled)
    mis_ids = {ev.ground_truth.id for ev in mislabelled}
    cor_ids = {ev.correction.target_example_id for ev in corrections}
    assert mis_ids == cor_ids


def test_determinism_across_runs(small_config):
    """Same seed must produce bit-for-bit identical datasets."""
    ds1 = build_dataset(small_config, seed=42)
    ds2 = build_dataset(small_config, seed=42)

    assert ds1.plan == ds2.plan
    assert [e.id for e in ds1.train.items] == [e.id for e in ds2.train.items]
    assert [e.tokens for e in ds1.train.items] == [e.tokens for e in ds2.train.items]
    assert [e.truth for e in ds1.train.items] == [e.truth for e in ds2.train.items]
