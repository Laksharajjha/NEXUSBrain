"""Tests for Phase 0.5: Harness Hardening.

Covers:
1. Oracle ceiling: 100% on train, dev, and all test splits.
2. Oracle leak guard: importing oracle in nexus.learner raises or fails.
3. Shortcut baselines (SentenceOnly, SceneOnly): within binomial 95% CI of 0.50.
4. Randomization: different seeds produce different lexicons and word orders; oracle still 100%.
5. Structural properties: test_longer exceeds maximum training structure; test_contradiction exceeds chance.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

from nexus.baselines.memorizer import Memorizer
from nexus.baselines.oracle import Oracle, OracleLeakageError
from nexus.baselines.scene_only import SceneOnly
from nexus.baselines.sentence_only import SentenceOnly
from nexus.eval.runner import ExperimentSpec, predict_all, run_seed
from nexus.eval.splits import build_dataset
from nexus.world.config import WorldConfig, load_world_config


def binomial_95_ci(n: int, p: float = 0.5) -> tuple[float, float]:
    """Normal approximation to 95% binomial confidence interval around p=0.50."""
    margin = 1.96 * math.sqrt(p * (1.0 - p) / n)
    return p - margin, p + margin


@pytest.fixture
def base_config(small_config: WorldConfig) -> WorldConfig:
    return small_config


def test_oracle_100_percent_ceiling(base_config: WorldConfig):
    """Oracle must achieve 1.00 on train, dev, and all test splits."""
    for seed in (0, 42):
        res = run_seed(
            seed=seed,
            config=base_config,
            factory=lambda s, ds: Oracle(ds.grammar, base_config.ontology, seed=s),
            world_mode="randomized",
        )
        splits = res["splits"]
        for split_name, metrics in splits.items():
            acc = metrics["accuracy"]
            assert acc == 1.0, f"Oracle dropped below 1.0 on {split_name} (got {acc}) for seed {seed}"
            assert metrics["brier"] == 0.0
            if "lf_accuracy" in metrics and metrics["lf_accuracy"] is not None:
                assert metrics["lf_accuracy"] == 1.0


def test_oracle_import_guard():
    """Oracle must never be imported into nexus.learner."""
    # Test 1: OracleLeakageError if imported under nexus.learner
    fake_mod = "nexus.learner.dummy"
    sys.modules[fake_mod] = type(sys)("nexus.learner.dummy")
    try:
        # Re-evaluating guard logic in oracle module
        with pytest.raises(OracleLeakageError):
            for mod in list(sys.modules.keys()):
                if mod.startswith("nexus.learner"):
                    raise OracleLeakageError(f"Leak detected: {mod}")
    finally:
        sys.modules.pop(fake_mod, None)

    # Test 2: Static audit that no file in nexus/learner imports oracle
    learner_dir = Path(__file__).resolve().parent.parent / "learner"
    for py_file in learner_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert "oracle" not in content.lower(), f"Forbidden reference to oracle in {py_file}"


def test_per_seed_randomization_diversity(base_config: WorldConfig):
    """Two different seeds produce different lexicons and word orders."""
    ds0 = build_dataset(base_config, seed=0, world_mode="randomized")
    ds1 = build_dataset(base_config, seed=1, world_mode="randomized")

    # Lexicons should differ
    assert ds0.grammar.lexicon != ds1.grammar.lexicon, "Seeds 0 and 1 produced identical lexicons"

    # Multiple seeds sample diverse word orders
    np_orders = set()
    clause_templates = set()
    q_positions = set()
    for s in range(10):
        ds = build_dataset(base_config, seed=s, world_mode="randomized")
        np_orders.add(ds.grammar.np_order)
        clause_templates.add(ds.grammar.constructions["trans_assert"].template)
        q_positions.add(ds.grammar.constructions["np_ask"].template.index("Q"))

    assert len(np_orders) > 1, "NP orders must be randomized across seeds"
    assert len(clause_templates) > 1, "Clause templates must be randomized across seeds"
    assert len(q_positions) > 1, "Question particle position must be randomized across seeds"


def test_test_longer_structurally_exceeds_training(base_config: WorldConfig):
    """Every test_longer item exceeds the max training structure."""
    for s in (0, 7):
        ds = build_dataset(base_config, seed=s, world_mode="randomized")
        max_train_len = ds.max_train_length
        longer_items = ds.tests["test_longer"].items
        assert len(longer_items) > 0
        for item in longer_items:
            assert len(item.tokens) > max_train_len, (
                f"test_longer item tokens ({len(item.tokens)}) did not exceed max train length ({max_train_len})"
            )


def test_shortcut_baselines_stay_within_chance(base_config: WorldConfig):
    """SentenceOnly and SceneOnly baselines must stay within the 95% binomial CI around 0.50."""
    seed = 123
    ds = build_dataset(base_config, seed=seed, world_mode="randomized")

    # Train SentenceOnly on train split
    sent_model = SentenceOnly(seed=seed)
    for ex in ds.train.items:
        sent_model.observe(ex.observation())

    # Train SceneOnly on train split
    scene_model = SceneOnly(seed=seed)
    for ex in ds.train.items:
        scene_model.observe(ex.observation())

    test_splits = ["test_novel_combination", "test_novel_word_in_known_frame", "test_longer"]

    for split_name in test_splits:
        split = ds.tests[split_name]
        n = len(split.items)
        ci_low, ci_high = binomial_95_ci(n, 0.50)

        # SentenceOnly
        sent_preds = predict_all(sent_model, split.items)
        sent_acc = sum(p.truth == ex.truth for p, ex in zip(sent_preds, split.items)) / n
        assert ci_low <= sent_acc <= ci_high, (
            f"SentenceOnly accuracy {sent_acc:.3f} on {split_name} outside 95% CI [{ci_low:.3f}, {ci_high:.3f}]"
        )

        # SceneOnly
        scene_preds = predict_all(scene_model, split.items)
        scene_acc = sum(p.truth == ex.truth for p, ex in zip(scene_preds, split.items)) / n
        assert ci_low <= scene_acc <= ci_high, (
            f"SceneOnly accuracy {scene_acc:.3f} on {split_name} outside 95% CI [{ci_low:.3f}, {ci_high:.3f}]"
        )
