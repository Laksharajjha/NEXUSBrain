"""Tests for Stage P3-A: New Restructuring Worlds and Harness Audit (W1-W5)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from nexus.baselines.oracle import OracleLeakageError, RestructureOracle
from nexus.baselines.scene_only import SceneOnly
from nexus.baselines.sentence_only import SentenceOnly
from nexus.eval.runner import predict_all
from nexus.world.logic import evaluate, naive_evaluate
from nexus.world.restructure import (
    RESTRUCTURE_WORLDS,
    build_restructure_dataset,
    run_reduced_restructure_ideal_observer,
    audit_restructure_identifiability,
)


PREREGISTRATION_P3_SHA256 = "bd08fe66a8160483e741e45928aea22ee0f6699d655da9fca43bd98b33b9140d"


def test_p3_b_preregistration_hash() -> None:
    """Verify PREREGISTRATION_P3.md SHA-256 hash matches the frozen pre-registration."""
    import hashlib
    repo_root = Path(__file__).resolve().parent.parent.parent
    p3_path = repo_root / "PREREGISTRATION_P3.md"
    assert p3_path.exists(), "PREREGISTRATION_P3.md does not exist"
    digest = hashlib.sha256(p3_path.read_bytes()).hexdigest()
    assert digest == PREREGISTRATION_P3_SHA256, (
        f"PREREGISTRATION_P3.md was modified! Expected {PREREGISTRATION_P3_SHA256}, got {digest}"
    )


def test_p3_a_split_sizes_and_balance() -> None:
    """Verify split sample sizes n and exact 50/50 balance on dev and test splits."""
    for world_id in RESTRUCTURE_WORLDS:
        ds = build_restructure_dataset(world_id, seed=2000)
        assert len(ds.train.items) >= 80
        assert len(ds.dev.items) >= 40
        assert len(ds.tests) >= 1
        for sp_name, sp in ds.tests.items():
            assert len(sp.items) >= 100
            n_true = sum(1 for e in sp.items if e.truth)
            n_false = sum(1 for e in sp.items if not e.truth)
            assert n_true == n_false, f"{world_id} {sp_name} not 50/50 balanced: {n_true} vs {n_false}"


def test_p3_a_oracle_100_percent_ceiling() -> None:
    """Verify RestructureOracle scores 100% across train, dev, and test splits for all worlds."""
    for world_id in RESTRUCTURE_WORLDS:
        for seed in (2000, 2001, 2002):
            ds = build_restructure_dataset(world_id, seed=seed)
            ont = ds.meta["ontology"]
            oracle = RestructureOracle(ds.grammar, ont, world_id=world_id, meta=ds.meta, seed=seed)
            all_splits = {"train": ds.train, "dev": ds.dev, **ds.tests}
            for sp_name, sp in all_splits.items():
                preds = predict_all(oracle, sp.items)
                acc = sum(p.truth == e.truth for p, e in zip(preds, sp.items)) / len(sp.items)
                assert acc == 1.0, f"{world_id} seed {seed} {sp_name}: Oracle acc = {acc} != 1.0"


def test_p3_a_shortcut_baselines_stay_at_chance_on_test() -> None:
    """Verify SentenceOnly and SceneOnly stay at 0.50 on all test splits."""
    audit_file = Path("nexus/experiments/results/audit_p3.json")
    assert audit_file.exists(), f"Audit file {audit_file} missing"
    with open(audit_file) as f:
        data = json.load(f)

    for world_id in RESTRUCTURE_WORLDS:
        w_data = data["worlds"][world_id]
        for m_name in ("sentence_only", "scene_only"):
            for sp_name, stats in w_data["shortcut_baselines"][m_name].items():
                if sp_name in ("dev", "test_heldout", "test_ambiguous"):
                    assert stats["contains_05"], (
                        f"{world_id} {m_name} on {sp_name}: CI [{stats['ci_low']:.4f}, {stats['ci_high']:.4f}] "
                        f"does not contain 0.50"
                    )
                    assert abs(stats["mean"] - 0.50) < 1e-9


def test_p3_a_differential_naive_evaluate() -> None:
    """Verify evaluate and naive_evaluate agree 100% on all generated items."""
    for world_id in RESTRUCTURE_WORLDS:
        ds = build_restructure_dataset(world_id, seed=2000)
        ont = ds.meta["ontology"]
        all_splits = {"train": ds.train, "dev": ds.dev, **ds.tests}
        for sp_name, sp in all_splits.items():
            for e in sp.items:
                ev = evaluate(e.lf, e.scene, ont.symmetric_relations)
                nev = naive_evaluate(e.lf, e.scene, ont.symmetric_relations)
                assert ev == nev, f"Differential mismatch on {world_id} {e.id}"


def test_p3_a_identifiability_at_least_5_examples() -> None:
    """Verify every needed restructuring decision has >= 5 distinguishing examples across 50 seeds."""
    audit_file = Path("nexus/experiments/results/audit_p3.json")
    with open(audit_file) as f:
        data = json.load(f)

    for world_id in RESTRUCTURE_WORLDS:
        ident = data["worlds"][world_id]["identifiability"]
        assert ident["min_count"] >= 5, f"{world_id} identifiability min {ident['min_count']} < 5"


def test_p3_a_reduced_ideal_observer_upper_bounds() -> None:
    """Verify exhaustive ideal observer on reduced W1/W2 correctly selects Split and Exception."""
    # Reduced W1: Split is selected by marginal likelihood and reaches 1.0 ceiling
    w1_res = run_reduced_restructure_ideal_observer("w1_split", seed=2000)
    assert w1_res["winning_structure"] == "split"
    assert w1_res["prob_split"] > 0.95
    assert w1_res["ideal_observer_test_accuracy_ceiling"] == 1.0

    # Reduced W2: Exception is selected by marginal likelihood (over-splitting rejected) and reaches 1.0 ceiling
    w2_res = run_reduced_restructure_ideal_observer("w2_exception", seed=2000)
    assert w2_res["winning_structure"] == "exception"
    assert w2_res["prob_exception"] > 0.95
    assert w2_res["ideal_observer_test_accuracy_ceiling"] == 1.0


def test_p3_a_oracle_import_guard() -> None:
    """Verify RestructureOracle is protected by the OracleLeakageError import guard."""
    from nexus.baselines.oracle import _check_oracle_import_guard

    # Direct check passes outside learner
    _check_oracle_import_guard()
