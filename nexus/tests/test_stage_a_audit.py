"""Tests for Stage A (Phase 0.6): Audit Patch (A1-A8)."""

from __future__ import annotations

import dataclasses
import hashlib
import random
from pathlib import Path

import pytest

from nexus.baselines.oracle import CorrectedOracle, StubbornOracle
from nexus.eval.audit import (
    audit_identifiability,
    audit_novel_word_support,
    feature_balance_audit,
    pooled_shortcut_audit,
    run_reduced_world_ideal_observer,
)
from nexus.eval.runner import run_seed
from nexus.eval.splits import derive_seed
from nexus.world.config import ConfigError, WorldConfig
from nexus.world.generator import AdjectivePolicy, WorldGenerator
from nexus.world.logic import Atom, evaluate, make_lf, naive_evaluate
from nexus.world.scene import build_scene, sample_scene

PREREGISTRATION_SHA256 = "ef09666eb2bb96ca481ce05dc3dc5ca50fbe6d2143441ed38b062968313f695b"


def test_a1_pooled_shortcut_audit(small_config: WorldConfig) -> None:
    """A1: Report n per split; pooled t-interval must contain 0.50 on all test splits."""
    res = pooled_shortcut_audit(small_config, n_seeds=10, base_seed=0, world_mode="randomized")
    assert set(res["split_n"].keys()) == {
        "train",
        "dev",
        "test_novel_combination",
        "test_novel_word_in_known_frame",
        "test_longer",
        "test_contradiction",
    }
    for sp_name, n_val in res["split_n"].items():
        assert n_val > 0, f"Split {sp_name} has invalid n={n_val}"

    test_splits = [
        "test_novel_combination",
        "test_novel_word_in_known_frame",
        "test_longer",
        "test_contradiction",
    ]
    for model_name in ("sentence_only", "scene_only"):
        for sp_name in test_splits:
            stat = res["models"][model_name][sp_name]
            assert stat["contains_05"], (
                f"{model_name} on {sp_name}: 95% t-CI [{stat['ci_low']:.4f}, {stat['ci_high']:.4f}] "
                f"does not contain 0.50 (mean={stat['mean']:.4f})"
            )


def test_a2_feature_balance_with_holm_and_bh(small_config: WorldConfig) -> None:
    """A2: Feature-level label-balance audit across every split with Holm and BH corrections."""
    report = feature_balance_audit(
        small_config, n_seeds=5, base_seed=0, n_permutations=200, world_mode="randomized"
    )
    for sp_name, feat_map in report.items():
        # Verify required feature families exist
        assert "n_objs" in feat_map
        assert "n_rels" in feat_map
        assert "topo:0" in feat_map
        assert "topo:2_cycle" in feat_map
        assert "novel_kind_present" in feat_map
        assert "sentence_length" in feat_map
        for f_name, row in feat_map.items():
            assert abs(row["diff"]) < 1e-9, (
                f"Split {sp_name} feature {f_name} imbalanced: "
                f"P(f|true)={row['mean_true']:.4f} vs P(f|false)={row['mean_false']:.4f}"
            )
            assert row["p_holm"] > 0.05
            assert row["q_bh"] > 0.05


def test_a3_novel_word_support_enumeration_and_balance(small_config: WorldConfig) -> None:
    """A3: Novel-kind presence is balanced in test scenes; 2-example support uniquely grounds novel kind."""
    for seed in range(5):
        res = audit_novel_word_support(small_config, seed=seed)
        assert abs(res["p_novel_kind_given_true"] - res["p_novel_kind_given_false"]) < 1e-9
        assert res["candidate_count"] == 1
        assert res["consistent_candidates_support_only"] == [res["novel_kind_predicate"]]
        assert res["ideal_observer_ceiling"] == 1.0


def test_a4_stubborn_vs_corrected_oracle(small_config: WorldConfig) -> None:
    """A4: StubbornOracle scores < 0.50 on diagnostic probes; CorrectedOracle scores 1.00."""
    for seed in (0, 1):
        stubborn_res = run_seed(
            seed=seed,
            config=small_config,
            factory=lambda s, ds: StubbornOracle(
                ds.grammar,
                small_config.ontology,
                ds.plan.contradiction_target,
                ds.plan.contradiction_confusion,
                seed=s,
            ),
            world_mode="randomized",
        )
        tc_stubborn = stubborn_res["splits"]["test_contradiction"]
        assert tc_stubborn["accuracy"] < 0.50, (
            f"StubbornOracle must score < 0.50 on test_contradiction, got {tc_stubborn['accuracy']}"
        )
        assert tc_stubborn["accuracy"] == 0.0

        corrected_res = run_seed(
            seed=seed,
            config=small_config,
            factory=lambda s, ds: CorrectedOracle(
                ds.grammar,
                small_config.ontology,
                ds.plan.contradiction_target,
                ds.plan.contradiction_confusion,
                seed=s,
            ),
            world_mode="randomized",
        )
        tc_corrected = corrected_res["splits"]["test_contradiction"]
        # Before corrections: corrupted reading -> < 0.50
        assert tc_corrected["checkpoints"]["before_corrections"]["accuracy"] < 0.50
        # After corrections: repaired reading -> 1.00
        assert tc_corrected["checkpoints"]["after_corrections"]["accuracy"] == 1.0
        assert tc_corrected["accuracy"] == 1.0


def test_a5_naive_model_checker_20_golden_cases() -> None:
    """A5 (part 1): 20 hand-verified golden cases comparing evaluate and naive_evaluate."""
    sym = frozenset({"next_to"})

    sc1 = build_scene(
        [
            {"kind": "cube", "color": "red", "size": "small"},
            {"kind": "ball", "color": "blue", "size": "large"},
        ],
        [("chases", 0, 1)],
    )
    sc2 = build_scene(
        [
            {"kind": "cone", "color": "green", "size": "medium"},
            {"kind": "ring", "color": "yellow", "size": "small"},
            {"kind": "star", "color": "red", "size": "large"},
        ],
        [("next_to", 0, 1), ("chases", 2, 0)],
    )
    sc3 = build_scene(
        [
            {"kind": "cup", "color": "blue", "size": "medium"},
            {"kind": "cube", "color": "blue", "size": "large"},
            {"kind": "ball", "color": "green", "size": "small"},
        ],
        [],
    )

    golden_cases = [
        # 1. Simple kind match (true)
        (sc1, make_lf("assert", ("x0",), [Atom("kind:cube", ("x0",))]), True),
        # 2. Simple kind mismatch (false)
        (sc1, make_lf("assert", ("x0",), [Atom("kind:cone", ("x0",))]), False),
        # 3. Kind + color match on same object (true)
        (sc1, make_lf("assert", ("x0",), [Atom("kind:cube", ("x0",)), Atom("color:red", ("x0",))]), True),
        # 4. Kind + color cross-object binding mismatch (false)
        (sc1, make_lf("assert", ("x0",), [Atom("kind:cube", ("x0",)), Atom("color:blue", ("x0",))]), False),
        # 5. Kind + color + size all match (true)
        (
            sc1,
            make_lf(
                "ask",
                ("x0",),
                [Atom("kind:ball", ("x0",)), Atom("color:blue", ("x0",)), Atom("size:large", ("x0",))],
            ),
            True,
        ),
        # 6. Kind + color match but size wrong (false)
        (
            sc1,
            make_lf(
                "ask",
                ("x0",),
                [Atom("kind:ball", ("x0",)), Atom("color:blue", ("x0",)), Atom("size:small", ("x0",))],
            ),
            False,
        ),
        # 7. Directed relation chases(cube, ball) holds (true)
        (
            sc1,
            make_lf(
                "assert",
                ("x0", "x1"),
                [Atom("kind:cube", ("x0",)), Atom("kind:ball", ("x1",)), Atom("rel:chases", ("x0", "x1"))],
            ),
            True,
        ),
        # 8. Directed relation chases(ball, cube) reversed fails (false)
        (
            sc1,
            make_lf(
                "assert",
                ("x0", "x1"),
                [Atom("kind:ball", ("x0",)), Atom("kind:cube", ("x1",)), Atom("rel:chases", ("x0", "x1"))],
            ),
            False,
        ),
        # 9. Symmetric relation next_to(cone, ring) forward holds (true)
        (
            sc2,
            make_lf(
                "assert",
                ("x0", "x1"),
                [Atom("kind:cone", ("x0",)), Atom("kind:ring", ("x1",)), Atom("rel:next_to", ("x0", "x1"))],
            ),
            True,
        ),
        # 10. Symmetric relation next_to(ring, cone) backward holds (true)
        (
            sc2,
            make_lf(
                "assert",
                ("x0", "x1"),
                [Atom("kind:ring", ("x0",)), Atom("kind:cone", ("x1",)), Atom("rel:next_to", ("x0", "x1"))],
            ),
            True,
        ),
        # 11. Symmetric relation next_to(star, ring) absent (false)
        (
            sc2,
            make_lf(
                "assert",
                ("x0", "x1"),
                [Atom("kind:star", ("x0",)), Atom("kind:ring", ("x1",)), Atom("rel:next_to", ("x0", "x1"))],
            ),
            False,
        ),
        # 12. Directed relation chases(star, cone) with modifiers holds (true)
        (
            sc2,
            make_lf(
                "ask",
                ("x0", "x1"),
                [
                    Atom("kind:star", ("x0",)),
                    Atom("color:red", ("x0",)),
                    Atom("kind:cone", ("x1",)),
                    Atom("size:medium", ("x1",)),
                    Atom("rel:chases", ("x0", "x1")),
                ],
            ),
            True,
        ),
        # 13. Directed relation chases(star, cone) with wrong modifier on object (false)
        (
            sc2,
            make_lf(
                "ask",
                ("x0", "x1"),
                [
                    Atom("kind:star", ("x0",)),
                    Atom("color:red", ("x0",)),
                    Atom("kind:cone", ("x1",)),
                    Atom("size:small", ("x1",)),
                    Atom("rel:chases", ("x0", "x1")),
                ],
            ),
            False,
        ),
        # 14. Relation type mismatch: chases where next_to holds (false)
        (
            sc2,
            make_lf(
                "assert",
                ("x0", "x1"),
                [Atom("kind:cone", ("x0",)), Atom("kind:ring", ("x1",)), Atom("rel:chases", ("x0", "x1"))],
            ),
            False,
        ),
        # 15. Relation type mismatch: next_to where chases holds (false)
        (
            sc2,
            make_lf(
                "assert",
                ("x0", "x1"),
                [Atom("kind:star", ("x0",)), Atom("kind:cone", ("x1",)), Atom("rel:next_to", ("x0", "x1"))],
            ),
            False,
        ),
        # 16. Scene with no relations: any relational LF is false
        (
            sc3,
            make_lf(
                "assert",
                ("x0", "x1"),
                [Atom("kind:cup", ("x0",)), Atom("kind:cube", ("x1",)), Atom("rel:next_to", ("x0", "x1"))],
            ),
            False,
        ),
        # 17. Shared color across two objects, disambiguated by size (true)
        (
            sc3,
            make_lf(
                "assert",
                ("x0",),
                [Atom("kind:cup", ("x0",)), Atom("color:blue", ("x0",)), Atom("size:medium", ("x0",))],
            ),
            True,
        ),
        # 18. Shared color across two objects, wrong kind-size pairing (false)
        (
            sc3,
            make_lf(
                "assert",
                ("x0",),
                [Atom("kind:cup", ("x0",)), Atom("color:blue", ("x0",)), Atom("size:large", ("x0",))],
            ),
            False,
        ),
        # 19. Single color atom only (true)
        (sc3, make_lf("assert", ("x0",), [Atom("color:green", ("x0",))]), True),
        # 20. Single color atom absent from scene (false)
        (sc3, make_lf("assert", ("x0",), [Atom("color:red", ("x0",))]), False),
    ]

    assert len(golden_cases) == 20
    for idx, (sc, lf, expected) in enumerate(golden_cases, start=1):
        res_main = evaluate(lf, sc, sym)
        res_naive = naive_evaluate(lf, sc, sym)
        assert res_main == expected, f"Golden case {idx} failed on main evaluate: got {res_main}, want {expected}"
        assert res_naive == expected, f"Golden case {idx} failed on naive_evaluate: got {res_naive}, want {expected}"


def test_a5_differential_1000_random_scenes_and_lfs(config: WorldConfig) -> None:
    """A5 (part 2): Differential property test of evaluate vs naive_evaluate on 1,000 random pairs."""
    rng = random.Random(2026)
    gen = WorldGenerator(config)
    policy = AdjectivePolicy(max_adjectives=2, p_adjective=0.6)
    sym = config.ontology.symmetric_relations
    all_kinds = list(config.ontology.kinds)

    for i in range(1000):
        c = gen.sample_construction(rng)
        m = gen.sample_meaning(rng, c, all_kinds, policy)
        lf = gen.grammar.to_lf(m)
        if i % 2 == 0:
            sc = sample_scene(rng, config.ontology, config.scene)
        else:
            sc = gen.true_scene(rng, m, all_kinds) if (i % 4 == 1) else gen.false_scene(rng, m, all_kinds)
        assert evaluate(lf, sc, sym) == naive_evaluate(lf, sc, sym), (
            f"Mismatch at iteration {i} on LF {lf.canonical()} and scene {sc.to_json()}"
        )


def test_a6_identifiability_across_constructions_and_orders(small_config: WorldConfig) -> None:
    """A6: Every construction and structural parameter has >= 5 distinguishing training examples."""
    for seed in range(6):
        rep = audit_identifiability(small_config, seed=seed)
        assert rep["flagged_under_5"] == [], (
            f"Seed {seed} had < 5 distinguishing examples for: {rep['flagged_under_5']}"
        )
        assert rep["clause_role_order_chases_contrasts"] >= 5
        assert rep["np_modifier_attachment_contrasts"] >= 5
        assert rep["q_particle_position_examples"] >= 5


def test_a7_preregistration_hash_and_small_fixes(config: WorldConfig) -> None:
    """A7: Verify PREREGISTRATION.md SHA-256, even dev count validator, and 0 stacked train NPs."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    prereg_path = repo_root / "PREREGISTRATION.md"
    digest = hashlib.sha256(prereg_path.read_bytes()).hexdigest()
    assert digest == PREREGISTRATION_SHA256, (
        f"PREREGISTRATION.md was modified! Expected SHA-256 {PREREGISTRATION_SHA256}, got {digest}"
    )

    # Even per-construction dev count enforced
    assert config.splits.n_dev_per_construction % 2 == 0
    bad_raw = config.to_json()
    bad_raw["world"]["splits"]["n_dev_per_construction"] = 15
    from nexus.world.config import parse_world

    with pytest.raises(ConfigError, match="even"):
        parse_world(bad_raw["world"], bad_raw["grammar"])

    # Confirm 0 training NPs have stacked size+color modifiers
    rep = audit_identifiability(config, seed=0)
    assert rep["n_train_stacked_nps"] == 0


def test_a8_exhaustive_bayesian_ideal_observer_reduced_world() -> None:
    """A8: Exhaustive Bayesian ideal observer over reduced world reaches 1.00 ceiling."""
    for seed in (0, 1):
        res = run_reduced_world_ideal_observer(seed=seed)
        assert res["n_hypotheses_enumerated"] == 720
        assert res["top_hypothesis_matches_ground_truth"] is True
        assert res["train"] == 1.0
        assert res["dev"] == 1.0
        assert res["test_novel_combination"] == 1.0
        assert res["test_novel_word_after_support"] == 1.0
