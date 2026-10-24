"""Tests for NEXUS-0 Phase 1.5 Adversarial Validation (EXPLORATORY).

Covers:
1. Part 1: SpyLearnerWrapper proving NEXUS never updates on held-out items before scoring
   and predict() never mutates learner state.
2. Part 1: Static AST audit confirming nexus/learner/ reads only Learner-protocol fields
   and never imports neural_ref or oracle.
3. Part 2: Negative controls (shuffled labels, shuffled scenes, cross-seed lexicon) drop
   NEXUS to chance (~0.50).
4. Part 3: Documented breaking points (homonymy/polysemy, positive-only training,
   multi-clause chains beyond slash depth 2).
5. Part 4: Baseline C bug diagnosis & fix, Baseline D (PositionAwareCrossSituationalLearner),
   and Baseline E (FromScratchNeuralBaseline + isolation guard).
6. Part 5: Parameter-free log2 bit-cost MDL prior vs linear alpha=0.05 prior.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from nexus.baselines.neural_ref import FromScratchNeuralBaseline, _enforce_neural_isolation
from nexus.baselines.no_revision import NoRevisionLearner
from nexus.baselines.position_aware_cs import PositionAwareCrossSituationalLearner
from nexus.eval.metrics import accuracy
from nexus.eval.splits import build_dataset, derive_seed
from nexus.experiments.phase1_5_adversarial import (
    SpyLearnerWrapper,
    _apply_token_mapping_to_dataset,
    _evaluate_beyond_depth_suites,
    _stratified_subsample,
)
from nexus.learner.nexus_learner import NEXUSLearner
from nexus.world.config import load_world_config


@pytest.fixture(scope="module")
def world_cfg():
    return load_world_config()


@pytest.fixture(scope="module")
def ds_seed0(world_cfg):
    return build_dataset(world_cfg, derive_seed(0, "world"), world_mode="randomized")


@pytest.fixture(scope="module")
def ds_seed1(world_cfg):
    return build_dataset(world_cfg, derive_seed(1, "world"), world_mode="randomized")


def test_part1_spy_wrapper_no_test_leakage_or_predict_mutation(ds_seed0) -> None:
    """Prove NEXUS never observes held-out items before scoring and predict() is pure."""
    spy = SpyLearnerWrapper(NEXUSLearner(seed=0))
    for ex in ds_seed0.train.items:
        spy.observe(ex.observation())

    # Score dev, test_novel_combination, test_longer
    for split in (
        ds_seed0.dev,
        ds_seed0.tests["test_novel_combination"],
        ds_seed0.tests["test_longer"],
    ):
        for ex in split.items:
            pred = spy.predict_for_example(ex)
            assert 0.0 <= pred.p_true <= 1.0

    # Score test_novel_word_in_known_frame after its 2 support observations
    nw_split = ds_seed0.tests["test_novel_word_in_known_frame"]
    for ev in nw_split.support:
        if ev.kind == "observe" and ev.observation is not None:
            spy.observe(ev.observation)
    for ex in nw_split.items:
        spy.predict_for_example(ex)

    # Score test_contradiction after its support + corrections
    ct_split = ds_seed0.tests["test_contradiction"]
    for ev in ct_split.support:
        if ev.kind == "observe" and ev.observation is not None:
            spy.observe(ev.observation)
        elif ev.kind == "correct" and ev.correction is not None:
            spy.correct(ev.correction)
    for ex in ct_split.items:
        spy.predict_for_example(ex)

    assert spy.leak_violations == []
    assert spy.mutation_violations == []
    assert spy.predict_calls > 500


def test_part1_ast_protocol_field_audit_and_neural_isolation() -> None:
    """Confirm nexus/learner/ never reads generator metadata or imports neural_ref."""
    learner_dir = Path(__file__).resolve().parents[1] / "learner"
    forbidden_attrs = {
        "lf",
        "construction",
        "contrast_group",
        "tags",
        "ground_truth",
        "mislabelled",
        "plan",
    }
    for py_file in sorted(learner_dir.glob("*.py")):
        src = py_file.read_text(encoding="utf-8")
        assert "neural_ref" not in src
        assert "FromScratchNeuralBaseline" not in src
        tree = ast.parse(src, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden_attrs, (
                    f"Forbidden attribute {node.attr!r} accessed in {py_file.name}:{node.lineno}"
                )


def test_part2_negative_controls_drop_to_chance(ds_seed0, ds_seed1) -> None:
    """Verify negative controls: 2b & 2c drop to ~0.50; 2a exposes near-miss unlabeled co-occurrence."""
    rng = np.random.default_rng(42)
    sub_train = _stratified_subsample(ds_seed0.train.items, 400, rng)
    eval_items = ds_seed0.tests["test_novel_combination"].items

    # 2a: Shuffled labels on near-miss training set exposes unlabeled (scene, tokens) co-occurrence leak
    labels = [ex.truth for ex in sub_train]
    perm_l = rng.permutation(len(labels))
    l_2a = NEXUSLearner(seed=0)
    for i, ex in enumerate(sub_train):
        l_2a.observe(ex.observation(label=bool(labels[int(perm_l[i])])))
    acc_2a = accuracy([l_2a.predict(e.scene, e.tokens) for e in eval_items], eval_items)
    assert acc_2a > 0.60  # Documents the Part 2a unlabeled scene-token co-occurrence leak

    # 2b: Shuffled scenes drops to chance (~0.50)
    scenes = [ex.scene for ex in sub_train]
    perm_s = rng.permutation(len(scenes))
    l_2b = NEXUSLearner(seed=0)
    for i, ex in enumerate(sub_train):
        shuf_ex = dataclasses.replace(ex, scene=scenes[int(perm_s[i])])
        l_2b.observe(shuf_ex.observation())
    acc_2b = accuracy([l_2b.predict(e.scene, e.tokens) for e in eval_items], eval_items)
    assert 0.35 <= acc_2b <= 0.62

    # 2c: Train on seed 1's lexicon, test on seed 0 drops to chance (~0.50)
    l_2c = NEXUSLearner(seed=0)
    for ex in _stratified_subsample(ds_seed1.train.items, 400, rng):
        l_2c.observe(ex.observation())
    acc_2c = accuracy([l_2c.predict(e.scene, e.tokens) for e in eval_items], eval_items)
    assert 0.35 <= acc_2c <= 0.62


def test_part3_breaking_points_homonymy_and_beyond_depth(world_cfg, ds_seed0) -> None:
    """Verify homonymy and multi-clause/3-adj same-family sentences expose NEXUS breaking points."""
    assert ds_seed0.grammar is not None
    lex = ds_seed0.grammar.lexicon
    map_h3 = {
        lex["size:small"]: lex["color:red"],
        lex["size:medium"]: lex["color:blue"],
        lex["size:large"]: lex["color:green"],
    }
    ds_h3 = _apply_token_mapping_to_dataset(ds_seed0, map_h3)
    learner_h3 = NEXUSLearner(seed=0)
    for ex in ds_h3.train.items:
        learner_h3.observe(ex.observation())
    longer_acc = accuracy(
        [learner_h3.predict(e.scene, e.tokens) for e in ds_h3.tests["test_longer"].items],
        ds_h3.tests["test_longer"].items,
    )
    # Under 3 homonym pairs, stacked size+color in test_longer collides on the same surface word
    assert longer_acc < 0.90

    # Verify beyond-depth suites on clean learner
    clean_learner = NEXUSLearner(seed=0)
    for ex in ds_seed0.train.items:
        clean_learner.observe(ex.observation())
    depth_res = _evaluate_beyond_depth_suites(clean_learner, ds_seed0, world_cfg, seed=0)
    assert depth_res["depth_2adj_diff_family_acc"] == 1.0
    assert depth_res["depth_3adj_same_family_parse_rate"] == 0.0
    assert depth_res["depth_2clause_chain_parse_rate"] == 0.0


def test_part4_baselines_c_d_e(ds_seed0) -> None:
    """Verify Baseline C fix, Baseline D (PositionAwareCS), and Baseline E (Neural)."""
    # Baseline C: legacy buggy vs fixed
    c_legacy = NoRevisionLearner(seed=0, legacy_buggy_fallback=True)
    c_fixed = NoRevisionLearner(seed=0, legacy_buggy_fallback=False)
    for ex in ds_seed0.train.items:
        obs = ex.observation()
        c_legacy.observe(obs)
        c_fixed.observe(obs)

    acc_legacy = accuracy(
        [c_legacy.predict(e.scene, e.tokens) for e in ds_seed0.train.items],
        ds_seed0.train.items,
    )
    acc_fixed = accuracy(
        [c_fixed.predict(e.scene, e.tokens) for e in ds_seed0.train.items],
        ds_seed0.train.items,
    )
    assert acc_legacy < 0.55
    assert acc_fixed > acc_legacy + 0.15

    # Baseline D: PositionAwareCrossSituationalLearner
    d_learner = PositionAwareCrossSituationalLearner(seed=0)
    for ex in ds_seed0.train.items:
        d_learner.observe(ex.observation())
    acc_d_nc = accuracy(
        [
            d_learner.predict(e.scene, e.tokens)
            for e in ds_seed0.tests["test_novel_combination"].items
        ],
        ds_seed0.tests["test_novel_combination"].items,
    )
    assert acc_d_nc > 0.70

    # Baseline E: FromScratchNeuralBaseline
    e_learner = FromScratchNeuralBaseline(seed=0, epochs=35)
    for ex in ds_seed0.train.items[:400]:
        e_learner.observe(ex.observation())
    acc_e_tr = accuracy(
        [e_learner.predict(e.scene, e.tokens) for e in ds_seed0.train.items[:400]],
        ds_seed0.train.items[:400],
    )
    assert acc_e_tr > 0.65
    assert e_learner.model_size() > 1000


def test_part5_log2_bit_cost_prior_matches_linear_mdl(ds_seed0) -> None:
    """Verify explicit log2 bit-cost MDL prior achieves 1.0 accuracy on held-out combinations."""
    learner_bits = NEXUSLearner(seed=0, prior_mode="log2_bits")
    for ex in ds_seed0.train.items:
        learner_bits.observe(ex.observation())
    eval_items = ds_seed0.tests["test_novel_combination"].items
    acc_bits = accuracy(
        [learner_bits.predict(e.scene, e.tokens) for e in eval_items], eval_items
    )
    assert acc_bits == 1.0
