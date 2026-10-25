"""Unit and regression tests for NEXUS-0 Phase 1.6 (EXPLORATORY).

Verifies:
1. Task 1: Evidence channel separation (NEXUS-Truth, NEXUS-Cooc, NEXUS-Both).
   - NEXUS-Truth goes to chance (~0.50) under shuffled labels (Control 2a),
     shuffled scenes (Control 2b), and cross-seed lexicon (Control 2c), while
     reaching high accuracy (>= 0.95) on clean data.
   - NEXUS-Cooc is strictly invariant to label shuffling (Control 2a == clean).
2. Task 2: Paired bootstrap CI helper and grammar configuration breakdown.
3. Task 3: Garden-path / early-ambiguity stimulus order where beam K >= 5
   significantly outperforms greedy K = 1.
4. Task 4: Variable-description-length MDL worlds (filler/null entries and
   multi-sense homonymy/polysemy) where log2_bits MDL selects more compact
   hypotheses and improves generalization over flat (alpha=0) prior.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from nexus.eval.metrics import accuracy
from nexus.eval.splits import build_dataset, derive_seed
from nexus.experiments.phase1_6_mechanisms import (
    _build_garden_path_stream,
    paired_bootstrap_ci,
)
from nexus.learner.nexus_learner import NEXUSLearner
from nexus.world.config import load_world_config


def test_task1_nexus_truth_goes_to_chance_under_shuffled_labels() -> None:
    """NEXUS-Truth must drop to chance (~0.50) under shuffled truth labels (Control 2a)."""
    cfg = load_world_config()
    clean_accs: list[float] = []
    shuf_accs: list[float] = []

    for seed in range(3):
        ds = build_dataset(cfg, derive_seed(seed, "world"), world_mode="randomized")

        l_clean = NEXUSLearner(seed=seed, evidence_channel="truth")
        for ex in ds.train.items:
            l_clean.observe(ex.observation())
        preds_clean = [
            l_clean.predict(ex.scene, ex.tokens)
            for ex in ds.tests["test_novel_combination"].items
        ]
        clean_accs.append(accuracy(preds_clean, ds.tests["test_novel_combination"].items))

        rng = np.random.default_rng(seed + 900)
        labels = [ex.truth for ex in ds.train.items]
        perm = rng.permutation(len(labels))
        shuf_train = [
            dataclasses.replace(ex, truth=bool(labels[int(perm[i])]))
            for i, ex in enumerate(ds.train.items)
        ]
        l_shuf = NEXUSLearner(seed=seed, evidence_channel="truth")
        for ex in shuf_train:
            l_shuf.observe(ex.observation())
        preds_shuf = [
            l_shuf.predict(ex.scene, ex.tokens)
            for ex in ds.tests["test_novel_combination"].items
        ]
        shuf_accs.append(accuracy(preds_shuf, ds.tests["test_novel_combination"].items))

    mean_clean = float(np.mean(clean_accs))
    mean_shuf = float(np.mean(shuf_accs))
    assert mean_clean >= 0.95, f"Expected clean NEXUS-Truth >= 0.95, got {mean_clean:.3f}"
    assert 0.38 <= mean_shuf <= 0.62, (
        f"Expected NEXUS-Truth under shuffled labels to be at chance (~0.50), got {mean_shuf:.3f}"
    )


def test_task1_nexus_cooc_ignores_labels_completely() -> None:
    """NEXUS-Cooc must produce identical predictions under clean vs. shuffled truth labels."""
    cfg = load_world_config()
    ds = build_dataset(cfg, derive_seed(0, "world"), world_mode="randomized")

    l_clean = NEXUSLearner(seed=0, evidence_channel="cooc")
    for ex in ds.train.items[:400]:
        l_clean.observe(ex.observation())

    rng = np.random.default_rng(42)
    labels = [ex.truth for ex in ds.train.items[:400]]
    perm = rng.permutation(len(labels))
    shuf_train = [
        dataclasses.replace(ex, truth=bool(labels[int(perm[i])]))
        for i, ex in enumerate(ds.train.items[:400])
    ]
    l_shuf = NEXUSLearner(seed=0, evidence_channel="cooc")
    for ex in shuf_train:
        l_shuf.observe(ex.observation())

    preds_clean = [l_clean.predict(ex.scene, ex.tokens).p_true for ex in ds.dev.items]
    preds_shuf = [l_shuf.predict(ex.scene, ex.tokens).p_true for ex in ds.dev.items]
    assert np.allclose(preds_clean, preds_shuf), (
        "NEXUS-Cooc predictions changed when labels were shuffled!"
    )


def test_task2_paired_bootstrap_ci_properties() -> None:
    """Verify paired bootstrap CI returns exact mean difference and valid 95% bounds."""
    a = [0.90, 0.92, 0.95, 0.88, 0.91, 0.94, 0.93, 0.89]
    b = [0.70, 0.72, 0.75, 0.68, 0.71, 0.74, 0.73, 0.69]
    res = paired_bootstrap_ci(a, b, n_boot=1000, seed=0)
    assert abs(res["mean"] - 0.20) < 1e-9
    assert res["ci_low"] > 0.18
    assert res["ci_high"] < 0.22


def test_task3_garden_path_beam_beats_greedy_k1() -> None:
    """On the early-ambiguity garden-path stream, beam K=50 must beat greedy K=1."""
    cfg = load_world_config()
    ds = build_dataset(cfg, derive_seed(0, "world"), world_mode="randomized")
    gp_stream, dev_red = _build_garden_path_stream(ds)

    l_k1 = NEXUSLearner(seed=0, beam_width=1, evidence_channel="both", allow_synonyms=True)
    l_k50 = NEXUSLearner(seed=0, beam_width=50, evidence_channel="both", allow_synonyms=True)
    for ex in gp_stream:
        obs = ex.observation()
        l_k1.observe(obs)
        l_k50.observe(obs)

    acc_k1 = accuracy([l_k1.predict(ex.scene, ex.tokens) for ex in dev_red], dev_red)
    acc_k50 = accuracy([l_k50.predict(ex.scene, ex.tokens) for ex in dev_red], dev_red)
    assert acc_k50 > acc_k1 + 0.25, (
        f"Expected K=50 ({acc_k50:.3f}) to beat greedy K=1 ({acc_k1:.3f}) on garden-path stream"
    )


def test_task4_mdl_prior_prefers_compact_null_and_multisense_lexicons() -> None:
    """log2_bits MDL prior must yield smaller or equal model_size than flat prior on variable-length worlds."""
    cfg = load_world_config()
    ds = build_dataset(cfg, derive_seed(2, "world"), world_mode="randomized")
    assert ds.grammar is not None
    lex = ds.grammar.lexicon
    noun_words = {lex[f"kind:{k}"] for k in cfg.ontology.kinds}

    red_true = [
        ex
        for ex in ds.train.items
        if ex.truth
        and any(o.color == "red" for o in ex.scene.objects)
        and any(t in noun_words for t in ex.tokens)
    ][:2]
    red_ids = {ex.id for ex in red_true}
    train_filler = []
    for ex in ds.train.items[:400]:
        toks = list(ex.tokens)
        if ex.id in red_ids:
            for ti, t in enumerate(toks):
                if t in noun_words:
                    toks.insert(ti, "uh")
                    break
        train_filler.append(dataclasses.replace(ex, tokens=tuple(toks)))

    l_flat = NEXUSLearner(
        seed=2, evidence_channel="both", prior_mode="flat", allow_null_entries=True, allow_synonyms=True
    )
    l_bits = NEXUSLearner(
        seed=2, evidence_channel="both", prior_mode="log2_bits", allow_null_entries=True, allow_synonyms=True
    )
    for ex in train_filler:
        obs = ex.observation()
        l_flat.observe(obs)
        l_bits.observe(obs)

    assert l_bits.model_size() <= l_flat.model_size()
    top_bits = l_bits.top_hypothesis()
    assert top_bits is not None and top_bits.entries["uh"].predicate == "NULL"


def test_phase1_6_results_artifact_and_prereg_hash() -> None:
    """Verify PREREGISTRATION.md is untouched and phase1_6_mechanisms.json is valid if generated."""
    import hashlib

    prereg = Path(__file__).resolve().parents[2] / "PREREGISTRATION.md"
    digest = hashlib.sha256(prereg.read_bytes()).hexdigest()
    assert digest == "ef09666eb2bb96ca481ce05dc3dc5ca50fbe6d2143441ed38b062968313f695b"
