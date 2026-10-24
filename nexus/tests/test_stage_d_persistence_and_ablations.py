"""Stage D Tests: SQLite Persistence Round-Trip (D9), Hypothesis-Space Ablations (D4), and Brier Decomposition (D7)."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from nexus.eval.splits import build_dataset
from nexus.experiments.stage_d_exploratory import brier_decomposition_and_diagram
from nexus.learner import NEXUSLearner
from nexus.storage import LearnerRepository, SQLiteLearnerRepository
from nexus.world.config import WorldConfig


def test_d9_sqlite_persistence_roundtrip_fresh_process(small_config: WorldConfig, tmp_path: Path):
    """Train NEXUSLearner, save to SQLite, reload in a fresh Python subprocess, verify identical predictions and provenance."""
    ds = build_dataset(small_config, seed=4, world_mode="randomized")
    learner = NEXUSLearner(seed=4, beam_width=50)
    for ex in ds.train.items:
        learner.observe(ex.observation())

    contra = ds.tests["test_contradiction"]
    corr_target_id = ""
    for ev in contra.support:
        if ev.kind == "observe":
            assert ev.observation is not None
            learner.observe(ev.observation)
        else:
            assert ev.correction is not None
            learner.correct(ev.correction)
            corr_target_id = ev.correction.target_example_id

    db_file = tmp_path / "test_nexus.sqlite3"
    repo = SQLiteLearnerRepository(db_file)
    assert isinstance(repo, LearnerRepository)
    repo.save_learner("run_seed4", learner)
    assert repo.list_runs() == ["run_seed4"]

    eval_items = ds.tests["test_novel_combination"].items[:15]
    orig_preds = [learner.predict(e.scene, e.tokens).p_true for e in eval_items]
    orig_exp = learner.explain(eval_items[0].tokens, eval_items[0].scene).to_json()
    orig_chg = learner.what_changed(corr_target_id).to_json()

    # Verify in a completely separate Python subprocess
    items_json = json.dumps([e.to_json() for e in eval_items])
    cmd = [
        sys.executable,
        "-c",
        (
            "import json, sys; "
            "from nexus.storage import SQLiteLearnerRepository; "
            "from nexus.world.scene import Scene; "
            "repo = SQLiteLearnerRepository(sys.argv[1]); "
            "loaded = repo.load_learner('run_seed4'); "
            "items = json.loads(sys.argv[2]); "
            "preds = [loaded.predict(Scene.from_json(it['scene']), tuple(it['tokens'])).p_true for it in items]; "
            "exp0 = loaded.explain(tuple(items[0]['tokens']), Scene.from_json(items[0]['scene'])).to_json(); "
            "chg = loaded.what_changed(sys.argv[3]).to_json(); "
            "print(json.dumps({'preds': preds, 'exp0': exp0, 'chg': chg}))"
        ),
        str(db_file),
        items_json,
        corr_target_id,
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    sub_data = json.loads(proc.stdout.strip())

    for p_orig, p_sub in zip(orig_preds, sub_data["preds"]):
        assert math.isclose(p_orig, p_sub, abs_tol=1e-12)
    assert orig_exp == sub_data["exp0"]
    assert orig_chg == sub_data["chg"]


def test_d4_slash_depth_1_ablation_fails_transitive(small_config: WorldConfig):
    """Removing slash depth 2 (max_slash_depth=1) succeeds on NP constructions but fails on transitive clauses."""
    ds = build_dataset(small_config, seed=0, world_mode="randomized")
    learner_d1 = NEXUSLearner(seed=0, max_slash_depth=1)
    for ex in ds.train.items:
        learner_d1.observe(ex.observation())

    np_items = [e for e in ds.dev.items if e.construction in ("np_assert", "np_ask")]
    tr_items = [e for e in ds.dev.items if e.construction in ("trans_assert", "trans_ask")]

    np_acc = sum(learner_d1.predict(e.scene, e.tokens).truth == e.truth for e in np_items) / len(np_items)
    tr_preds = [learner_d1.predict(e.scene, e.tokens) for e in tr_items]
    tr_acc = sum(p.truth == e.truth for p, e in zip(tr_preds, tr_items)) / len(tr_items)

    assert np_acc >= 0.90
    # Transitive clauses cannot parse without depth-2 verb categories ((S\NP)\NP, (S\NP)/NP, (S/NP)/NP)
    assert all(p.lf is None and p.p_true == 0.5 for p in tr_preds)
    assert math.isclose(tr_acc, 0.5, abs_tol=1e-12)


def test_d7_brier_decomposition_identity(small_config: WorldConfig):
    """Murphy Brier decomposition satisfies Brier ~= Reliability - Resolution + Uncertainty within bin discretization."""
    ds = build_dataset(small_config, seed=0, world_mode="randomized")
    learner = NEXUSLearner(seed=0)
    for ex in ds.train.items:
        learner.observe(ex.observation())

    items = ds.tests["test_novel_combination"].items
    preds = [learner.predict(e.scene, e.tokens) for e in items]
    truths = [e.truth for e in items]
    decomp = brier_decomposition_and_diagram(preds, truths, n_bins=10)

    assert math.isclose(decomp["brier_score"], decomp["reconstructed_brier"], abs_tol=1e-4)
    assert decomp["uncertainty"] == 0.25
    assert decomp["resolution"] > 0.20
