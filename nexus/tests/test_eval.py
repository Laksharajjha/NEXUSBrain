"""Tests for evaluation metrics, runner, and Baseline A (memorizer)."""

from __future__ import annotations

import pytest

from nexus.baselines.memorizer import Memorizer, UnknownExampleError
from nexus.eval.learner_api import Prediction
from nexus.eval.metrics import (
    accuracy,
    brier,
    expected_calibration_error,
    lf_accuracy,
    split_metrics,
    summarize,
)
from nexus.eval.runner import ExperimentSpec, predict_all, run_experiment
from nexus.world.example import Correction, Example
from nexus.world.logic import Atom, make_lf
from nexus.world.scene import build_scene


def _dummy_example(truth: bool, ex_id: str = "ex-0") -> Example:
    scene = build_scene([{"kind": "cube", "color": "red", "size": "small"}], [])
    lf = make_lf("assert", ("x0",), [Atom("kind:cube", ("x0",))])
    return Example(id=ex_id, scene=scene, tokens=("dax",), lf=lf, truth=truth, construction="np_assert")


def test_metrics_computations():
    exs = [_dummy_example(True, "1"), _dummy_example(False, "2"), _dummy_example(True, "3"), _dummy_example(False, "4")]
    preds = [Prediction(0.9), Prediction(0.2), Prediction(0.4), Prediction(0.8)]

    # preds truths: [True (0.9), False (0.2), False (0.4), True (0.8)]
    # actual truths: [True, False, True, False]
    # correct: [True, True, False, False] -> 2 / 4 = 0.5
    assert accuracy(preds, exs) == 0.5
    # brier: mean((0.9-1)^2 + (0.2-0)^2 + (0.4-1)^2 + (0.8-0)^2)
    # = (0.01 + 0.04 + 0.36 + 0.64) / 4 = 1.05 / 4 = 0.2625
    assert abs(brier(preds, exs) - 0.2625) < 1e-6

    ece = expected_calibration_error(preds, exs, n_bins=5)
    assert 0.0 <= ece <= 1.0


def test_lf_accuracy():
    exs = [_dummy_example(True, "1"), _dummy_example(False, "2")]
    lf1 = exs[0].lf
    lf_wrong = make_lf("assert", ("x0",), [Atom("kind:ball", ("x0",))])

    # No LFs returned
    preds_no_lf = [Prediction(1.0), Prediction(0.0)]
    assert lf_accuracy(preds_no_lf, exs) is None

    # Partial match
    preds_with_lf = [Prediction(1.0, lf=lf1), Prediction(0.0, lf=lf_wrong)]
    assert lf_accuracy(preds_with_lf, exs) == 0.5


def test_memorizer_learn_and_predict():
    mem = Memorizer()
    ex1 = _dummy_example(True, "1")
    ex2 = _dummy_example(False, "2")

    # Before seeing anything: Laplace smoothed prior on 0 observations -> (0+1)/(0+2) = 0.5
    pred0 = mem.predict(ex1.scene, ex1.tokens)
    assert pred0.p_true == 0.5

    # Observe ex1
    mem.observe(ex1.observation())
    pred1 = mem.predict(ex1.scene, ex1.tokens)
    # Exact match: 1 true out of 1 -> (1+1)/(1+2) = 2/3
    assert abs(pred1.p_true - 2 / 3) < 1e-6
    assert pred1.truth is True

    # Observe duplicate ID raises ValueError
    with pytest.raises(ValueError, match="duplicate"):
        mem.observe(ex1.observation())

    # Correction test
    cor = Correction("1", corrected_label=False, source="teacher", reason="test")
    mem.correct(cor)
    pred_after_cor = mem.predict(ex1.scene, ex1.tokens)
    # Now exact count is 0 true out of 1 -> (0+1)/(1+2) = 1/3
    assert abs(pred_after_cor.p_true - 1 / 3) < 1e-6
    assert pred_after_cor.truth is False

    # Unknown correction raises UnknownExampleError
    with pytest.raises(UnknownExampleError):
        mem.correct(Correction("999", False, "teacher", "bogus"))


def test_runner_on_memorizer(small_config):
    """End-to-end runner test with memorizer on 2 seeds."""
    spec = ExperimentSpec("test_exp", "memorizer", (0, 1))
    result = run_experiment(spec, small_config, lambda s: Memorizer(seed=s))

    assert result["experiment"] == "test_exp"
    assert result["n_seeds"] == 2
    agg = result["aggregate"]
    assert "splits.train.accuracy" in agg
    # Train accuracy for memorizer must be 1.0
    assert agg["splits.train.accuracy"]["mean"] == 1.0
    # Novel splits should be ~0.5
    assert abs(agg["splits.test_novel_combination.accuracy"]["mean"] - 0.5) < 0.05
