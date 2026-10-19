"""Evaluation metrics.

* accuracy: fraction of items whose predicted truth equals ground truth.
* ECE (expected calibration error): items binned by confidence (probability
  of the predicted label) into equal-width bins on [0.5, 1]; ECE is the
  item-weighted mean of |accuracy - mean confidence| per bin.
* Brier score: mean of (p_true - truth)^2.
* LF accuracy: exact match of predicted logical form (only when the learner
  returns parses; otherwise ``None``).
* examples-to-criterion: the first learning-curve checkpoint at which dev
  accuracy for a construction reaches the criterion (``None`` if never).
* retention: dev accuracy of a construction at the end of its curriculum
  stage vs. at the end of training.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from nexus.eval.learner_api import Prediction
from nexus.world.example import Example


@dataclass(frozen=True)
class CurvePoint:
    """Dev accuracy for one construction after ``n_seen`` training examples."""

    n_seen: int
    n_seen_construction: int
    accuracy: float


def accuracy(preds: Sequence[Prediction], items: Sequence[Example]) -> float:
    """Truth-judgment accuracy."""
    _check_lengths(preds, items)
    return float(np.mean([p.truth == e.truth for p, e in zip(preds, items)]))


def brier(preds: Sequence[Prediction], items: Sequence[Example]) -> float:
    """Mean squared error of ``p_true``."""
    _check_lengths(preds, items)
    return float(np.mean([(p.p_true - float(e.truth)) ** 2 for p, e in zip(preds, items)]))


def expected_calibration_error(preds: Sequence[Prediction], items: Sequence[Example], n_bins: int) -> float:
    """ECE over equal-width confidence bins on [0.5, 1]."""
    _check_lengths(preds, items)
    conf = np.array([p.confidence for p in preds])
    correct = np.array([p.truth == e.truth for p, e in zip(preds, items)], dtype=float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def lf_accuracy(preds: Sequence[Prediction], items: Sequence[Example]) -> float | None:
    """Exact-match LF accuracy, or ``None`` if the learner returned no parses."""
    _check_lengths(preds, items)
    if all(p.lf is None for p in preds):
        return None
    return float(np.mean([p.lf == e.lf for p, e in zip(preds, items)]))


def split_metrics(preds: Sequence[Prediction], items: Sequence[Example], n_bins: int) -> dict[str, Any]:
    """All per-split metrics, including a per-construction accuracy breakdown."""
    by_c: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(items):
        by_c[e.construction].append(i)
    return {
        "n": len(items),
        "accuracy": accuracy(preds, items),
        "ece": expected_calibration_error(preds, items, n_bins),
        "brier": brier(preds, items),
        "lf_accuracy": lf_accuracy(preds, items),
        "mean_confidence": float(np.mean([p.confidence for p in preds])),
        "frac_true": float(np.mean([e.truth for e in items])),
        "accuracy_by_construction": {
            c: accuracy([preds[i] for i in idx], [items[i] for i in idx]) for c, idx in sorted(by_c.items())
        },
    }


def examples_to_criterion(curve: Sequence[CurvePoint], criterion: float) -> dict[str, int] | None:
    """First checkpoint reaching ``criterion``; ``None`` if never reached."""
    for point in curve:
        if point.accuracy >= criterion:
            return {"n_seen": point.n_seen, "n_seen_construction": point.n_seen_construction}
    return None


def retention(curve: Sequence[CurvePoint], stage_end: int) -> dict[str, float]:
    """Accuracy at the end of the construction's stage vs. at the end of training."""
    at_stage = [p for p in curve if p.n_seen == stage_end]
    if not at_stage:
        raise ValueError(f"no checkpoint at stage end {stage_end}")
    a, b = at_stage[0].accuracy, curve[-1].accuracy
    return {"at_stage_end": a, "final": b, "delta": b - a}


def summarize(values: Iterable[float | None]) -> dict[str, Any]:
    """Mean and sample std over seeds, ignoring ``None`` (counted separately)."""
    vals = list(values)
    valid = [float(v) for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not valid:
        return {"mean": None, "std": None, "n": 0, "n_missing": len(vals)}
    std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
    return {"mean": float(np.mean(valid)), "std": std, "n": len(valid), "n_missing": len(vals) - len(valid)}


def _check_lengths(preds: Sequence[Any], items: Sequence[Any]) -> None:
    if len(preds) != len(items):
        raise ValueError(f"{len(preds)} predictions for {len(items)} items")
    if not items:
        raise ValueError("empty split")
