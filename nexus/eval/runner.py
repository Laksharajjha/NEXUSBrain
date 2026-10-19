"""Experiment runner: trains a learner per seed and evaluates every split.

Protocol for one seed ``s``:

1. Build the dataset from ``derive_seed(s, "world")``.
2. Create the learner from ``derive_seed(s, "learner")``.
3. Stream the train split through ``observe``; at every ``checkpoint_every``
   examples and at each curriculum-stage boundary, evaluate dev accuracy per
   construction (learning curves).
4. Evaluate train, dev and every test split. For splits with a support
   stream, a deep copy of the trained learner processes the stream and items
   are evaluated at each support checkpoint, so splits never contaminate
   each other.
5. Compute metrics; aggregate mean +/- sample std across seeds.
"""

from __future__ import annotations

import copy
import json
import platform
import time
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from nexus.eval.learner_api import Learner, Prediction
from nexus.eval.metrics import (
    CurvePoint,
    examples_to_criterion,
    retention,
    split_metrics,
    summarize,
)
from nexus.eval.splits import Dataset, Split, build_dataset, derive_seed
from nexus.world.config import WorldConfig
from nexus.world.example import Example

LearnerFactory = Callable[..., Any]
RESULTS_DIR = Path(__file__).resolve().parent.parent / "experiments" / "results"


@dataclass(frozen=True)
class ExperimentSpec:
    """What to run."""

    name: str
    learner_name: str
    seeds: tuple[int, ...]
    world_mode: str = "randomized"


def predict_all(learner: Learner, items: Sequence[Example]) -> list[Prediction]:
    """Predictions for every item (learner only sees scene and tokens)."""
    return [learner.predict(e.scene, e.tokens) for e in items]


def checkpoint_positions(n_train: int, every: int, boundaries: dict[str, int]) -> list[int]:
    """Sorted training positions at which to evaluate dev."""
    points = set(range(0, n_train + 1, every)) | set(boundaries.values()) | {n_train}
    return sorted(points)


def train_with_curves(
    learner: Learner, ds: Dataset, config: WorldConfig
) -> dict[str, list[CurvePoint]]:
    """Stream train into the learner, recording dev accuracy per construction."""
    train = ds.train.items
    positions = set(checkpoint_positions(len(train), config.evaluation.checkpoint_every, ds.stage_boundaries))
    constructions = sorted(config.grammar.constructions)
    dev_by_c = {c: [e for e in ds.dev.items if e.construction == c] for c in constructions}
    seen_by_c = {c: 0 for c in constructions}
    curves: dict[str, list[CurvePoint]] = {c: [] for c in constructions}

    def record(n_seen: int) -> None:
        for c in constructions:
            preds = predict_all(learner, dev_by_c[c])
            acc = float(np.mean([p.truth == e.truth for p, e in zip(preds, dev_by_c[c])]))
            curves[c].append(CurvePoint(n_seen, seen_by_c[c], acc))

    if 0 in positions:
        record(0)
    for i, ex in enumerate(train, start=1):
        learner.observe(ex.observation())
        seen_by_c[ex.construction] += 1
        if i in positions:
            record(i)
    return curves


def evaluate_support_split(learner: Learner, split: Split, n_bins: int) -> dict[str, Any]:
    """Process a support stream on a copy of the learner, evaluating at checkpoints."""
    local = copy.deepcopy(learner)
    out: dict[str, Any] = {"checkpoints": {}}
    processed = 0
    for label, position in split.support_checkpoints:
        while processed < position:
            event = split.support[processed]
            if event.kind == "observe":
                local.observe(event.observation)
            else:
                local.correct(event.correction)
            processed += 1
        out["checkpoints"][label] = split_metrics(predict_all(local, split.items), split.items, n_bins)
    if processed != len(split.support):
        raise ValueError(f"{split.name}: support checkpoints do not cover the stream")
    final_label = split.support_checkpoints[-1][0]
    out.update(out["checkpoints"][final_label])
    out["model_size_after_support"] = local.model_size()
    return out


def _create_learner(factory: LearnerFactory, seed: int, ds: Dataset) -> Learner:
    """Instantiate a learner, passing (seed, ds) if accepted, else (seed)."""
    try:
        sig = inspect.signature(factory)
        params = list(sig.parameters.values())
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
        if has_varargs or len(params) >= 2:
            return factory(seed, ds)
        return factory(seed)
    except Exception:
        try:
            return factory(seed, ds)
        except TypeError:
            return factory(seed)


def run_seed(seed: int, config: WorldConfig, factory: LearnerFactory, world_mode: str = "randomized") -> dict[str, Any]:
    """Run one seed end to end."""
    t0 = time.perf_counter()
    world_seed, learner_seed = derive_seed(seed, "world"), derive_seed(seed, "learner")
    ds = build_dataset(config, world_seed, world_mode=world_mode)
    learner = _create_learner(factory, learner_seed, ds)
    curves = train_with_curves(learner, ds, config)
    n_bins = config.evaluation.calibration_bins
    splits: dict[str, Any] = {
        "train": split_metrics(predict_all(learner, ds.train.items), ds.train.items, n_bins),
        "dev": split_metrics(predict_all(learner, ds.dev.items), ds.dev.items, n_bins),
    }
    for name, split in ds.tests.items():
        if split.support:
            splits[name] = evaluate_support_split(learner, split, n_bins)
        else:
            splits[name] = split_metrics(predict_all(learner, split.items), split.items, n_bins)
    return {
        "seed": seed,
        "world_seed": world_seed,
        "learner_seed": learner_seed,
        "plan": ds.plan.to_json(),
        "max_train_length": ds.max_train_length,
        "model_size": learner.model_size(),
        "splits": splits,
        "examples_to_criterion": {
            c: examples_to_criterion(curve, config.evaluation.criterion) for c, curve in curves.items()
        },
        "retention": {c: retention(curve, ds.stage_boundaries[c]) for c, curve in curves.items()},
        "curves": {c: [p.__dict__ for p in curve] for c, curve in curves.items()},
        "seconds": time.perf_counter() - t0,
    }


def _numeric_leaves(obj: Any, prefix: str = "") -> dict[str, float | None]:
    """Flatten nested dicts to ``a.b.c -> number`` (skipping curves and lists)."""
    out: dict[str, float | None] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_numeric_leaves(v, f"{prefix}{k}."))
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)) or obj is None:
        out[prefix[:-1]] = obj
    return out


def aggregate(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean +/- std across seeds for every numeric metric.

    For examples-to-criterion, seeds that never reach criterion are reported
    via ``n_missing`` rather than imputed.
    """
    skip = ("seed", "world_seed", "learner_seed", "seconds")
    flats = []
    for run in per_seed:
        body = {k: v for k, v in run.items() if k not in skip and k not in ("curves", "plan")}
        flats.append(_numeric_leaves(body))
    keys = sorted(set().union(*flats))
    agg = {}
    for k in keys:
        values = [f.get(k) for f in flats]
        if k.startswith("examples_to_criterion.") or any(v is not None for v in values):
            agg[k] = summarize(values)
    return agg


def run_experiment(
    spec: ExperimentSpec,
    config: WorldConfig,
    factory: LearnerFactory,
    world_mode: str | None = None,
) -> dict[str, Any]:
    """Run all seeds and return a full result record."""
    mode = world_mode if world_mode is not None else spec.world_mode
    per_seed = [run_seed(s, config, factory, world_mode=mode) for s in spec.seeds]
    return {
        "experiment": spec.name,
        "learner": spec.learner_name,
        "world_mode": mode,
        "seeds": list(spec.seeds),
        "n_seeds": len(spec.seeds),
        "config": config.to_json(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "created_at_unix": time.time(),
        "aggregate": aggregate(per_seed),
        "per_seed": per_seed,
    }


def save_results(result: dict[str, Any], path: Path) -> Path:
    """Write a result record as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    return path
