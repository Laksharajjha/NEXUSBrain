"""Phase 3 Restructuring Evaluation Runner (Seeds 2000-2049).

Executes the frozen pre-registered evaluation:
- Learners: NEXUS-Restructure, NEXUS-NoRestructure, NEXUS-ExceptionsOnly,
  NEXUS-SplitAlways, Baseline D, Oracle, and 3 Ablations (NoSplit, NoMerge, NoException).
- Worlds: W1 (Split-required), W2 (Exception-sufficient), W3 (Merge-required),
  W4 (Ambiguous-evidence), W5 (Exceptions-cannot-match).
- Metrics: Test Accuracy, Dev Retention, Model Size, Operator Counts, Confidence, Brier Score.
- Paired bootstrap 95% CIs (1,000 resamples).
- Error collection and forensic categorization via explain().
- Delivers raw data to nexus/experiments/results/phase3_eval.json.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from nexus.baselines.oracle import RestructureOracle
from nexus.baselines.position_aware_cs import PositionAwareCrossSituationalLearner
from nexus.learner.restructure_learner import (
    NEXUSAblationLearner,
    NEXUSExceptionsOnlyLearner,
    NEXUSNoRestructureLearner,
    NEXUSRestructureLearner,
    NEXUSSplitAlwaysLearner,
)
from nexus.world.restructure import build_restructure_dataset, load_world_config


WORLDS = ["w1_split", "w2_exception", "w3_merge", "w4_ambiguous", "w5_exceptions_cannot_match"]
SEEDS = list(range(2000, 2050))  # 50 fresh seeds
N_BOOTSTRAP = 1000

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "phase3_eval.json"


def compute_paired_bootstrap(
    diffs: list[float], n_resamples: int = N_BOOTSTRAP, seed: int = 42
) -> tuple[float, float, float]:
    """Compute mean and 95% percentile bootstrap confidence interval for paired differences."""
    rng = np.random.RandomState(seed)
    n = len(diffs)
    if n == 0:
        return 0.0, 0.0, 0.0
    arr = np.array(diffs, dtype=np.float64)
    boot_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        boot_means[i] = np.mean(arr[indices])
    mean_val = float(np.mean(arr))
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return mean_val, ci_low, ci_high


def run_evaluation() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = load_world_config()

    learners_config = [
        ("nexus_restructure", lambda s: NEXUSRestructureLearner(seed=s, evidence_channel="legacy")),
        ("nexus_no_restructure", lambda s: NEXUSNoRestructureLearner(seed=s, evidence_channel="legacy")),
        ("nexus_exceptions_only", lambda s: NEXUSExceptionsOnlyLearner(seed=s, evidence_channel="legacy")),
        ("nexus_split_always", lambda s: NEXUSSplitAlwaysLearner(seed=s, evidence_channel="legacy")),
        ("baseline_d", lambda s: PositionAwareCrossSituationalLearner(seed=s)),
        ("oracle", None),  # RestructureOracle instantiated per dataset
        ("nexus_minus_split", lambda s: NEXUSAblationLearner("split", seed=s, evidence_channel="legacy")),
        ("nexus_minus_merge", lambda s: NEXUSAblationLearner("merge", seed=s, evidence_channel="legacy")),
        ("nexus_minus_exception", lambda s: NEXUSAblationLearner("exception", seed=s, evidence_channel="legacy")),
    ]

    all_results: dict[str, Any] = {
        "metadata": {
            "worlds": WORLDS,
            "seeds": SEEDS,
            "n_seeds": len(SEEDS),
            "n_bootstrap": N_BOOTSTRAP,
            "prereg_tag": "prereg-p3",
            "pre_run_tag": "pre-run-p3",
        },
        "per_world": {},
        "worst_errors": [],
    }

    all_test_errors: list[dict[str, Any]] = []

    print(f"[Phase 3] Starting pre-registered evaluation across {len(SEEDS)} seeds and {len(WORLDS)} worlds...")

    for world_id in WORLDS:
        print(f"\n--- Evaluating World: {world_id} ---")
        world_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        op_counts: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        retention_scores: list[float] = []

        for seed in SEEDS:
            ds = build_restructure_dataset(world_id, seed=seed)
            test_split = ds.tests.get("test_heldout") or ds.tests.get("test_ambiguous")
            assert test_split is not None
            test_items = test_split.items

            for learner_name, factory in learners_config:
                if learner_name == "oracle":
                    learner = RestructureOracle(
                        ds.grammar, config.ontology, world_id, meta=ds.meta, seed=seed
                    )
                else:
                    assert factory is not None
                    learner = factory(seed)

                # Train
                for ex in ds.train.items:
                    learner.observe(ex.observation())

                # Evaluate on test
                correct = 0
                confidences = []
                briers = []

                for ex in test_items:
                    pred = learner.predict(ex.scene, ex.tokens)
                    if pred.truth == ex.truth:
                        correct += 1
                    else:
                        if learner_name == "nexus_restructure" and len(all_test_errors) < 50:
                            # Capture explanation for forensic error categorization
                            expl = learner.explain(ex.scene, ex.tokens)
                            all_test_errors.append({
                                "world_id": world_id,
                                "seed": seed,
                                "example_id": ex.id,
                                "tokens": list(ex.tokens),
                                "ground_truth": ex.truth,
                                "pred_truth": pred.truth,
                                "pred_p": pred.p_true,
                                "confidence": pred.confidence,
                                "failure_reason": expl.failure_reason,
                                "rule_history": expl.rule_application_history if hasattr(expl, "rule_application_history") else [],
                            })

                    confidences.append(pred.confidence)
                    y = 1.0 if ex.truth else 0.0
                    briers.append((pred.p_true - y) ** 2)

                acc = correct / len(test_items)
                world_metrics[learner_name]["acc"].append(acc)
                world_metrics[learner_name]["model_size"].append(float(learner.model_size()))
                world_metrics[learner_name]["confidence"].append(float(np.mean(confidences)))
                world_metrics[learner_name]["brier"].append(float(np.mean(briers)))

                # Record operator counts if learner maintains revision_log
                if hasattr(learner, "revision_log"):
                    ops = [r.operator for r in learner.revision_log if r.operator != "initial"]
                    split_c = sum(1 for op in ops if op == "split")
                    merge_c = sum(1 for op in ops if op == "merge")
                    exc_c = sum(1 for op in ops if op == "retain_exception")
                    op_counts[learner_name]["split"].append(split_c)
                    op_counts[learner_name]["merge"].append(merge_c)
                    op_counts[learner_name]["retain_exception"].append(exc_c)

            # Measure untouched concept retention on dev for NEXUSRestructure
            if world_id == "w1_split":
                target_word = ds.meta["shared_word"]
                untouched_dev = [ex for ex in ds.dev.items if target_word not in ex.tokens]
                if untouched_dev:
                    nr_learner = NEXUSRestructureLearner(seed=seed, evidence_channel="legacy")
                    for ex in ds.train.items:
                        nr_learner.observe(ex.observation())
                    r_cor = sum(1 for ex in untouched_dev if nr_learner.predict(ex.scene, ex.tokens).truth == ex.truth)
                    retention_scores.append(r_cor / len(untouched_dev))

        # Summarize per-world results across 50 seeds
        world_summary: dict[str, Any] = {"metrics": {}, "operators": {}, "pairwise_differences": {}}
        for lname in world_metrics:
            world_summary["metrics"][lname] = {
                "acc_mean": float(np.mean(world_metrics[lname]["acc"])),
                "acc_std": float(np.std(world_metrics[lname]["acc"])),
                "model_size_mean": float(np.mean(world_metrics[lname]["model_size"])),
                "model_size_std": float(np.std(world_metrics[lname]["model_size"])),
                "confidence_mean": float(np.mean(world_metrics[lname]["confidence"])),
                "brier_mean": float(np.mean(world_metrics[lname]["brier"])),
            }
            if lname in op_counts:
                world_summary["operators"][lname] = {
                    "split_mean": float(np.mean(op_counts[lname]["split"])),
                    "merge_mean": float(np.mean(op_counts[lname]["merge"])),
                    "retain_exception_mean": float(np.mean(op_counts[lname]["retain_exception"])),
                }

        # Pairwise differences against nexus_restructure
        nr_accs = world_metrics["nexus_restructure"]["acc"]
        for other in ["nexus_no_restructure", "nexus_exceptions_only", "nexus_split_always", "baseline_d", "nexus_minus_split", "nexus_minus_merge", "nexus_minus_exception"]:
            diffs = [a - b for a, b in zip(nr_accs, world_metrics[other]["acc"])]
            mean_d, ci_lo, ci_hi = compute_paired_bootstrap(diffs)
            world_summary["pairwise_differences"][f"restructure_minus_{other}"] = {
                "diff_mean": mean_d,
                "ci_low": ci_lo,
                "ci_high": ci_hi,
            }

        # Model size difference against SplitAlways in W2
        if world_id == "w2_exception":
            size_diffs = [
                a - b for a, b in zip(world_metrics["nexus_restructure"]["model_size"], world_metrics["nexus_split_always"]["model_size"])
            ]
            s_mean, s_lo, s_hi = compute_paired_bootstrap(size_diffs)
            world_summary["w2_size_diff_vs_split_always"] = {
                "diff_mean": s_mean,
                "ci_low": s_lo,
                "ci_high": s_hi,
            }

        if retention_scores:
            world_summary["retention_dev_acc"] = {
                "mean": float(np.mean(retention_scores)),
                "std": float(np.std(retention_scores)),
            }

        all_results["per_world"][world_id] = world_summary
        print(f"  NEXUS-Restructure Acc: {world_summary['metrics']['nexus_restructure']['acc_mean']:.4f} +/- {world_summary['metrics']['nexus_restructure']['acc_std']:.4f}")
        print(f"  NEXUS-NoRestructure Acc: {world_summary['metrics']['nexus_no_restructure']['acc_mean']:.4f}")
        print(f"  NEXUS-ExceptionsOnly Acc: {world_summary['metrics']['nexus_exceptions_only']['acc_mean']:.4f}")

    all_results["worst_errors"] = all_test_errors[:15]

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[Phase 3] Evaluation complete. Saved raw data to {RESULTS_FILE}")
    return all_results


if __name__ == "__main__":
    run_evaluation()
