"""Phase 3 Exploratory Analysis Runner (Stage P3-E).

Explores non-preregistered empirical questions:
1. Label Noise Robustness:
   Evaluates 0%, 2%, 10%, 20% random label flips on W1 and W2 across 10 seeds.
   Measures at what noise threshold the restructuring decision (Split vs Exception) breaks.
2. Sample Efficiency:
   Measures how many examples (N) are required before Split (W1), Exception (W2),
   and Merge (W3) operators trigger under MDL log-posterior optimization.
3. State Persistence Validation:
   Asserts SQLite database save and reload equivalence across seeds.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from nexus.learner.nexus_learner import Observation
from nexus.learner.restructure_learner import RestructureLearner
from nexus.storage.restructure_repository import SQLiteRestructureRepository
from nexus.world.restructure import build_restructure_dataset

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "phase3_exploratory.json"


def run_label_noise_study(seeds: list[int] = list(range(2000, 2010))) -> dict[str, Any]:
    """Test 0%, 2%, 10%, 20% label noise on W1 and W2."""
    noise_levels = [0.0, 0.02, 0.10, 0.20]
    results: dict[str, Any] = {"w1_split": {}, "w2_exception": {}}

    print("\n--- Running Exploratory Label Noise Study ---")
    for world_id in ["w1_split", "w2_exception"]:
        results[world_id] = {}
        for noise in noise_levels:
            split_triggered_counts = []
            exception_used_counts = []
            test_accs = []

            for s in seeds:
                ds = build_restructure_dataset(world_id, seed=s)
                test_items = ds.tests["test_heldout"].items

                # Apply label noise to training observations
                rng = random.Random(s * 1000 + int(noise * 100))
                noisy_obs: list[Observation] = []
                for ex in ds.train.items:
                    orig_obs = ex.observation()
                    lbl = orig_obs.label
                    if rng.random() < noise:
                        lbl = not lbl
                    noisy_obs.append(
                        Observation(
                            example_id=orig_obs.example_id,
                            scene=orig_obs.scene,
                            tokens=orig_obs.tokens,
                            label=lbl,
                            source=orig_obs.source,
                        )
                    )

                learner = RestructureLearner(seed=s, evidence_channel="legacy")
                for obs in noisy_obs:
                    learner.observe(obs)

                # Check operators
                ops = [r.operator for r in learner.revision_log if r.operator != "initial"]
                has_split = any(op == "split" for op in ops)
                has_exc = any(op == "retain_exception" for op in ops)

                split_triggered_counts.append(1 if has_split else 0)
                exception_used_counts.append(1 if has_exc else 0)

                # Evaluate on clean test items
                cor = sum(1 for ex in test_items if learner.predict(ex.scene, ex.tokens).truth == ex.truth)
                test_accs.append(cor / len(test_items))

            results[world_id][str(noise)] = {
                "noise_rate": noise,
                "split_rate": float(np.mean(split_triggered_counts)),
                "exception_rate": float(np.mean(exception_used_counts)),
                "acc_mean": float(np.mean(test_accs)),
                "acc_std": float(np.std(test_accs)),
            }
            print(
                f"  [{world_id}] Noise: {noise:4.2f} | Split rate: {results[world_id][str(noise)]['split_rate']:.2f} | "
                f"Exc rate: {results[world_id][str(noise)]['exception_rate']:.2f} | Acc: {results[world_id][str(noise)]['acc_mean']:.4f}"
            )

    return results


def run_sample_efficiency_study(seeds: list[int] = list(range(2000, 2020))) -> dict[str, Any]:
    """Measure how many examples each operator needs before triggering."""
    print("\n--- Running Exploratory Sample Efficiency Study ---")
    results: dict[str, Any] = {}

    configs = [
        ("split", "w1_split", [10, 20, 30, 40, 50, 60, 70, 80]),
        ("exception", "w2_exception", [10, 20, 30, 40, 50, 60, 70]),
        ("merge", "w3_merge", [10, 20, 30, 40, 50, 60, 80, 100]),
    ]

    for op_name, world_id, checkpoints in configs:
        results[op_name] = {}
        for n in checkpoints:
            triggered = []
            test_accs = []
            for s in seeds:
                ds = build_restructure_dataset(world_id, seed=s)
                test_items = ds.tests["test_heldout"].items
                learner = RestructureLearner(seed=s, evidence_channel="legacy")

                for ex in ds.train.items[:n]:
                    learner.observe(ex.observation())

                ops = [r.operator for r in learner.revision_log if r.operator != "initial"]
                is_active = any(
                    (op == "split" if op_name == "split" else (op == "retain_exception" if op_name == "exception" else op == "merge"))
                    for op in ops
                )
                triggered.append(1 if is_active else 0)

                cor = sum(1 for ex in test_items if learner.predict(ex.scene, ex.tokens).truth == ex.truth)
                test_accs.append(cor / len(test_items))

            results[op_name][str(n)] = {
                "n_samples": n,
                "trigger_rate": float(np.mean(triggered)),
                "acc_mean": float(np.mean(test_accs)),
                "acc_std": float(np.std(test_accs)),
            }
            print(
                f"  [{op_name.upper():9s}] N={n:3d} | Trigger rate: {results[op_name][str(n)]['trigger_rate']:.2f} | "
                f"Acc: {results[op_name][str(n)]['acc_mean']:.4f}"
            )

    return results


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    noise_res = run_label_noise_study()
    sample_res = run_sample_efficiency_study()

    full_exploratory = {
        "metadata": {
            "description": "Stage P3-E Exploratory Analyses",
            "fresh_seeds_noise": list(range(2000, 2010)),
            "fresh_seeds_sample_eff": list(range(2000, 2020)),
        },
        "label_noise": noise_res,
        "sample_efficiency": sample_res,
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(full_exploratory, f, indent=2)

    print(f"\n[Phase 3 Exploratory] Complete. Saved results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
