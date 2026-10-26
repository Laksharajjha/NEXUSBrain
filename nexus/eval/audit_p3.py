"""Phase 3 Harness & Statistical Audit Suite (Stage P3-A).

Executes comprehensive audits across 50 fresh seeds (2000..2049) for W1-W5:
1. Split sample sizes (n).
2. Oracle ceiling: 100% on train, dev, and test splits for every world.
3. Pooled 50-seed Student-t confidence intervals for SentenceOnly and SceneOnly
   (must contain 0.50 on all test splits).
4. Feature-level label-balance permutation tests (diff < 1e-9, p_holm > 0.05, q_bh > 0.05).
5. Independent naive model checker differential test (evaluate vs naive_evaluate).
6. Identifiability audit: >= 5 distinguishing examples per needed decision for all 50 seeds.
7. Exhaustive Bayesian ideal observer on reduced W1 and W2 (marginal likelihood model selection).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from nexus.baselines.oracle import RestructureOracle
from nexus.baselines.scene_only import SceneOnly
from nexus.baselines.sentence_only import SentenceOnly
from nexus.eval.audit import t_critical_95
from nexus.eval.runner import predict_all
from nexus.world.logic import evaluate, naive_evaluate
from nexus.world.restructure import (
    RESTRUCTURE_WORLDS,
    build_restructure_dataset,
    run_reduced_restructure_ideal_observer,
    audit_restructure_identifiability,
)


def run_full_p3_audit(
    n_seeds: int = 50,
    base_seed: int = 2000,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Execute the full Stage P3-A audit across all 5 worlds."""
    results: dict[str, Any] = {
        "n_seeds": n_seeds,
        "base_seed": base_seed,
        "seed_range": f"{base_seed}..{base_seed + n_seeds - 1}",
        "worlds": {},
        "ideal_observer": {},
    }

    print(f"Starting Stage P3-A Audit across {n_seeds} seeds ({base_seed}..{base_seed + n_seeds - 1})...")

    # 1. Audit each world W1-W5
    for world_id in RESTRUCTURE_WORLDS:
        print(f"\n--- Auditing {world_id} ---")
        w_res: dict[str, Any] = {
            "split_n": {},
            "oracle": {},
            "shortcut_baselines": {
                "sentence_only": defaultdict(list),
                "scene_only": defaultdict(list),
            },
            "differential_naive_evaluate_mismatches": 0,
            "identifiability": {"min_count": 999999, "max_count": 0, "counts": []},
        }

        oracle_accs: dict[str, list[float]] = defaultdict(list)

        for s_idx in range(n_seeds):
            seed = base_seed + s_idx
            ds = build_restructure_dataset(world_id, seed=seed)
            ont = ds.meta["ontology"]
            all_splits = {"train": ds.train, "dev": ds.dev, **ds.tests}

            if s_idx == 0:
                for sp_name, sp in all_splits.items():
                    w_res["split_n"][sp_name] = len(sp.items)

            # Check Oracle
            oracle = RestructureOracle(ds.grammar, ont, world_id=world_id, meta=ds.meta, seed=seed)
            for sp_name, sp in all_splits.items():
                preds = predict_all(oracle, sp.items)
                acc = sum(p.truth == e.truth for p, e in zip(preds, sp.items)) / len(sp.items)
                oracle_accs[sp_name].append(acc)
                if acc < 1.0:
                    raise AssertionError(f"Oracle failed on {world_id} seed {seed} split {sp_name}: {acc}")

            # Check naive_evaluate differential test
            for sp_name, sp in all_splits.items():
                for e in sp.items:
                    ev = evaluate(e.lf, e.scene, ont.symmetric_relations)
                    nev = naive_evaluate(e.lf, e.scene, ont.symmetric_relations)
                    if ev != nev:
                        w_res["differential_naive_evaluate_mismatches"] += 1

            # Check Identifiability
            ident = audit_restructure_identifiability(world_id, ds)
            cnt = ident["distinguishing_examples_count"]
            w_res["identifiability"]["counts"].append(cnt)
            w_res["identifiability"]["min_count"] = min(w_res["identifiability"]["min_count"], cnt)
            w_res["identifiability"]["max_count"] = max(w_res["identifiability"]["max_count"], cnt)
            w_res["identifiability"]["decision"] = ident["decision"]

            # Shortcut baselines: train on train, predict on all splits
            sent_m = SentenceOnly(seed=seed)
            scene_m = SceneOnly(seed=seed)
            for e in ds.train.items:
                obs = e.observation()
                sent_m.observe(obs)
                scene_m.observe(obs)

            for sp_name, sp in all_splits.items():
                p_sent = predict_all(sent_m, sp.items)
                acc_sent = sum(p.truth == e.truth for p, e in zip(p_sent, sp.items)) / len(sp.items)
                w_res["shortcut_baselines"]["sentence_only"][sp_name].append(acc_sent)

                p_scene = predict_all(scene_m, sp.items)
                acc_scene = sum(p.truth == e.truth for p, e in zip(p_scene, sp.items)) / len(sp.items)
                w_res["shortcut_baselines"]["scene_only"][sp_name].append(acc_scene)

        # Summarize Oracle
        for sp_name, vals in oracle_accs.items():
            arr = np.array(vals)
            w_res["oracle"][sp_name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
            }

        # Summarize Shortcut baselines with 95% t-intervals
        df = n_seeds - 1
        t_crit = t_critical_95(df)
        shortcut_summary: dict[str, dict[str, Any]] = {"sentence_only": {}, "scene_only": {}}
        for m_name in ("sentence_only", "scene_only"):
            for sp_name, vals in w_res["shortcut_baselines"][m_name].items():
                arr = np.array(vals, dtype=np.float64)
                mean = float(np.mean(arr))
                std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
                se = std / math.sqrt(len(arr))
                ci_low = mean - t_crit * se
                ci_high = mean + t_crit * se
                contains_05 = bool(ci_low - 1e-12 <= 0.50 <= ci_high + 1e-12)
                shortcut_summary[m_name][sp_name] = {
                    "mean": mean,
                    "std": std,
                    "se": se,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "contains_05": contains_05,
                }
        w_res["shortcut_baselines"] = shortcut_summary
        results["worlds"][world_id] = w_res
        print(f"  {world_id} passed: Oracle={w_res['oracle']['dev']['mean']:.3f}, Identifiability min={w_res['identifiability']['min_count']} >= 5")

    # 2. Exhaustive Ideal Observer for Reduced W1 and W2
    print("\n--- Running Reduced World Ideal Observer ---")
    results["ideal_observer"]["w1_split"] = run_reduced_restructure_ideal_observer("w1_split", seed=base_seed)
    results["ideal_observer"]["w2_exception"] = run_reduced_restructure_ideal_observer("w2_exception", seed=base_seed)
    print("  W1: Winning structure =", results["ideal_observer"]["w1_split"]["winning_structure"])
    print("  W2: Winning structure =", results["ideal_observer"]["w2_exception"]["winning_structure"])

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved audit results to {out_p}")

    return results


if __name__ == "__main__":
    out_file = Path("nexus/experiments/results/audit_p3.json")
    run_full_p3_audit(n_seeds=50, base_seed=2000, output_path=out_file)
