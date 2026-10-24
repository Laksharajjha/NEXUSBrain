"""Stage A (Phase 0.6) comprehensive audit runner.

Executes:
- A1: 50-seed pooled t-interval test for SentenceOnly and SceneOnly + split size reporting.
- A2: 20-seed feature-level label-balance audit with permutation tests + Holm & BH corrections,
      plus 20-seed benchmark rerun table (Oracle, StubbornOracle, CorrectedOracle,
      SentenceOnly, SceneOnly, Memorizer).
- A3: 20-seed novel-word support stream candidate enumeration and ideal-observer ceiling.
- A6: 20-seed identifiability report across constructions, NP orders, clause orders, and Q position.
- A8: 20-seed exhaustive Bayesian ideal observer on the reduced world.
"""

from __future__ import annotations

import json
from pathlib import Path

from nexus.baselines.memorizer import Memorizer
from nexus.baselines.oracle import CorrectedOracle, Oracle, StubbornOracle
from nexus.baselines.scene_only import SceneOnly
from nexus.baselines.sentence_only import SentenceOnly
from nexus.eval.audit import (
    audit_identifiability,
    audit_novel_word_support,
    feature_balance_audit,
    pooled_shortcut_audit,
    run_reduced_world_ideal_observer,
)
from nexus.eval.report import full_report
from nexus.eval.runner import RESULTS_DIR, ExperimentSpec, run_experiment, save_results
from nexus.world.config import DEFAULT_WORLD_CONFIG, load_world_config


def main() -> None:
    config = load_world_config(DEFAULT_WORLD_CONFIG)

    print("1. Running A1: 50-seed pooled shortcut audit...")
    a1_res = pooled_shortcut_audit(config, n_seeds=50, base_seed=0, world_mode="randomized")

    print("2. Running A2: 20-seed feature-level label-balance audit...")
    a2_res = feature_balance_audit(
        config, n_seeds=20, base_seed=0, n_permutations=1000, world_mode="randomized"
    )

    print("3. Running A2/A4: 20-seed benchmark rerun table...")
    seeds20 = tuple(range(20))
    models = [
        ("oracle", lambda s, ds: Oracle(ds.grammar, config.ontology, seed=s)),
        (
            "stubborn_oracle",
            lambda s, ds: StubbornOracle(
                ds.grammar,
                config.ontology,
                ds.plan.contradiction_target,
                ds.plan.contradiction_confusion,
                seed=s,
            ),
        ),
        (
            "corrected_oracle",
            lambda s, ds: CorrectedOracle(
                ds.grammar,
                config.ontology,
                ds.plan.contradiction_target,
                ds.plan.contradiction_confusion,
                seed=s,
            ),
        ),
        ("sentence_only", lambda s: SentenceOnly(seed=s)),
        ("scene_only", lambda s: SceneOnly(seed=s)),
        ("memorizer", lambda s: Memorizer(seed=s)),
    ]
    table_20: dict[str, dict[str, str]] = {}
    for name, factory in models:
        spec = ExperimentSpec(f"phase06_audit_{name}", name, seeds20, world_mode="randomized")
        res = run_experiment(spec, config, factory)
        out_path = RESULTS_DIR / f"phase05_{name}.json"
        save_results(res, out_path)
        out_path.with_suffix(".md").write_text(full_report(res), encoding="utf-8")
        agg = res["aggregate"]
        table_20[name] = {}
        for sp in (
            "train",
            "dev",
            "test_novel_combination",
            "test_novel_word_in_known_frame",
            "test_longer",
            "test_contradiction",
        ):
            st = agg[f"splits.{sp}.accuracy"]
            table_20[name][sp] = f"{st['mean']:.3f} +/- {st['std']:.3f}"
        if "splits.test_contradiction.checkpoints.before_corrections.accuracy" in agg:
            st_b = agg["splits.test_contradiction.checkpoints.before_corrections.accuracy"]
            st_a = agg["splits.test_contradiction.checkpoints.after_corrections.accuracy"]
            table_20[name]["tc_before_corrections"] = f"{st_b['mean']:.3f} +/- {st_b['std']:.3f}"
            table_20[name]["tc_after_corrections"] = f"{st_a['mean']:.3f} +/- {st_a['std']:.3f}"

    print("4. Running A3: 20-seed novel-word support enumeration...")
    a3_res = [audit_novel_word_support(config, seed=s) for s in range(20)]

    print("5. Running A6: 20-seed identifiability audit...")
    a6_res = [audit_identifiability(config, seed=s) for s in range(20)]

    print("6. Running A8: 20-seed exhaustive Bayesian ideal observer on reduced world...")
    a8_res = [run_reduced_world_ideal_observer(seed=s) for s in range(20)]

    out = {
        "A1_pooled_50_seed": a1_res,
        "A2_feature_balance": a2_res,
        "A2_A4_table_20_seed": table_20,
        "A3_novel_word_support": a3_res,
        "A6_identifiability": a6_res,
        "A8_reduced_world_ideal_observer": a8_res,
    }
    out_file = RESULTS_DIR / "stage_a_audit.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved Stage A audit results to {out_file}")


if __name__ == "__main__":
    main()
