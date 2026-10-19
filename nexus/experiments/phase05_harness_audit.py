"""Phase 0.5 experiment: Harness audit.

Evaluates:
- Oracle ceiling (should hit 1.00 on all splits)
- SentenceOnly shortcut baseline (should stay within chance on all test splits)
- SceneOnly shortcut baseline (should stay within chance on all test splits)
- Memorizer baseline (should be near 1.0 on train, chance on generalization splits)

All evaluated under randomized world mode (per-seed lexicon, NP order, clause order, Q-particle).

Usage::

    python -m nexus.experiments.phase05_harness_audit --seeds 20 --base-seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nexus.baselines.memorizer import Memorizer
from nexus.baselines.oracle import Oracle
from nexus.baselines.scene_only import SceneOnly
from nexus.baselines.sentence_only import SentenceOnly
from nexus.eval.report import full_report
from nexus.eval.runner import RESULTS_DIR, ExperimentSpec, run_experiment, save_results
from nexus.eval.splits import Dataset
from nexus.world.config import DEFAULT_WORLD_CONFIG, load_world_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=20, help="number of seeds (default 20)")
    p.add_argument("--base-seed", type=int, default=0, help="first seed; seeds are base..base+N-1")
    p.add_argument("--world-mode", choices=["fixed", "randomized"], default="randomized", help="world mode")
    p.add_argument("--config", type=Path, default=DEFAULT_WORLD_CONFIG, help="world config JSON")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_world_config(args.config)
    seeds = tuple(range(args.base_seed, args.base_seed + args.seeds))

    models = [
        ("oracle", lambda s, ds: Oracle(ds.grammar, config.ontology, seed=s)),
        ("sentence_only", lambda s: SentenceOnly(seed=s)),
        ("scene_only", lambda s: SceneOnly(seed=s)),
        ("memorizer", lambda s: Memorizer(seed=s)),
    ]

    all_results = {}
    print(f"Running Phase 0.5 audit across {len(seeds)} seeds in mode '{args.world_mode}'...\n")

    for name, factory in models:
        print(f"--> Evaluating {name}...")
        spec = ExperimentSpec(
            name=f"phase05_audit_{name}",
            learner_name=name,
            seeds=seeds,
            world_mode=args.world_mode,
        )
        res = run_experiment(spec, config, factory)
        out_path = args.out_dir / f"phase05_{name}.json"
        save_results(res, out_path)
        report = full_report(res)
        out_path.with_suffix(".md").write_text(report, encoding="utf-8")
        all_results[name] = res
        print(f"    Saved {out_path} and {out_path.with_suffix('.md')}")

    # Build comparison summary table
    print("\n" + "=" * 80)
    print("PHASE 0.5 AUDIT SUMMARY (Mean +/- Std Accuracy across 20 seeds)")
    print("=" * 80)
    splits = [
        "train",
        "dev",
        "test_novel_combination",
        "test_novel_word_in_known_frame",
        "test_longer",
        "test_contradiction",
    ]
    header = f"{'Model':<16} | " + " | ".join(f"{s:<15}" for s in splits)
    print(header)
    print("-" * len(header))

    summary_rows = []
    for name, _ in models:
        row = [f"{name:<16}"]
        agg = all_results[name]["aggregate"]
        row_dict = {"model": name}
        for s in splits:
            stat = agg.get(f"splits.{s}.accuracy")
            if stat and stat.get("mean") is not None:
                val_str = f"{stat['mean']:.3f} +/- {stat['std']:.3f}"
            else:
                val_str = "N/A"
            row.append(f"{val_str:<15}")
            row_dict[s] = val_str
        summary_rows.append(row_dict)
        print(" | ".join(row))

    summary_file = args.out_dir / "phase05_summary.json"
    summary_file.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print("\nAudit complete.")


if __name__ == "__main__":
    main()
