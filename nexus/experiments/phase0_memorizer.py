"""Phase 0 experiment: run Baseline A (memorizer) through the harness.

Purpose: validate the harness. The memorizer should be near-perfect on train
and at chance on every split that requires generalization.

Usage::

    python -m nexus.experiments.phase0_memorizer --seeds 20 --base-seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nexus.baselines.memorizer import Memorizer
from nexus.eval.report import full_report
from nexus.eval.runner import RESULTS_DIR, ExperimentSpec, run_experiment, save_results
from nexus.world.config import DEFAULT_WORLD_CONFIG, load_world_config


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=20, help="number of seeds (default 20)")
    p.add_argument("--base-seed", type=int, default=0, help="first seed; seeds are base..base+N-1")
    p.add_argument("--config", type=Path, default=DEFAULT_WORLD_CONFIG, help="world config JSON")
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "phase0_memorizer.json")
    return p.parse_args()


def main() -> None:
    """Run the experiment, save JSON and a markdown report, print the report."""
    args = parse_args()
    config = load_world_config(args.config)
    spec = ExperimentSpec("phase0_harness_check", "memorizer",
                          tuple(range(args.base_seed, args.base_seed + args.seeds)))
    result = run_experiment(spec, config, lambda seed: Memorizer(seed=seed))
    path = save_results(result, args.out)
    report = full_report(result)
    path.with_suffix(".md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved {path} and {path.with_suffix('.md')}")


if __name__ == "__main__":
    main()
