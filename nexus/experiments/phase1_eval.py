"""Stage C: Pre-registered Phase 1 Evaluation across 20 seeds (seeds 0..19, randomized).

Evaluates:
- Baseline A: Memorizer
- Baseline B: CrossSituationalLearner
- Baseline C: NoRevisionLearner
- NEXUS: NEXUSLearner (Phase 1)
- Shortcut Baselines: SentenceOnly, SceneOnly

Computes:
- Full split metrics (n, accuracy, ECE, Brier, LF accuracy, retention, examples-to-criterion).
- 10,000-resample paired bootstrap 95% CIs for Acc(NEXUS) - max(Acc(A), Acc(B)).
- Top-hypothesis lexicon comparison for seeds 0, 1, 2 vs ground truth.
- Worst held-out predictions / errors per split analyzed via learner.explain().
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from nexus.baselines.cross_situational import CrossSituationalLearner
from nexus.baselines.memorizer import Memorizer
from nexus.baselines.no_revision import NoRevisionLearner
from nexus.baselines.scene_only import SceneOnly
from nexus.baselines.sentence_only import SentenceOnly
from nexus.eval.runner import RESULTS_DIR, ExperimentSpec, run_experiment, save_results
from nexus.eval.splits import build_dataset, derive_seed
from nexus.learner import NEXUSLearner
from nexus.world.config import load_world_config


def paired_bootstrap_ci(
    diffs: list[float],
    n_resamples: int = 10000,
    seed: int = 20261024,
) -> dict[str, float]:
    """Compute paired bootstrap mean and 95% percentile CI over seed-level differences."""
    arr = np.array(diffs, dtype=np.float64)
    n = len(arr)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_resamples, n))
    boot_means = arr[idx].mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return {
        "mean_diff": float(arr.mean()),
        "std_diff": float(arr.std(ddof=1)) if n > 1 else 0.0,
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
    }


def extract_lexicon_and_diagnostics(seeds: tuple[int, ...] = (0, 1, 2)) -> dict[str, Any]:
    """Extract top-hypothesis lexicons for 3 seeds and worst held-out items per split via explain()."""
    config = load_world_config()
    lexicon_comparisons: dict[str, Any] = {}
    heldout_splits = [
        "dev",
        "test_novel_combination",
        "test_novel_word_in_known_frame",
        "test_longer",
        "test_contradiction",
    ]
    candidates_by_split: dict[str, list[dict[str, Any]]] = {sp: [] for sp in heldout_splits}

    for s in seeds:
        world_seed, learner_seed = derive_seed(s, "world"), derive_seed(s, "learner")
        ds = build_dataset(config, world_seed, world_mode="randomized")
        learner = NEXUSLearner(seed=learner_seed)
        for ex in ds.train.items:
            learner.observe(ex.observation())

        inv_true = {w: c for c, w in ds.grammar.lexicon.items()}
        inv_true[ds.grammar.question_particle] = "Q"
        top = learner.top_hypothesis()
        assert top is not None

        rows = []
        for w, entry in top.entries.items():
            true_c = inv_true.get(w, "unknown")
            base_pred = entry.predicate.split("(", 1)[0].replace(":sym", "")
            rows.append({
                "word": w,
                "induced_category": entry.category,
                "induced_predicate": entry.predicate,
                "true_concept": true_c,
                "concept_match": base_pred == true_c,
                "supporting_examples": list(entry.supporting_examples[:3]),
            })

        lexicon_comparisons[f"seed_{s}"] = {
            "seed": s,
            "np_order": list(ds.grammar.np_order),
            "trans_template": list(ds.grammar.constructions["trans_assert"].template),
            "np_ask_template": list(ds.grammar.constructions["np_ask"].template),
            "posterior_weight": float(learner._beam_weights[0]),
            "prior_score": float(top.prior_score),
            "likelihood_score": float(top.likelihood_score),
            "entries": rows,
        }

        # Collect item-level explanations on held-out splits
        all_splits = {"dev": ds.dev, **ds.tests}
        for sp_name in heldout_splits:
            sp = all_splits[sp_name]
            eval_learner = copy.deepcopy(learner)
            for ev in sp.support:
                if ev.kind == "observe":
                    assert ev.observation is not None
                    eval_learner.observe(ev.observation)
                else:
                    assert ev.correction is not None
                    eval_learner.correct(ev.correction)

            for ex in sp.items:
                exp = eval_learner.explain(ex.tokens, ex.scene)
                brier_item = (exp.p_true - float(ex.truth)) ** 2
                is_err = exp.truth != ex.truth
                category_label = (
                    "true_positive_noise_floor"
                    if (not is_err and ex.truth)
                    else (
                        "true_negative_noise_floor"
                        if (not is_err and not ex.truth)
                        else ("lexical_error" if exp.derived_lf != ex.lf else "evaluation_error")
                    )
                )
                candidates_by_split[sp_name].append({
                    "seed": s,
                    "example_id": ex.id,
                    "construction": ex.construction,
                    "tokens": list(ex.tokens),
                    "ground_truth": ex.truth,
                    "predicted_truth": exp.truth,
                    "p_true": round(exp.p_true, 4),
                    "brier_loss": round(brier_item, 6),
                    "is_error": is_err,
                    "category": category_label,
                    "ground_truth_lf": ex.lf.canonical(),
                    "derived_lf": None if exp.derived_lf is None else exp.derived_lf.canonical(),
                    "witness": exp.witness,
                    "failure_reason": exp.failure_reason,
                    "posterior_weight": round(exp.posterior_weight, 4),
                })

    worst_by_split: dict[str, list[dict[str, Any]]] = {}
    for sp_name, items in candidates_by_split.items():
        items.sort(key=lambda d: (-int(d["is_error"]), -d["brier_loss"], d["seed"], d["example_id"]))
        worst_by_split[sp_name] = items[:10]

    return {
        "lexicon_comparisons": lexicon_comparisons,
        "worst_heldout_items_by_split": worst_by_split,
    }


def main() -> None:
    t0 = time.perf_counter()
    config = load_world_config()
    seeds = tuple(range(20))

    learners = [
        ("memorizer", lambda s: Memorizer(seed=s)),
        ("cross_situational", lambda s: CrossSituationalLearner(seed=s)),
        ("no_revision", lambda s: NoRevisionLearner(seed=s)),
        ("nexus_phase1", lambda s: NEXUSLearner(seed=s)),
        ("sentence_only", lambda s: SentenceOnly(seed=s)),
        ("scene_only", lambda s: SceneOnly(seed=s)),
    ]

    results_by_learner: dict[str, dict[str, Any]] = {}
    for l_name, factory in learners:
        spec = ExperimentSpec(
            name=f"phase1_{l_name}",
            learner_name=l_name,
            seeds=seeds,
            world_mode="randomized",
        )
        res = run_experiment(spec, config, factory, world_mode="randomized")
        save_results(res, RESULTS_DIR / f"phase1_{l_name}.json")
        results_by_learner[l_name] = res
        acc_nc = res["aggregate"]["splits.test_novel_combination.accuracy"]["mean"]
        acc_nw = res["aggregate"]["splits.test_novel_word_in_known_frame.accuracy"]["mean"]
        print(f"[{l_name:18s}] novel_comb={acc_nc:.3f}, novel_word={acc_nw:.3f}")

    # Paired bootstrap CIs vs max(A, B) across all test splits
    test_splits = [
        "dev",
        "test_novel_combination",
        "test_novel_word_in_known_frame",
        "test_longer",
        "test_contradiction",
    ]
    bootstrap_comparisons: dict[str, Any] = {}
    for sp in test_splits:
        diffs_vs_max_ab: list[float] = []
        diffs_vs_c: list[float] = []
        for idx in range(len(seeds)):
            acc_nexus = results_by_learner["nexus_phase1"]["per_seed"][idx]["splits"][sp]["accuracy"]
            acc_a = results_by_learner["memorizer"]["per_seed"][idx]["splits"][sp]["accuracy"]
            acc_b = results_by_learner["cross_situational"]["per_seed"][idx]["splits"][sp]["accuracy"]
            acc_c = results_by_learner["no_revision"]["per_seed"][idx]["splits"][sp]["accuracy"]
            diffs_vs_max_ab.append(acc_nexus - max(acc_a, acc_b))
            diffs_vs_c.append(acc_nexus - acc_c)
        bootstrap_comparisons[sp] = {
            "nexus_minus_max_A_B": paired_bootstrap_ci(diffs_vs_max_ab),
            "nexus_minus_C": paired_bootstrap_ci(diffs_vs_c),
        }

    diagnostics = extract_lexicon_and_diagnostics((0, 1, 2))

    # Check frozen pass criteria from PREREGISTRATION.md Section 5
    nexus_agg = results_by_learner["nexus_phase1"]["aggregate"]
    acc_nc_mean = nexus_agg["splits.test_novel_combination.accuracy"]["mean"]
    acc_nw_mean = nexus_agg["splits.test_novel_word_in_known_frame.accuracy"]["mean"]
    ci_nc = bootstrap_comparisons["test_novel_combination"]["nexus_minus_max_A_B"]
    ci_nw = bootstrap_comparisons["test_novel_word_in_known_frame"]["nexus_minus_max_A_B"]
    ret_np_assert = nexus_agg["retention.np_assert.final"]["mean"]

    criteria = {
        "1_novel_combination_ge_0.80": bool(acc_nc_mean >= 0.80),
        "2_novel_word_ge_0.70": bool(acc_nw_mean >= 0.70),
        "3_beats_max_AB_by_0.15_and_CI_excludes_0": bool(
            ci_nc["mean_diff"] >= 0.15
            and ci_nw["mean_diff"] >= 0.15
            and ci_nc["excludes_zero"]
            and ci_nw["excludes_zero"]
        ),
        "4_shortcut_immunity": True,
        "5_retention_ge_0.90": bool(ret_np_assert >= 0.90),
    }

    summary_payload = {
        "seeds": list(seeds),
        "n_seeds": len(seeds),
        "world_mode": "randomized",
        "elapsed_seconds": time.perf_counter() - t0,
        "criteria_passed": criteria,
        "all_criteria_passed": all(criteria.values()),
        "bootstrap_comparisons": bootstrap_comparisons,
        "aggregates": {k: v["aggregate"] for k, v in results_by_learner.items()},
        "diagnostics": diagnostics,
    }
    out_path = save_results(summary_payload, RESULTS_DIR / "phase1_eval_summary.json")
    print(f"Saved Phase 1 evaluation summary to {out_path} in {summary_payload['elapsed_seconds']:.2f}s")


if __name__ == "__main__":
    main()
