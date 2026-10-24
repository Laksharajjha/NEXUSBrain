"""NEXUS-0 Phase 1.6 Mechanisms Evaluation Suite (EXPLORATORY).

Executes Tasks 0..4 on the fresh 50-seed evaluation range (1000..1049) in randomized
world mode, while recording every held-out test split evaluation for RERUNS.md:

- Task 0: Fresh evaluation data (seeds 1000..1049, n_seeds=50). Test splits on 1000..1049
  are evaluated ONCE at the end of the pipeline.
- Task 1: Separate the evidence channels (NEXUS-Truth, NEXUS-Cooc, NEXUS-Both) with
  identical hypothesis space and beam; run Negative Controls 2a/2b/2c on each; report
  accuracy and examples-to-criterion.
- Task 2: Matched-N comparison (N = 20, 50, 100, 200, 400, 1600, 50 seeds, paired
  bootstrap 95% CIs) across NEXUS-Both, NEXUS-Truth, NEXUS-Cooc, Baseline D, Baseline F
  (given_syntax, given_lexicon), and Baseline C; plus grammatical configuration breakdown
  by NP order, clause order, and question particle position.
- Task 3: Does the beam matter? Compare K = 1, 5, 50, 200 over 50 seeds with paired
  bootstrap CIs on both the standard curriculum order (N=100) and a garden-path /
  early-ambiguity stimulus order; plus Baseline C commit_after_n sensitivity on dev.
- Task 4: Does MDL matter? Compare explicit log2 bit-cost MDL prior, linear prior
  (alpha=0.05), and flat prior (alpha=0.0) on dev across variable-description-length
  worlds: (a) null-meaning / filler-word world, (b) homonym / polysemy multi-sense world,
  and (c) synonym world.
"""

from __future__ import annotations

import dataclasses
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from nexus.baselines.no_revision import NoRevisionLearner
from nexus.baselines.position_aware_cs import PositionAwareCrossSituationalLearner
from nexus.eval.learner_api import Learner
from nexus.eval.metrics import accuracy
from nexus.eval.splits import Dataset, Split, build_dataset, derive_seed
from nexus.learner.nexus_learner import NEXUSLearner
from nexus.world.config import WorldConfig, load_world_config
from nexus.world.example import Example, StreamEvent
from nexus.world.scene import Scene, SceneObject

SEED_START = 1000
SEED_END = 1050  # 1000..1049 inclusive (50 seeds)


@dataclass
class Phase16EvalCounter:
    """Tracks held-out test split evaluations on fresh seeds 1000..1049 and dev evaluations."""

    fresh_test_split_evaluations: int = 0
    dev_evaluations: int = 0
    test_runs_by_task: dict[str, int] = field(default_factory=dict)

    def record_test(self, task_name: str, num_test_splits: int = 4) -> None:
        self.fresh_test_split_evaluations += num_test_splits
        self.test_runs_by_task[task_name] = (
            self.test_runs_by_task.get(task_name, 0) + num_test_splits
        )

    def record_dev(self, count: int = 1) -> None:
        self.dev_evaluations += count


def eval_split_items(learner: Learner, items: Sequence[Example]) -> float:
    if not items:
        return 0.0
    preds = [learner.predict(ex.scene, ex.tokens) for ex in items]
    return accuracy(preds, items)


def paired_bootstrap_ci(
    vals_a: Sequence[float],
    vals_b: Sequence[float] | None = None,
    n_boot: int = 2000,
    seed: int = 10600,
) -> dict[str, float]:
    """Compute mean, std, and 95% (paired) bootstrap CI using vectorized NumPy."""
    arr_a = np.asarray(vals_a, dtype=np.float64)
    if vals_b is not None:
        arr_b = np.asarray(vals_b, dtype=np.float64)
        diff = arr_a - arr_b
    else:
        diff = arr_a
    n = len(diff)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[boot_idx].mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return {
        "mean": float(np.mean(diff)),
        "std": float(np.std(diff, ddof=1)) if n > 1 else 0.0,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def summarize_seed_dicts(seed_dicts: Sequence[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not seed_dicts:
        return {}
    keys = list(seed_dicts[0].keys())
    out: dict[str, dict[str, float]] = {}
    for idx, k in enumerate(keys):
        vals = [d[k] for d in seed_dicts]
        out[k] = paired_bootstrap_ci(vals, None, seed=20000 + idx)
    return out


def evaluate_learner_on_dataset(
    learner_factory: Callable[[], Learner],
    ds: Dataset,
    counter: Phase16EvalCounter,
    task_name: str,
    train_items_override: Sequence[Example] | None = None,
    eval_test_splits: bool = True,
) -> dict[str, float]:
    """Train once and evaluate on dev (and optionally once on the 4 test splits)."""
    train_items = (
        list(train_items_override)
        if train_items_override is not None
        else list(ds.train.items)
    )
    learner = learner_factory()
    for ex in train_items:
        learner.observe(ex.observation())

    res: dict[str, float] = {}
    res["train"] = eval_split_items(learner, ds.train.items)
    res["dev"] = eval_split_items(learner, ds.dev.items)
    counter.record_dev(1)

    if not eval_test_splits:
        return res

    res["test_novel_combination"] = eval_split_items(
        learner, ds.tests["test_novel_combination"].items
    )
    res["test_longer"] = eval_split_items(learner, ds.tests["test_longer"].items)

    if isinstance(learner, NEXUSLearner):
        n_obs = len(learner._observations)
        n_log = len(learner.experience_log_jsonl)
        saved_word_order = dict(learner._word_order)
        saved_obs_by_id = dict(learner._obs_by_id)

        nw_split = ds.tests["test_novel_word_in_known_frame"]
        for ev in nw_split.support:
            if ev.kind == "observe" and ev.observation is not None:
                learner.observe(ev.observation)
            elif ev.kind == "correct" and ev.correction is not None:
                learner.correct(ev.correction)
        res["test_novel_word_in_known_frame"] = eval_split_items(
            learner, nw_split.items
        )

        del learner._observations[n_obs:]
        del learner.experience_log_jsonl[n_log:]
        learner._word_order = saved_word_order
        learner._obs_by_id = saved_obs_by_id
        learner._change_records.clear()
        for obs_entry in learner._observations:
            obs_entry.corrected = False
        learner._beam_dirty = True

        ct_split = ds.tests["test_contradiction"]
        for ev in ct_split.support:
            if ev.kind == "observe" and ev.observation is not None:
                learner.observe(ev.observation)
            elif ev.kind == "correct" and ev.correction is not None:
                learner.correct(ev.correction)
        res["test_contradiction"] = eval_split_items(learner, ct_split.items)
    else:
        learner_nw = learner_factory()
        for ex in train_items:
            learner_nw.observe(ex.observation())
        nw_split = ds.tests["test_novel_word_in_known_frame"]
        for ev in nw_split.support:
            if ev.kind == "observe" and ev.observation is not None:
                learner_nw.observe(ev.observation)
            elif ev.kind == "correct" and ev.correction is not None:
                learner_nw.correct(ev.correction)
        res["test_novel_word_in_known_frame"] = eval_split_items(
            learner_nw, nw_split.items
        )

        learner_ct = learner_factory()
        for ex in train_items:
            learner_ct.observe(ex.observation())
        ct_split = ds.tests["test_contradiction"]
        for ev in ct_split.support:
            if ev.kind == "observe" and ev.observation is not None:
                learner_ct.observe(ev.observation)
            elif ev.kind == "correct" and ev.correction is not None:
                learner_ct.correct(ev.correction)
        res["test_contradiction"] = eval_split_items(learner_ct, ct_split.items)

    counter.record_test(task_name, num_test_splits=4)
    return res


def _stratified_subsample(
    items: Sequence[Example], n: int, rng: np.random.Generator
) -> list[Example]:
    """Subsample ``n`` training items while preserving 1-token / 2-token / >=3-token curriculum ratio."""
    if n >= len(items):
        return list(items)
    len1 = [ex for ex in items if len(ex.tokens) <= 2 and ex.lf.act == "assert" and len(ex.lf.atoms) == 1]
    len2 = [ex for ex in items if len(ex.lf.atoms) == 2]
    rest = [ex for ex in items if len(ex.lf.atoms) >= 3]

    n1 = min(len(len1), max(6, int(round(0.25 * n))))
    n2 = min(len(len2), max(6, int(round(0.35 * n))))
    n3 = max(0, n - n1 - n2)

    idx1 = rng.choice(len(len1), size=n1, replace=False) if n1 > 0 else []
    idx2 = rng.choice(len(len2), size=n2, replace=False) if n2 > 0 else []
    idx3 = rng.choice(len(rest), size=min(len(rest), n3), replace=False) if n3 > 0 else []

    out = [len1[int(i)] for i in idx1] + [len2[int(i)] for i in idx2] + [rest[int(i)] for i in idx3]
    return out


# ----------------------------------------------------------------------
# Task 1: Separate the Evidence Channels (Truth, Cooc, Both)
# ----------------------------------------------------------------------


def run_task1_evidence_channels(
    datasets: Sequence[Dataset], counter: Phase16EvalCounter
) -> dict[str, Any]:
    channels = ("truth", "cooc", "both")
    clean_runs: dict[str, list[dict[str, float]]] = {ch: [] for ch in channels}
    ctrl_2a_runs: dict[str, list[dict[str, float]]] = {ch: [] for ch in channels}
    ctrl_2b_runs: dict[str, list[dict[str, float]]] = {ch: [] for ch in channels}
    ctrl_2c_runs: dict[str, list[dict[str, float]]] = {ch: [] for ch in channels}
    etc_runs: dict[str, list[int]] = {ch: [] for ch in channels}

    checkpoints = (50, 100, 200, 400, 800, 1600)
    n_seeds = len(datasets)

    for idx, ds in enumerate(datasets):
        seed = SEED_START + idx
        rng = np.random.default_rng(seed + 11000)

        labels = [ex.truth for ex in ds.train.items]
        perm_l = rng.permutation(len(labels))
        train_2a = [
            dataclasses.replace(ex, truth=bool(labels[int(perm_l[i])]))
            for i, ex in enumerate(ds.train.items)
        ]

        scenes = [ex.scene for ex in ds.train.items]
        perm_s = rng.permutation(len(scenes))
        train_2b = [
            dataclasses.replace(ex, scene=scenes[int(perm_s[i])])
            for i, ex in enumerate(ds.train.items)
        ]

        other_ds = datasets[(idx + 1) % n_seeds]

        for ch in channels:
            # 1. Clean evaluation (dev only here; full test splits evaluated in Task 2 at N=1600
            # so each fresh test split is evaluated only once for clean N=1600, plus once here for
            # the 3 channel variants and negative controls)
            res_clean = evaluate_learner_on_dataset(
                lambda s=seed, c=ch: NEXUSLearner(seed=s, evidence_channel=c),
                ds,
                counter,
                f"task1_clean_{ch}",
                eval_test_splits=True,
            )
            clean_runs[ch].append(res_clean)

            # 2. Control 2a (shuffled labels)
            res_2a = evaluate_learner_on_dataset(
                lambda s=seed, c=ch: NEXUSLearner(seed=s, evidence_channel=c),
                ds,
                counter,
                f"task1_ctrl2a_shuffled_labels_{ch}",
                train_items_override=train_2a,
                eval_test_splits=True,
            )
            ctrl_2a_runs[ch].append(res_2a)

            # 3. Control 2b (shuffled scenes)
            res_2b = evaluate_learner_on_dataset(
                lambda s=seed, c=ch: NEXUSLearner(seed=s, evidence_channel=c),
                ds,
                counter,
                f"task1_ctrl2b_shuffled_scenes_{ch}",
                train_items_override=train_2b,
                eval_test_splits=True,
            )
            ctrl_2b_runs[ch].append(res_2b)

            # 4. Control 2c (cross-seed lexicon)
            res_2c = evaluate_learner_on_dataset(
                lambda s=seed, c=ch: NEXUSLearner(seed=s, evidence_channel=c),
                ds,
                counter,
                f"task1_ctrl2c_cross_seed_{ch}",
                train_items_override=other_ds.train.items,
                eval_test_splits=True,
            )
            ctrl_2c_runs[ch].append(res_2c)

            # 5. Examples-to-criterion (dev >= 0.90) across stratified checkpoints
            reached = 1600
            for cp in checkpoints:
                if cp == 1600:
                    dev_acc = res_clean["dev"]
                else:
                    rng_cp = np.random.default_rng(seed + 15000 + cp)
                    sub_cp = _stratified_subsample(ds.train.items, cp, rng_cp)
                    l_cp = NEXUSLearner(seed=seed, evidence_channel=ch)
                    for ex in sub_cp:
                        l_cp.observe(ex.observation())
                    dev_acc = eval_split_items(l_cp, ds.dev.items)
                    counter.record_dev(1)
                if dev_acc >= 0.90:
                    reached = cp
                    break
            etc_runs[ch].append(reached)

    comparisons = {
        "both_minus_truth_novel_comb": paired_bootstrap_ci(
            [r["test_novel_combination"] for r in clean_runs["both"]],
            [r["test_novel_combination"] for r in clean_runs["truth"]],
        ),
        "both_minus_cooc_novel_comb": paired_bootstrap_ci(
            [r["test_novel_combination"] for r in clean_runs["both"]],
            [r["test_novel_combination"] for r in clean_runs["cooc"]],
        ),
        "both_minus_cooc_novel_word": paired_bootstrap_ci(
            [r["test_novel_word_in_known_frame"] for r in clean_runs["both"]],
            [r["test_novel_word_in_known_frame"] for r in clean_runs["cooc"]],
        ),
        "both_minus_cooc_contradiction": paired_bootstrap_ci(
            [r["test_contradiction"] for r in clean_runs["both"]],
            [r["test_contradiction"] for r in clean_runs["cooc"]],
        ),
    }

    return {
        "clean": {ch: summarize_seed_dicts(clean_runs[ch]) for ch in channels},
        "control_2a_shuffled_labels": {
            ch: summarize_seed_dicts(ctrl_2a_runs[ch]) for ch in channels
        },
        "control_2b_shuffled_scenes": {
            ch: summarize_seed_dicts(ctrl_2b_runs[ch]) for ch in channels
        },
        "control_2c_cross_seed_lexicon": {
            ch: summarize_seed_dicts(ctrl_2c_runs[ch]) for ch in channels
        },
        "examples_to_criterion_dev90": {
            ch: paired_bootstrap_ci([float(x) for x in etc_runs[ch]]) for ch in channels
        },
        "paired_comparisons": comparisons,
        "raw_clean_runs": clean_runs,
    }


# ----------------------------------------------------------------------
# Task 2: Matched-N Comparison & Grammar Configuration Breakdown
# ----------------------------------------------------------------------


def _build_given_lexicon_and_syntax(ds: Dataset, config: WorldConfig) -> tuple[dict[str, str], dict[str, str]]:
    assert ds.grammar is not None
    g = ds.grammar
    lex = g.lexicon
    sym_rels = config.ontology.symmetric_relations

    given_lex: dict[str, str] = {}
    given_syn: dict[str, str] = {}

    noun_pos = g.np_order.index("noun")
    color_pos = g.np_order.index("color")
    size_pos = g.np_order.index("size")
    color_cat = "N/N" if color_pos < noun_pos else "N\\N"
    size_cat = "N/N" if size_pos < noun_pos else "N\\N"

    if g.clause_order == "SOV":
        verb_cat = "(S\\NP)\\NP"
    elif g.clause_order == "SVO":
        verb_cat = "(S\\NP)/NP"
    else:
        verb_cat = "(S/NP)/NP"

    q_cat = "S/S" if g.question_pos == "initial" else "S\\S"

    for k in config.ontology.kinds:
        w = lex[f"kind:{k}"]
        given_lex[w] = f"kind:{k}"
        given_syn[w] = "N"
    for c in config.ontology.colors:
        w = lex[f"color:{c}"]
        given_lex[w] = f"color:{c}"
        given_syn[w] = color_cat
    for s in config.ontology.sizes:
        w = lex[f"size:{s}"]
        given_lex[w] = f"size:{s}"
        given_syn[w] = size_cat
    for r in config.ontology.relation_names:
        w = lex[f"rel:{r}"]
        if r in sym_rels:
            given_lex[w] = f"rel:{r}:sym(x0,x1)"
        else:
            given_lex[w] = f"rel:{r}(x0,x1)"
        given_syn[w] = verb_cat

    w_q = g.question_particle
    given_lex[w_q] = "Q"
    given_syn[w_q] = q_cat
    return given_lex, given_syn


def run_task2_matched_n_comparison(
    datasets: Sequence[Dataset],
    base_config: WorldConfig,
    counter: Phase16EvalCounter,
    task1_clean_runs: dict[str, list[dict[str, float]]],
) -> dict[str, Any]:
    sample_sizes = (20, 50, 100, 200, 400, 1600)
    models = (
        "nexus_both",
        "nexus_truth",
        "nexus_cooc",
        "baseline_d_pos_cs",
        "baseline_f_given_syntax",
        "baseline_f_given_lexicon",
        "baseline_c_no_revision",
    )

    raw_by_n: dict[int, dict[str, list[dict[str, float]]]] = {
        n: {m: [] for m in models} for n in sample_sizes
    }

    for idx, ds in enumerate(datasets):
        seed = SEED_START + idx
        given_lex, given_syn = _build_given_lexicon_and_syntax(ds, base_config)

        for n in sample_sizes:
            if n == 1600:
                sub_train = list(ds.train.items)
                # Reuse Task 1 N=1600 runs for nexus_both, nexus_truth, nexus_cooc so we NEVER
                # evaluate the fresh test splits twice on the same model & data!
                raw_by_n[n]["nexus_both"].append(task1_clean_runs["both"][idx])
                raw_by_n[n]["nexus_truth"].append(task1_clean_runs["truth"][idx])
                raw_by_n[n]["nexus_cooc"].append(task1_clean_runs["cooc"][idx])
            else:
                rng_sub = np.random.default_rng(seed + 22000 + n)
                sub_train = _stratified_subsample(ds.train.items, n, rng_sub)

                raw_by_n[n]["nexus_both"].append(
                    evaluate_learner_on_dataset(
                        lambda s=seed: NEXUSLearner(seed=s, evidence_channel="both"),
                        ds,
                        counter,
                        f"task2_N{n}_nexus_both",
                        train_items_override=sub_train,
                        eval_test_splits=True,
                    )
                )
                raw_by_n[n]["nexus_truth"].append(
                    evaluate_learner_on_dataset(
                        lambda s=seed: NEXUSLearner(seed=s, evidence_channel="truth"),
                        ds,
                        counter,
                        f"task2_N{n}_nexus_truth",
                        train_items_override=sub_train,
                        eval_test_splits=True,
                    )
                )
                raw_by_n[n]["nexus_cooc"].append(
                    evaluate_learner_on_dataset(
                        lambda s=seed: NEXUSLearner(seed=s, evidence_channel="cooc"),
                        ds,
                        counter,
                        f"task2_N{n}_nexus_cooc",
                        train_items_override=sub_train,
                        eval_test_splits=True,
                    )
                )

            raw_by_n[n]["baseline_d_pos_cs"].append(
                evaluate_learner_on_dataset(
                    lambda s=seed: PositionAwareCrossSituationalLearner(seed=s),
                    ds,
                    counter,
                    f"task2_N{n}_baseline_d",
                    train_items_override=sub_train,
                    eval_test_splits=True,
                )
            )
            raw_by_n[n]["baseline_f_given_syntax"].append(
                evaluate_learner_on_dataset(
                    lambda s=seed, gs=given_syn: NEXUSLearner(
                        seed=s, evidence_channel="both", given_syntax=gs
                    ),
                    ds,
                    counter,
                    f"task2_N{n}_baseline_f_syntax",
                    train_items_override=sub_train,
                    eval_test_splits=True,
                )
            )
            raw_by_n[n]["baseline_f_given_lexicon"].append(
                evaluate_learner_on_dataset(
                    lambda s=seed, gl=given_lex: NEXUSLearner(
                        seed=s, evidence_channel="both", given_lexicon=gl
                    ),
                    ds,
                    counter,
                    f"task2_N{n}_baseline_f_lexicon",
                    train_items_override=sub_train,
                    eval_test_splits=True,
                )
            )
            raw_by_n[n]["baseline_c_no_revision"].append(
                evaluate_learner_on_dataset(
                    lambda s=seed: NoRevisionLearner(seed=s, commit_after_n=12),
                    ds,
                    counter,
                    f"task2_N{n}_baseline_c",
                    train_items_override=sub_train,
                    eval_test_splits=True,
                )
            )

    summary_by_n: dict[str, Any] = {}
    paired_vs_d_by_n: dict[str, Any] = {}
    for n in sample_sizes:
        summary_by_n[f"N_{n}"] = {
            m: summarize_seed_dicts(raw_by_n[n][m]) for m in models
        }
        paired_vs_d_by_n[f"N_{n}"] = {}
        for split_name in (
            "dev",
            "test_novel_combination",
            "test_novel_word_in_known_frame",
            "test_longer",
            "test_contradiction",
        ):
            vals_both = [r[split_name] for r in raw_by_n[n]["nexus_both"]]
            vals_d = [r[split_name] for r in raw_by_n[n]["baseline_d_pos_cs"]]
            paired_vs_d_by_n[f"N_{n}"][split_name] = paired_bootstrap_ci(
                vals_both, vals_d, seed=25000 + n
            )

    # Breakdown by NP order, clause order, and particle position at N = 50, 100, 200
    breakdown: dict[str, Any] = {}
    for n in (50, 100, 200):
        np_groups: dict[str, list[float]] = {
            "uniform_prenominal": [],
            "uniform_postnominal": [],
            "split_adj_noun_adj": [],
        }
        clause_groups: dict[str, list[float]] = {"SOV": [], "SVO": [], "VSO": []}
        q_groups: dict[str, list[float]] = {"initial": [], "final": []}

        for idx, ds in enumerate(datasets):
            assert ds.grammar is not None
            g = ds.grammar
            acc_val = raw_by_n[n]["nexus_both"][idx]["test_novel_combination"]
            noun_idx = g.np_order.index("noun")
            if noun_idx == 2:
                np_groups["uniform_prenominal"].append(acc_val)
            elif noun_idx == 0:
                np_groups["uniform_postnominal"].append(acc_val)
            else:
                np_groups["split_adj_noun_adj"].append(acc_val)
            clause_groups[g.clause_order].append(acc_val)
            q_groups[g.question_pos].append(acc_val)

        breakdown[f"N_{n}"] = {
            "by_np_order": {
                k: {"n_seeds": len(v), **paired_bootstrap_ci(v)}
                for k, v in np_groups.items()
            },
            "by_clause_order": {
                k: {"n_seeds": len(v), **paired_bootstrap_ci(v)}
                for k, v in clause_groups.items()
            },
            "by_particle_pos": {
                k: {"n_seeds": len(v), **paired_bootstrap_ci(v)}
                for k, v in q_groups.items()
            },
        }

    return {
        "summary_by_n": summary_by_n,
        "paired_nexus_both_minus_baseline_d": paired_vs_d_by_n,
        "grammar_configuration_breakdown": breakdown,
    }


# ----------------------------------------------------------------------
# Task 3: Does the Beam Matter? (Standard vs Garden-Path Stimulus Order + Baseline C Sensitivity)
# ----------------------------------------------------------------------


def _render_np(
    np_order: tuple[str, ...], w_noun: str, w_color: str | None, w_size: str | None
) -> tuple[str, ...]:
    out: list[str] = []
    for slot in np_order:
        if slot == "noun":
            out.append(w_noun)
        elif slot == "color" and w_color is not None:
            out.append(w_color)
        elif slot == "size" and w_size is not None:
            out.append(w_size)
    return tuple(out)


def _build_garden_path_stream(ds: Dataset) -> tuple[list[Example], list[Example]]:
    """Construct an early-ambiguity / garden-path stream where color:red is initially
    confounded with size:small (with 1 noise example favoring size:small), followed by
    3-token NPs (w_red + w_small + noun) that disambiguate via categorial mutual-exclusivity.
    """
    assert ds.grammar is not None
    lex = ds.grammar.lexicon
    w_red = lex["color:red"]
    w_small = lex["size:small"]
    w_ball = lex["kind:ball"]
    w_cup = lex["kind:cup"]
    np_order = ds.grammar.np_order

    toks_red_ball = _render_np(np_order, w_ball, w_red, None)
    toks_red_cup = _render_np(np_order, w_cup, w_red, None)
    toks_red_small_ball = _render_np(np_order, w_ball, w_red, w_small)
    toks_red_small_cup = _render_np(np_order, w_cup, w_red, w_small)

    base_train = [ex for ex in ds.train.items[:400] if w_red not in ex.tokens]
    template_ex = ds.train.items[0]
    gp_examples: list[Example] = []

    for i in range(6):
        k = "ball" if i % 2 == 0 else "cup"
        toks = toks_red_ball if i % 2 == 0 else toks_red_cup
        sc = Scene(
            objects=(SceneObject(id=0, kind=k, color="red", size="small"),),
            relations=(),
        )
        gp_examples.append(
            dataclasses.replace(
                template_ex, id=f"gp_conf_{i}", scene=sc, tokens=toks, truth=True
            )
        )

    sc_lead = Scene(
        objects=(SceneObject(id=0, kind="ball", color="blue", size="small"),),
        relations=(),
    )
    gp_examples.append(
        dataclasses.replace(
            template_ex, id="gp_lead", scene=sc_lead, tokens=toks_red_ball, truth=True
        )
    )

    for i in range(4):
        k = "ball" if i % 2 == 0 else "cup"
        toks = toks_red_ball if i % 2 == 0 else toks_red_cup
        sc = Scene(
            objects=(SceneObject(id=0, kind=k, color="green", size="medium"),),
            relations=(),
        )
        gp_examples.append(
            dataclasses.replace(
                template_ex, id=f"gp_false_{i}", scene=sc, tokens=toks, truth=False
            )
        )

    for i in range(8):
        k = "ball" if i % 2 == 0 else "cup"
        toks = toks_red_small_ball if i % 2 == 0 else toks_red_small_cup
        sc = Scene(
            objects=(SceneObject(id=0, kind=k, color="red", size="small"),),
            relations=(),
        )
        gp_examples.append(
            dataclasses.replace(
                template_ex, id=f"gp_np3_{i}", scene=sc, tokens=toks, truth=True
            )
        )

    dev_red = [ex for ex in ds.dev.items if w_red in ex.tokens]
    return base_train + gp_examples, dev_red


def run_task3_beam_and_commitment(
    datasets: Sequence[Dataset], counter: Phase16EvalCounter
) -> dict[str, Any]:
    beam_widths = (1, 5, 50, 200)
    std_n100_dev: dict[int, list[float]] = {k: [] for k in beam_widths}
    gp_all_dev: dict[int, list[float]] = {k: [] for k in beam_widths}
    gp_target_dev: dict[int, list[float]] = {k: [] for k in beam_widths}

    for idx, ds in enumerate(datasets):
        seed = SEED_START + idx
        rng_100 = np.random.default_rng(seed + 31000)
        sub_100 = _stratified_subsample(ds.train.items, 100, rng_100)
        gp_stream, dev_red = _build_garden_path_stream(ds)

        for k in beam_widths:
            # Standard curriculum at N=100 (evaluated on dev)
            l_std = NEXUSLearner(seed=seed, beam_width=k, evidence_channel="both")
            for ex in sub_100:
                l_std.observe(ex.observation())
            std_n100_dev[k].append(eval_split_items(l_std, ds.dev.items))
            counter.record_dev(1)

            # Garden-path / early-ambiguity stimulus order (evaluated on dev and target word subset)
            l_gp = NEXUSLearner(
                seed=seed, beam_width=k, evidence_channel="both", allow_synonyms=True
            )
            for ex in gp_stream:
                l_gp.observe(ex.observation())
            gp_all_dev[k].append(eval_split_items(l_gp, ds.dev.items))
            gp_target_dev[k].append(eval_split_items(l_gp, dev_red))
            counter.record_dev(2)

    # Baseline C commit_after_n sensitivity on dev (4, 8, 12, 24, 48)
    commit_thresholds = (4, 8, 12, 24, 48)
    commit_sensitivity: dict[str, Any] = {}
    for m_thresh in commit_thresholds:
        dev_vals: list[float] = []
        train_vals: list[float] = []
        for idx, ds in enumerate(datasets):
            seed = SEED_START + idx
            learner_c = NoRevisionLearner(seed=seed, commit_after_n=m_thresh)
            for ex in ds.train.items:
                learner_c.observe(ex.observation())
            train_vals.append(eval_split_items(learner_c, ds.train.items))
            dev_vals.append(eval_split_items(learner_c, ds.dev.items))
            counter.record_dev(1)
        commit_sensitivity[f"commit_after_{m_thresh}"] = {
            "train": paired_bootstrap_ci(train_vals),
            "dev": paired_bootstrap_ci(dev_vals),
        }

    return {
        "standard_order_N100_dev": {
            f"K_{k}": paired_bootstrap_ci(std_n100_dev[k]) for k in beam_widths
        },
        "standard_order_K200_minus_K1": paired_bootstrap_ci(
            std_n100_dev[200], std_n100_dev[1]
        ),
        "garden_path_overall_dev": {
            f"K_{k}": paired_bootstrap_ci(gp_all_dev[k]) for k in beam_widths
        },
        "garden_path_target_word_dev": {
            f"K_{k}": paired_bootstrap_ci(gp_target_dev[k]) for k in beam_widths
        },
        "garden_path_target_K200_minus_K1": paired_bootstrap_ci(
            gp_target_dev[200], gp_target_dev[1]
        ),
        "garden_path_target_K5_minus_K1": paired_bootstrap_ci(
            gp_target_dev[5], gp_target_dev[1]
        ),
        "baseline_c_commit_after_n_sensitivity_dev": commit_sensitivity,
    }


# ----------------------------------------------------------------------
# Task 4: Does MDL Matter? (Variable-Length Hypotheses on Dev)
# ----------------------------------------------------------------------


def _apply_token_map_to_examples(
    items: Sequence[Example], token_map: dict[str, str]
) -> list[Example]:
    return [
        dataclasses.replace(ex, tokens=tuple(token_map.get(t, t) for t in ex.tokens))
        for ex in items
    ]


def _apply_synonym_split_to_examples(
    items: Sequence[Example], word_to_syn: dict[str, str], rng: np.random.Generator
) -> list[Example]:
    out: list[Example] = []
    for ex in items:
        new_toks = tuple(
            word_to_syn[t] if (t in word_to_syn and float(rng.random()) < 0.5) else t
            for t in ex.tokens
        )
        out.append(dataclasses.replace(ex, tokens=new_toks))
    return out


def run_task4_mdl_variable_length(
    datasets: Sequence[Dataset],
    base_config: WorldConfig,
    counter: Phase16EvalCounter,
) -> dict[str, Any]:
    priors = ("flat", "linear", "log2_bits")
    filler_dev: dict[str, list[float]] = {p: [] for p in priors}
    filler_size: dict[str, list[float]] = {p: [] for p in priors}
    homonym_dev: dict[str, list[float]] = {p: [] for p in priors}
    homonym_size: dict[str, list[float]] = {p: [] for p in priors}
    synonym_dev: dict[str, list[float]] = {p: [] for p in priors}
    synonym_size: dict[str, list[float]] = {p: [] for p in priors}

    for idx, ds in enumerate(datasets):
        seed = SEED_START + idx
        assert ds.grammar is not None
        lex = ds.grammar.lexicon
        noun_words = {lex[f"kind:{k}"] for k in base_config.ontology.kinds}

        # World 4a: Variable-size lexicon with rare optional discourse filler words ("uh", "um")
        red_true = [
            ex
            for ex in ds.train.items
            if ex.truth
            and any(o.color == "red" for o in ex.scene.objects)
            and any(t in noun_words for t in ex.tokens)
        ][:2]
        small_true = [
            ex
            for ex in ds.train.items
            if ex.truth
            and any(o.size == "small" for o in ex.scene.objects)
            and any(t in noun_words for t in ex.tokens)
        ][2:4]
        red_ids = {ex.id for ex in red_true}
        small_ids = {ex.id for ex in small_true}

        train_filler: list[Example] = []
        for ex in ds.train.items[:400]:
            toks = list(ex.tokens)
            if ex.id in red_ids:
                for ti, t in enumerate(toks):
                    if t in noun_words:
                        toks.insert(ti, "uh")
                        break
            elif ex.id in small_ids:
                for ti, t in enumerate(toks):
                    if t in noun_words:
                        toks.insert(ti, "um")
                        break
            train_filler.append(dataclasses.replace(ex, tokens=tuple(toks)))

        dev_filler: list[Example] = []
        for di, ex in enumerate(ds.dev.items):
            toks = list(ex.tokens)
            filler = "uh" if di % 2 == 0 else "um"
            for ti, t in enumerate(toks):
                if t in noun_words:
                    toks.insert(ti, filler)
                    break
            dev_filler.append(dataclasses.replace(ex, tokens=tuple(toks)))

        # World 4b: Homonym / Polysemy world (size:small shares surface word with color:red)
        w_red = lex["color:red"]
        map_h1 = {lex["size:small"]: w_red}
        train_hom = _apply_token_map_to_examples(ds.train.items[:400], map_h1)
        dev_hom = _apply_token_map_to_examples(ds.dev.items, map_h1)

        # World 4c: Synonym world (2 surface words per kind)
        word_to_syn = {
            lex[f"kind:{k}"]: f"{lex[f'kind:{k}']}_syn2"
            for k in base_config.ontology.kinds
        }
        rng_syn_tr = np.random.default_rng(seed + 41000)
        rng_syn_dv = np.random.default_rng(seed + 42000)
        train_syn = _apply_synonym_split_to_examples(
            ds.train.items[:400], word_to_syn, rng_syn_tr
        )
        dev_syn = _apply_synonym_split_to_examples(
            ds.dev.items, word_to_syn, rng_syn_dv
        )

        for pm in priors:
            # Evaluate 4a (Filler words)
            l_f = NEXUSLearner(
                seed=seed,
                evidence_channel="both",
                prior_mode=pm,
                allow_null_entries=True,
                allow_synonyms=True,
            )
            for ex in train_filler:
                l_f.observe(ex.observation())
            filler_dev[pm].append(eval_split_items(l_f, dev_filler))
            filler_size[pm].append(float(l_f.model_size()))
            counter.record_dev(1)

            # Evaluate 4b (Homonymy / Polysemy with multi-sense lexicon)
            l_h = NEXUSLearner(
                seed=seed,
                evidence_channel="both",
                prior_mode=pm,
                allow_multi_sense=True,
            )
            for ex in train_hom:
                l_h.observe(ex.observation())
            homonym_dev[pm].append(eval_split_items(l_h, dev_hom))
            homonym_size[pm].append(float(l_h.model_size()))
            counter.record_dev(1)

            # Evaluate 4c (Synonyms)
            l_s = NEXUSLearner(
                seed=seed,
                evidence_channel="both",
                prior_mode=pm,
                allow_synonyms=True,
            )
            for ex in train_syn:
                l_s.observe(ex.observation())
            synonym_dev[pm].append(eval_split_items(l_s, dev_syn))
            synonym_size[pm].append(float(l_s.model_size()))
            counter.record_dev(1)

    return {
        "world_4a_filler_null_words": {
            pm: {
                "dev_accuracy": paired_bootstrap_ci(filler_dev[pm]),
                "model_size": paired_bootstrap_ci(filler_size[pm]),
            }
            for pm in priors
        },
        "world_4a_log2_minus_flat_dev": paired_bootstrap_ci(
            filler_dev["log2_bits"], filler_dev["flat"]
        ),
        "world_4b_homonym_multi_sense": {
            pm: {
                "dev_accuracy": paired_bootstrap_ci(homonym_dev[pm]),
                "model_size": paired_bootstrap_ci(homonym_size[pm]),
            }
            for pm in priors
        },
        "world_4b_log2_minus_flat_dev": paired_bootstrap_ci(
            homonym_dev["log2_bits"], homonym_dev["flat"]
        ),
        "world_4c_synonyms": {
            pm: {
                "dev_accuracy": paired_bootstrap_ci(synonym_dev[pm]),
                "model_size": paired_bootstrap_ci(synonym_size[pm]),
            }
            for pm in priors
        },
        "world_4c_log2_minus_flat_dev": paired_bootstrap_ci(
            synonym_dev["log2_bits"], synonym_dev["flat"]
        ),
    }


def run_all_phase1_6() -> dict[str, Any]:
    t0 = time.time()
    base_config = load_world_config()
    counter = Phase16EvalCounter()

    print(f"[Phase 1.6] Building fresh datasets for 50 seeds ({SEED_START}..{SEED_END - 1})...", flush=True)
    datasets: list[Dataset] = [
        build_dataset(base_config, derive_seed(s, "world"), world_mode="randomized")
        for s in range(SEED_START, SEED_END)
    ]

    print("[Phase 1.6] Running Task 1: Separate the evidence channels...", flush=True)
    t1_res = run_task1_evidence_channels(datasets, counter)
    raw_clean_runs = t1_res.pop("raw_clean_runs")

    print("[Phase 1.6] Running Task 2: Matched-N comparison & grammar breakdown...", flush=True)
    t2_res = run_task2_matched_n_comparison(datasets, base_config, counter, raw_clean_runs)

    print("[Phase 1.6] Running Task 3: Beam width & commitment threshold sensitivity...", flush=True)
    t3_res = run_task3_beam_and_commitment(datasets, counter)

    print("[Phase 1.6] Running Task 4: Variable-length MDL prior comparison...", flush=True)
    t4_res = run_task4_mdl_variable_length(datasets, base_config, counter)

    elapsed = time.time() - t0
    results = {
        "phase": "1.6_exploratory_mechanisms",
        "seed_range": [SEED_START, SEED_END - 1],
        "n_seeds": SEED_END - SEED_START,
        "elapsed_seconds": round(elapsed, 2),
        "evaluation_counter": {
            "fresh_test_split_evaluations": counter.fresh_test_split_evaluations,
            "dev_evaluations": counter.dev_evaluations,
            "test_runs_by_task": counter.test_runs_by_task,
        },
        "task1_evidence_channels": t1_res,
        "task2_matched_n": t2_res,
        "task3_beam_and_commitment": t3_res,
        "task4_mdl_variable_length": t4_res,
    }

    out_path = (
        Path(__file__).resolve().parent / "results" / "phase1_6_mechanisms.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[Phase 1.6] Complete in {elapsed:.1f}s. "
        f"Fresh test split evaluations: {counter.fresh_test_split_evaluations}. "
        f"Saved to {out_path}"
    )
    return results


if __name__ == "__main__":
    run_all_phase1_6()
