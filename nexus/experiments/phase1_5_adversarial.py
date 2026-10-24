"""NEXUS-0 Phase 1.5 Adversarial Validation Runner (EXPLORATORY).

Executes all 5 parts of Phase 1.5 across 20 seeds (0..19) in randomized world mode,
without tuning NEXUSLearner:
- Part 1: Leakage & Protocol Audit (SpyLearnerWrapper + static AST field audit + twin audit)
- Part 2: Negative Controls (2a shuffled labels, 2b shuffled scenes, 2c cross-seed lexicon)
- Part 3: Difficulty Ladder (3.1 sample size N, 3.2 random negatives only, 3.3 positive-only,
  3.4 label noise, 3.5 lexicon scale 6..48 kinds, 3.6 homonymy & synonymy,
  3.7 nested modification & multi-clause chains beyond training depth, 3.8 beam width K=1..200)
- Part 4: Stronger Baselines (Baseline C before/after fix, D PositionAwareCS,
  E FromScratchNeuralBaseline, F NEXUS with given_syntax vs given_lexicon)
- Part 5: MDL Prior Audit (alpha=0.05 linear vs explicit log2 bit-cost prior)
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from nexus.baselines.cross_situational import CrossSituationalLearner
from nexus.baselines.memorizer import Memorizer
from nexus.baselines.neural_ref import FromScratchNeuralBaseline
from nexus.baselines.no_revision import NoRevisionLearner
from nexus.baselines.position_aware_cs import PositionAwareCrossSituationalLearner
from nexus.eval.learner_api import Learner, Prediction
from nexus.eval.metrics import accuracy
from nexus.eval.splits import Dataset, Split, build_dataset, derive_seed
from nexus.learner.categorial import parse_tokens
from nexus.learner.nexus_learner import NEXUSLearner
from nexus.world.config import (
    GrammarConfig,
    Ontology,
    WorldConfig,
    load_world_config,
)
from nexus.world.example import Correction, Example, Observation, StreamEvent
from nexus.world.logic import evaluate
from nexus.world.scene import Scene, SceneObject, SceneRelation, build_scene, sample_scene


@dataclass
class EvalCounter:
    """Tracks every held-out test split evaluation for RERUNS.md logging."""

    test_split_evaluations: int = 0
    runs_by_experiment: dict[str, int] = field(default_factory=dict)

    def record(self, exp_name: str, num_test_splits: int = 4) -> None:
        self.test_split_evaluations += num_test_splits
        self.runs_by_experiment[exp_name] = (
            self.runs_by_experiment.get(exp_name, 0) + num_test_splits
        )


class SpyLearnerWrapper:
    """Spy wrapper proving a learner never observes held-out items before scoring."""

    def __init__(self, inner: NEXUSLearner) -> None:
        self.inner = inner
        self.observed_ids: set[str] = set()
        self.leak_violations: list[str] = []
        self.mutation_violations: list[str] = []
        self.predict_calls: int = 0
        self.observe_calls: int = 0

    def observe(self, observation: Observation) -> None:
        self.observe_calls += 1
        self.observed_ids.add(observation.example_id)
        self.inner.observe(observation)

    def correct(self, correction: Correction) -> None:
        self.inner.correct(correction)

    def predict_for_example(self, ex: Example) -> Prediction:
        self.predict_calls += 1
        if ex.id in self.observed_ids:
            self.leak_violations.append(
                f"Example ID {ex.id} was already observed before predict!"
            )
        if ex.id in self.inner._obs_by_id:
            self.leak_violations.append(
                f"Example ID {ex.id} found in learner._obs_by_id before predict!"
            )

        if self.inner._beam_dirty:
            self.inner._ensure_beam()
        obs_len_before = len(self.inner._observations)
        log_len_before = len(self.inner.experience_log_jsonl)
        top_sig_before = self.inner._beam[0].signature if self.inner._beam else ()

        pred = self.inner.predict(ex.scene, ex.tokens)

        obs_len_after = len(self.inner._observations)
        log_len_after = len(self.inner.experience_log_jsonl)
        top_sig_after = self.inner._beam[0].signature if self.inner._beam else ()

        if (
            obs_len_before != obs_len_after
            or log_len_before != log_len_after
            or top_sig_before != top_sig_after
        ):
            self.mutation_violations.append(
                f"predict() mutated learner state on {ex.id}"
            )
        return pred


def eval_split_items(learner: Learner, items: Sequence[Example]) -> float:
    if not items:
        return 0.0
    preds = [learner.predict(ex.scene, ex.tokens) for ex in items]
    return accuracy(preds, items)


def evaluate_learner_fast(
    learner_factory: Callable[[], Learner],
    ds: Dataset,
    counter: EvalCounter,
    exp_name: str,
    train_items_override: Sequence[Example] | None = None,
) -> dict[str, float]:
    """Train once and evaluate across all splits without intermediate curve rebuilds."""
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
    res["test_novel_combination"] = eval_split_items(
        learner, ds.tests["test_novel_combination"].items
    )
    res["test_longer"] = eval_split_items(learner, ds.tests["test_longer"].items)

    if isinstance(learner, NEXUSLearner):
        # Fast snapshot/restore without copying _eval_cache
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

        # Restore to post-train state for contradiction split
        del learner._observations[n_obs:]
        del learner.experience_log_jsonl[n_log:]
        learner._word_order = saved_word_order
        learner._obs_by_id = saved_obs_by_id
        learner._beam_dirty = True

        ct_split = ds.tests["test_contradiction"]
        cp_map = dict(ct_split.support_checkpoints)
        after_mis_idx = cp_map.get("after_misleading", 0)
        for ev in ct_split.support[:after_mis_idx]:
            if ev.kind == "observe" and ev.observation is not None:
                learner.observe(ev.observation)
            elif ev.kind == "correct" and ev.correction is not None:
                learner.correct(ev.correction)
        res["test_contradiction_after_misleading"] = eval_split_items(
            learner, ct_split.items
        )
        for ev in ct_split.support[after_mis_idx:]:
            if ev.kind == "observe" and ev.observation is not None:
                learner.observe(ev.observation)
            elif ev.kind == "correct" and ev.correction is not None:
                learner.correct(ev.correction)
        res["test_contradiction"] = eval_split_items(learner, ct_split.items)
    else:
        nw_split = ds.tests["test_novel_word_in_known_frame"]
        fs_learner = copy.deepcopy(learner)
        for ev in nw_split.support:
            if ev.kind == "observe" and ev.observation is not None:
                fs_learner.observe(ev.observation)
            elif ev.kind == "correct" and ev.correction is not None:
                fs_learner.correct(ev.correction)
        res["test_novel_word_in_known_frame"] = eval_split_items(
            fs_learner, nw_split.items
        )

        ct_split = ds.tests["test_contradiction"]
        ct_learner = copy.deepcopy(learner)
        cp_map = dict(ct_split.support_checkpoints)
        after_mis_idx = cp_map.get("after_misleading", 0)
        for ev in ct_split.support[:after_mis_idx]:
            if ev.kind == "observe" and ev.observation is not None:
                ct_learner.observe(ev.observation)
            elif ev.kind == "correct" and ev.correction is not None:
                ct_learner.correct(ev.correction)
        res["test_contradiction_after_misleading"] = eval_split_items(
            ct_learner, ct_split.items
        )
        for ev in ct_split.support[after_mis_idx:]:
            if ev.kind == "observe" and ev.observation is not None:
                ct_learner.observe(ev.observation)
            elif ev.kind == "correct" and ev.correction is not None:
                ct_learner.correct(ev.correction)
        res["test_contradiction"] = eval_split_items(ct_learner, ct_split.items)

    counter.record(exp_name, num_test_splits=4)
    return res


def summarize_seed_dicts(
    seed_dicts: Sequence[dict[str, float]]
) -> dict[str, dict[str, float]]:
    keys = list(seed_dicts[0].keys()) if seed_dicts else []
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = np.array([d[k] for d in seed_dicts], dtype=np.float64)
        out[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return out


# ----------------------------------------------------------------------
# Part 1: Leakage, Protocol, and Twin-Pairing Audit
# ----------------------------------------------------------------------


def run_part1_audit(
    datasets: Sequence[Dataset], counter: EvalCounter
) -> dict[str, Any]:
    learner_dir = Path(__file__).resolve().parents[1] / "learner"
    accessed_fields_by_var: dict[str, set[str]] = {
        "observation": set(),
        "correction": set(),
        "scene": set(),
        "object": set(),
        "relation": set(),
    }
    forbidden_fields = {
        "lf",
        "construction",
        "contrast_group",
        "tags",
        "ground_truth",
        "mislabelled",
        "plan",
    }
    forbidden_hits: list[str] = []

    for py_file in sorted(learner_dir.glob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                attr = node.attr
                if attr in forbidden_fields:
                    forbidden_hits.append(f"{py_file.name}:{node.lineno}:{attr}")
                if isinstance(node.value, ast.Name):
                    vname = node.value.id
                    if vname in ("observation", "obs", "o"):
                        if attr in (
                            "example_id",
                            "scene",
                            "tokens",
                            "label",
                            "source",
                            "corrected",
                        ):
                            accessed_fields_by_var["observation"].add(attr)
                        elif attr in ("id", "kind", "color", "size", "value"):
                            accessed_fields_by_var["object"].add(attr)
                    elif vname in ("correction", "corr"):
                        accessed_fields_by_var["correction"].add(attr)
                    elif vname in ("scene",):
                        accessed_fields_by_var["scene"].add(attr)
                    elif vname in ("obj",):
                        accessed_fields_by_var["object"].add(attr)
                    elif vname in ("rel", "r"):
                        accessed_fields_by_var["relation"].add(attr)
                elif isinstance(node.value, ast.Attribute) and node.value.attr == "scene":
                    accessed_fields_by_var["scene"].add(attr)

    total_predict_calls = 0
    total_observe_calls = 0
    leak_violations: list[str] = []
    mutation_violations: list[str] = []

    for seed, ds in enumerate(datasets):
        spy = SpyLearnerWrapper(NEXUSLearner(seed=seed))
        for ex in ds.train.items:
            spy.observe(ex.observation())

        for split in (
            ds.dev,
            ds.tests["test_novel_combination"],
            ds.tests["test_longer"],
        ):
            for ex in split.items:
                spy.predict_for_example(ex)

        nw_split = ds.tests["test_novel_word_in_known_frame"]
        for ev in nw_split.support:
            if ev.kind == "observe" and ev.observation is not None:
                spy.observe(ev.observation)
        for ex in nw_split.items:
            spy.predict_for_example(ex)

        ct_split = ds.tests["test_contradiction"]
        for ev in ct_split.support:
            if ev.kind == "observe" and ev.observation is not None:
                spy.observe(ev.observation)
            elif ev.kind == "correct" and ev.correction is not None:
                spy.correct(ev.correction)
        for ex in ct_split.items:
            spy.predict_for_example(ex)

        total_predict_calls += spy.predict_calls
        total_observe_calls += spy.observe_calls
        leak_violations.extend(spy.leak_violations)
        mutation_violations.extend(spy.mutation_violations)
        counter.record("part1_spy_audit", num_test_splits=4)

    # Shuffled training order check (separating adjacent true/false twins)
    shuffled_order_results: list[dict[str, float]] = []
    for seed, ds in enumerate(datasets):
        rng = np.random.default_rng(seed + 9000)
        perm = rng.permutation(len(ds.train.items))
        shuffled_train = [ds.train.items[int(i)] for i in perm]
        res = evaluate_learner_fast(
            lambda s=seed: NEXUSLearner(seed=s),
            ds,
            counter,
            "part1_shuffled_train_order",
            train_items_override=shuffled_train,
        )
        shuffled_order_results.append(res)

    return {
        "accessed_fields": {k: sorted(v) for k, v in accessed_fields_by_var.items()},
        "forbidden_generator_field_hits": forbidden_hits,
        "spy_total_observe_calls": total_observe_calls,
        "spy_total_predict_calls": total_predict_calls,
        "spy_leak_violations": len(leak_violations),
        "spy_mutation_violations": len(mutation_violations),
        "shuffled_train_order_summary": summarize_seed_dicts(shuffled_order_results),
    }


# ----------------------------------------------------------------------
# Part 2: Negative Controls (2a, 2b, 2c)
# ----------------------------------------------------------------------


def run_part2_negative_controls(
    datasets: Sequence[Dataset], counter: EvalCounter
) -> dict[str, Any]:
    res_2a: list[dict[str, float]] = []
    res_2b: list[dict[str, float]] = []
    res_2c: list[dict[str, float]] = []

    n_seeds = len(datasets)
    for seed, ds in enumerate(datasets):
        rng = np.random.default_rng(seed + 1000)

        # 2a: Shuffle truth labels in training
        labels = [ex.truth for ex in ds.train.items]
        perm_l = rng.permutation(len(labels))
        train_2a = [
            dataclasses.replace(ex, truth=bool(labels[int(perm_l[i])]))
            for i, ex in enumerate(ds.train.items)
        ]
        res_2a.append(
            evaluate_learner_fast(
                lambda s=seed: NEXUSLearner(seed=s),
                ds,
                counter,
                "part2a_shuffled_labels",
                train_items_override=train_2a,
            )
        )

        # 2b: Shuffle scenes between training examples
        scenes = [ex.scene for ex in ds.train.items]
        perm_s = rng.permutation(len(scenes))
        train_2b = [
            dataclasses.replace(ex, scene=scenes[int(perm_s[i])])
            for i, ex in enumerate(ds.train.items)
        ]
        res_2b.append(
            evaluate_learner_fast(
                lambda s=seed: NEXUSLearner(seed=s),
                ds,
                counter,
                "part2b_shuffled_scenes",
                train_items_override=train_2b,
            )
        )

        # 2c: Train on next seed's lexicon/world, test on this seed's held-out splits
        other_ds = datasets[(seed + 1) % n_seeds]
        res_2c.append(
            evaluate_learner_fast(
                lambda s=seed: NEXUSLearner(seed=s),
                ds,
                counter,
                "part2c_cross_seed_lexicon",
                train_items_override=other_ds.train.items,
            )
        )

    return {
        "2a_shuffled_labels": summarize_seed_dicts(res_2a),
        "2b_shuffled_scenes": summarize_seed_dicts(res_2b),
        "2c_cross_seed_lexicon": summarize_seed_dicts(res_2c),
    }


# ----------------------------------------------------------------------
# Part 3: Difficulty Ladder (Find the Breaking Point)
# ----------------------------------------------------------------------


def _stratified_subsample(
    items: Sequence[Example], n: int, rng: np.random.Generator
) -> list[Example]:
    """Select ``n`` training items stratified across constructions."""
    if n >= len(items):
        return list(items)
    by_c: dict[str, list[Example]] = {}
    for ex in items:
        by_c.setdefault(ex.construction, []).append(ex)
    constructions = ["np_assert", "np_ask", "trans_assert", "trans_ask"]
    buckets = {
        c: [by_c[c][int(i)] for i in rng.permutation(len(by_c.get(c, [])))]
        for c in constructions
        if c in by_c
    }
    ptr = {c: 0 for c in buckets}
    selected: list[Example] = []
    while len(selected) < n:
        progressed = False
        for c in constructions:
            if c not in buckets or len(selected) >= n:
                continue
            if ptr[c] < len(buckets[c]):
                selected.append(buckets[c][ptr[c]])
                ptr[c] += 1
                progressed = True
        if not progressed:
            break
    return selected


def _replace_negatives_with_random_scenes(
    items: Sequence[Example], config: WorldConfig, seed: int
) -> list[Example]:
    """Replace paired near-miss False scenes with independent random falsifying scenes."""
    rng = random.Random(seed + 32000)
    out: list[Example] = []
    for ex in items:
        if ex.truth:
            out.append(ex)
        else:
            rand_scene = ex.scene
            for _ in range(250):
                cand = sample_scene(rng, config.ontology, config.scene)
                if not evaluate(ex.lf, cand, config.ontology.symmetric_relations):
                    rand_scene = cand
                    break
            out.append(dataclasses.replace(ex, scene=rand_scene, contrast_group=None))
    return out


def _make_scaled_world_config(base: WorldConfig, scale: int) -> WorldConfig:
    if scale == 1:
        return base
    ont = base.ontology
    new_kinds = tuple(f"{k}_{i}" for i in range(scale) for k in ont.kinds)
    new_colors = tuple(f"{c}_{i}" for i in range(scale) for c in ont.colors)
    new_sizes = tuple(f"{s}_{i}" for i in range(scale) for s in ont.sizes)
    new_ont = Ontology(
        kinds=new_kinds,
        colors=new_colors,
        sizes=new_sizes,
        relations=ont.relations,
    )
    new_lex: dict[str, str] = {}
    for i in range(scale):
        for k in ont.kinds:
            w = base.grammar.lexicon[f"kind:{k}"]
            new_lex[f"kind:{k}_{i}"] = w if i == 0 else f"{w}{i}"
        for c in ont.colors:
            w = base.grammar.lexicon[f"color:{c}"]
            new_lex[f"color:{c}_{i}"] = w if i == 0 else f"{w}{i}"
        for s in ont.sizes:
            w = base.grammar.lexicon[f"size:{s}"]
            new_lex[f"size:{s}_{i}"] = w if i == 0 else f"{w}{i}"
    for r in ont.relation_names:
        new_lex[f"rel:{r}"] = base.grammar.lexicon[f"rel:{r}"]

    new_grammar = dataclasses.replace(base.grammar, lexicon=new_lex)
    new_splits = dataclasses.replace(
        base.splits,
        n_train=400,
        n_dev_per_construction=20,
        n_test_novel_combination=80,
        n_test_novel_word_queries=40,
        n_test_longer=80,
        n_contradiction_probes=40,
    )
    return dataclasses.replace(
        base, ontology=new_ont, grammar=new_grammar, splits=new_splits
    )


def _apply_token_mapping_to_dataset(
    ds: Dataset, token_map: dict[str, str]
) -> Dataset:
    """Map surface words via ``token_map`` (used to create homonyms/polysemy)."""

    def map_toks(toks: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(token_map.get(t, t) for t in toks)

    def map_ex(ex: Example) -> Example:
        return dataclasses.replace(ex, tokens=map_toks(ex.tokens))

    def map_ev(ev: StreamEvent) -> StreamEvent:
        if ev.kind == "observe" and ev.observation is not None:
            new_obs = dataclasses.replace(
                ev.observation, tokens=map_toks(ev.observation.tokens)
            )
            new_gt = map_ex(ev.ground_truth) if ev.ground_truth is not None else None
            return dataclasses.replace(ev, observation=new_obs, ground_truth=new_gt)
        return ev

    def map_split(sp: Split) -> Split:
        return dataclasses.replace(
            sp,
            items=tuple(map_ex(e) for e in sp.items),
            support=tuple(map_ev(ev) for ev in sp.support),
        )

    return dataclasses.replace(
        ds,
        train=map_split(ds.train),
        dev=map_split(ds.dev),
        tests={k: map_split(v) for k, v in ds.tests.items()},
    )


def _apply_synonym_splitting_to_dataset(
    ds: Dataset, concepts_to_split: Sequence[str], seed: int
) -> Dataset:
    """Give each concept in ``concepts_to_split`` two surface forms (w and w_syn2)."""
    rng = np.random.default_rng(seed + 66000)
    assert ds.grammar is not None
    word_to_syn = {
        ds.grammar.lexicon[c]: f"{ds.grammar.lexicon[c]}_syn2"
        for c in concepts_to_split
        if c in ds.grammar.lexicon
    }

    def map_toks(toks: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            (
                word_to_syn[t]
                if (t in word_to_syn and float(rng.random()) < 0.5)
                else t
            )
            for t in toks
        )

    def map_ex(ex: Example) -> Example:
        return dataclasses.replace(ex, tokens=map_toks(ex.tokens))

    def map_ev(ev: StreamEvent) -> StreamEvent:
        if ev.kind == "observe" and ev.observation is not None:
            new_obs = dataclasses.replace(
                ev.observation, tokens=map_toks(ev.observation.tokens)
            )
            new_gt = map_ex(ev.ground_truth) if ev.ground_truth is not None else None
            return dataclasses.replace(ev, observation=new_obs, ground_truth=new_gt)
        return ev

    def map_split(sp: Split) -> Split:
        return dataclasses.replace(
            sp,
            items=tuple(map_ex(e) for e in sp.items),
            support=tuple(map_ev(ev) for ev in sp.support),
        )

    return dataclasses.replace(
        ds,
        train=map_split(ds.train),
        dev=map_split(ds.dev),
        tests={k: map_split(v) for k, v in ds.tests.items()},
    )


def _evaluate_beyond_depth_suites(
    learner: NEXUSLearner,
    ds: Dataset,
    config: WorldConfig,
    seed: int,
) -> dict[str, float]:
    """Evaluate trained NEXUSLearner on 3 structural depth tiers:
    1) depth_2adj_diff_family: standard 2-adj NPs (size + color + kind, test_longer)
    2) depth_3adj_same_family: 3-adj NPs with 2 colors + 1 size + kind (sortal clash)
    3) depth_2clause_chain: 2-clause relational chains (NP1 rel1 NP2 rel2 NP3)
    """
    rng = np.random.default_rng(seed + 77000)
    py_rng = random.Random(seed + 77000)
    assert ds.grammar is not None
    lex = ds.grammar.lexicon
    kinds = [k for k in config.ontology.kinds if k != ds.plan.novel_kind]
    colors = config.ontology.colors
    sizes = config.ontology.sizes
    rels = config.ontology.relation_names

    longer_items = ds.tests["test_longer"].items
    acc_2adj = eval_split_items(learner, longer_items)
    top_h = learner.top_hypothesis()
    entries = top_h.entries if top_h is not None else {}
    parse_rate_2adj = (
        sum(parse_tokens(ex.tokens, entries) is not None for ex in longer_items)
        / len(longer_items)
    )

    # 2. depth_3adj_same_family: 50 balanced examples where an NP has 2 colors + 1 size + kind
    n_idx = ds.grammar.np_order.index("noun")
    c_idx = ds.grammar.np_order.index("color")
    adj_before = c_idx < n_idx
    hits_3adj = 0
    parsed_3adj = 0
    for i in range(50):
        k = str(rng.choice(kinds))
        c1, c2 = [str(x) for x in rng.choice(colors, size=2, replace=False)]
        sz = str(rng.choice(sizes))
        w_k = lex[f"kind:{k}"]
        w_c1 = lex[f"color:{c1}"]
        w_c2 = lex[f"color:{c2}"]
        w_sz = lex[f"size:{sz}"]
        toks = (w_sz, w_c1, w_c2, w_k) if adj_before else (w_k, w_c1, w_c2, w_sz)
        scene = sample_scene(py_rng, config.ontology, config.scene)
        target_label = i % 2 == 0
        if parse_tokens(toks, entries) is not None:
            parsed_3adj += 1
        pred = learner.predict(scene, toks)
        if pred.truth == target_label:
            hits_3adj += 1
    acc_3adj = hits_3adj / 50.0
    parse_rate_3adj = parsed_3adj / 50.0

    # 3. depth_2clause_chain: 50 balanced examples of 2-relation chains (NP1 rel1 NP2 rel2 NP3)
    tr_tpl = ds.grammar.constructions["trans_assert"].template
    v_idx = tr_tpl.index("VERB")
    hits_chain = 0
    parsed_chain = 0
    for i in range(50):
        k1, k2, k3 = [str(x) for x in rng.choice(kinds, size=3, replace=False)]
        r1 = str(rng.choice(rels))
        r2 = str(rng.choice(rels))
        w_k1, w_k2, w_k3 = lex[f"kind:{k1}"], lex[f"kind:{k2}"], lex[f"kind:{k3}"]
        w_r1, w_r2 = lex[f"rel:{r1}"], lex[f"rel:{r2}"]
        if v_idx == 1:  # SVO
            toks = (w_k1, w_r1, w_k2, w_r2, w_k3)
        elif v_idx == 2:  # SOV
            toks = (w_k1, w_k2, w_r1, w_k3, w_r2)
        else:  # VSO
            toks = (w_r1, w_k1, w_k2, w_r2, w_k3)

        target_label = i % 2 == 0
        attrs = [
            {"kind": k1, "color": str(rng.choice(colors)), "size": str(rng.choice(sizes))},
            {"kind": k2, "color": str(rng.choice(colors)), "size": str(rng.choice(sizes))},
            {"kind": k3, "color": str(rng.choice(colors)), "size": str(rng.choice(sizes))},
        ]
        s_rels = [(r1, 0, 1), (r2, 1, 2)] if target_label else [(r1, 0, 1)]
        scene = build_scene(attrs, s_rels)
        if parse_tokens(toks, entries) is not None:
            parsed_chain += 1
        pred = learner.predict(scene, toks)
        if pred.truth == target_label:
            hits_chain += 1

    acc_chain = hits_chain / 50.0
    parse_rate_chain = parsed_chain / 50.0

    return {
        "depth_2adj_diff_family_acc": acc_2adj,
        "depth_2adj_diff_family_parse_rate": parse_rate_2adj,
        "depth_3adj_same_family_acc": acc_3adj,
        "depth_3adj_same_family_parse_rate": parse_rate_3adj,
        "depth_2clause_chain_acc": acc_chain,
        "depth_2clause_chain_parse_rate": parse_rate_chain,
    }


def run_part3_difficulty_ladder(
    base_config: WorldConfig,
    datasets: Sequence[Dataset],
    counter: EvalCounter,
) -> dict[str, Any]:
    # 3.1 Training set size N = 10, 20, 50, 100, 200, 400 (plus full N=1600 reference)
    sample_sizes = (10, 20, 50, 100, 200, 400, 1600)
    ladder_3_1_stratified: dict[str, Any] = {}
    ladder_3_1_prefix: dict[str, Any] = {}

    for n in sample_sizes:
        s_runs: list[dict[str, float]] = []
        p_runs: list[dict[str, float]] = []
        for seed, ds in enumerate(datasets):
            rng = np.random.default_rng(seed + 31000 + n)
            strat_items = _stratified_subsample(ds.train.items, n, rng)
            s_runs.append(
                evaluate_learner_fast(
                    lambda s=seed: NEXUSLearner(seed=s),
                    ds,
                    counter,
                    f"part3_1_stratified_N{n}",
                    train_items_override=strat_items,
                )
            )
            prefix_items = ds.train.items[:n]
            p_runs.append(
                evaluate_learner_fast(
                    lambda s=seed: NEXUSLearner(seed=s),
                    ds,
                    counter,
                    f"part3_1_prefix_N{n}",
                    train_items_override=prefix_items,
                )
            )
        ladder_3_1_stratified[f"N={n}"] = summarize_seed_dicts(s_runs)
        ladder_3_1_prefix[f"N={n}"] = summarize_seed_dicts(p_runs)

    # 3.2 Remove paired-contrast examples from training (random negatives only, evaluated at N=200, 400, 1600)
    runs_3_2_N200: list[dict[str, float]] = []
    runs_3_2_N400: list[dict[str, float]] = []
    runs_3_2_N1600: list[dict[str, float]] = []
    for seed, ds in enumerate(datasets):
        rand_neg_all = _replace_negatives_with_random_scenes(
            ds.train.items, base_config, seed
        )
        rng200 = np.random.default_rng(seed + 32200)
        rng400 = np.random.default_rng(seed + 32400)
        runs_3_2_N200.append(
            evaluate_learner_fast(
                lambda s=seed: NEXUSLearner(seed=s),
                ds,
                counter,
                "part3_2_random_negatives_N200",
                train_items_override=_stratified_subsample(rand_neg_all, 200, rng200),
            )
        )
        runs_3_2_N400.append(
            evaluate_learner_fast(
                lambda s=seed: NEXUSLearner(seed=s),
                ds,
                counter,
                "part3_2_random_negatives_N400",
                train_items_override=_stratified_subsample(rand_neg_all, 400, rng400),
            )
        )
        runs_3_2_N1600.append(
            evaluate_learner_fast(
                lambda s=seed: NEXUSLearner(seed=s),
                ds,
                counter,
                "part3_2_random_negatives_N1600",
                train_items_override=rand_neg_all,
            )
        )

    # 3.3 Positive-only training (no False examples in training)
    runs_3_3: list[dict[str, float]] = []
    for seed, ds in enumerate(datasets):
        pos_train = [ex for ex in ds.train.items if ex.truth]
        runs_3_3.append(
            evaluate_learner_fast(
                lambda s=seed: NEXUSLearner(seed=s),
                ds,
                counter,
                "part3_3_positive_only",
                train_items_override=pos_train,
            )
        )

    # 3.4 Label noise: 0, 2, 10, 20, 30 percent (evaluated at both N=400 and full N=1600)
    noise_levels = (0.0, 0.02, 0.10, 0.20, 0.30)
    ladder_3_4_N1600: dict[str, Any] = {}
    ladder_3_4_N200: dict[str, Any] = {}
    for eta in noise_levels:
        n_runs_1600: list[dict[str, float]] = []
        n_runs_200: list[dict[str, float]] = []
        for seed, ds in enumerate(datasets):
            rng = np.random.default_rng(seed + 34000 + int(eta * 1000))
            n_flip = int(round(eta * len(ds.train.items)))
            flip_set = (
                set(
                    int(i)
                    for i in rng.choice(
                        len(ds.train.items), size=n_flip, replace=False
                    )
                )
                if n_flip > 0
                else set()
            )
            noisy_train = [
                dataclasses.replace(
                    ex, truth=(not ex.truth) if idx in flip_set else ex.truth
                )
                for idx, ex in enumerate(ds.train.items)
            ]
            n_runs_1600.append(
                evaluate_learner_fast(
                    lambda s=seed: NEXUSLearner(seed=s),
                    ds,
                    counter,
                    f"part3_4_noise_{int(eta*100)}_N1600",
                    train_items_override=noisy_train,
                )
            )
            rng_sub = np.random.default_rng(seed + 34500 + int(eta * 1000))
            sub_noisy = _stratified_subsample(noisy_train, 200, rng_sub)
            n_runs_200.append(
                evaluate_learner_fast(
                    lambda s=seed: NEXUSLearner(seed=s),
                    ds,
                    counter,
                    f"part3_4_noise_{int(eta*100)}_N200",
                    train_items_override=sub_noisy,
                )
            )
        ladder_3_4_N1600[f"noise_{int(eta*100)}pct"] = summarize_seed_dicts(n_runs_1600)
        ladder_3_4_N200[f"noise_{int(eta*100)}pct"] = summarize_seed_dicts(n_runs_200)

    # 3.5 Lexicon size: 6, 12, 24, 48 kinds/words with proportionally scaled attributes (at fixed N=400 budget)
    ladder_3_5_N400: dict[str, Any] = {}
    for scale, n_kinds in ((1, 6), (2, 12), (4, 24), (8, 48)):
        l_runs_400: list[dict[str, float]] = []
        scaled_cfg = _make_scaled_world_config(base_config, scale)
        for seed in range(len(datasets)):
            if scale == 1:
                ds_scaled = datasets[seed]
                rng400 = np.random.default_rng(seed + 35000 + n_kinds)
                sub400 = _stratified_subsample(ds_scaled.train.items, 400, rng400)
                l_runs_400.append(
                    evaluate_learner_fast(
                        lambda s=seed: NEXUSLearner(seed=s),
                        ds_scaled,
                        counter,
                        f"part3_5_kinds_{n_kinds}_N400",
                        train_items_override=sub400,
                    )
                )
            else:
                ds_scaled = build_dataset(
                    scaled_cfg, derive_seed(seed, f"world_s{scale}"), world_mode="randomized"
                )
                l_runs_400.append(
                    evaluate_learner_fast(
                        lambda s=seed: NEXUSLearner(seed=s),
                        ds_scaled,
                        counter,
                        f"part3_5_kinds_{n_kinds}_N400",
                    )
                )
        ladder_3_5_N400[f"kinds_{n_kinds}"] = summarize_seed_dicts(l_runs_400)

    # 3.6 Ambiguous words (homonymy/polysemy) and Synonyms
    runs_homonym_1: list[dict[str, float]] = []
    runs_homonym_3: list[dict[str, float]] = []
    runs_synonym_nouns: list[dict[str, float]] = []
    runs_synonym_all_N1600: list[dict[str, float]] = []
    runs_synonym_all_N200: list[dict[str, float]] = []

    for seed, ds in enumerate(datasets):
        assert ds.grammar is not None
        lex = ds.grammar.lexicon
        # Homonym 1: size:small shares surface word with color:red
        map_h1 = {lex["size:small"]: lex["color:red"]}
        ds_h1 = _apply_token_mapping_to_dataset(ds, map_h1)
        runs_homonym_1.append(
            evaluate_learner_fast(
                lambda s=seed: NEXUSLearner(seed=s),
                ds_h1,
                counter,
                "part3_6_homonym_1pair",
            )
        )

        # Homonym 3: all 3 size words share surface forms with 3 color words
        map_h3 = {
            lex["size:small"]: lex["color:red"],
            lex["size:medium"]: lex["color:blue"],
            lex["size:large"]: lex["color:green"],
        }
        ds_h3 = _apply_token_mapping_to_dataset(ds, map_h3)
        runs_homonym_3.append(
            evaluate_learner_fast(
                lambda s=seed: NEXUSLearner(seed=s),
                ds_h3,
                counter,
                "part3_6_homonym_3pairs",
            )
        )

        # Synonyms: 2 forms per kind
        kind_concepts = [f"kind:{k}" for k in base_config.ontology.kinds]
        ds_syn_n = _apply_synonym_splitting_to_dataset(ds, kind_concepts, seed)
        runs_synonym_nouns.append(
            evaluate_learner_fast(
                lambda s=seed: NEXUSLearner(seed=s),
                ds_syn_n,
                counter,
                "part3_6_synonyms_nouns",
            )
        )

        # Synonyms: 2 forms for every content concept (kinds, colors, sizes, relations)
        all_content_concepts = list(lex.keys())
        ds_syn_all = _apply_synonym_splitting_to_dataset(
            ds, all_content_concepts, seed + 500
        )
        runs_synonym_all_N1600.append(
            evaluate_learner_fast(
                lambda s=seed: NEXUSLearner(seed=s),
                ds_syn_all,
                counter,
                "part3_6_synonyms_all_N1600",
            )
        )
        rng_syn200 = np.random.default_rng(seed + 36200)
        runs_synonym_all_N200.append(
            evaluate_learner_fast(
                lambda s=seed: NEXUSLearner(seed=s),
                ds_syn_all,
                counter,
                "part3_6_synonyms_all_N200",
                train_items_override=_stratified_subsample(
                    ds_syn_all.train.items, 200, rng_syn200
                ),
            )
        )

    # 3.7 Longer sentences and nested modification beyond training depth
    runs_3_7: list[dict[str, float]] = []
    for seed, ds in enumerate(datasets):
        learner = NEXUSLearner(seed=seed)
        for ex in ds.train.items:
            learner.observe(ex.observation())
        runs_3_7.append(_evaluate_beyond_depth_suites(learner, ds, base_config, seed))
        counter.record("part3_7_beyond_depth", num_test_splits=3)

    # 3.8 Reduce beam K = 1, 2, 5, 10, 50, 200 (both at full N=1600 and low-sample N=100)
    ladder_3_8_N1600: dict[str, Any] = {}
    ladder_3_8_N100: dict[str, Any] = {}
    for k in (1, 2, 5, 10, 50, 200):
        k_runs_1600: list[dict[str, float]] = []
        k_runs_100: list[dict[str, float]] = []
        for seed, ds in enumerate(datasets):
            k_runs_1600.append(
                evaluate_learner_fast(
                    lambda s=seed, bw=k: NEXUSLearner(seed=s, beam_width=bw),
                    ds,
                    counter,
                    f"part3_8_beam_K{k}_N1600",
                )
            )
            rng100 = np.random.default_rng(seed + 38100 + k)
            sub100 = _stratified_subsample(ds.train.items, 100, rng100)
            k_runs_100.append(
                evaluate_learner_fast(
                    lambda s=seed, bw=k: NEXUSLearner(seed=s, beam_width=bw),
                    ds,
                    counter,
                    f"part3_8_beam_K{k}_N100",
                    train_items_override=sub100,
                )
            )
        ladder_3_8_N1600[f"K={k}"] = summarize_seed_dicts(k_runs_1600)
        ladder_3_8_N100[f"K={k}"] = summarize_seed_dicts(k_runs_100)

    return {
        "3_1_sample_size_stratified": ladder_3_1_stratified,
        "3_1_sample_size_prefix": ladder_3_1_prefix,
        "3_2_random_negatives_only": {
            "N=200": summarize_seed_dicts(runs_3_2_N200),
            "N=400": summarize_seed_dicts(runs_3_2_N400),
            "N=1600": summarize_seed_dicts(runs_3_2_N1600),
        },
        "3_3_positive_only": summarize_seed_dicts(runs_3_3),
        "3_4_label_noise_N1600": ladder_3_4_N1600,
        "3_4_label_noise_N200": ladder_3_4_N200,
        "3_5_lexicon_scaling_N400": ladder_3_5_N400,
        "3_6_ambiguity_and_synonyms": {
            "homonym_1_pair": summarize_seed_dicts(runs_homonym_1),
            "homonym_3_pairs": summarize_seed_dicts(runs_homonym_3),
            "synonyms_nouns_2x": summarize_seed_dicts(runs_synonym_nouns),
            "synonyms_all_2x_N1600": summarize_seed_dicts(runs_synonym_all_N1600),
            "synonyms_all_2x_N200": summarize_seed_dicts(runs_synonym_all_N200),
        },
        "3_7_beyond_training_depth": summarize_seed_dicts(runs_3_7),
        "3_8_beam_width_N1600": ladder_3_8_N1600,
        "3_8_beam_width_N100": ladder_3_8_N100,
    }


# ----------------------------------------------------------------------
# Part 4: Stronger Baselines (Baselines A, B, C-legacy, C-fixed, D, E, F)
# ----------------------------------------------------------------------


def _true_syntax_and_lexicon(ds: Dataset) -> tuple[dict[str, str], dict[str, str]]:
    assert ds.grammar is not None
    lex = ds.grammar.lexicon
    np_order = ds.grammar.np_order
    n_idx = np_order.index("noun")
    c_idx = np_order.index("color")
    s_idx = np_order.index("size")
    cat_color = "N/N" if c_idx < n_idx else "N\\N"
    cat_size = "N/N" if s_idx < n_idx else "N\\N"

    tr_tpl = ds.grammar.constructions["trans_assert"].template
    v_idx = tr_tpl.index("VERB")
    if v_idx == 2:
        cat_verb = "(S\\NP)\\NP"
    elif v_idx == 1:
        cat_verb = "(S\\NP)/NP"
    else:
        cat_verb = "(S/NP)/NP"

    ask_tpl = ds.grammar.constructions["np_ask"].template
    cat_q = "S/S" if ask_tpl[0] == "Q" else "S\\S"

    given_syntax: dict[str, str] = {ds.grammar.question_particle: cat_q}
    given_lexicon: dict[str, str] = {ds.grammar.question_particle: "Q"}
    for concept, word in lex.items():
        fam = concept.split(":", 1)[0]
        if fam == "kind":
            given_syntax[word] = "N"
            given_lexicon[word] = concept
        elif fam == "color":
            given_syntax[word] = cat_color
            given_lexicon[word] = concept
        elif fam == "size":
            given_syntax[word] = cat_size
            given_lexicon[word] = concept
        elif fam == "rel":
            given_syntax[word] = cat_verb
            given_lexicon[word] = concept
    return given_syntax, given_lexicon


def run_part4_baselines(
    datasets: Sequence[Dataset], counter: EvalCounter
) -> dict[str, Any]:
    runs_a: list[dict[str, float]] = []
    runs_b: list[dict[str, float]] = []
    runs_c_legacy: list[dict[str, float]] = []
    runs_c_fixed: list[dict[str, float]] = []
    runs_d: list[dict[str, float]] = []
    runs_e: list[dict[str, float]] = []
    runs_f_syntax: list[dict[str, float]] = []
    runs_f_lexicon: list[dict[str, float]] = []

    for seed, ds in enumerate(datasets):
        given_syn, given_lex = _true_syntax_and_lexicon(ds)
        runs_a.append(
            evaluate_learner_fast(
                lambda s=seed: Memorizer(seed=s), ds, counter, "part4_baseline_A"
            )
        )
        runs_b.append(
            evaluate_learner_fast(
                lambda s=seed: CrossSituationalLearner(seed=s),
                ds,
                counter,
                "part4_baseline_B",
            )
        )
        runs_c_legacy.append(
            evaluate_learner_fast(
                lambda s=seed: NoRevisionLearner(seed=s, legacy_buggy_fallback=True),
                ds,
                counter,
                "part4_baseline_C_legacy",
            )
        )
        runs_c_fixed.append(
            evaluate_learner_fast(
                lambda s=seed: NoRevisionLearner(seed=s, legacy_buggy_fallback=False),
                ds,
                counter,
                "part4_baseline_C_fixed",
            )
        )
        runs_d.append(
            evaluate_learner_fast(
                lambda s=seed: PositionAwareCrossSituationalLearner(seed=s),
                ds,
                counter,
                "part4_baseline_D_position_aware_cs",
            )
        )
        runs_e.append(
            evaluate_learner_fast(
                lambda s=seed: FromScratchNeuralBaseline(seed=s),
                ds,
                counter,
                "part4_baseline_E_neural",
            )
        )
        runs_f_syntax.append(
            evaluate_learner_fast(
                lambda s=seed, gs=given_syn: NEXUSLearner(seed=s, given_syntax=gs),
                ds,
                counter,
                "part4_baseline_F_given_syntax",
            )
        )
        runs_f_lexicon.append(
            evaluate_learner_fast(
                lambda s=seed, gl=given_lex: NEXUSLearner(seed=s, given_lexicon=gl),
                ds,
                counter,
                "part4_baseline_F_given_lexicon",
            )
        )

    # Also evaluate F (given_syntax vs given_lexicon vs full NEXUS) at small sample sizes N=10, 20, 50, 100
    f_sample_ladder: dict[str, Any] = {}
    for n in (10, 20, 50, 100):
        f_full_n: list[dict[str, float]] = []
        f_syn_n: list[dict[str, float]] = []
        f_lex_n: list[dict[str, float]] = []
        for seed, ds in enumerate(datasets):
            rng = np.random.default_rng(seed + 31000 + n)
            sub_items = _stratified_subsample(ds.train.items, n, rng)
            given_syn, given_lex = _true_syntax_and_lexicon(ds)
            f_full_n.append(
                evaluate_learner_fast(
                    lambda s=seed: NEXUSLearner(seed=s),
                    ds,
                    counter,
                    f"part4_F_full_N{n}",
                    train_items_override=sub_items,
                )
            )
            f_syn_n.append(
                evaluate_learner_fast(
                    lambda s=seed, gs=given_syn: NEXUSLearner(seed=s, given_syntax=gs),
                    ds,
                    counter,
                    f"part4_F_given_syntax_N{n}",
                    train_items_override=sub_items,
                )
            )
            f_lex_n.append(
                evaluate_learner_fast(
                    lambda s=seed, gl=given_lex: NEXUSLearner(seed=s, given_lexicon=gl),
                    ds,
                    counter,
                    f"part4_F_given_lexicon_N{n}",
                    train_items_override=sub_items,
                )
            )
        f_sample_ladder[f"N={n}"] = {
            "nexus_full": summarize_seed_dicts(f_full_n),
            "given_syntax": summarize_seed_dicts(f_syn_n),
            "given_lexicon": summarize_seed_dicts(f_lex_n),
        }

    return {
        "A_memorizer": summarize_seed_dicts(runs_a),
        "B_cross_situational": summarize_seed_dicts(runs_b),
        "C_no_revision_legacy_buggy": summarize_seed_dicts(runs_c_legacy),
        "C_no_revision_fixed": summarize_seed_dicts(runs_c_fixed),
        "D_position_aware_cs": summarize_seed_dicts(runs_d),
        "E_neural_from_scratch": summarize_seed_dicts(runs_e),
        "F_given_syntax_N1600": summarize_seed_dicts(runs_f_syntax),
        "F_given_lexicon_N1600": summarize_seed_dicts(runs_f_lexicon),
        "F_small_sample_comparison": f_sample_ladder,
    }


# ----------------------------------------------------------------------
# Part 5: MDL Prior Audit (alpha=0.05 vs explicit log2 bit-cost prior)
# ----------------------------------------------------------------------


def run_part5_mdl_prior_audit(
    datasets: Sequence[Dataset], counter: EvalCounter
) -> dict[str, Any]:
    runs_linear_1600: list[dict[str, float]] = []
    runs_log2_nats_1600: list[dict[str, float]] = []
    runs_log2_bits_1600: list[dict[str, float]] = []
    runs_linear_50: list[dict[str, float]] = []
    runs_log2_nats_50: list[dict[str, float]] = []

    for seed, ds in enumerate(datasets):
        runs_linear_1600.append(
            evaluate_learner_fast(
                lambda s=seed: NEXUSLearner(seed=s, prior_mode="linear"),
                ds,
                counter,
                "part5_prior_linear_N1600",
            )
        )
        runs_log2_nats_1600.append(
            evaluate_learner_fast(
                lambda s=seed: NEXUSLearner(seed=s, prior_mode="log2_bits"),
                ds,
                counter,
                "part5_prior_log2_nats_N1600",
            )
        )
        runs_log2_bits_1600.append(
            evaluate_learner_fast(
                lambda s=seed: NEXUSLearner(seed=s, prior_mode="log2_raw_bits"),
                ds,
                counter,
                "part5_prior_log2_raw_bits_N1600",
            )
        )

        rng = np.random.default_rng(seed + 31000 + 50)
        sub_50 = _stratified_subsample(ds.train.items, 50, rng)
        runs_linear_50.append(
            evaluate_learner_fast(
                lambda s=seed: NEXUSLearner(seed=s, prior_mode="linear"),
                ds,
                counter,
                "part5_prior_linear_N50",
                train_items_override=sub_50,
            )
        )
        runs_log2_nats_50.append(
            evaluate_learner_fast(
                lambda s=seed: NEXUSLearner(seed=s, prior_mode="log2_bits"),
                ds,
                counter,
                "part5_prior_log2_nats_N50",
                train_items_override=sub_50,
            )
        )

    return {
        "N1600_linear_alpha_0_05": summarize_seed_dicts(runs_linear_1600),
        "N1600_log2_bit_cost_nats": summarize_seed_dicts(runs_log2_nats_1600),
        "N1600_log2_bit_cost_raw_bits": summarize_seed_dicts(runs_log2_bits_1600),
        "N50_linear_alpha_0_05": summarize_seed_dicts(runs_linear_50),
        "N50_log2_bit_cost_nats": summarize_seed_dicts(runs_log2_nats_50),
    }


def run_all_phase1_5(n_seeds: int = 20) -> dict[str, Any]:
    t0 = time.time()
    counter = EvalCounter()
    base_config = load_world_config()
    print(f"[Phase 1.5] Building datasets for {n_seeds} seeds (world_mode=randomized)...")
    datasets = [
        build_dataset(base_config, derive_seed(s, "world"), world_mode="randomized")
        for s in range(n_seeds)
    ]

    print("[Phase 1.5] Running Part 1: Leakage & Protocol Audit...")
    part1 = run_part1_audit(datasets, counter)

    print("[Phase 1.5] Running Part 2: Negative Controls (2a, 2b, 2c)...")
    part2 = run_part2_negative_controls(datasets, counter)

    print("[Phase 1.5] Running Part 3: Difficulty Ladder (3.1 - 3.8)...")
    part3 = run_part3_difficulty_ladder(base_config, datasets, counter)

    print("[Phase 1.5] Running Part 4: Stronger Baselines (A, B, C, D, E, F)...")
    part4 = run_part4_baselines(datasets, counter)

    print("[Phase 1.5] Running Part 5: MDL Prior Audit...")
    part5 = run_part5_mdl_prior_audit(datasets, counter)

    elapsed = time.time() - t0
    results = {
        "label": "EXPLORATORY",
        "phase": "1.5_adversarial_validation",
        "n_seeds": n_seeds,
        "world_mode": "randomized",
        "elapsed_sec": round(elapsed, 2),
        "eval_counter": {
            "total_test_split_evaluations": counter.test_split_evaluations,
            "by_experiment": counter.runs_by_experiment,
        },
        "part1_leakage_and_protocol_audit": part1,
        "part2_negative_controls": part2,
        "part3_difficulty_ladder": part3,
        "part4_stronger_baselines": part4,
        "part5_mdl_prior_audit": part5,
    }

    out_path = Path(__file__).resolve().parent / "results" / "phase1_5_adversarial.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[Phase 1.5] Complete in {elapsed:.1f}s. Total test split evaluations: "
        f"{counter.test_split_evaluations}. Saved to {out_path}"
    )
    return results


if __name__ == "__main__":
    run_all_phase1_5(20)
