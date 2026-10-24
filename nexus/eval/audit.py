"""Stage A (Phase 0.6) Harness & Statistical Audit Tools.

Implements programmatic checks for:
- A1: Split sizes ``n`` and 50-seed pooled Student-t confidence interval audit for
      ``SentenceOnly`` and ``SceneOnly``.
- A2: Feature-level label-balance audit (``P(feature|true)`` vs ``P(feature|false)``)
      across object count, relation count and graph topology, kind/color/size counts,
      novel-kind presence, token presence, and sentence length, with permutation tests
      and Holm-Bonferroni + Benjamini-Hochberg (BH) corrections.
- A3: ``test_novel_word_in_known_frame`` novel-kind balance, exact enumeration of
      candidate meanings consistent with the 2-example support stream in the Phase 1
      hypothesis space, and ideal-observer accuracy ceiling.
- A6: Word-order and construction identifiability audit across NP orders, clause orders,
      and question particle positions.
- A8: Exhaustive Bayesian ideal observer on a reduced world (2 kinds, 2 colors,
      1 relation, fixed order) small enough for exact enumeration.
"""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from nexus.baselines.scene_only import SceneOnly
from nexus.baselines.sentence_only import SentenceOnly
from nexus.eval.runner import predict_all
from nexus.eval.splits import Dataset, build_dataset, derive_seed
from nexus.world.config import RelationSpec, WorldConfig
from nexus.world.example import Example
from nexus.world.logic import Atom, LogicalForm, evaluate, make_lf
from nexus.world.scene import Scene, build_scene


# ---------------------------------------------------------------------------
# A1: Pooled 50-seed t-interval test for SentenceOnly and SceneOnly
# ---------------------------------------------------------------------------

def t_critical_95(df: int) -> float:
    """Two-sided 95% Student-t critical value for ``df`` degrees of freedom."""
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        9: 2.262, 10: 2.228, 19: 2.093, 20: 2.086, 29: 2.045,
        49: 2.0096, 50: 2.0086,
    }
    if df in table:
        return table[df]
    z = 1.959963984540054
    return z + (z**3 + z) / (4.0 * df) + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)


def pooled_shortcut_audit(
    config: WorldConfig,
    n_seeds: int = 50,
    base_seed: int = 0,
    world_mode: str = "randomized",
) -> dict[str, Any]:
    """Run SentenceOnly and SceneOnly across ``n_seeds`` seeds and compute 95% t-intervals."""
    split_names = [
        "train",
        "dev",
        "test_novel_combination",
        "test_novel_word_in_known_frame",
        "test_longer",
        "test_contradiction",
    ]
    accs: dict[str, dict[str, list[float]]] = {
        "sentence_only": defaultdict(list),
        "scene_only": defaultdict(list),
    }
    split_ns: dict[str, int] = {}

    for s in range(base_seed, base_seed + n_seeds):
        ds = build_dataset(config, seed=s, world_mode=world_mode)
        sent_m = SentenceOnly(seed=s)
        scene_m = SceneOnly(seed=s)
        for ex in ds.train.items:
            obs = ex.observation()
            sent_m.observe(obs)
            scene_m.observe(obs)

        all_splits = {"train": ds.train, "dev": ds.dev, **ds.tests}
        for sp_name in split_names:
            sp = all_splits[sp_name]
            split_ns[sp_name] = len(sp.items)
            for m_name, model in [("sentence_only", sent_m), ("scene_only", scene_m)]:
                preds = predict_all(model, sp.items)
                acc = sum(p.truth == e.truth for p, e in zip(preds, sp.items)) / len(sp.items)
                accs[m_name][sp_name].append(acc)

    df = max(1, n_seeds - 1)
    t_crit = t_critical_95(df)
    summary: dict[str, Any] = {"n_seeds": n_seeds, "split_n": split_ns, "models": {}}

    for m_name, sp_dict in accs.items():
        summary["models"][m_name] = {}
        for sp_name, vals in sp_dict.items():
            arr = np.array(vals, dtype=np.float64)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            se = std / math.sqrt(len(arr))
            ci_low = mean - t_crit * se
            ci_high = mean + t_crit * se
            contains_05 = (ci_low - 1e-12 <= 0.50 <= ci_high + 1e-12)
            summary["models"][m_name][sp_name] = {
                "n": split_ns[sp_name],
                "mean": mean,
                "std": std,
                "se": se,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "contains_05": contains_05,
            }
    return summary


# ---------------------------------------------------------------------------
# A2: Feature-level label-balance audit with permutation tests & Holm/BH
# ---------------------------------------------------------------------------

def scene_relation_topology(scene: Scene) -> str:
    """Classify the directed graph topology of ``scene.relations``."""
    rels = scene.relations
    if len(rels) == 0:
        return "topo:0"
    if len(rels) == 1:
        return "topo:1"
    r0, r1 = rels[0], rels[1]
    e0 = (r0.subject, r0.object)
    e1 = (r1.subject, r1.object)
    if e0 == e1:
        return "topo:2_parallel"
    if e0 == (e1[1], e1[0]):
        return "topo:2_cycle"
    if len({e0[0], e0[1], e1[0], e1[1]}) == 4:
        return "topo:2_disjoint"
    if e0[0] == e1[0]:
        return "topo:2_shared_src"
    if e0[1] == e1[1]:
        return "topo:2_shared_dst"
    return "topo:2_chain"


def extract_example_audit_features(ex: Example, ds: Dataset) -> dict[str, float]:
    """Extract all scene and sentence features required by Stage A2."""
    sc = ex.scene
    feats: dict[str, float] = {}
    n_objs = len(sc.objects)
    n_rels = len(sc.relations)
    feats["n_objs"] = float(n_objs)
    for k in (2, 3, 4):
        feats[f"n_objs=={k}"] = 1.0 if n_objs == k else 0.0
    feats["n_rels"] = float(n_rels)

    topo = scene_relation_topology(sc)
    for t_name in (
        "topo:0",
        "topo:1",
        "topo:2_parallel",
        "topo:2_cycle",
        "topo:2_chain",
        "topo:2_disjoint",
        "topo:2_shared_src",
        "topo:2_shared_dst",
    ):
        feats[t_name] = 1.0 if topo == t_name else 0.0

    counts: Counter[str] = Counter()
    for obj in sc.objects:
        for f in obj.features():
            counts[f] += 1
    for r in sc.relations:
        counts[f"rel:{r.name}"] += 1

    all_preds = list(ds.grammar.lexicon.keys()) if ds.grammar is not None else []
    for p in all_preds:
        feats[f"count:{p}"] = float(counts[p])
        feats[f"present:{p}"] = 1.0 if counts[p] > 0 else 0.0

    novel_p = ds.plan.novel_kind_predicate
    feats["novel_kind_present"] = 1.0 if counts[novel_p] > 0 else 0.0

    feats["sentence_length"] = float(len(ex.tokens))
    tok_set = set(ex.tokens)
    if ds.grammar is not None:
        for p, w in ds.grammar.lexicon.items():
            feats[f"token_pred:{p}"] = 1.0 if w in tok_set else 0.0
        feats["token_pred:Q"] = 1.0 if ds.grammar.question_particle in tok_set else 0.0
    return feats


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, p_values[idx] * (m - rank))
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg FDR adjusted q-values."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_min = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        adj = min(1.0, p_values[idx] * m / (rank + 1))
        running_min = min(running_min, adj)
        adjusted[idx] = running_min
    return adjusted


def feature_balance_audit(
    config: WorldConfig,
    n_seeds: int = 20,
    base_seed: int = 0,
    n_permutations: int = 1000,
    world_mode: str = "randomized",
) -> dict[str, dict[str, dict[str, float]]]:
    """Compare P(feature|true) vs P(feature|false) on every split with permutation tests."""
    split_names = [
        "train",
        "dev",
        "test_novel_combination",
        "test_novel_word_in_known_frame",
        "test_longer",
        "test_contradiction",
    ]
    by_split_feats: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_split_labels: dict[str, list[bool]] = defaultdict(list)

    for s in range(base_seed, base_seed + n_seeds):
        ds = build_dataset(config, seed=s, world_mode=world_mode)
        all_splits = {"train": ds.train, "dev": ds.dev, **ds.tests}
        for sp_name in split_names:
            for ex in all_splits[sp_name].items:
                by_split_feats[sp_name].append(extract_example_audit_features(ex, ds))
                by_split_labels[sp_name].append(ex.truth)

    rng = np.random.RandomState(42)
    report: dict[str, dict[str, dict[str, float]]] = {}

    for sp_name in split_names:
        feat_dicts = by_split_feats[sp_name]
        labels = np.array(by_split_labels[sp_name], dtype=bool)
        feat_names = sorted(feat_dicts[0].keys())
        raw_ps: list[float] = []
        rows: list[dict[str, float]] = []

        for f_name in feat_names:
            vals = np.array([d[f_name] for d in feat_dicts], dtype=np.float64)
            v_true = vals[labels]
            v_false = vals[~labels]
            mean_true = float(np.mean(v_true))
            mean_false = float(np.mean(v_false))
            diff = abs(mean_true - mean_false)
            if diff < 1e-12:
                p_val = 1.0
            else:
                n_total = len(vals)
                n_t = int(np.sum(labels))
                sum_all = float(np.sum(vals))
                exceed = 0
                for _ in range(n_permutations):
                    perm_idx = rng.choice(n_total, size=n_t, replace=False)
                    perm_sum_t = float(np.sum(vals[perm_idx]))
                    perm_mean_t = perm_sum_t / n_t
                    perm_mean_f = (sum_all - perm_sum_t) / (n_total - n_t)
                    if abs(perm_mean_t - perm_mean_f) >= diff - 1e-12:
                        exceed += 1
                p_val = (exceed + 1.0) / (n_permutations + 1.0)
            raw_ps.append(p_val)
            rows.append({
                "mean_true": mean_true,
                "mean_false": mean_false,
                "diff": mean_true - mean_false,
                "p_raw": p_val,
            })

        holm_ps = holm_bonferroni(raw_ps)
        bh_qs = benjamini_hochberg(raw_ps)
        report[sp_name] = {}
        for f_name, row, hp, bq in zip(feat_names, rows, holm_ps, bh_qs):
            row["p_holm"] = hp
            row["q_bh"] = bq
            report[sp_name][f_name] = row

    return report


# ---------------------------------------------------------------------------
# A3: Novel-word support stream candidate enumeration & ceiling
# ---------------------------------------------------------------------------

def audit_novel_word_support(config: WorldConfig, seed: int = 0) -> dict[str, Any]:
    """Enumerate Phase 1 candidate meanings consistent with the 2-example support stream."""
    ds = build_dataset(config, seed=seed, world_mode="randomized")
    nw_split = ds.tests["test_novel_word_in_known_frame"]
    assert len(nw_split.support) == 2
    ev_true, ev_false = nw_split.support[0], nw_split.support[1]
    sc_true = ev_true.observation.scene
    sc_false = ev_false.observation.scene
    assert ev_true.observation.label is True
    assert ev_false.observation.label is False

    all_preds = list(config.ontology.all_predicates())
    sym = config.ontology.symmetric_relations

    def pred_holds_in_scene(pred: str, scene: Scene) -> bool:
        if pred.startswith("rel:"):
            lf = make_lf("assert", ("x0", "x1"), [Atom(pred, ("x0", "x1"))])
        else:
            lf = make_lf("assert", ("x0",), [Atom(pred, ("x0",))])
        return evaluate(lf, scene, sym)

    consistent_from_support_alone = [
        p for p in all_preds
        if pred_holds_in_scene(p, sc_true) and not pred_holds_in_scene(p, sc_false)
    ]

    novel_pred = ds.plan.novel_kind_predicate
    true_items = [e for e in nw_split.items if e.truth]
    false_items = [e for e in nw_split.items if not e.truth]
    p_novel_true = sum(pred_holds_in_scene(novel_pred, e.scene) for e in true_items) / len(true_items)
    p_novel_false = sum(pred_holds_in_scene(novel_pred, e.scene) for e in false_items) / len(false_items)

    n_candidates = len(consistent_from_support_alone)
    ceiling = 1.0 if consistent_from_support_alone == [novel_pred] else (0.5 + 0.5 / max(1, n_candidates))

    return {
        "seed": seed,
        "novel_kind_predicate": novel_pred,
        "consistent_candidates_support_only": consistent_from_support_alone,
        "candidate_count": n_candidates,
        "ideal_observer_ceiling": ceiling,
        "p_novel_kind_given_true": p_novel_true,
        "p_novel_kind_given_false": p_novel_false,
    }


# ---------------------------------------------------------------------------
# A6: Identifiability report across constructions and word orders
# ---------------------------------------------------------------------------

def audit_identifiability(config: WorldConfig, seed: int = 0) -> dict[str, Any]:
    """Count training examples that distinguish the true grammar from main alternatives."""
    ds = build_dataset(config, seed=seed, world_mode="randomized")
    assert ds.grammar is not None
    g = ds.grammar
    sym = config.ontology.symmetric_relations
    q_word = g.question_particle

    construction_counts: Counter[str] = Counter()
    construction_distinguishing: Counter[str] = Counter()
    role_swap_chases_contrasts = 0
    np_attachment_contrasts = 0
    q_position_examples = 0

    for ex in ds.train.items:
        construction_counts[ex.construction] += 1
        construction_distinguishing[ex.construction] += 1

        if ex.construction in ("np_ask", "trans_ask"):
            if (ex.tokens[0] == q_word) != (ex.tokens[-1] == q_word):
                q_position_examples += 1

        if ex.construction in ("trans_assert", "trans_ask"):
            has_chases = any(a.predicate == "rel:chases" for a in ex.lf.atoms)
            if has_chases:
                swapped_atoms = []
                for a in ex.lf.atoms:
                    if a.predicate == "rel:chases":
                        swapped_atoms.append(Atom(a.predicate, (a.args[1], a.args[0])))
                    else:
                        swapped_atoms.append(a)
                swapped_lf = make_lf(ex.lf.act, ex.lf.variables, swapped_atoms)
                if evaluate(ex.lf, ex.scene, sym) != evaluate(swapped_lf, ex.scene, sym):
                    role_swap_chases_contrasts += 1

            adj_atoms = [a for a in ex.lf.atoms if a.family in ("color", "size")]
            if adj_atoms:
                reattached_atoms = []
                for a in ex.lf.atoms:
                    if a.family in ("color", "size"):
                        other_var = "x1" if a.args[0] == "x0" else "x0"
                        reattached_atoms.append(Atom(a.predicate, (other_var,)))
                    else:
                        reattached_atoms.append(a)
                reattached_lf = make_lf(ex.lf.act, ex.lf.variables, reattached_atoms)
                if evaluate(ex.lf, ex.scene, sym) != evaluate(reattached_lf, ex.scene, sym):
                    np_attachment_contrasts += 1

    n_train_stacked_nps = 0
    for ex in ds.train.items:
        by_var: defaultdict[str, list[str]] = defaultdict(list)
        for a in ex.lf.atoms:
            if a.family in ("color", "size"):
                by_var[a.args[0]].append(a.family)
        if any(len(v) >= 2 for v in by_var.values()):
            n_train_stacked_nps += 1

    flagged = [
        k for k, v in {
            **{f"construction:{c}": construction_distinguishing[c] for c in g.constructions},
            "clause_role_order_chases": role_swap_chases_contrasts,
            "np_modifier_attachment": np_attachment_contrasts,
            "q_particle_position": q_position_examples,
        }.items()
        if v < 5
    ]

    return {
        "seed": seed,
        "np_order": g.np_order,
        "trans_template": g.constructions["trans_assert"].template,
        "q_position": "initial" if g.constructions["np_ask"].template[0] == "Q" else "final",
        "construction_counts": dict(construction_counts),
        "clause_role_order_chases_contrasts": role_swap_chases_contrasts,
        "np_modifier_attachment_contrasts": np_attachment_contrasts,
        "q_particle_position_examples": q_position_examples,
        "n_train_stacked_nps": n_train_stacked_nps,
        "flagged_under_5": flagged,
    }


# ---------------------------------------------------------------------------
# A8: Exhaustive Bayesian ideal observer on a reduced world
# ---------------------------------------------------------------------------

class ExhaustiveIdealObserver:
    """Exact enumerative Bayesian ideal observer over the reduced world (Stage A8).

    Reduced world: 2 known kinds + 1 novel kind, 2 colors, 1 asymmetric relation,
    fixed word order (NP order: noun -> color; clause order: NP_subj NP_obj VERB; Q final).
    Enumerates all ``6! = 720`` bijections between the 6 content words and 6
    predicates (`kind:cube`, `kind:ball`, `kind:cone`, `color:red`, `color:blue`,
    `rel:chases`) and computes exact Bayesian posterior updates:
        log P(H | D) = log P(H) + sum_i log((1 - eps) if H(s_i)==y_i else eps)
    """

    def __init__(
        self,
        vocab_words: Sequence[str],
        predicates: Sequence[str],
        q_word: str,
        epsilon: float = 0.02,
    ) -> None:
        self.vocab_words = tuple(vocab_words)
        self.predicates = tuple(predicates)
        self.q_word = q_word
        self.epsilon = epsilon
        self.hypotheses: list[dict[str, str]] = [
            dict(zip(self.vocab_words, perm))
            for perm in itertools.permutations(self.predicates)
        ]
        self.log_weights = np.zeros(len(self.hypotheses), dtype=np.float64)
        self._obs_history: dict[str, tuple[Scene, tuple[str, ...], bool]] = {}

    @staticmethod
    def parse_with_map(tokens: tuple[str, ...], w2p: Mapping[str, str], q_word: str) -> LogicalForm | None:
        """Attempt to parse ``tokens`` under fixed order (noun color, SOV, Q-final)."""
        act = "ask" if (tokens and tokens[-1] == q_word) else "assert"
        lex = [t for t in tokens if t != q_word]
        if not lex or any(t not in w2p for t in lex):
            return None
        preds = [w2p[t] for t in lex]
        rel_preds = [p for p in preds if p.startswith("rel:")]
        if len(rel_preds) == 0:
            if not preds[0].startswith("kind:"):
                return None
            if any(not p.startswith("color:") for p in preds[1:]):
                return None
            atoms = [Atom(p, ("x0",)) for p in preds]
            return make_lf(act, ("x0",), atoms)
        if len(rel_preds) == 1 and preds[-1].startswith("rel:"):
            np_preds = preds[:-1]
            kind_pos = [i for i, p in enumerate(np_preds) if p.startswith("kind:")]
            if len(kind_pos) != 2 or kind_pos[0] != 0:
                return None
            k2 = kind_pos[1]
            subj_preds = np_preds[:k2]
            obj_preds = np_preds[k2:]
            if any(not p.startswith("color:") for p in subj_preds[1:]):
                return None
            if any(not p.startswith("color:") for p in obj_preds[1:]):
                return None
            atoms = [Atom(p, ("x0",)) for p in subj_preds]
            atoms.extend(Atom(p, ("x1",)) for p in obj_preds)
            atoms.append(Atom(preds[-1], ("x0", "x1")))
            return make_lf(act, ("x0", "x1"), atoms)
        return None

    def _eval_hyp(self, h: Mapping[str, str], scene: Scene, tokens: tuple[str, ...]) -> bool | None:
        lf = self.parse_with_map(tokens, h, self.q_word)
        if lf is None:
            return None
        return evaluate(lf, scene, frozenset())

    def observe(self, example_id: str, scene: Scene, tokens: tuple[str, ...], label: bool) -> None:
        self._obs_history[example_id] = (scene, tokens, label)
        self._apply_likelihood(scene, tokens, label, sign=1.0)

    def correct(self, example_id: str, new_label: bool) -> None:
        sc, toks, old_label = self._obs_history[example_id]
        self._apply_likelihood(sc, toks, old_label, sign=-1.0)
        self._obs_history[example_id] = (sc, toks, new_label)
        self._apply_likelihood(sc, toks, new_label, sign=1.0)

    def _apply_likelihood(self, scene: Scene, tokens: tuple[str, ...], label: bool, sign: float) -> None:
        log_hit = math.log(1.0 - self.epsilon)
        log_miss = math.log(self.epsilon)
        log_invalid = math.log(1e-6)
        for idx, h in enumerate(self.hypotheses):
            res = self._eval_hyp(h, scene, tokens)
            if res is None:
                ll = log_invalid
            else:
                ll = log_hit if (res == label) else log_miss
            self.log_weights[idx] += sign * ll

    def predict_p_true(self, scene: Scene, tokens: tuple[str, ...]) -> float:
        m = float(np.max(self.log_weights))
        w = np.exp(self.log_weights - m)
        w /= np.sum(w)
        p_true = 0.0
        for idx, h in enumerate(self.hypotheses):
            if w[idx] < 1e-12:
                continue
            res = self._eval_hyp(h, scene, tokens)
            p_h = 0.5 if res is None else ((1.0 - self.epsilon) if res else self.epsilon)
            p_true += float(w[idx]) * p_h
        return p_true


def run_reduced_world_ideal_observer(seed: int = 0) -> dict[str, Any]:
    """Build a reduced world (2 kinds + 1 novel kind, 2 colors, 1 relation, fixed order)
    and evaluate the exhaustive Bayesian ideal observer across splits (Stage A8).
    """
    rng = random.Random(derive_seed(seed, "reduced_world"))
    kinds = ("cube", "ball", "cone")
    colors = ("red", "blue")
    lexicon = {
        "kind:cube": "dax",
        "kind:ball": "wug",
        "kind:cone": "zib",
        "color:red": "mip",
        "color:blue": "sorn",
        "rel:chases": "zorp",
    }
    q_word = "ko"
    sym: frozenset[str] = frozenset()

    def eval_meaning(
        act: str,
        nps: list[tuple[str, list[str]]],
        rel: str | None,
        sc: Scene,
    ) -> tuple[tuple[str, ...], LogicalForm, bool]:
        toks: list[str] = []
        atoms: list[Atom] = []
        vars_used = tuple(f"x{i}" for i in range(len(nps)))
        for idx, (k, cols) in enumerate(nps):
            v = f"x{idx}"
            toks.append(lexicon[f"kind:{k}"])
            atoms.append(Atom(f"kind:{k}", (v,)))
            for c in cols:
                toks.append(lexicon[f"color:{c}"])
                atoms.append(Atom(f"color:{c}", (v,)))
        if rel is not None:
            toks.append(lexicon[f"rel:{rel}"])
            atoms.append(Atom(f"rel:{rel}", ("x0", "x1")))
        if act == "ask":
            toks.append(q_word)
        lf = make_lf(act, vars_used, atoms)
        return tuple(toks), lf, evaluate(lf, sc, sym)

    def make_pair(
        act: str,
        nps: list[tuple[str, list[str]]],
        rel: str | None,
        allowed_k: Sequence[str],
    ) -> list[tuple[Scene, tuple[str, ...], bool]]:
        k0, cols0 = nps[0]
        c0 = cols0[0] if cols0 else rng.choice(colors)
        other_c0 = "blue" if c0 == "red" else "red"
        if len(nps) == 1 and rel is None:
            other_k = [k for k in allowed_k if k != k0][0]
            attrs_t = [
                {"kind": k0, "color": c0, "size": "small"},
                {"kind": other_k, "color": other_c0, "size": "small"},
            ]
            if not cols0:
                attrs_f = [
                    {"kind": other_k, "color": c0, "size": "small"},
                    {"kind": other_k, "color": other_c0, "size": "small"},
                ]
            else:
                attrs_f = [
                    {"kind": k0, "color": other_c0, "size": "small"},
                    {"kind": other_k, "color": c0, "size": "small"},
                ]
            st = build_scene(attrs_t, [])
            sf = build_scene(attrs_f, [])
        else:
            k1, cols1 = nps[1]
            c1 = cols1[0] if cols1 else rng.choice(colors)
            attrs_t = [
                {"kind": k0, "color": c0, "size": "small"},
                {"kind": k1, "color": c1, "size": "small"},
            ]
            st = build_scene(attrs_t, [("chases", 0, 1)])
            sf = build_scene(attrs_t, [("chases", 1, 0)])
        toks, _, _ = eval_meaning(act, nps, rel, st)
        return [(st, toks, True), (sf, toks, False)]

    known_k = ("cube", "ball")
    train_items: list[tuple[Scene, tuple[str, ...], bool]] = []
    specs_train = [
        ("assert", [("cube", [])], None),
        ("assert", [("ball", [])], None),
        ("ask", [("cube", [])], None),
        ("assert", [("cube", ["red"])], None),
        ("assert", [("cube", ["blue"])], None),
        ("assert", [("ball", ["red"])], None),
        ("assert", [("cube", []), ("ball", [])], "chases"),
        ("ask", [("ball", []), ("cube", [])], "chases"),
        ("assert", [("cube", ["red"]), ("ball", [])], "chases"),
        ("assert", [("ball", ["red"]), ("cube", ["blue"])], "chases"),
    ]
    for act, nps, rel in specs_train:
        train_items.extend(make_pair(act, nps, rel, known_k))

    dev_items = list(train_items)
    novel_comb_items = (
        make_pair("assert", [("ball", ["blue"])], None, known_k)
        + make_pair("ask", [("ball", ["blue"])], None, known_k)
        + make_pair("assert", [("ball", ["blue"]), ("cube", ["red"])], "chases", known_k)
    )
    novel_word_support = make_pair("assert", [("cone", [])], None, kinds)
    novel_word_queries = (
        make_pair("assert", [("cone", ["red"])], None, kinds)
        + make_pair("assert", [("cone", []), ("cube", [])], "chases", kinds)
        + make_pair("ask", [("ball", []), ("cone", ["blue"])], "chases", kinds)
    )

    all_words = ["dax", "wug", "zib", "mip", "sorn", "zorp"]
    all_preds = ["kind:cube", "kind:ball", "kind:cone", "color:red", "color:blue", "rel:chases"]
    observer = ExhaustiveIdealObserver(all_words, all_preds, q_word="ko")

    for idx, (sc, toks, lbl) in enumerate(train_items):
        observer.observe(f"train-{idx}", sc, toks, lbl)

    def eval_split(items: Sequence[tuple[Scene, tuple[str, ...], bool]]) -> float:
        hits = 0
        for sc, toks, lbl in items:
            p = observer.predict_p_true(sc, toks)
            if (p >= 0.5) == lbl:
                hits += 1
        return hits / len(items)

    acc_train = eval_split(train_items)
    acc_dev = eval_split(dev_items)
    acc_novel_comb = eval_split(novel_comb_items)
    acc_novel_word_before = eval_split(novel_word_queries)

    for idx, (sc, toks, lbl) in enumerate(novel_word_support):
        observer.observe(f"nw-sup-{idx}", sc, toks, lbl)
    acc_novel_word_after = eval_split(novel_word_queries)

    best_idx = int(np.argmax(observer.log_weights))
    best_hyp = observer.hypotheses[best_idx]
    exact_match = all(best_hyp[w] == p for p, w in lexicon.items())

    return {
        "n_hypotheses_enumerated": len(observer.hypotheses),
        "train": acc_train,
        "dev": acc_dev,
        "test_novel_combination": acc_novel_comb,
        "test_novel_word_before_support": acc_novel_word_before,
        "test_novel_word_after_support": acc_novel_word_after,
        "top_hypothesis_matches_ground_truth": exact_match,
    }
