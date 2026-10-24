"""STAGE D (EXPLORATORY): Post-Preregistration Ablations, Stress Tests, Calibration, and SQLite Persistence.

All analyses in this script are explicitly labeled ``EXPLORATORY`` (post-gate-C)
and do not modify ``PREREGISTRATION.md`` or the Stage C results.

Covers:
- D1: Beam width K in {1, 5, 25, 100, 200, 500}
- D2: Noise parameter epsilon in {0.001, 0.01, 0.02, 0.05, 0.1}
- D3: Learning curves at N in {50, 100, 200, 400, 800, 1600} vs baselines + gap to A8 ideal observer
- D4: Hypothesis-space ablations (given_lexicon, given_syntax, max_slash_depth=1)
- D5: Label noise {0%, 2%, 5%, 10%, 20%} before/after corrections
- D6: New held-out probes (role reversal on symmetric vs asymmetric relations, 3-adjective NPs, unseen adj + novel_noun)
- D7: Calibration reliability diagrams + Murphy Brier decomposition (Reliability - Resolution + Uncertainty)
- D8: Failure taxonomy across low-N, high-noise, and structural boundary regimes
- D9: SQLite persistence round-trip in a fresh subprocess
"""

from __future__ import annotations

import copy
import json
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from nexus.baselines.cross_situational import CrossSituationalLearner
from nexus.baselines.memorizer import Memorizer
from nexus.baselines.no_revision import NoRevisionLearner
from nexus.eval.audit import ExhaustiveIdealObserver
from nexus.eval.learner_api import Prediction
from nexus.eval.metrics import split_metrics
from nexus.eval.runner import RESULTS_DIR, predict_all, save_results
from nexus.eval.splits import Dataset, build_dataset, derive_seed
from nexus.learner import NEXUSLearner
from nexus.storage import SQLiteLearnerRepository
from nexus.world.config import WorldConfig, load_world_config
from nexus.world.example import Correction, Example, Observation
from nexus.world.grammar import Grammar, Meaning, NPSpec
from nexus.world.logic import evaluate
from nexus.world.scene import Scene, SceneObject, SceneRelation, build_scene


def stratified_prefix(items: Sequence[Example], n_total: int, seed: int = 0) -> list[Example]:
    """Select ``n_total`` training examples stratified across the 4 curriculum constructions."""
    if n_total >= len(items):
        return list(items)
    by_c: dict[str, list[Example]] = {}
    for ex in items:
        by_c.setdefault(ex.construction, []).append(ex)
    constructions = ["np_assert", "np_ask", "trans_assert", "trans_ask"]
    per_c = n_total // len(constructions)
    rem = n_total % len(constructions)
    out: list[Example] = []
    for idx, c in enumerate(constructions):
        take = per_c + (1 if idx < rem else 0)
        out.extend(by_c.get(c, [])[:take])
    return out


# ---------------------------------------------------------------------------
# D1: Beam width K in {1, 5, 25, 100, 200, 500}
# ---------------------------------------------------------------------------

def run_d1_beam_width(config: WorldConfig, seeds: Sequence[int] = (0, 1, 2, 3, 4)) -> dict[str, Any]:
    beam_widths = [1, 5, 25, 100, 200, 500]
    results: dict[str, Any] = {"analysis": "EXPLORATORY", "beam_widths": {}}
    datasets = [build_dataset(config, derive_seed(s, "world"), world_mode="randomized") for s in seeds]

    for k in beam_widths:
        accs_low_n: list[float] = []
        accs_full_n: list[float] = []
        accs_nw: list[float] = []
        briers_nw: list[float] = []
        for s, ds in zip(seeds, datasets):
            # Low-N regime (N=120 stratified) to expose beam width sensitivity before full convergence
            l_low = NEXUSLearner(seed=s, beam_width=k)
            for ex in stratified_prefix(ds.train.items, 120, seed=s):
                l_low.observe(ex.observation())
            preds_low = predict_all(l_low, ds.tests["test_novel_combination"].items[:80])
            accs_low_n.append(
                float(np.mean([p.truth == e.truth for p, e in zip(preds_low, ds.tests["test_novel_combination"].items[:80])]))
            )

            # Full-N regime (N=1600)
            l_full = NEXUSLearner(seed=s, beam_width=k)
            for ex in ds.train.items:
                l_full.observe(ex.observation())
            preds_nc = predict_all(l_full, ds.tests["test_novel_combination"].items[:100])
            accs_full_n.append(
                float(np.mean([p.truth == e.truth for p, e in zip(preds_nc, ds.tests["test_novel_combination"].items[:100])]))
            )

            # Novel word split after support
            nw_split = ds.tests["test_novel_word_in_known_frame"]
            for ev in nw_split.support:
                if ev.kind == "observe":
                    assert ev.observation is not None
                    l_full.observe(ev.observation)
            preds_nw = predict_all(l_full, nw_split.items[:80])
            accs_nw.append(float(np.mean([p.truth == e.truth for p, e in zip(preds_nw, nw_split.items[:80])])))
            briers_nw.append(float(np.mean([(p.p_true - float(e.truth)) ** 2 for p, e in zip(preds_nw, nw_split.items[:80])])))

        results["beam_widths"][str(k)] = {
            "beam_width": k,
            "novel_comb_acc_N120_mean": float(np.mean(accs_low_n)),
            "novel_comb_acc_N120_std": float(np.std(accs_low_n, ddof=1)),
            "novel_comb_acc_N1600_mean": float(np.mean(accs_full_n)),
            "novel_word_acc_N1600_mean": float(np.mean(accs_nw)),
            "novel_word_brier_N1600_mean": float(np.mean(briers_nw)),
        }
    return results


# ---------------------------------------------------------------------------
# D2: Noise parameter epsilon in {0.001, 0.01, 0.02, 0.05, 0.1}
# ---------------------------------------------------------------------------

def run_d2_noise_epsilon(config: WorldConfig, seeds: Sequence[int] = (0, 1, 2, 3, 4)) -> dict[str, Any]:
    epsilons = [0.001, 0.01, 0.02, 0.05, 0.1]
    results: dict[str, Any] = {"analysis": "EXPLORATORY", "epsilons": {}}
    datasets = [build_dataset(config, derive_seed(s, "world"), world_mode="randomized") for s in seeds]

    for eps in epsilons:
        acc_list: list[float] = []
        ece_list: list[float] = []
        brier_list: list[float] = []
        noisy_acc_list: list[float] = []
        for s, ds in zip(seeds, datasets):
            learner = NEXUSLearner(seed=s, epsilon=eps)
            for ex in ds.train.items:
                learner.observe(ex.observation())
            items = ds.tests["test_novel_combination"].items[:100]
            preds = predict_all(learner, items)
            m = split_metrics(preds, items, 10)
            acc_list.append(m["accuracy"])
            ece_list.append(m["ece"])
            brier_list.append(m["brier"])

            # Also evaluate under 10% un-corrected training label noise
            rng = random.Random(s + 900)
            l_noisy = NEXUSLearner(seed=s, epsilon=eps)
            for ex in stratified_prefix(ds.train.items, 400, seed=s):
                flip = rng.random() < 0.10
                l_noisy.observe(ex.observation(label=(not ex.truth if flip else ex.truth)))
            preds_n = predict_all(l_noisy, items)
            noisy_acc_list.append(float(np.mean([p.truth == e.truth for p, e in zip(preds_n, items)])))

        results["epsilons"][str(eps)] = {
            "epsilon": eps,
            "clean_novel_comb_acc": float(np.mean(acc_list)),
            "clean_ece": float(np.mean(ece_list)),
            "clean_brier": float(np.mean(brier_list)),
            "noisy_10pct_novel_comb_acc": float(np.mean(noisy_acc_list)),
        }
    return results


# ---------------------------------------------------------------------------
# D3: Learning curves at N in {50, 100, 200, 400, 800, 1600} + A8 ideal observer gap
# ---------------------------------------------------------------------------

def run_d3_learning_curves(config: WorldConfig, seeds: Sequence[int] = (0, 1, 2, 3, 4)) -> dict[str, Any]:
    sample_sizes = [50, 100, 200, 400, 800, 1600]
    datasets = [build_dataset(config, derive_seed(s, "world"), world_mode="randomized") for s in seeds]
    curves: dict[str, Any] = {}

    for n_sz in sample_sizes:
        acc_nexus, acc_a, acc_b, acc_c = [], [], [], []
        for s, ds in zip(seeds, datasets):
            sub_train = stratified_prefix(ds.train.items, n_sz, seed=s)
            l_n = NEXUSLearner(seed=s)
            l_a = Memorizer(seed=s)
            l_b = CrossSituationalLearner(seed=s)
            l_c = NoRevisionLearner(seed=s)
            for ex in sub_train:
                obs = ex.observation()
                l_n.observe(obs)
                l_a.observe(obs)
                l_b.observe(obs)
                l_c.observe(obs)
            eval_items = ds.tests["test_novel_combination"].items[:100]
            acc_nexus.append(float(np.mean([l_n.predict(e.scene, e.tokens).truth == e.truth for e in eval_items])))
            acc_a.append(float(np.mean([l_a.predict(e.scene, e.tokens).truth == e.truth for e in eval_items])))
            acc_b.append(float(np.mean([l_b.predict(e.scene, e.tokens).truth == e.truth for e in eval_items])))
            acc_c.append(float(np.mean([l_c.predict(e.scene, e.tokens).truth == e.truth for e in eval_items])))

        curves[str(n_sz)] = {
            "n_train": n_sz,
            "nexus_acc_mean": float(np.mean(acc_nexus)),
            "nexus_acc_std": float(np.std(acc_nexus, ddof=1)),
            "memorizer_acc_mean": float(np.mean(acc_a)),
            "cross_situational_acc_mean": float(np.mean(acc_b)),
            "no_revision_acc_mean": float(np.mean(acc_c)),
        }

    # Reduced world comparison against A8 ExhaustiveIdealObserver (720 bijections)
    all_words = ["dax", "wug", "zib", "mip", "sorn", "zorp"]
    all_preds = ["kind:cube", "kind:ball", "kind:cone", "color:red", "color:blue", "rel:chases"]
    red_sizes = [4, 8, 12, 16, 20]
    ideal_gap: dict[str, Any] = {}

    for n_red in red_sizes:
        io_accs, nx_accs = [], []
        for s in seeds:
            rng = random.Random(derive_seed(s, "reduced_world"))
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

            def make_red_pair(
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
                toks: list[str] = []
                for k, cols in nps:
                    toks.append(lexicon[f"kind:{k}"])
                    for c in cols:
                        toks.append(lexicon[f"color:{c}"])
                if rel is not None:
                    toks.append(lexicon[f"rel:{rel}"])
                if act == "ask":
                    toks.append(q_word)
                return [(st, tuple(toks), True), (sf, tuple(toks), False)]

            known_k = ("cube", "ball")
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
            tr_pairs: list[tuple[Scene, tuple[str, ...], bool]] = []
            for act, nps, rel in specs_train:
                tr_pairs.extend(make_red_pair(act, nps, rel, known_k))

            eval_pairs = (
                make_red_pair("assert", [("ball", ["blue"])], None, known_k)
                + make_red_pair("ask", [("ball", ["blue"])], None, known_k)
                + make_red_pair("assert", [("ball", ["blue"]), ("cube", ["red"])], "chases", known_k)
            )

            io = ExhaustiveIdealObserver(all_words, all_preds, q_word="ko", epsilon=0.02)
            nx = NEXUSLearner(seed=s, epsilon=0.02)
            for idx, (sc, toks, lbl) in enumerate(tr_pairs[:n_red]):
                io.observe(f"red-{idx}", sc, toks, lbl)
                nx.observe(Observation(f"red-{idx}", sc, toks, lbl, "teacher"))

            io_hits = sum((io.predict_p_true(sc, toks) >= 0.5) == lbl for sc, toks, lbl in eval_pairs)
            nx_hits = sum(nx.predict(sc, toks).truth == lbl for sc, toks, lbl in eval_pairs)
            io_accs.append(io_hits / len(eval_pairs))
            nx_accs.append(nx_hits / len(eval_pairs))

        ideal_gap[str(n_red)] = {
            "n_train_reduced": n_red,
            "ideal_observer_acc": float(np.mean(io_accs)),
            "nexus_acc": float(np.mean(nx_accs)),
            "gap_ideal_minus_nexus": float(np.mean(io_accs) - np.mean(nx_accs)),
        }

    return {
        "analysis": "EXPLORATORY",
        "full_world_learning_curves": curves,
        "reduced_world_ideal_observer_comparison": ideal_gap,
    }


# ---------------------------------------------------------------------------
# D4: Hypothesis-space ablations
# ---------------------------------------------------------------------------

def _true_syntax_map(ds: Dataset) -> dict[str, str]:
    """Derive ground-truth categorial syntax mapping for a dataset seed."""
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

    syn: dict[str, str] = {ds.grammar.question_particle: cat_q}
    for concept, w in lex.items():
        fam = concept.split(":", 1)[0]
        if fam == "kind":
            syn[w] = "N"
        elif fam == "color":
            syn[w] = cat_color
        elif fam == "size":
            syn[w] = cat_size
        elif fam == "rel":
            syn[w] = cat_verb
    return syn


def _true_lexicon_map(ds: Dataset) -> dict[str, str]:
    inv = {w: c for c, w in ds.grammar.lexicon.items()}
    inv[ds.grammar.question_particle] = "Q"
    return inv


def run_d4_ablations(config: WorldConfig, seeds: Sequence[int] = (0, 1, 2, 3, 4)) -> dict[str, Any]:
    datasets = [build_dataset(config, derive_seed(s, "world"), world_mode="randomized") for s in seeds]
    modes = [
        "full_nexus",
        "given_lexicon_induce_syntax",
        "given_syntax_induce_lexicon",
        "slash_depth_le_1",
    ]
    out: dict[str, Any] = {"analysis": "EXPLORATORY", "ablations": {}}

    for mode in modes:
        nc_accs, np_accs, tr_accs = [], [], []
        low_n_accs = []
        for s, ds in zip(seeds, datasets):
            g_lex = _true_lexicon_map(ds) if mode == "given_lexicon_induce_syntax" else None
            g_syn = _true_syntax_map(ds) if mode == "given_syntax_induce_lexicon" else None
            max_d = 1 if mode == "slash_depth_le_1" else 2

            # Evaluate at N=100 (low-sample) and N=1600 (full)
            l_low = NEXUSLearner(seed=s, max_slash_depth=max_d, given_lexicon=g_lex, given_syntax=g_syn)
            for ex in stratified_prefix(ds.train.items, 100, seed=s):
                l_low.observe(ex.observation())
            items_nc = ds.tests["test_novel_combination"].items[:100]
            low_n_accs.append(float(np.mean([l_low.predict(e.scene, e.tokens).truth == e.truth for e in items_nc])))

            l_full = NEXUSLearner(seed=s, max_slash_depth=max_d, given_lexicon=g_lex, given_syntax=g_syn)
            for ex in ds.train.items:
                l_full.observe(ex.observation())

            preds_nc = predict_all(l_full, items_nc)
            nc_accs.append(float(np.mean([p.truth == e.truth for p, e in zip(preds_nc, items_nc)])))

            np_items = [e for e in ds.dev.items if e.construction in ("np_assert", "np_ask")]
            tr_items = [e for e in ds.dev.items if e.construction in ("trans_assert", "trans_ask")]
            np_accs.append(float(np.mean([l_full.predict(e.scene, e.tokens).truth == e.truth for e in np_items])))
            tr_accs.append(float(np.mean([l_full.predict(e.scene, e.tokens).truth == e.truth for e in tr_items])))

        out["ablations"][mode] = {
            "novel_comb_acc_N100": float(np.mean(low_n_accs)),
            "novel_comb_acc_N1600": float(np.mean(nc_accs)),
            "dev_np_constructions_acc": float(np.mean(np_accs)),
            "dev_transitive_constructions_acc": float(np.mean(tr_accs)),
        }
    return out


# ---------------------------------------------------------------------------
# D5: Label noise {0%, 2%, 5%, 10%, 20%} before/after corrections
# ---------------------------------------------------------------------------

def run_d5_label_noise(config: WorldConfig, seeds: Sequence[int] = (0, 1, 2, 3, 4)) -> dict[str, Any]:
    noise_rates = [0.00, 0.02, 0.05, 0.10, 0.20]
    datasets = [build_dataset(config, derive_seed(s, "world"), world_mode="randomized") for s in seeds]
    out: dict[str, Any] = {"analysis": "EXPLORATORY", "noise_rates": {}}

    for rate in noise_rates:
        before_N120, after_N120 = [], []
        before_N1600, after_N1600 = [], []
        for s, ds in zip(seeds, datasets):
            eval_items = ds.tests["test_novel_combination"].items[:80]

            for n_tr, b_list, a_list in [
                (120, before_N120, after_N120),
                (400, before_N1600, after_N1600),
            ]:
                rng = random.Random(s * 100 + int(rate * 1000) + n_tr)
                learner = NEXUSLearner(seed=s)
                corrupted_ex: list[Example] = []
                sub_items = stratified_prefix(ds.train.items, n_tr, seed=s)
                for ex in sub_items:
                    if rng.random() < rate:
                        learner.observe(ex.observation(label=not ex.truth, source="noisy_teacher"))
                        corrupted_ex.append(ex)
                    else:
                        learner.observe(ex.observation())

                b_acc = float(np.mean([learner.predict(e.scene, e.tokens).truth == e.truth for e in eval_items]))
                b_list.append(b_acc)

                for ex in corrupted_ex:
                    learner.correct(Correction(ex.id, ex.truth, "teacher", "retract noisy training label"))
                a_acc = float(np.mean([learner.predict(e.scene, e.tokens).truth == e.truth for e in eval_items]))
                a_list.append(a_acc)

        out["noise_rates"][f"{int(rate * 100)}pct"] = {
            "noise_rate": rate,
            "N120_before_correction_acc": float(np.mean(before_N120)),
            "N120_after_correction_acc": float(np.mean(after_N120)),
            "N400_before_correction_acc": float(np.mean(before_N1600)),
            "N400_after_correction_acc": float(np.mean(after_N1600)),
        }
    return out


# ---------------------------------------------------------------------------
# D6: New held-out probes (role reversal, 3-adjective NPs, unseen adj + novel_noun)
# ---------------------------------------------------------------------------

def run_d6_new_probes(config: WorldConfig, seeds: Sequence[int] = (0, 1, 2, 3, 4)) -> dict[str, Any]:
    datasets = [build_dataset(config, derive_seed(s, "world"), world_mode="randomized") for s in seeds]
    asym_accs, sym_accs = [], []
    two_adj_np_accs, four_adj_trans_accs, three_adj_conflict_parse_rates = [], [], []
    adj_novel_noun_accs = []

    for s, ds in zip(seeds, datasets):
        grammar_engine = Grammar(ds.grammar)
        learner = NEXUSLearner(seed=s)
        for ex in ds.train.items:
            learner.observe(ex.observation())

        # 1. Role reversal on asymmetric (chases) vs symmetric (next_to) relations
        known_kinds = [k for k in config.ontology.kinds if k != ds.plan.novel_kind]
        k0, k1 = known_kinds[0], known_kinds[1]
        c0, c1 = config.ontology.colors[0], config.ontology.colors[1]
        z0 = config.ontology.sizes[0]
        scene_asym = build_scene(
            [{"kind": k0, "color": c0, "size": z0}, {"kind": k1, "color": c1, "size": z0}],
            [("chases", 0, 1)],
        )
        m_ab_asym = Meaning("trans_assert", (("subj", NPSpec(k0, ())), ("obj", NPSpec(k1, ()))), "chases")
        m_ba_asym = Meaning("trans_assert", (("subj", NPSpec(k1, ())), ("obj", NPSpec(k0, ()))), "chases")
        tok_ab_asym = grammar_engine.realize(m_ab_asym)
        tok_ba_asym = grammar_engine.realize(m_ba_asym)
        p_ab = learner.predict(scene_asym, tok_ab_asym).truth
        p_ba = learner.predict(scene_asym, tok_ba_asym).truth
        asym_accs.append(0.5 * (float(p_ab is True) + float(p_ba is False)))

        scene_sym = build_scene(
            [{"kind": k0, "color": c0, "size": z0}, {"kind": k1, "color": c1, "size": z0}],
            [("next_to", 0, 1)],
        )
        m_ab_sym = Meaning("trans_assert", (("subj", NPSpec(k0, ())), ("obj", NPSpec(k1, ()))), "next_to")
        m_ba_sym = Meaning("trans_assert", (("subj", NPSpec(k1, ())), ("obj", NPSpec(k0, ()))), "next_to")
        p_sym_ab = learner.predict(scene_sym, grammar_engine.realize(m_ab_sym)).truth
        p_sym_ba = learner.predict(scene_sym, grammar_engine.realize(m_ba_sym)).truth
        sym_accs.append(0.5 * (float(p_sym_ab is True) + float(p_sym_ba is True)))

        # 2. Stacked modifier probes:
        # (a) 2-adjective single NP (color + size on 1 noun)
        # (b) 4-adjective transitive clause (color + size on both subject and object)
        # (c) 3-adjective single NP with 2 colors + 1 size (same-family sortal conflict)
        m_2adj = Meaning(
            "np_assert",
            (("np", NPSpec(k0, (("color", c0), ("size", z0)))),),
            None,
        )
        tok_2adj = grammar_engine.realize(m_2adj)
        p2_true = learner.predict(scene_asym, tok_2adj).truth
        scene_2adj_false = build_scene(
            [{"kind": k0, "color": c1, "size": z0}, {"kind": k1, "color": c0, "size": z0}],
            [],
        )
        p2_false = learner.predict(scene_2adj_false, tok_2adj).truth
        two_adj_np_accs.append(0.5 * (float(p2_true is True) + float(p2_false is False)))

        m_4adj = Meaning(
            "trans_assert",
            (
                ("subj", NPSpec(k0, (("color", c0), ("size", z0)))),
                ("obj", NPSpec(k1, (("color", c1), ("size", z0)))),
            ),
            "chases",
        )
        tok_4adj = grammar_engine.realize(m_4adj)
        p4_true = learner.predict(scene_asym, tok_4adj).truth
        p4_false = learner.predict(scene_2adj_false, tok_4adj).truth
        four_adj_trans_accs.append(0.5 * (float(p4_true is True) + float(p4_false is False)))

        # Construct a 3-adjective NP token sequence by adding a second color modifier adjacent to c0
        w_c1 = ds.grammar.lexicon[f"color:{c1}"]
        w_c0 = ds.grammar.lexicon[f"color:{c0}"]
        tok_2adj_list = list(tok_2adj)
        c0_pos = tok_2adj_list.index(w_c0)
        tok_3adj_conflict = tuple(tok_2adj_list[:c0_pos] + [w_c1] + tok_2adj_list[c0_pos:])
        exp_3adj = learner.explain(tok_3adj_conflict, scene_asym)
        three_adj_conflict_parse_rates.append(float(exp_3adj.derived_lf is not None))

        # 3. Unseen adj + novel_noun after 2-example bare-NP support stream
        nw_split = ds.tests["test_novel_word_in_known_frame"]
        nw_learner = copy.deepcopy(learner)
        for ev in nw_split.support:
            if ev.kind == "observe":
                assert ev.observation is not None
                nw_learner.observe(ev.observation)
        adj_novel_items = [
            e for e in nw_split.items
            if any(a.family in ("color", "size") for a in e.lf.atoms)
        ]
        adj_novel_noun_accs.append(
            float(np.mean([nw_learner.predict(e.scene, e.tokens).truth == e.truth for e in adj_novel_items]))
        )

    return {
        "analysis": "EXPLORATORY",
        "role_reversal_asymmetric_chases_acc": float(np.mean(asym_accs)),
        "role_reversal_symmetric_next_to_acc": float(np.mean(sym_accs)),
        "stacked_2adj_single_np_acc": float(np.mean(two_adj_np_accs)),
        "stacked_4adj_transitive_acc": float(np.mean(four_adj_trans_accs)),
        "conflict_3adj_same_family_parse_rate": float(np.mean(three_adj_conflict_parse_rates)),
        "unseen_adj_plus_novel_noun_acc": float(np.mean(adj_novel_noun_accs)),
    }


# ---------------------------------------------------------------------------
# D7: Calibration reliability diagrams + Murphy Brier decomposition
# ---------------------------------------------------------------------------

def brier_decomposition_and_diagram(
    preds: Sequence[Prediction],
    truths: Sequence[bool],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute 10-bin reliability diagram on [0, 1] and Murphy Brier decomposition."""
    p_arr = np.array([p.p_true for p in preds], dtype=np.float64)
    y_arr = np.array([float(y) for y in truths], dtype=np.float64)
    n_total = len(p_arr)
    o_bar = float(np.mean(y_arr))
    uncertainty = o_bar * (1.0 - o_bar)
    brier_raw = float(np.mean((p_arr - y_arr) ** 2))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(p_arr, edges[1:-1], right=True), 0, n_bins - 1)

    reliability = 0.0
    resolution = 0.0
    bins_out: list[dict[str, Any]] = []

    for b in range(n_bins):
        mask = bin_indices == b
        cnt = int(np.sum(mask))
        if cnt > 0:
            f_k = float(np.mean(p_arr[mask]))
            o_k = float(np.mean(y_arr[mask]))
            w_k = cnt / n_total
            reliability += w_k * ((f_k - o_k) ** 2)
            resolution += w_k * ((o_k - o_bar) ** 2)
            bins_out.append({
                "bin": f"[{edges[b]:.1f}, {edges[b + 1]:.1f}]",
                "count": cnt,
                "mean_predicted_prob": round(f_k, 4),
                "empirical_true_rate": round(o_k, 4),
                "abs_gap": round(abs(f_k - o_k), 4),
            })
        else:
            bins_out.append({
                "bin": f"[{edges[b]:.1f}, {edges[b + 1]:.1f}]",
                "count": 0,
                "mean_predicted_prob": None,
                "empirical_true_rate": None,
                "abs_gap": None,
            })

    return {
        "n_items": n_total,
        "brier_score": round(brier_raw, 6),
        "reliability": round(reliability, 6),
        "resolution": round(resolution, 6),
        "uncertainty": round(uncertainty, 6),
        "reconstructed_brier": round(reliability - resolution + uncertainty, 6),
        "reliability_diagram_bins": bins_out,
    }


def run_d7_calibration(config: WorldConfig, seeds: Sequence[int] = (0, 1, 2, 3, 4)) -> dict[str, Any]:
    datasets = [build_dataset(config, derive_seed(s, "world"), world_mode="randomized") for s in seeds]
    all_preds_nexus: list[Prediction] = []
    all_preds_nw: list[Prediction] = []
    all_preds_low_n: list[Prediction] = []
    all_preds_b: list[Prediction] = []
    all_truths_nc: list[bool] = []
    all_truths_nw: list[bool] = []

    for s, ds in zip(seeds, datasets):
        l_full = NEXUSLearner(seed=s)
        l_b = CrossSituationalLearner(seed=s)
        for ex in ds.train.items:
            obs = ex.observation()
            l_full.observe(obs)
            l_b.observe(obs)

        nc_items = ds.tests["test_novel_combination"].items
        all_preds_nexus.extend(predict_all(l_full, nc_items))
        all_preds_b.extend(predict_all(l_b, nc_items))
        all_truths_nc.extend([e.truth for e in nc_items])

        l_low = NEXUSLearner(seed=s)
        for ex in stratified_prefix(ds.train.items, 100, seed=s):
            l_low.observe(ex.observation())
        all_preds_low_n.extend(predict_all(l_low, nc_items))

        nw_split = ds.tests["test_novel_word_in_known_frame"]
        l_nw = copy.deepcopy(l_full)
        for ev in nw_split.support:
            if ev.kind == "observe":
                assert ev.observation is not None
                l_nw.observe(ev.observation)
        all_preds_nw.extend(predict_all(l_nw, nw_split.items))
        all_truths_nw.extend([e.truth for e in nw_split.items])

    return {
        "analysis": "EXPLORATORY",
        "nexus_N1600_novel_combination": brier_decomposition_and_diagram(all_preds_nexus, all_truths_nc),
        "nexus_N1600_novel_word_after_support": brier_decomposition_and_diagram(all_preds_nw, all_truths_nw),
        "nexus_N100_novel_combination": brier_decomposition_and_diagram(all_preds_low_n, all_truths_nc),
        "baseline_B_N1600_novel_combination": brier_decomposition_and_diagram(all_preds_b, all_truths_nc),
    }


# ---------------------------------------------------------------------------
# D8: Failure taxonomy across low-N, noisy, and structural boundary regimes
# ---------------------------------------------------------------------------

def run_d8_failure_taxonomy(config: WorldConfig, seeds: Sequence[int] = (0, 1, 2, 3, 4)) -> dict[str, Any]:
    datasets = [build_dataset(config, derive_seed(s, "world"), world_mode="randomized") for s in seeds]
    counts: Counter[str] = Counter()
    examples_by_mode: dict[str, list[dict[str, Any]]] = {}

    for s, ds in zip(seeds, datasets):
        # Regime 1: Low-resource N=60
        l_low = NEXUSLearner(seed=s)
        for ex in stratified_prefix(ds.train.items, 60, seed=s):
            l_low.observe(ex.observation())

        for ex in ds.tests["test_novel_combination"].items[:60]:
            exp = l_low.explain(ex.tokens, ex.scene)
            if exp.truth != ex.truth:
                if exp.derived_lf is None and "Unknown words" in exp.failure_reason:
                    mode = "1_unseen_word_at_low_N"
                elif exp.derived_lf is None:
                    mode = "2_unparseable_slash_direction_at_low_N"
                else:
                    mode = "3_underdetermined_lexical_mapping_at_low_N"
                counts[mode] += 1
                if len(examples_by_mode.setdefault(mode, [])) < 3:
                    examples_by_mode[mode].append({
                        "seed": s,
                        "example_id": ex.id,
                        "tokens": list(ex.tokens),
                        "ground_truth_lf": ex.lf.canonical(),
                        "derived_lf": None if exp.derived_lf is None else exp.derived_lf.canonical(),
                        "failure_reason": exp.failure_reason,
                    })

        # Regime 2: Slash depth <= 1 ablation on transitive sentences
        l_d1 = NEXUSLearner(seed=s, max_slash_depth=1)
        for ex in ds.train.items[:200]:
            l_d1.observe(ex.observation())
        tr_items = [e for e in ds.dev.items if e.construction == "trans_assert"][:10]
        for ex in tr_items:
            exp = l_d1.explain(ex.tokens, ex.scene)
            if exp.derived_lf is None:
                mode = "4_depth2_functor_expressivity_boundary"
                counts[mode] += 1
                if len(examples_by_mode.setdefault(mode, [])) < 3:
                    examples_by_mode[mode].append({
                        "seed": s,
                        "example_id": ex.id,
                        "tokens": list(ex.tokens),
                        "ground_truth_lf": ex.lf.canonical(),
                        "derived_lf": None,
                        "failure_reason": exp.failure_reason,
                    })

    return {
        "analysis": "EXPLORATORY",
        "failure_counts": dict(counts),
        "sample_failures_by_category": examples_by_mode,
    }


# ---------------------------------------------------------------------------
# D9: SQLite persistence round-trip in a fresh subprocess
# ---------------------------------------------------------------------------

def run_d9_sqlite_roundtrip(config: WorldConfig, seed: int = 0) -> dict[str, Any]:
    ds = build_dataset(config, derive_seed(seed, "world"), world_mode="randomized")
    learner = NEXUSLearner(seed=seed)
    for ex in ds.train.items:
        learner.observe(ex.observation())

    # Also apply a contradiction stream and record what_changed
    contra = ds.tests["test_contradiction"]
    target_corr_id = ""
    for ev in contra.support:
        if ev.kind == "observe":
            assert ev.observation is not None
            learner.observe(ev.observation)
        else:
            assert ev.correction is not None
            learner.correct(ev.correction)
            target_corr_id = ev.correction.target_example_id

    test_items = ds.tests["test_novel_combination"].items[:25]
    orig_preds = [learner.predict(e.scene, e.tokens).p_true for e in test_items]
    orig_exp = learner.explain(test_items[0].tokens, test_items[0].scene).to_json()
    orig_change = learner.what_changed(target_corr_id).to_json()

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "nexus_state.sqlite3"
        repo = SQLiteLearnerRepository(db_path)
        repo.save_learner("seed0_run", learner)

        # Reload inside a fresh Python subprocess to verify zero in-memory state sharing
        items_json = json.dumps([e.to_json() for e in test_items])
        cmd = [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "from nexus.storage import SQLiteLearnerRepository; "
                "from nexus.world.scene import Scene; "
                "repo = SQLiteLearnerRepository(sys.argv[1]); "
                "loaded = repo.load_learner('seed0_run'); "
                "items = json.loads(sys.argv[2]); "
                "preds = [loaded.predict(Scene.from_json(it['scene']), tuple(it['tokens'])).p_true for it in items]; "
                "exp0 = loaded.explain(tuple(items[0]['tokens']), Scene.from_json(items[0]['scene'])).to_json(); "
                "chg = loaded.what_changed(sys.argv[3]).to_json(); "
                "print(json.dumps({'preds': preds, 'exp0': exp0, 'chg': chg, 'runs': repo.list_runs()}))"
            ),
            str(db_path),
            items_json,
            target_corr_id,
        ]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        sub_out = json.loads(proc.stdout.strip())

    max_pred_diff = float(np.max(np.abs(np.array(orig_preds) - np.array(sub_out["preds"]))))
    exp_identical = orig_exp == sub_out["exp0"]
    change_identical = orig_change == sub_out["chg"]

    return {
        "analysis": "EXPLORATORY",
        "subprocess_roundtrip_success": bool(max_pred_diff == 0.0 and exp_identical and change_identical),
        "max_prediction_abs_diff": max_pred_diff,
        "explanation_provenance_identical": exp_identical,
        "what_changed_identical": change_identical,
        "stored_runs": sub_out["runs"],
    }


def main() -> None:
    t0 = time.perf_counter()
    config = load_world_config()
    seeds = (0, 1, 2, 3, 4)

    print("Running D1 (Beam width ablation)...")
    d1 = run_d1_beam_width(config, seeds)
    print("Running D2 (Noise epsilon ablation)...")
    d2 = run_d2_noise_epsilon(config, seeds)
    print("Running D3 (Learning curves & A8 Ideal Observer gap)...")
    d3 = run_d3_learning_curves(config, seeds)
    print("Running D4 (Hypothesis-space ablations)...")
    d4 = run_d4_ablations(config, seeds)
    print("Running D5 (Label noise before/after corrections)...")
    d5 = run_d5_label_noise(config, seeds)
    print("Running D6 (New held-out probes)...")
    d6 = run_d6_new_probes(config, seeds)
    print("Running D7 (Calibration reliability diagrams & Brier decomposition)...")
    d7 = run_d7_calibration(config, seeds)
    print("Running D8 (Failure taxonomy)...")
    d8 = run_d8_failure_taxonomy(config, seeds)
    print("Running D9 (SQLite persistence fresh-subprocess round-trip)...")
    d9 = run_d9_sqlite_roundtrip(config, seed=0)

    payload = {
        "stage": "STAGE_D_EXPLORATORY",
        "label": "EXPLORATORY",
        "seeds": list(seeds),
        "elapsed_seconds": time.perf_counter() - t0,
        "D1_beam_width": d1,
        "D2_noise_epsilon": d2,
        "D3_learning_curves": d3,
        "D4_hypothesis_space_ablations": d4,
        "D5_label_noise": d5,
        "D6_new_heldout_probes": d6,
        "D7_calibration_brier_decomposition": d7,
        "D8_failure_taxonomy": d8,
        "D9_sqlite_persistence_roundtrip": d9,
    }
    out_path = save_results(payload, RESULTS_DIR / "stage_d_exploratory.json")
    print(f"Saved Stage D exploratory results to {out_path} in {payload['elapsed_seconds']:.2f}s")


if __name__ == "__main__":
    main()
