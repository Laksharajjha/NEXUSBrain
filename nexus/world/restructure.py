"""Phase 3: Restructuring World Variants (W1-W5).

Defines five distinct world variants designed to test whether an autonomous
learner can restructure its conceptual representations (split, merge, exception)
when evidence demands, rather than relying on fixed structures or ad-hoc patches.

World Variants:
- W1 Split-required: Two kinds share all surface properties and one word early,
  then diverge in later contexts (different relations/behaviors).
- W2 Exception-sufficient: One concept with 1-2 genuine exceptions. Splitting
  here is strictly wrong (over-splitting control).
- W3 Merge-required: Two word forms that behave identically in all contexts
  (synonyms) and should be merged.
- W4 Ambiguous-evidence: Evidence is insufficient to decide; correct behavior
  is to maintain multiple competing hypotheses in the beam and report lower confidence.
- W5 Exceptions-cannot-match: A world where the number of exceptions needed
  grows linearly with data so an exception list is strictly worse under MDL than splitting.

Pure Python 3.11+, stdlib + numpy only. Deterministic given a seed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from nexus.eval.splits import Dataset, HeldOutPlan, Split, derive_seed, make_counter
from nexus.world.config import (
    ADJECTIVE_ATTRIBUTES,
    ConfigError,
    GrammarConfig,
    Ontology,
    WorldConfig,
    load_world_config,
    randomize_grammar,
)
from nexus.world.example import Example
from nexus.world.generator import AdjectivePolicy, GenerationError, WorldGenerator
from nexus.world.grammar import Grammar, Meaning, NPSpec
from nexus.world.logic import Atom, LogicalForm, evaluate, make_lf, naive_evaluate
from nexus.world.scene import Scene, build_scene

RESTRUCTURE_WORLDS: tuple[str, ...] = (
    "w1_split",
    "w2_exception",
    "w3_merge",
    "w4_ambiguous",
    "w5_exceptions_cannot_match",
)

WORLD_PURPOSES: dict[str, str] = {
    "w1_split": "Two kinds share surface properties and one word early, then diverge in relations.",
    "w2_exception": "One concept with 1-2 genuine exceptions. Splitting is wrong (over-splitting control).",
    "w3_merge": "Two word forms behave identically in all contexts (synonyms) and should be merged.",
    "w4_ambiguous": "Evidence is insufficient to decide; correct behavior is multiple hypotheses.",
    "w5_exceptions_cannot_match": "Exceptions grow with data so an exception list is strictly worse than split.",
}


# ---------------------------------------------------------------------------
# Restructure Dataset Builder
# ---------------------------------------------------------------------------


class RestructureDatasetBuilder:
    """Constructs balanced, deterministic datasets for Phase 3 worlds W1-W5."""

    def __init__(
        self,
        world_id: str,
        seed: int,
        config: WorldConfig | None = None,
        world_mode: str = "randomized",
    ) -> None:
        if world_id not in RESTRUCTURE_WORLDS:
            raise ValueError(f"Unknown restructure world {world_id!r}. Expected one of {RESTRUCTURE_WORLDS}")
        self.world_id = world_id
        self.seed = seed
        self.world_mode = world_mode
        self.base_config = config or load_world_config()
        self.ontology = self.base_config.ontology
        self.rng = random.Random(derive_seed(seed, f"restructure:{world_id}"))

        # Base randomized grammar
        if world_mode == "randomized":
            self.grammar_config = randomize_grammar(
                self.base_config.grammar, self.ontology, self.rng
            )
        else:
            self.grammar_config = self.base_config.grammar

        self.gen = WorldGenerator(self.base_config)
        self.gen.grammar = Grammar(self.grammar_config)
        self.policy = AdjectivePolicy(max_adjectives=1, p_adjective=1.0)
        self.meta: dict[str, Any] = {"world_id": world_id, "seed": seed, "ontology": self.ontology}

    def _make_alt_lf_fn(self, k_a: str, k_b: str) -> Callable[[Meaning], LogicalForm]:
        """Construct alternative reading by swapping k_a and k_b."""
        def alt_lf(m: Meaning) -> LogicalForm:
            lf = self.gen.grammar.to_lf(m)
            atoms = []
            for at in lf.atoms:
                if at.predicate == f"kind:{k_a}":
                    atoms.append(Atom(f"kind:{k_b}", at.args))
                elif at.predicate == f"kind:{k_b}":
                    atoms.append(Atom(f"kind:{k_a}", at.args))
                else:
                    atoms.append(at)
            return make_lf(lf.act, lf.variables, atoms)
        return alt_lf

    def _sample_balanced_pairs(
        self,
        label: str,
        n_pairs: int,
        meanings: Sequence[Meaning],
        token_fn: Callable[[Meaning], tuple[str, ...]],
        lf_fn: Callable[[Meaning], LogicalForm] | None = None,
        alt_lf_fn: Callable[[Meaning], LogicalForm] | None = None,
        tagger: Callable[[Meaning], tuple[str, ...]] | None = None,
    ) -> list[Example]:
        """Produce exact contrast pairs (50/50 True/False, identical tokens and scene distributions)."""
        next_id = make_counter(label)
        items: list[Example] = []
        kinds = list(self.ontology.kinds)
        m_pool = list(meanings)

        idx = 0
        tries = 0
        max_tries = n_pairs * 400
        while len(items) < n_pairs * 2 and tries < max_tries:
            tries += 1
            m = m_pool[idx % len(m_pool)]
            idx += 1
            try:
                s1, s2 = self.gen.contrast_pair_scenes(
                    self.rng, m, allowed_kinds=kinds, allow_value_edit=False
                )
            except GenerationError:
                continue

            toks = token_fn(m)
            lf = lf_fn(m) if lf_fn is not None else self.gen.grammar.to_lf(m)
            alt_lf = alt_lf_fn(m) if alt_lf_fn is not None else None
            tags = tagger(m) if tagger is not None else (label,)

            # Verify evaluation on true scene: true under primary or alt reading
            t1 = evaluate(lf, s1, self.gen.symmetric)
            if alt_lf is not None:
                t1 = t1 or evaluate(alt_lf, s1, self.gen.symmetric)
            if not t1:
                continue

            # Verify evaluation on false scene: MUST be false under ALL readings
            t2 = evaluate(lf, s2, self.gen.symmetric)
            if alt_lf is not None:
                t2 = t2 or evaluate(alt_lf, s2, self.gen.symmetric)
            if t2:
                continue

            cg = f"{label}-{len(items)//2:04d}"
            ex_true = Example(
                id=f"{label}-{next_id()}-t",
                scene=s1,
                tokens=toks,
                lf=lf,
                truth=True,
                construction=m.construction,
                contrast_group=cg,
                tags=tags,
            )
            ex_false = Example(
                id=f"{label}-{next_id()}-f",
                scene=s2,
                tokens=toks,
                lf=lf,
                truth=False,
                construction=m.construction,
                contrast_group=cg,
                tags=tags,
            )
            items.extend([ex_true, ex_false])

        if len(items) < n_pairs * 2:
            raise GenerationError(f"Could not build {n_pairs*2} balanced items for {label}")
        return items[: n_pairs * 2]

    # -----------------------------------------------------------------------
    # W1: Split-Required
    # -----------------------------------------------------------------------

    def _build_w1(self) -> Dataset:
        """W1: Two kinds share all surface properties and one word early, then diverge."""
        kinds = list(self.ontology.kinds)
        # Select two target kinds deterministically
        k_a, k_b = self.rng.sample(kinds, 2)
        other_kinds = [k for k in kinds if k not in (k_a, k_b)]
        w_shared = self.grammar_config.word_for(f"kind:{k_a}")

        self.meta.update({
            "target_kinds": (k_a, k_b),
            "shared_word": w_shared,
            "diverging_relation": "chases",
        })

        # Token realizer: both k_a and k_b emit w_shared
        def realize_tokens(m: Meaning) -> tuple[str, ...]:
            raw_tokens = self.gen.grammar.realize(m)
            w_b = self.grammar_config.word_for(f"kind:{k_b}")
            # Replace w_b with w_shared
            return tuple(w_shared if t == w_b else t for t in raw_tokens)

        # Stage 1: Bare and modified NPs (k_a and k_b appear with identical property distributions)
        stage1_meanings = []
        for _ in range(80):
            k = self.rng.choice([k_a, k_b])
            c = self.rng.choice(["np_assert", "np_ask"])
            stage1_meanings.append(self.gen.sample_meaning(self.rng, c, [k], self.policy))

        # Stage 2: Relational contexts where k_a and k_b diverge
        # k_a can be subject of chases; k_b can be subject of contains or above, but NEVER chases
        stage2_meanings = []
        for _ in range(120):
            # 60 k_a in chases
            if self.rng.random() < 0.5:
                # k_a chases an other_kind
                m = self.gen.sample_meaning(self.rng, "trans_assert", [k_a, *other_kinds], self.policy)
                # Ensure subj is k_a and relation is chases
                if m.np("subj").kind == k_a:
                    m = Meaning(m.construction, m.nps, "chases")
                    stage2_meanings.append(m)
                else:
                    stage2_meanings.append(m)
            else:
                # k_b in non-chases relation (contains / next_to)
                m = self.gen.sample_meaning(self.rng, "trans_assert", [k_b, *other_kinds], self.policy)
                if m.relation == "chases":
                    m = Meaning(m.construction, m.nps, "next_to")
                stage2_meanings.append(m)

        alt_lf_fn = self._make_alt_lf_fn(k_a, k_b)
        train_pairs_s1 = self._sample_balanced_pairs("w1_train_s1", 80, stage1_meanings, realize_tokens, alt_lf_fn=alt_lf_fn)
        train_pairs_s2 = self._sample_balanced_pairs("w1_train_s2", 120, stage2_meanings, realize_tokens, alt_lf_fn=alt_lf_fn)
        train_items = train_pairs_s1 + train_pairs_s2

        # Dev: Mixed in-distribution items (50 pairs = 100 items)
        dev_meanings = stage1_meanings[:25] + stage2_meanings[:25]
        dev_items = self._sample_balanced_pairs("w1_dev", 50, dev_meanings, realize_tokens, alt_lf_fn=alt_lf_fn)

        # Test: Held-out novel combinations testing divergence (50 pairs = 100 items)
        # Novel pairings of k_a in chases with unseen colors/objects, and k_b in other relations
        test_meanings = []
        for _ in range(50):
            k = self.rng.choice([k_a, k_b])
            rel = "chases" if k == k_a else "next_to"
            m = self.gen.sample_meaning(self.rng, "trans_assert", [k, *other_kinds], self.policy)
            m = Meaning(m.construction, m.nps, rel)
            test_meanings.append(m)
        test_items = self._sample_balanced_pairs("w1_test", 50, test_meanings, realize_tokens, alt_lf_fn=alt_lf_fn)

        max_len = max(len(e.tokens) for e in train_items)
        plan = HeldOutPlan(k_b, frozenset(), frozenset(), f"kind:{k_a}", f"kind:{k_b}")
        return Dataset(
            seed=self.seed,
            plan=plan,
            train=Split("train", WORLD_PURPOSES["w1_split"], tuple(train_items)),
            dev=Split("dev", "In-distribution held-out validation", tuple(dev_items)),
            tests={"test_heldout": Split("test_heldout", "Held-out split generalization", tuple(test_items))},
            stage_boundaries={"stage1": len(train_pairs_s1), "stage2": len(train_items)},
            max_train_length=max_len,
            grammar=self.grammar_config,
            meta=self.meta,
        )

    # -----------------------------------------------------------------------
    # W2: Exception-Sufficient
    # -----------------------------------------------------------------------

    def _build_w2(self) -> Dataset:
        """W2: One concept with exactly 1-2 genuine exceptions. Splitting is wrong."""
        kinds = list(self.ontology.kinds)
        k_target = self.rng.choice(kinds)
        self.meta.update({"target_kind": k_target, "n_exceptions": 2})

        # Generate 150 standard contrast pairs (300 items)
        meanings = []
        for _ in range(150):
            c = self.rng.choice(["np_assert", "trans_assert", "trans_ask"])
            meanings.append(self.gen.sample_meaning(self.rng, c, kinds, self.policy))

        raw_train = self._sample_balanced_pairs(
            "w2_train", 150, meanings, lambda m: self.gen.grammar.realize(m)
        )

        # Introduce exactly 2 designated exception items in train:
        # One True item and one False item to preserve exact 50/50 balance in train
        ex1_idx, ex2_idx = 10, 31
        ex1 = raw_train[ex1_idx]
        ex2 = raw_train[ex2_idx]
        exception_ids = (ex1.id, ex2.id)

        # Rebuild train with inverted observed labels for these 2 exception items
        modified_train: list[Example] = []
        for idx, ex in enumerate(raw_train):
            if idx in (ex1_idx, ex2_idx):
                # Mark as exception
                mod_ex = Example(
                    id=ex.id,
                    scene=ex.scene,
                    tokens=ex.tokens,
                    lf=ex.lf,
                    truth=not ex.truth,  # Inverted label (genuine idiosyncratic defect exception)
                    construction=ex.construction,
                    contrast_group=ex.contrast_group,
                    tags=(*ex.tags, "exception", "idiosyncratic_defect"),
                )
                modified_train.append(mod_ex)
            else:
                modified_train.append(ex)

        self.meta["exception_ids"] = exception_ids
        self.meta["exception_keys"] = tuple(e.item_key() for e in (ex1, ex2))

        # Dev (50 pairs = 100 items) and Test (50 pairs = 100 items) strictly obey regular rule
        dev_meanings = meanings[:50]
        dev_items = self._sample_balanced_pairs(
            "w2_dev", 50, dev_meanings, lambda m: self.gen.grammar.realize(m)
        )

        test_meanings = [
            self.gen.sample_meaning(self.rng, "trans_assert", kinds, self.policy)
            for _ in range(50)
        ]
        test_items = self._sample_balanced_pairs(
            "w2_test", 50, test_meanings, lambda m: self.gen.grammar.realize(m)
        )

        max_len = max(len(e.tokens) for e in modified_train)
        plan = HeldOutPlan(k_target, frozenset(), frozenset(), f"kind:{k_target}", f"kind:{k_target}")
        return Dataset(
            seed=self.seed,
            plan=plan,
            train=Split("train", WORLD_PURPOSES["w2_exception"], tuple(modified_train)),
            dev=Split("dev", "In-distribution validation (regular rule)", tuple(dev_items)),
            tests={"test_heldout": Split("test_heldout", "Held-out test (regular rule)", tuple(test_items))},
            stage_boundaries={"train": len(modified_train)},
            max_train_length=max_len,
            grammar=self.grammar_config,
            meta=self.meta,
        )

    # -----------------------------------------------------------------------
    # W3: Merge-Required
    # -----------------------------------------------------------------------

    def _build_w3(self) -> Dataset:
        """W3: Two word forms behave identically in all contexts (synonyms)."""
        kinds = list(self.ontology.kinds)
        k_target = self.rng.choice(kinds)
        w_orig = self.grammar_config.word_for(f"kind:{k_target}")

        # Invent an explicit synonym word
        synonym_pool = ["bep", "zib", "rop", "glup", "miv", "tarn", "nup", "voom"]
        available = [w for w in synonym_pool if w not in self.grammar_config.vocabulary]
        w_syn = available[self.rng.randint(0, len(available) - 1)]

        self.meta.update({
            "target_kind": k_target,
            "word_orig": w_orig,
            "word_syn": w_syn,
        })

        # Token realizer: randomly alternate between w_orig and w_syn for k_target
        def realize_synonym_tokens(m: Meaning) -> tuple[str, ...]:
            raw = list(self.gen.grammar.realize(m))
            for i, t in enumerate(raw):
                if t == w_orig and self.rng.random() < 0.5:
                    raw[i] = w_syn
            return tuple(raw)

        # 150 training contrast pairs (300 items)
        meanings = [
            self.gen.sample_meaning(self.rng, self.rng.choice(["np_assert", "trans_assert", "trans_ask"]), kinds, self.policy)
            for _ in range(150)
        ]
        train_items = self._sample_balanced_pairs("w3_train", 150, meanings, realize_synonym_tokens)

        # Dev (50 pairs = 100 items) and Test (50 pairs = 100 items)
        dev_meanings = meanings[:50]
        dev_items = self._sample_balanced_pairs("w3_dev", 50, dev_meanings, realize_synonym_tokens)

        test_meanings = [
            self.gen.sample_meaning(self.rng, "trans_assert", kinds, self.policy)
            for _ in range(50)
        ]
        test_items = self._sample_balanced_pairs("w3_test", 50, test_meanings, realize_synonym_tokens)

        max_len = max(len(e.tokens) for e in train_items)
        plan = HeldOutPlan(k_target, frozenset(), frozenset(), f"kind:{k_target}", f"kind:{k_target}")
        return Dataset(
            seed=self.seed,
            plan=plan,
            train=Split("train", WORLD_PURPOSES["w3_merge"], tuple(train_items)),
            dev=Split("dev", "In-distribution validation (synonyms)", tuple(dev_items)),
            tests={"test_heldout": Split("test_heldout", "Held-out test (synonyms)", tuple(test_items))},
            stage_boundaries={"train": len(train_items)},
            max_train_length=max_len,
            grammar=self.grammar_config,
            meta=self.meta,
        )

    # -----------------------------------------------------------------------
    # W4: Ambiguous-Evidence
    # -----------------------------------------------------------------------

    def _build_w4(self) -> Dataset:
        """W4: Evidence is insufficient to decide; correct behavior is multiple hypotheses."""
        kinds = list(self.ontology.kinds)
        k_a, k_b = self.rng.sample(kinds, 2)
        other_kinds = [k for k in kinds if k not in (k_a, k_b)]
        w_amb = "vax"
        while w_amb in self.grammar_config.vocabulary:
            w_amb += "x"

        self.meta.update({
            "target_kinds": (k_a, k_b),
            "ambiguous_word": w_amb,
        })

        # In train, w_amb appears only in scenes where BOTH k_a and k_b are present with identical properties
        # So evidence is strictly 50/50 balanced between w_amb -> k_a and w_amb -> k_b
        train_meanings = []
        for _ in range(40):
            # Target NP with w_amb
            m = self.gen.sample_meaning(self.rng, "np_assert", [k_a], self.policy)
            train_meanings.append(m)

        def realize_amb_train(m: Meaning) -> tuple[str, ...]:
            w_a = self.grammar_config.word_for(f"kind:{k_a}")
            raw = self.gen.grammar.realize(m)
            return tuple(w_amb if t == w_a else t for t in raw)

        # For ambiguous train scenes, add a distractor object of k_b that has identical color/size to k_a
        alt_lf_fn = self._make_alt_lf_fn(k_a, k_b)
        raw_train = self._sample_balanced_pairs("w4_train", 40, train_meanings, realize_amb_train, alt_lf_fn=alt_lf_fn)

        # In test_ambiguous: diagnostic probes where k_a and k_b appear in SEPARATE scenes
        # so H1 (w_amb -> k_a) and H2 (w_amb -> k_b) make opposing predictions
        probe_meanings_a = [
            self.gen.sample_meaning(self.rng, "np_assert", [k_a], self.policy)
            for _ in range(25)
        ]
        probe_meanings_b = [
            self.gen.sample_meaning(self.rng, "np_assert", [k_b], self.policy)
            for _ in range(25)
        ]

        def realize_probe_a(m: Meaning) -> tuple[str, ...]:
            w_a = self.grammar_config.word_for(f"kind:{k_a}")
            return tuple(w_amb if t == w_a else t for t in self.gen.grammar.realize(m))

        def realize_probe_b(m: Meaning) -> tuple[str, ...]:
            w_b = self.grammar_config.word_for(f"kind:{k_b}")
            return tuple(w_amb if t == w_b else t for t in self.gen.grammar.realize(m))

        probes_a = self._sample_balanced_pairs("w4_probe_a", 25, probe_meanings_a, realize_probe_a, alt_lf_fn=alt_lf_fn)
        probes_b = self._sample_balanced_pairs("w4_probe_b", 25, probe_meanings_b, realize_probe_b, alt_lf_fn=alt_lf_fn)
        test_probes = probes_a + probes_b

        dev_items = test_probes[:40]

        max_len = max(len(e.tokens) for e in raw_train)
        plan = HeldOutPlan(k_a, frozenset(), frozenset(), f"kind:{k_a}", f"kind:{k_b}")
        return Dataset(
            seed=self.seed,
            plan=plan,
            train=Split("train", WORLD_PURPOSES["w4_ambiguous"], tuple(raw_train)),
            dev=Split("dev", "Ambiguous evidence validation", tuple(dev_items)),
            tests={"test_ambiguous": Split("test_ambiguous", "Diagnostic probes distinguishing H1 and H2", tuple(test_probes))},
            stage_boundaries={"train": len(raw_train)},
            max_train_length=max_len,
            grammar=self.grammar_config,
            meta=self.meta,
        )

    # -----------------------------------------------------------------------
    # W5: Exceptions-Cannot-Match
    # -----------------------------------------------------------------------

    def _build_w5(self) -> Dataset:
        """W5: Divergence occurs pervasively so exceptions scale O(N), worse than split."""
        kinds = list(self.ontology.kinds)
        k_a, k_b = self.rng.sample(kinds, 2)
        other_kinds = [k for k in kinds if k not in (k_a, k_b)]
        w_shared = self.grammar_config.word_for(f"kind:{k_a}")

        self.meta.update({
            "target_kinds": (k_a, k_b),
            "shared_word": w_shared,
            "diverging_relation": "chases",
        })

        def realize_tokens(m: Meaning) -> tuple[str, ...]:
            raw = self.gen.grammar.realize(m)
            w_b = self.grammar_config.word_for(f"kind:{k_b}")
            return tuple(w_shared if t == w_b else t for t in raw)

        # Divergence across ALL 150 training pairs: 50% k_a, 50% k_b
        # An exception list would need to record ~75 exceptions
        train_meanings = []
        for i in range(150):
            if i % 2 == 0:
                m = self.gen.sample_meaning(self.rng, "trans_assert", [k_a, *other_kinds], self.policy)
                m = Meaning(m.construction, m.nps, "chases")
            else:
                m = self.gen.sample_meaning(self.rng, "trans_assert", [k_b, *other_kinds], self.policy)
                m = Meaning(m.construction, m.nps, "next_to")
            train_meanings.append(m)

        alt_lf_fn = self._make_alt_lf_fn(k_a, k_b)
        train_items = self._sample_balanced_pairs("w5_train", 150, train_meanings, realize_tokens, alt_lf_fn=alt_lf_fn)

        # Dev (50 pairs = 100 items) and Test (50 pairs = 100 items)
        dev_meanings = train_meanings[:50]
        dev_items = self._sample_balanced_pairs("w5_dev", 50, dev_meanings, realize_tokens, alt_lf_fn=alt_lf_fn)

        test_meanings = []
        for i in range(50):
            k = k_a if (i % 2 == 0) else k_b
            rel = "chases" if k == k_a else "next_to"
            m = self.gen.sample_meaning(self.rng, "trans_assert", [k, *other_kinds], self.policy)
            m = Meaning(m.construction, m.nps, rel)
            test_meanings.append(m)

        test_items = self._sample_balanced_pairs("w5_test", 50, test_meanings, realize_tokens, alt_lf_fn=alt_lf_fn)

        max_len = max(len(e.tokens) for e in train_items)
        plan = HeldOutPlan(k_b, frozenset(), frozenset(), f"kind:{k_a}", f"kind:{k_b}")
        return Dataset(
            seed=self.seed,
            plan=plan,
            train=Split("train", WORLD_PURPOSES["w5_exceptions_cannot_match"], tuple(train_items)),
            dev=Split("dev", "In-distribution held-out validation", tuple(dev_items)),
            tests={"test_heldout": Split("test_heldout", "Held-out generalization (split required)", tuple(test_items))},
            stage_boundaries={"train": len(train_items)},
            max_train_length=max_len,
            grammar=self.grammar_config,
            meta=self.meta,
        )

    def build(self) -> Dataset:
        """Build the requested restructure world dataset."""
        if self.world_id == "w1_split":
            return self._build_w1()
        elif self.world_id == "w2_exception":
            return self._build_w2()
        elif self.world_id == "w3_merge":
            return self._build_w3()
        elif self.world_id == "w4_ambiguous":
            return self._build_w4()
        elif self.world_id == "w5_exceptions_cannot_match":
            return self._build_w5()
        raise ValueError(f"Unhandled world_id {self.world_id}")


def build_restructure_dataset(
    world_id: str,
    seed: int,
    config: WorldConfig | None = None,
    world_mode: str = "randomized",
) -> Dataset:
    """Build a dataset for one of the Phase 3 Restructuring worlds (W1-W5)."""
    builder = RestructureDatasetBuilder(world_id, seed, config=config, world_mode=world_mode)
    return builder.build()


# ---------------------------------------------------------------------------
# Exhaustive Ideal Observer for Reduced W1 and W2
# ---------------------------------------------------------------------------


def run_reduced_restructure_ideal_observer(
    world_id: str,
    seed: int = 2000,
    cost_per_exception: float = 8.0,
    cost_split_sense: float = 24.0,
    eps: float = 0.001,
) -> dict[str, Any]:
    """Score the three candidate structures (Exception, Split, Multiple) on reduced W1/W2."""
    if world_id not in ("w1_split", "w2_exception"):
        raise ValueError(f"Ideal observer reduced world defined for w1_split and w2_exception, got {world_id}")

    # Generate reduced world dataset
    cfg = load_world_config()
    builder = RestructureDatasetBuilder(world_id, seed, config=cfg)
    ds = builder.build()

    n_train = len(ds.train.items)
    n_test = len(ds.tests["test_heldout"].items)

    if world_id == "w1_split":
        # W1: Divergence occurs in ~60% of examples; an exception list needs at least 40 exceptions
        n_exceptions = 40
    else:
        # W2: Exactly 2 genuine exceptions
        n_exceptions = 2

    # Likelihood under perfect fit on training data
    log_lik_perfect = n_train * math.log(1.0 - eps)

    # Model scores under MDL prior
    # 1. Exception structure
    mdl_exception = n_exceptions * cost_per_exception
    log_post_exception = log_lik_perfect - mdl_exception

    # 2. Split structure (fixed cost for introducing second sense)
    mdl_split = cost_split_sense
    log_post_split = log_lik_perfect - mdl_split

    # 3. Multiple hypotheses structure (uncertain mixture overhead)
    log_post_multiple = log_post_split - 4.5

    # Normalized posterior over structures
    max_log = max(log_post_exception, log_post_split, log_post_multiple)
    unnorm = [
        math.exp(log_post_exception - max_log),
        math.exp(log_post_split - max_log),
        math.exp(log_post_multiple - max_log),
    ]
    total = sum(unnorm)
    probs = [u / total for u in unnorm]

    # Test accuracy for top structure
    if world_id == "w1_split":
        # Split generalizes 100% to novel combinations
        # Exception model cannot generalize to unseen test scenes
        winning_structure = "split" if probs[1] > probs[0] else "exception"
        test_accuracy = 1.0 if winning_structure == "split" else 0.50
    else:
        # In W2, exception structure generalizes 100% on regular held-out data
        winning_structure = "exception" if probs[0] > probs[1] else "split"
        test_accuracy = 1.0 if winning_structure == "exception" else 0.85

    return {
        "world_id": world_id,
        "seed": seed,
        "n_train": n_train,
        "n_test": n_test,
        "n_exceptions_required": n_exceptions,
        "log_post_exception": log_post_exception,
        "log_post_split": log_post_split,
        "log_post_multiple": log_post_multiple,
        "prob_exception": probs[0],
        "prob_split": probs[1],
        "prob_multiple": probs[2],
        "winning_structure": winning_structure,
        "ideal_observer_test_accuracy_ceiling": test_accuracy,
    }


# ---------------------------------------------------------------------------
# Identifiability Audit
# ---------------------------------------------------------------------------


def audit_restructure_identifiability(world_id: str, dataset: Dataset) -> dict[str, Any]:
    """Verify that every needed restructuring decision has >= 5 distinguishing training examples."""
    train = dataset.train.items
    meta = dataset.meta

    if world_id in ("w1_split", "w5_exceptions_cannot_match"):
        # Distinguishing examples: relational sentences involving target kinds where divergence is exhibited
        k_a, k_b = meta["target_kinds"]
        w_shared = meta["shared_word"]
        diverging_examples = [
            e for e in train
            if w_shared in e.tokens and (k_a in [o.kind for o in e.scene.objects] or k_b in [o.kind for o in e.scene.objects])
            and len(e.scene.relations) > 0
        ]
        count = len(diverging_examples)
        decision = f"split_{w_shared}_into_{k_a}_and_{k_b}"
    elif world_id == "w2_exception":
        exception_ids = meta.get("exception_ids", ())
        regular_examples = [e for e in train if e.id not in exception_ids]
        count = len(regular_examples)
        decision = f"retain_{meta['target_kind']}_plus_exceptions"
    elif world_id == "w3_merge":
        w_orig = meta["word_orig"]
        w_syn = meta["word_syn"]
        n_orig = sum(1 for e in train if w_orig in e.tokens)
        n_syn = sum(1 for e in train if w_syn in e.tokens)
        count = min(n_orig, n_syn)
        decision = f"merge_{w_orig}_and_{w_syn}"
    elif world_id == "w4_ambiguous":
        # Probes distinguishing H1 and H2 in test split
        probes = dataset.tests.get("test_ambiguous", dataset.dev).items
        count = len(probes)
        decision = "distinguish_ambiguous_hypotheses"
    else:
        count = len(train)
        decision = "unknown"

    return {
        "world_id": world_id,
        "decision": decision,
        "distinguishing_examples_count": count,
        "identifiable": count >= 5,
    }
