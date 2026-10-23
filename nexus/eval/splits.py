"""Split builders.

Every split has a name and a stated purpose (see :data:`SPLIT_PURPOSES`).
Held-out structure (novel noun, held-out adjective-noun pairs, held-out
noun-role pairs, contradiction target word) is chosen per seed by
:func:`choose_plan`, so results average over many different hold-outs.

Splits that allow learning before evaluation (novel word, contradiction)
carry a ``support`` stream of :class:`StreamEvent`; the harness evaluates
``items`` at each of the ``support_checkpoints``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from nexus.world.config import ADJECTIVE_ATTRIBUTES, GrammarConfig, WorldConfig, randomize_grammar
from nexus.world.example import Correction, Example, StreamEvent
from nexus.world.generator import AdjectivePolicy, GenerationError, WorldGenerator
from nexus.world.grammar import Meaning, NPSpec
from nexus.world.logic import evaluate

SPLIT_PURPOSES: dict[str, str] = {
    "train": "Learning stream (curriculum-ordered). Excludes the novel noun, held-out "
    "adjective-noun pairs, held-out noun-role pairs, and NPs with >1 adjective.",
    "dev": "In-distribution held-out items (same constraints as train, unseen (scene, "
    "sentence) pairs). Used for learning curves, examples-to-criterion and retention.",
    "test_novel_combination": "Known words in combinations never seen in train: a held-out "
    "adjective-noun pair or a noun in a held-out grammatical role.",
    "test_novel_word_in_known_frame": "A noun never seen in train. Support: one contrastive "
    "pair teaching it in a bare-NP frame. Items: the noun in other seen constructions.",
    "test_longer": "Sentences strictly longer than any training sentence (NPs with stacked "
    "adjectives). Only known words and seen combinations.",
    "test_contradiction": "Support stream where a teacher systematically mislabels sentences "
    "containing one adjective (as if it meant another color), later sends corrections. "
    "Items: diagnostic probes whose truth differs between the two readings.",
}


@dataclass(frozen=True)
class HeldOutPlan:
    """Per-seed choice of what is held out of training."""

    novel_kind: str
    adj_noun_pairs: frozenset[tuple[str, str]]  # (adjective predicate, kind predicate)
    noun_roles: frozenset[tuple[str, str]]  # (kind predicate, role)
    contradiction_target: str  # adjective predicate whose word is mislabelled
    contradiction_confusion: str  # predicate the teacher wrongly uses instead

    @property
    def novel_kind_predicate(self) -> str:
        """E.g. ``kind:cup``."""
        return f"kind:{self.novel_kind}"

    def to_json(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "novel_kind": self.novel_kind,
            "adj_noun_pairs": sorted(list(p) for p in self.adj_noun_pairs),
            "noun_roles": sorted(list(p) for p in self.noun_roles),
            "contradiction_target": self.contradiction_target,
            "contradiction_confusion": self.contradiction_confusion,
        }


@dataclass(frozen=True)
class Split:
    """A named evaluation (or training) split."""

    name: str
    purpose: str
    items: tuple[Example, ...]
    support: tuple[StreamEvent, ...] = ()
    support_checkpoints: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class Dataset:
    """All splits for one seed."""

    seed: int
    plan: HeldOutPlan
    train: Split
    dev: Split
    tests: dict[str, Split]
    stage_boundaries: dict[str, int]  # construction -> train index where its stage ends
    max_train_length: int
    grammar: GrammarConfig | None = None
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def derive_seed(seed: int, label: str) -> int:
    """Deterministic, platform-independent sub-seed (not Python's salted hash)."""
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def make_counter(prefix: str) -> Callable[[], str]:
    """Return a function yielding ``prefix-00000``, ``prefix-00001``, ..."""
    state = {"i": 0}

    def next_id() -> str:
        value = f"{prefix}-{state['i']:05d}"
        state["i"] += 1
        return value

    return next_id


def meaning_kinds(m: Meaning) -> set[str]:
    """Kind predicates used by a meaning."""
    return {spec.kind_predicate for _, spec in m.nps}


def meaning_adjectives(m: Meaning) -> set[str]:
    """Adjective predicates used by a meaning."""
    return {p for _, spec in m.nps for p in spec.adjective_predicates}


def meaning_adj_noun_pairs(m: Meaning) -> set[tuple[str, str]]:
    """(adjective, kind) pairs that co-occur inside one NP."""
    return {(p, spec.kind_predicate) for _, spec in m.nps for p in spec.adjective_predicates}


def meaning_noun_roles(m: Meaning) -> set[tuple[str, str]]:
    """(kind, role) pairs for transitive roles."""
    return {(spec.kind_predicate, role) for role, spec in m.nps if role in ("subj", "obj")}


def max_np_adjectives(m: Meaning) -> int:
    """Largest number of adjectives on any NP."""
    return max(len(spec.adjectives) for _, spec in m.nps)


def heldout_hits(m: Meaning, plan: HeldOutPlan) -> tuple[str, ...]:
    """Tags for every held-out combination present in ``m``."""
    tags = [f"heldout_adj_noun:{a}+{k}" for a, k in sorted(meaning_adj_noun_pairs(m) & plan.adj_noun_pairs)]
    tags += [f"heldout_noun_role:{k}@{r}" for k, r in sorted(meaning_noun_roles(m) & plan.noun_roles)]
    return tuple(tags)


def train_allowed(m: Meaning, plan: HeldOutPlan, max_adjectives: int) -> bool:
    """Whether a meaning may appear in train (and train-like splits)."""
    return (
        plan.novel_kind_predicate not in meaning_kinds(m)
        and max_np_adjectives(m) <= max_adjectives
        and not heldout_hits(m, plan)
    )


def substitute_adjective(m: Meaning, old: str, new: str) -> Meaning:
    """Replace adjective predicate ``old`` by ``new`` in every NP."""
    out = m
    for role, spec in m.nps:
        if old in spec.adjective_predicates:
            attr, value = new.split(":", 1)
            adjectives = tuple(sorted((a, value if f"{a}:{v}" == old else v) for a, v in spec.adjectives))
            if attr != old.split(":", 1)[0]:
                raise GenerationError("substitution must stay within one attribute")
            out = out.with_np(role, NPSpec(spec.kind, adjectives))
    return out


def choose_plan(rng: random.Random, config: WorldConfig) -> HeldOutPlan:
    """Choose held-out structure for one seed. Fails loudly if infeasible."""
    ont = config.ontology
    novel = rng.choice(list(ont.kinds))
    known = [f"kind:{k}" for k in ont.kinds if k != novel]
    adjectives = [f"{a}:{v}" for a in ADJECTIVE_ATTRIBUTES for v in ont.values(a)]
    pairs = rng.sample([(a, k) for a in adjectives for k in known], config.splits.n_heldout_adj_noun_pairs)
    _check_pairs_leave_coverage(pairs, adjectives, known)
    role_kinds = rng.sample(known, config.splits.n_heldout_noun_roles)
    roles = [(k, rng.choice(["subj", "obj"])) for k in role_kinds]
    colors = [f"color:{c}" for c in ont.colors]
    target, confusion = rng.sample(colors, 2)
    return HeldOutPlan(novel, frozenset(pairs), frozenset(roles), target, confusion)


def _check_pairs_leave_coverage(pairs: Iterable[tuple[str, str]], adjectives: list[str], known: list[str]) -> None:
    pairs = set(pairs)
    for a in adjectives:
        if all((a, k) in pairs for k in known):
            raise GenerationError(f"held-out pairs remove every noun for adjective {a}")
    for k in known:
        if all((a, k) in pairs for a in adjectives):
            raise GenerationError(f"held-out pairs remove every adjective for noun {k}")


# --------------------------------------------------------------------------
# split builders
# --------------------------------------------------------------------------


class SplitBuilder:
    """Builds every split for one seed."""

    def __init__(self, config: WorldConfig, seed: int) -> None:
        self.config = config
        self.seed = seed
        self.gen = WorldGenerator(config)
        self.plan = choose_plan(self._rng("plan"), config)
        self.known_kinds = [k for k in config.ontology.kinds if k != self.plan.novel_kind]
        self.train_policy = AdjectivePolicy(config.generation.max_adjectives_train, config.generation.p_adjective)
        self.longer_policy = AdjectivePolicy(
            config.generation.max_adjectives_longer, config.generation.p_adjective_longer
        )

    def _rng(self, label: str) -> random.Random:
        return random.Random(derive_seed(self.seed, label))

    def _train_ok(self, m: Meaning) -> bool:
        return train_allowed(m, self.plan, self.config.generation.max_adjectives_train)

    def _allowed_pair(self, adj_pred: str, kind_pred: str) -> bool:
        return (adj_pred, kind_pred) not in self.plan.adj_noun_pairs

    def _allowed_role(self, kind_pred: str, role: str) -> bool:
        return (kind_pred, role) not in self.plan.noun_roles

    # ---------------------------------------------------------------- train

    def build_train(self) -> tuple[list[Example], dict[str, int]]:
        """Train stream, ordered by curriculum stage."""
        rng = self._rng("train")
        next_id = make_counter("train")
        examples: list[Example] = []
        keys: set[tuple[tuple[str, ...], str]] = set()
        n = self.config.splits.n_train
        while len(examples) < n:
            rem = n - len(examples)
            m = self.gen.sample_meaning_where(
                rng,
                self._train_ok,
                self.known_kinds,
                self.train_policy,
                allowed_pair=self._allowed_pair,
                allowed_role=self._allowed_role,
            )
            is_bare_np = len(m.nps) == 1 and len(m.nps[0][1].adjectives) == 0 and m.relation is None
            if is_bare_np and rem < 4:
                continue

            use_quad = is_bare_np or (rem >= 4 and rng.random() < 0.5)
            if use_quad:
                quad = self.gen.value_swap_quad_scenes(
                    rng, m, allowed_kinds=self.known_kinds, accept_m2=self._train_ok
                )
                if quad is not None:
                    m2, s1, s2 = quad
                    use_cg = rng.random() < self.config.generation.p_contrast_pair
                    id1, id2, id3, id4 = next_id(), next_id(), next_id(), next_id()
                    cg1 = f"cg:{id1}" if use_cg else None
                    cg2 = f"cg:{id3}" if use_cg else None
                    batch = [
                        self.gen.make_example(id1, m, s1, cg1, ("train",)),
                        self.gen.make_example(id2, m, s2, cg1, ("train",)),
                        self.gen.make_example(id3, m2, s2, cg2, ("train",)),
                        self.gen.make_example(id4, m2, s1, cg2, ("train",)),
                    ]
                    b_keys = [ex.item_key() for ex in batch]
                    if len(set(b_keys)) == 4 and all(k not in keys for k in b_keys):
                        keys.update(b_keys)
                        examples.extend(batch)
                        continue
                if is_bare_np:
                    continue

            if rem >= 2:
                try:
                    s1, s2 = self.gen.contrast_pair_scenes(
                        rng, m, allowed_kinds=self.known_kinds, allow_value_edit=False
                    )
                except GenerationError:
                    continue
                use_cg = rng.random() < self.config.generation.p_contrast_pair
                id1, id2 = next_id(), next_id()
                cg = f"cg:{id1}" if use_cg else None
                batch = [
                    self.gen.make_example(id1, m, s1, cg, ("train",)),
                    self.gen.make_example(id2, m, s2, cg, ("train",)),
                ]
                b_keys = [ex.item_key() for ex in batch]
                if len(set(b_keys)) == 2 and all(k not in keys for k in b_keys):
                    keys.update(b_keys)
                    examples.extend(batch)
            else:
                ex = self.gen.single_example(
                    rng, m, next_id(), True, ("train",), allowed_kinds=self.known_kinds, allow_value_edit=False
                )
                if ex.item_key() not in keys:
                    keys.add(ex.item_key())
                    examples.append(ex)
        return self._order_by_curriculum(rng, examples)

    def _order_by_curriculum(self, rng: random.Random, examples: list[Example]) -> tuple[list[Example], dict[str, int]]:
        stages = self.config.splits.curriculum
        stage_of = {c: i for i, stage in enumerate(stages) for c in stage}
        groups: list[list[Example]] = [[] for _ in stages]
        for ex in examples:
            groups[stage_of[ex.construction]].append(ex)
        ordered: list[Example] = []
        boundaries: dict[str, int] = {}
        for i, group in enumerate(groups):
            if self.config.splits.shuffle_within_stage:
                rng.shuffle(group)
            ordered.extend(group)
            for c in stages[i]:
                boundaries[c] = len(ordered)
        return ordered, boundaries

    # ---------------------------------------------------------------- generic

    def _balanced_items(
        self,
        label: str,
        n: int,
        accept: Callable[[Meaning], bool],
        kinds: list[str],
        policy: AdjectivePolicy,
        forbidden_keys: set[tuple[tuple[str, ...], str]],
        forbidden_tokens: set[tuple[str, ...]] = frozenset(),  # type: ignore[assignment]
        constructions: list[str] | None = None,
        tagger: Callable[[Meaning], tuple[str, ...]] = lambda m: (),
        allowed_pair: Callable[[str, str], bool] | None = None,
        allowed_role: Callable[[str, str], bool] | None = None,
    ) -> list[Example]:
        """``n`` items with exact 50/50 truth, feature, and token balance."""
        rng = self._rng(label)
        next_id = make_counter(label)
        items: list[Example] = []
        keys: set[tuple[tuple[str, ...], str]] = set()
        tries = 0
        while len(items) < n:
            tries += 1
            if tries > self.config.generation.max_rejection_tries * 4:
                raise GenerationError(f"{label}: could not build {n} items")
            rem = n - len(items)
            m = self.gen.sample_meaning_where(
                rng, accept, kinds, policy, constructions, allowed_pair=allowed_pair, allowed_role=allowed_role
            )
            if self.gen.grammar.realize(m) in forbidden_tokens:
                continue
            is_bare_np = len(m.nps) == 1 and len(m.nps[0][1].adjectives) == 0 and m.relation is None
            if is_bare_np:
                if rem < 4:
                    continue
                quad = self.gen.value_swap_quad_scenes(rng, m, allowed_kinds=kinds, accept_m2=accept)
                if quad is None:
                    continue
                m2, s1, s2 = quad
                if self.gen.grammar.realize(m2) in forbidden_tokens:
                    continue
                batch = [
                    self.gen.make_example(next_id(), m, s1, None, (label, *tagger(m))),
                    self.gen.make_example(next_id(), m, s2, None, (label, *tagger(m))),
                    self.gen.make_example(next_id(), m2, s2, None, (label, *tagger(m2))),
                    self.gen.make_example(next_id(), m2, s1, None, (label, *tagger(m2))),
                ]
                b_keys = [ex.item_key() for ex in batch]
                if len(set(b_keys)) == 4 and all(k not in forbidden_keys and k not in keys for k in b_keys):
                    keys.update(b_keys)
                    items.extend(batch)
            else:
                try:
                    s1, s2 = self.gen.contrast_pair_scenes(rng, m, allowed_kinds=kinds, allow_value_edit=False)
                except GenerationError:
                    continue
                batch = [
                    self.gen.make_example(next_id(), m, s1, None, (label, *tagger(m))),
                    self.gen.make_example(next_id(), m, s2, None, (label, *tagger(m))),
                ]
                b_keys = [ex.item_key() for ex in batch]
                if len(set(b_keys)) == 2 and all(k not in forbidden_keys and k not in keys for k in b_keys):
                    keys.update(b_keys)
                    items.extend(batch)
        return items

    def build_dev(self, train_keys: set) -> list[Example]:
        """In-distribution items, stratified by construction."""
        items: list[Example] = []
        for c in sorted(self.config.grammar.constructions):
            items += self._balanced_items(
                f"dev_{c}",
                self.config.splits.n_dev_per_construction,
                self._train_ok,
                self.known_kinds,
                self.train_policy,
                train_keys,
                constructions=[c],
                allowed_pair=self._allowed_pair,
                allowed_role=self._allowed_role,
            )
        return items

    def build_novel_combination(self, train_keys: set, train_tokens: set) -> list[Example]:
        """Items containing at least one held-out combination of known words."""
        max_adj = self.config.generation.max_adjectives_train

        def accept(m: Meaning) -> bool:
            return (self.plan.novel_kind_predicate not in meaning_kinds(m)
                    and max_np_adjectives(m) <= max_adj and bool(heldout_hits(m, self.plan)))

        return self._balanced_items(
            "test_novel_combination", self.config.splits.n_test_novel_combination, accept,
            self.known_kinds, self.train_policy, train_keys, train_tokens,
            tagger=lambda m: heldout_hits(m, self.plan),
        )

    def build_novel_word(self, train_keys: set, train_tokens: set) -> Split:
        """Novel noun: contrastive support pair(s), then queries in other frames."""
        rng = self._rng("novel_word_support")
        next_id = make_counter("novel_support")
        single_np = sorted(c for c, s in self.config.grammar.constructions.items() if not s.has_relation)
        support_examples: list[Example] = []
        all_kinds = list(self.config.ontology.kinds)
        for _ in range(self.config.splits.n_novel_word_support_pairs):
            m = Meaning(rng.choice(single_np), (("np", NPSpec(self.plan.novel_kind, ())),), None)
            support_examples += self.gen.contrast_pair(rng, m, next_id, ("novel_word_support",), allowed_kinds=all_kinds)
        support = tuple(StreamEvent("observe", observation=e.observation(), ground_truth=e) for e in support_examples)
        support_tokens = {e.tokens for e in support_examples}
        max_adj = self.config.generation.max_adjectives_train
        novel = self.plan.novel_kind_predicate

        def accept(m: Meaning) -> bool:
            has_frame = len(m.nps) > 1 or max_np_adjectives(m) >= 1
            return (novel in meaning_kinds(m) and has_frame and max_np_adjectives(m) <= max_adj
                    and not heldout_hits(m, self.plan))

        items = self._balanced_items(
            "test_novel_word_in_known_frame", self.config.splits.n_test_novel_word_queries, accept,
            all_kinds, self.train_policy, train_keys,
            train_tokens | support_tokens,
        )
        return Split(
            "test_novel_word_in_known_frame", SPLIT_PURPOSES["test_novel_word_in_known_frame"],
            tuple(items), support, (("before_support", 0), ("after_support", len(support))),
        )

    def build_longer(self, train_keys: set, train_tokens: set, max_train_length: int) -> list[Example]:
        """Items strictly longer than every training sentence."""
        def accept(m: Meaning) -> bool:
            return (self.plan.novel_kind_predicate not in meaning_kinds(m)
                    and not heldout_hits(m, self.plan)
                    and len(self.gen.grammar.realize(m)) > max_train_length)

        return self._balanced_items(
            "test_longer",
            self.config.splits.n_test_longer,
            accept,
            self.known_kinds,
            self.longer_policy,
            train_keys,
            train_tokens,
            allowed_pair=self._allowed_pair,
            allowed_role=self._allowed_role,
        )

    # ---------------------------------------------------------------- contradiction

    def build_contradiction(self, train_keys: set, train_tokens: set) -> Split:
        """Mislabelled stream + corrections, evaluated on diagnostic probes."""
        rng = self._rng("contradiction_stream")
        target, confusion = self.plan.contradiction_target, self.plan.contradiction_confusion
        misleading = self._misleading_examples(rng, target, confusion)
        fillers = self._filler_examples(rng)
        part1 = [StreamEvent("observe", observation=e.observation(label=not e.truth, source="teacher"),
                             ground_truth=e, mislabelled=True) for e in misleading]
        part1 += [StreamEvent("observe", observation=e.observation(), ground_truth=e) for e in fillers]
        rng.shuffle(part1)
        corrections = [
            StreamEvent("correct", correction=Correction(e.id, e.truth, "teacher",
                                                         "teacher retracts an earlier label"), ground_truth=e)
            for e in misleading
        ]
        support = tuple(part1 + corrections)
        stream_tokens = {e.tokens for e in misleading + fillers}
        probes = self._probes(target, confusion, train_keys, train_tokens | stream_tokens)
        checkpoints = (("before_stream", 0), ("before_corrections", len(part1)), ("after_corrections", len(support)))
        return Split("test_contradiction", SPLIT_PURPOSES["test_contradiction"], tuple(probes), support, checkpoints)

    def _with_target(self, target: str) -> Callable[[Meaning], bool]:
        return lambda m: self._train_ok(m) and target in meaning_adjectives(m)

    def _misleading_examples(self, rng: random.Random, target: str, confusion: str) -> list[Example]:
        """Examples whose teacher label matches the *confusion* reading, not the truth."""
        next_id = make_counter("contradiction_misleading")
        out: list[Example] = []
        for _ in range(self.config.generation.max_rejection_tries):
            if len(out) >= self.config.splits.n_contradiction_misleading:
                return out[: self.config.splits.n_contradiction_misleading]
            m = self.gen.sample_meaning_where(
                rng,
                self._with_target(target),
                self.known_kinds,
                self.train_policy,
                allowed_pair=self._allowed_pair,
                allowed_role=self._allowed_role,
            )
            pair = self._diagnostic_pair(rng, m, target, confusion, next_id, ("contradiction_misleading",))
            if pair is not None:
                # Emit False then True to match alternating parity
                out.extend([pair[1], pair[0]])
        raise GenerationError("could not build misleading examples")

    def _diagnostic_pair(
        self,
        rng: random.Random,
        m: Meaning,
        target: str,
        confusion: str,
        next_id: Callable[[], str],
        tags: tuple[str, ...],
    ) -> tuple[Example, Example] | None:
        """Build a (True, False) pair for ``m`` whose truth flips under the confusion reading.

        Both scenes contain both ``target`` and ``confusion`` colors (swapped between the
        witness object and a distractor object in the False scene), guaranteeing 100%
        identical feature and token distributions between True and False examples.
        """
        from nexus.world.config import ATTRIBUTES
        from nexus.world.scene import build_scene

        alt_m = substitute_adjective(m, target, confusion)
        lf = self.gen.grammar.to_lf(m)
        alt_lf = self.gen.grammar.to_lf(alt_m)
        t_attr, t_val = target.split(":", 1)
        _, c_val = confusion.split(":", 1)
        other_vals = [v for v in self.config.ontology.values(t_attr) if v not in (t_val, c_val)]
        target_kinds = {spec.kind for _, spec in m.nps if target in spec.adjective_predicates}

        for _ in range(30):
            ts = self.gen.true_scene(rng, m, self.known_kinds)
            attrs = [{a: o.value(a) for a in ATTRIBUTES} for o in ts.objects]
            rels = [(r.name, r.subject, r.object) for r in ts.relations]

            witness_idxs = [i for i, d in enumerate(attrs) if d["kind"] in target_kinds and d[t_attr] == t_val]
            distractor_idxs = [i for i, d in enumerate(attrs) if d["kind"] not in target_kinds]
            if not witness_idxs or not distractor_idxs:
                continue
            w_idx = witness_idxs[0]
            d_idx = rng.choice(distractor_idxs)

            for i, d in enumerate(attrs):
                if i == w_idx:
                    d[t_attr] = t_val
                elif i == d_idx:
                    d[t_attr] = c_val
                elif other_vals:
                    d[t_attr] = rng.choice(other_vals)

            attrs_true = [dict(x) for x in attrs]
            attrs_false = [dict(x) for x in attrs]
            attrs_false[w_idx][t_attr], attrs_false[d_idx][t_attr] = (
                attrs_false[d_idx][t_attr],
                attrs_false[w_idx][t_attr],
            )

            scene_true = build_scene(attrs_true, rels)
            scene_false = build_scene(attrs_false, rels)
            ok_true = evaluate(lf, scene_true, self.gen.symmetric) and not evaluate(alt_lf, scene_true, self.gen.symmetric)
            ok_false = not evaluate(lf, scene_false, self.gen.symmetric) and evaluate(alt_lf, scene_false, self.gen.symmetric)
            if ok_true and ok_false:
                ex_true = self.gen.make_example(next_id(), m, scene_true, None, tags)
                ex_false = self.gen.make_example(next_id(), m, scene_false, None, tags)
                return ex_true, ex_false
        return None

    def _filler_examples(self, rng: random.Random) -> list[Example]:
        next_id = make_counter("contradiction_filler")
        out: list[Example] = []
        while len(out) < self.config.splits.n_contradiction_filler:
            m = self.gen.sample_meaning_where(
                rng,
                lambda x: self._train_ok(x) and (len(x.nps) > 1 or max_np_adjectives(x) >= 1),
                self.known_kinds,
                self.train_policy,
                allowed_pair=self._allowed_pair,
                allowed_role=self._allowed_role,
            )
            try:
                s1, s2 = self.gen.contrast_pair_scenes(rng, m, allowed_kinds=self.known_kinds, allow_value_edit=False)
            except GenerationError:
                continue
            out.append(self.gen.make_example(next_id(), m, s1, None, ("contradiction_filler",)))
            if len(out) < self.config.splits.n_contradiction_filler:
                out.append(self.gen.make_example(next_id(), m, s2, None, ("contradiction_filler",)))
        return out

    def _probes(self, target: str, confusion: str, train_keys: set, forbidden_tokens: set) -> list[Example]:
        rng = self._rng("contradiction_probes")
        next_id = make_counter("test_contradiction")
        out: list[Example] = []
        keys: set = set()
        for _ in range(self.config.generation.max_rejection_tries * 4):
            if len(out) >= self.config.splits.n_contradiction_probes:
                return out[: self.config.splits.n_contradiction_probes]
            m = self.gen.sample_meaning_where(
                rng,
                self._with_target(target),
                self.known_kinds,
                self.train_policy,
                allowed_pair=self._allowed_pair,
                allowed_role=self._allowed_role,
            )
            if self.gen.grammar.realize(m) in forbidden_tokens:
                continue
            pair = self._diagnostic_pair(rng, m, target, confusion, next_id, ("test_contradiction",))
            if pair is not None:
                ex_true, ex_false = pair
                k1, k2 = ex_true.item_key(), ex_false.item_key()
                if k1 != k2 and k1 not in train_keys and k2 not in train_keys and k1 not in keys and k2 not in keys:
                    keys.add(k1)
                    keys.add(k2)
                    out.extend([ex_true, ex_false])
        raise GenerationError("could not build contradiction probes")


    # ---------------------------------------------------------------- all

    def build(self) -> Dataset:
        """Build every split."""
        train, boundaries = self.build_train()
        train_keys = {e.item_key() for e in train}
        train_tokens = {e.tokens for e in train}
        max_len = max(len(e.tokens) for e in train)
        dev = self.build_dev(train_keys)
        tests = {
            "test_novel_combination": Split(
                "test_novel_combination", SPLIT_PURPOSES["test_novel_combination"],
                tuple(self.build_novel_combination(train_keys, train_tokens))),
            "test_novel_word_in_known_frame": self.build_novel_word(train_keys, train_tokens),
            "test_longer": Split(
                "test_longer", SPLIT_PURPOSES["test_longer"],
                tuple(self.build_longer(train_keys, train_tokens, max_len))),
            "test_contradiction": self.build_contradiction(train_keys, train_tokens),
        }
        return Dataset(
            seed=self.seed,
            plan=self.plan,
            train=Split("train", SPLIT_PURPOSES["train"], tuple(train)),
            dev=Split("dev", SPLIT_PURPOSES["dev"], tuple(dev)),
            tests=tests,
            stage_boundaries=boundaries,
            max_train_length=max_len,
            grammar=self.config.grammar,
        )


def build_dataset(config: WorldConfig, seed: int, world_mode: str = "randomized") -> Dataset:
    """Build all splits for ``seed``. Deterministic."""
    if world_mode == "randomized":
        grammar_rng = random.Random(derive_seed(seed, "grammar"))
        rand_grammar = randomize_grammar(config.grammar, config.ontology, grammar_rng)
        config = dataclasses.replace(config, grammar=rand_grammar)
    return SplitBuilder(config, seed).build()
