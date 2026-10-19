"""Meaning-first example generation.

Pipeline for one example:

1. sample a structured :class:`Meaning` (construction + NP specs + relation),
2. realize it as tokens and a logical form via the hidden grammar,
3. build a scene that makes it true (witness objects + random distractors),
4. optionally falsify the scene with a minimal edit (near-miss contrast) or
   sample an unrelated scene in which it is false,
5. compute the truth value with the model checker (never trusted from step 3/4).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence

from nexus.world.config import ADJECTIVE_ATTRIBUTES, ATTRIBUTES, WorldConfig
from nexus.world.example import Example
from nexus.world.grammar import Grammar, Meaning, NPSpec
from nexus.world.logic import LogicalForm, evaluate
from nexus.world.scene import (
    Scene,
    add_random_relations,
    build_scene,
    random_object_attrs,
    sample_scene,
)


class GenerationError(RuntimeError):
    """Raised when generation cannot satisfy its constraints."""


@dataclass(frozen=True)
class AdjectivePolicy:
    """How many adjectives an NP may carry and how likely each one is."""

    max_adjectives: int
    p_adjective: float


class WorldGenerator:
    """Generates examples from a :class:`WorldConfig`."""

    def __init__(self, config: WorldConfig) -> None:
        self.config = config
        self.ontology = config.ontology
        self.grammar = Grammar(config.grammar)
        self.symmetric = config.ontology.symmetric_relations

    # ------------------------------------------------------------------
    # meanings
    # ------------------------------------------------------------------

    def sample_construction(self, rng: random.Random, allowed: Sequence[str] | None = None) -> str:
        """Sample a construction name by configured weight."""
        specs = self.config.grammar.constructions
        names = sorted(allowed if allowed is not None else specs)
        return rng.choices(names, weights=[specs[n].weight for n in names])[0]

    def sample_np(self, rng: random.Random, kinds: Sequence[str], policy: AdjectivePolicy) -> NPSpec:
        """Sample an NP: a kind and up to ``max_adjectives`` adjectives."""
        attrs = list(ADJECTIVE_ATTRIBUTES)
        rng.shuffle(attrs)
        chosen = []
        for attr in attrs:
            if len(chosen) < policy.max_adjectives and rng.random() < policy.p_adjective:
                chosen.append((attr, rng.choice(self.ontology.values(attr))))
        return NPSpec(rng.choice(list(kinds)), tuple(sorted(chosen)))

    def sample_meaning(
        self, rng: random.Random, construction: str, kinds: Sequence[str], policy: AdjectivePolicy
    ) -> Meaning:
        """Sample a full meaning for ``construction``."""
        spec = self.config.grammar.constructions[construction]
        nps = tuple((role, self.sample_np(rng, kinds, policy)) for role in spec.roles)
        relation = rng.choice(self.ontology.relation_names) if spec.has_relation else None
        return Meaning(construction, nps, relation)

    def sample_meaning_where(
        self,
        rng: random.Random,
        accept: Callable[[Meaning], bool],
        kinds: Sequence[str],
        policy: AdjectivePolicy,
        constructions: Sequence[str] | None = None,
    ) -> Meaning:
        """Rejection-sample a meaning satisfying ``accept``. Fails loudly."""
        for _ in range(self.config.generation.max_rejection_tries):
            construction = self.sample_construction(rng, constructions)
            meaning = self.sample_meaning(rng, construction, kinds, policy)
            if accept(meaning):
                return meaning
        raise GenerationError("could not sample an acceptable meaning; constraints too tight")

    # ------------------------------------------------------------------
    # scenes
    # ------------------------------------------------------------------

    def true_scene(self, rng: random.Random, meaning: Meaning, allowed_kinds: Sequence[str] | None = None) -> Scene:
        """Build a scene that makes ``meaning`` true."""
        roles = [r for r, _ in meaning.nps]
        n = rng.randint(max(self.config.scene.min_objects, len(roles)), self.config.scene.max_objects)
        kinds = list(allowed_kinds) if allowed_kinds is not None else list(self.ontology.kinds)
        attrs = [random_object_attrs(rng, self.ontology, spec.fixed_attributes()) for _, spec in meaning.nps]
        for _ in range(n - len(attrs)):
            raw = random_object_attrs(rng, self.ontology)
            raw["kind"] = rng.choice(kinds)
            attrs.append(raw)
        order = list(range(n))
        rng.shuffle(order)  # order[new_id] = old index
        new_id = {old: new for new, old in enumerate(order)}
        shuffled = [attrs[old] for old in order]
        relations: list[tuple[str, int, int]] = []
        if meaning.relation is not None:
            s, o = new_id[roles.index("subj")], new_id[roles.index("obj")]
            if meaning.relation in self.symmetric and rng.random() < 0.5:
                s, o = o, s
            relations.append((meaning.relation, s, o))
        relations = add_random_relations(rng, relations, n, self.ontology, self.config.scene)
        scene = build_scene(shuffled, relations)
        if not evaluate(self.grammar.to_lf(meaning), scene, self.symmetric):
            raise GenerationError("internal error: true_scene produced a false scene")
        return scene

    def single_edits(self, scene: Scene, allowed_kinds: Sequence[str] | None = None) -> list[Scene]:
        """All scenes one minimal edit away, strictly preserving object count and relation count."""
        attrs = [{a: o.value(a) for a in ATTRIBUTES} for o in scene.objects]
        rels = [(r.name, r.subject, r.object) for r in scene.relations]
        out: list[Scene] = []
        kinds = list(allowed_kinds) if allowed_kinds is not None else list(self.ontology.kinds)
        for i, obj_attrs in enumerate(attrs):
            for a in ATTRIBUTES:
                vals = kinds if a == "kind" else self.ontology.values(a)
                for v in vals:
                    if v != obj_attrs[a]:
                        new_attrs = [dict(x) for x in attrs]
                        new_attrs[i][a] = v
                        out.append(build_scene(new_attrs, rels))
        n_objs = len(scene.objects)
        for j, (name, s, o) in enumerate(rels):
            base = rels[:j] + rels[j + 1:]
            # 1. Swap endpoints if asymmetric
            if (name, o, s) not in rels and name not in self.symmetric:
                out.append(build_scene(attrs, base + [(name, o, s)]))
            # 2. Change relation type
            for other in self.ontology.relation_names:
                if other != name and (other, s, o) not in rels:
                    out.append(build_scene(attrs, base + [(other, s, o)]))
            # 3. Change endpoints to another object
            if n_objs >= 3:
                for k in range(n_objs):
                    if k != s and k != o:
                        if (name, s, k) not in rels:
                            out.append(build_scene(attrs, base + [(name, s, k)]))
                        if (name, k, o) not in rels:
                            out.append(build_scene(attrs, base + [(name, k, o)]))
        return out

    def matched_false_scene(
        self,
        rng: random.Random,
        meaning: Meaning,
        lf: LogicalForm,
        allowed_kinds: Sequence[str] | None = None,
    ) -> Scene:
        """Sample a false scene matching the exact object count and relation count distribution of a true scene."""
        roles = [r for r, _ in meaning.nps]
        n = rng.randint(max(self.config.scene.min_objects, len(roles)), self.config.scene.max_objects)
        kinds = list(allowed_kinds) if allowed_kinds is not None else list(self.ontology.kinds)
        has_rel = meaning.relation is not None
        n_rels = (1 if has_rel else 0) + (1 if rng.random() < self.config.scene.p_relation else 0)
        for _ in range(self.config.generation.max_rejection_tries):
            attrs = []
            for _ in range(n):
                raw = random_object_attrs(rng, self.ontology)
                raw["kind"] = rng.choice(kinds)
                attrs.append(raw)
            rels: list[tuple[str, int, int]] = []
            if n_rels > 0 and n >= 2:
                pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
                chosen = rng.sample(pairs, min(n_rels, len(pairs)))
                for s, o in chosen:
                    rels.append((rng.choice(self.ontology.relation_names), s, o))
            s = build_scene(attrs, rels)
            if not evaluate(lf, s, self.symmetric):
                return s
        raise GenerationError(f"could not sample matched false scene for {lf.canonical()}")

    def false_scene(self, rng: random.Random, meaning: Meaning, allowed_kinds: Sequence[str] | None = None) -> Scene:
        """A false scene: near-miss with prob ``p_near_miss_false``, else matched."""
        lf = self.grammar.to_lf(meaning)
        if rng.random() < self.config.generation.p_near_miss_false:
            ts = self.true_scene(rng, meaning, allowed_kinds)
            cands = [s for s in self.single_edits(ts, allowed_kinds) if not evaluate(lf, s, self.symmetric)]
            if cands:
                return rng.choice(cands)
        return self.matched_false_scene(rng, meaning, lf, allowed_kinds)

    def contrast_pair_scenes(
        self, rng: random.Random, meaning: Meaning, allowed_kinds: Sequence[str] | None = None
    ) -> tuple[Scene, Scene]:
        """A true scene and a minimally edited false scene for ``meaning``."""
        lf = self.grammar.to_lf(meaning)
        for _ in range(self.config.generation.max_rejection_tries):
            true_scene = self.true_scene(rng, meaning, allowed_kinds)
            candidates = [s for s in self.single_edits(true_scene, allowed_kinds) if not evaluate(lf, s, self.symmetric)]
            if candidates:
                return true_scene, rng.choice(candidates)
        return true_scene, self.matched_false_scene(rng, meaning, lf, allowed_kinds)

    # ------------------------------------------------------------------
    # examples
    # ------------------------------------------------------------------

    def make_example(
        self,
        example_id: str,
        meaning: Meaning,
        scene: Scene,
        contrast_group: str | None = None,
        tags: tuple[str, ...] = (),
    ) -> Example:
        """Assemble an example; the truth value comes from the model checker."""
        lf = self.grammar.to_lf(meaning)
        return Example(
            id=example_id,
            scene=scene,
            tokens=self.grammar.realize(meaning),
            lf=lf,
            truth=evaluate(lf, scene, self.symmetric),
            construction=meaning.construction,
            contrast_group=contrast_group,
            tags=tags,
        )

    def single_example(
        self,
        rng: random.Random,
        meaning: Meaning,
        example_id: str,
        want: bool,
        tags: tuple[str, ...] = (),
        allowed_kinds: Sequence[str] | None = None,
    ) -> Example:
        """One example of ``meaning`` whose truth value is ``want``."""
        scene = self.true_scene(rng, meaning, allowed_kinds) if want else self.false_scene(rng, meaning, allowed_kinds)
        example = self.make_example(example_id, meaning, scene, None, tags)
        if example.truth != want:
            raise GenerationError("internal error: single example truth mismatch")
        return example

    def contrast_pair(
        self,
        rng: random.Random,
        meaning: Meaning,
        next_id: Callable[[], str],
        tags: tuple[str, ...] = (),
        allowed_kinds: Sequence[str] | None = None,
    ) -> list[Example]:
        """Same sentence in a true scene and in a near-miss false scene."""
        true_scene, false_scene = self.contrast_pair_scenes(rng, meaning, allowed_kinds)
        true_id, false_id = next_id(), next_id()
        group = f"cg:{true_id}"
        out = [
            self.make_example(true_id, meaning, true_scene, group, tags),
            self.make_example(false_id, meaning, false_scene, group, tags),
        ]
        if [e.truth for e in out] != [True, False]:
            raise GenerationError("internal error: contrast pair truth mismatch")
        return out

    def examples_for_meaning(
        self,
        rng: random.Random,
        meaning: Meaning,
        next_id: Callable[[], str],
        tags: tuple[str, ...] = (),
        allowed_kinds: Sequence[str] | None = None,
    ) -> list[Example]:
        """One or two examples for ``meaning`` (training-stream generation)."""
        if rng.random() < self.config.generation.p_contrast_pair:
            return self.contrast_pair(rng, meaning, next_id, tags, allowed_kinds)
        return [self.single_example(rng, meaning, next_id(), rng.random() < 0.5, tags, allowed_kinds)]
