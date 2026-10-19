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

    def true_scene(self, rng: random.Random, meaning: Meaning) -> Scene:
        """Build a scene that makes ``meaning`` true."""
        roles = [r for r, _ in meaning.nps]
        n = rng.randint(max(self.config.scene.min_objects, len(roles)), self.config.scene.max_objects)
        attrs = [random_object_attrs(rng, self.ontology, spec.fixed_attributes()) for _, spec in meaning.nps]
        attrs += [random_object_attrs(rng, self.ontology) for _ in range(n - len(attrs))]
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

    def single_edits(self, scene: Scene) -> list[Scene]:
        """All scenes one minimal edit away (attribute change, relation edit)."""
        attrs = [{a: o.value(a) for a in ATTRIBUTES} for o in scene.objects]
        rels = [(r.name, r.subject, r.object) for r in scene.relations]
        out: list[Scene] = []
        for i, obj_attrs in enumerate(attrs):
            for a in ATTRIBUTES:
                for v in self.ontology.values(a):
                    if v != obj_attrs[a]:
                        new_attrs = [dict(x) for x in attrs]
                        new_attrs[i][a] = v
                        out.append(build_scene(new_attrs, rels))
        for j, (name, s, o) in enumerate(rels):
            rest = rels[:j] + rels[j + 1:]
            out.append(build_scene(attrs, rest))
            if (name, o, s) not in rels:
                out.append(build_scene(attrs, rest + [(name, o, s)]))
            for other in self.ontology.relation_names:
                if other != name and (other, s, o) not in rels:
                    out.append(build_scene(attrs, rest + [(other, s, o)]))
        return out

    def near_miss_false_scene(self, rng: random.Random, scene: Scene, lf: LogicalForm) -> Scene:
        """A minimally edited copy of ``scene`` in which ``lf`` is false."""
        candidates = [s for s in self.single_edits(scene) if not evaluate(lf, s, self.symmetric)]
        if not candidates:
            raise GenerationError(f"no single edit falsifies {lf.canonical()}")
        return rng.choice(candidates)

    def random_false_scene(self, rng: random.Random, lf: LogicalForm) -> Scene:
        """A random scene in which ``lf`` is false."""
        for _ in range(self.config.generation.max_rejection_tries):
            scene = sample_scene(rng, self.ontology, self.config.scene)
            if not evaluate(lf, scene, self.symmetric):
                return scene
        raise GenerationError(f"could not find a random scene falsifying {lf.canonical()}")

    def false_scene(self, rng: random.Random, meaning: Meaning) -> Scene:
        """A false scene: near-miss with prob ``p_near_miss_false``, else random."""
        lf = self.grammar.to_lf(meaning)
        if rng.random() < self.config.generation.p_near_miss_false:
            return self.contrast_pair_scenes(rng, meaning)[1]
        return self.random_false_scene(rng, lf)

    def contrast_pair_scenes(self, rng: random.Random, meaning: Meaning) -> tuple[Scene, Scene]:
        """A true scene and a minimally edited false scene for ``meaning``.

        Retries with a fresh true scene when no single edit falsifies the
        meaning (e.g. two distinct witnesses are present).
        """
        lf = self.grammar.to_lf(meaning)
        for _ in range(self.config.generation.max_rejection_tries):
            true_scene = self.true_scene(rng, meaning)
            candidates = [s for s in self.single_edits(true_scene) if not evaluate(lf, s, self.symmetric)]
            if candidates:
                return true_scene, rng.choice(candidates)
        raise GenerationError(f"could not build a contrast pair for {lf.canonical()}")

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
    ) -> Example:
        """One example of ``meaning`` whose truth value is ``want``."""
        scene = self.true_scene(rng, meaning) if want else self.false_scene(rng, meaning)
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
    ) -> list[Example]:
        """Same sentence in a true scene and in a near-miss false scene."""
        true_scene, false_scene = self.contrast_pair_scenes(rng, meaning)
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
    ) -> list[Example]:
        """One or two examples for ``meaning`` (training-stream generation).

        With probability ``p_contrast_pair``: a contrastive pair. Otherwise a
        single example whose truth is a fair coin flip.
        """
        if rng.random() < self.config.generation.p_contrast_pair:
            return self.contrast_pair(rng, meaning, next_id, tags)
        return [self.single_example(rng, meaning, next_id(), rng.random() < 0.5, tags)]
