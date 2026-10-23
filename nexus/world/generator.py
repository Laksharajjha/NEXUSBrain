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

    def sample_np(
        self,
        rng: random.Random,
        kinds: Sequence[str],
        policy: AdjectivePolicy,
        allowed_pair: Callable[[str, str], bool] | None = None,
    ) -> NPSpec:
        """Sample an NP: sample adjectives uniformly first, then an allowed kind."""
        attrs = list(ADJECTIVE_ATTRIBUTES)
        rng.shuffle(attrs)
        chosen: list[tuple[str, str]] = []
        for attr in attrs:
            if len(chosen) < policy.max_adjectives and rng.random() < policy.p_adjective:
                chosen.append((attr, rng.choice(self.ontology.values(attr))))
        valid_kinds = list(kinds)
        if allowed_pair is not None and chosen:
            filtered = [k for k in valid_kinds if all(allowed_pair(f"{a}:{v}", f"kind:{k}") for a, v in chosen)]
            if filtered:
                valid_kinds = filtered
        return NPSpec(rng.choice(valid_kinds), tuple(sorted(chosen)))

    def sample_meaning(
        self,
        rng: random.Random,
        construction: str,
        kinds: Sequence[str],
        policy: AdjectivePolicy,
        allowed_pair: Callable[[str, str], bool] | None = None,
        allowed_role: Callable[[str, str], bool] | None = None,
    ) -> Meaning:
        """Sample a full meaning for ``construction`` with distinct role kinds."""
        spec = self.config.grammar.constructions[construction]
        nps: list[tuple[str, NPSpec]] = []
        used_kinds: set[str] = set()
        for role in spec.roles:
            role_kinds = [k for k in kinds if k not in used_kinds] or list(kinds)
            if allowed_role is not None and role in ("subj", "obj"):
                filtered_rk = [k for k in role_kinds if allowed_role(f"kind:{k}", role)]
                if filtered_rk:
                    role_kinds = filtered_rk
            np_spec = self.sample_np(rng, role_kinds, policy, allowed_pair=allowed_pair)
            nps.append((role, np_spec))
            used_kinds.add(np_spec.kind)
        relation = rng.choice(self.ontology.relation_names) if spec.has_relation else None
        return Meaning(construction, tuple(nps), relation)

    def sample_meaning_where(
        self,
        rng: random.Random,
        accept: Callable[[Meaning], bool],
        kinds: Sequence[str],
        policy: AdjectivePolicy,
        constructions: Sequence[str] | None = None,
        allowed_pair: Callable[[str, str], bool] | None = None,
        allowed_role: Callable[[str, str], bool] | None = None,
    ) -> Meaning:
        """Rejection-sample a meaning satisfying ``accept``. Fails loudly."""
        for _ in range(self.config.generation.max_rejection_tries):
            construction = self.sample_construction(rng, constructions)
            meaning = self.sample_meaning(
                rng, construction, kinds, policy, allowed_pair=allowed_pair, allowed_role=allowed_role
            )
            if accept(meaning):
                return meaning
        raise GenerationError("could not sample an acceptable meaning; constraints too tight")

    # ------------------------------------------------------------------
    # scenes
    # ------------------------------------------------------------------

    def true_scene(self, rng: random.Random, meaning: Meaning, allowed_kinds: Sequence[str] | None = None) -> Scene:
        """Build a scene that makes ``meaning`` true and supports clean falsification."""
        roles = [r for r, _ in meaning.nps]
        min_objs = max(self.config.scene.min_objects, len(roles))
        if meaning.relation is not None and meaning.relation in self.symmetric:
            min_objs = max(min_objs, min(3, self.config.scene.max_objects))
        n = rng.randint(min_objs, self.config.scene.max_objects)
        kinds = list(allowed_kinds) if allowed_kinds is not None else list(self.ontology.kinds)

        witness_specs = [spec for _, spec in meaning.nps]
        witness_kinds = {spec.kind for spec in witness_specs}
        attrs = [random_object_attrs(rng, self.ontology, spec.fixed_attributes()) for spec in witness_specs]

        # Ensure that in transitive sentences with an adjective on one NP, the other NP does not accidentally
        # share both the same adjective value if needed, and distractors have distinct kinds/adjectives.
        other_kinds = [k for k in kinds if k not in witness_kinds] or kinds
        fixed_by_single_np = witness_specs[0].fixed_attributes() if len(witness_specs) == 1 else {}

        for _ in range(n - len(attrs)):
            raw: dict[str, str] = {}
            raw["kind"] = rng.choice(other_kinds)
            for a in ADJECTIVE_ATTRIBUTES:
                if a in fixed_by_single_np:
                    other_vals = [v for v in self.ontology.values(a) if v != fixed_by_single_np[a]]
                    raw[a] = rng.choice(other_vals)
                else:
                    raw[a] = rng.choice(self.ontology.values(a))
            attrs.append(raw)

        order = list(range(n))
        rng.shuffle(order)  # order[new_id] = old index
        new_id = {old: new for new, old in enumerate(order)}
        shuffled = [attrs[old] for old in order]
        relations: list[tuple[str, int, int]] = []
        s_id = o_id = -1
        if meaning.relation is not None:
            s_id, o_id = new_id[roles.index("subj")], new_id[roles.index("obj")]
            if meaning.relation in self.symmetric and rng.random() < 0.5:
                s_id, o_id = o_id, s_id
            relations.append((meaning.relation, s_id, o_id))

        # Add optional random relation while preserving clean falsifiability
        if len(relations) < self.config.scene.max_relations and rng.random() < self.config.scene.p_relation:
            r_name = rng.choice(self.ontology.relation_names)
            candidate_pairs = [(i, j) for i in range(n) for j in range(n) if i != j and (r_name, i, j) not in relations]
            if meaning.relation is not None:
                # Avoid adding the reverse witness relation or parallel relation on (s_id, o_id)/(o_id, s_id)
                filtered_pairs = [
                    (i, j) for (i, j) in candidate_pairs
                    if {i, j} != {s_id, o_id}
                ]
                if meaning.relation in self.symmetric and n >= 3:
                    # Leave at least one distractor node unlinked by meaning.relation to s_id
                    distractor_ids = [idx for idx in range(n) if idx not in (s_id, o_id)]
                    if len(distractor_ids) == 1 and r_name == meaning.relation:
                        d_id = distractor_ids[0]
                        filtered_pairs = [(i, j) for (i, j) in filtered_pairs if {i, j} != {s_id, d_id}]
                candidate_pairs = filtered_pairs
            if candidate_pairs:
                a_idx, b_idx = rng.choice(candidate_pairs)
                relations.append((r_name, a_idx, b_idx))

        scene = build_scene(shuffled, relations)
        if not evaluate(self.grammar.to_lf(meaning), scene, self.symmetric):
            raise GenerationError("internal error: true_scene produced a false scene")
        return scene

    def binding_edits(self, scene: Scene) -> list[Scene]:
        """All scenes formed by swapping whole objects or single attributes between two objects.

        Strictly preserves object count, every kind/color/size count, relation count,
        and exact relation graph topology.
        """
        attrs = [{a: o.value(a) for a in ATTRIBUTES} for o in scene.objects]
        rels = [(r.name, r.subject, r.object) for r in scene.relations]
        n = len(attrs)
        out: list[Scene] = []
        # 1. Swap full object attribute bundles between i and j
        for i in range(n):
            for j in range(i + 1, n):
                if attrs[i] != attrs[j]:
                    new_attrs = [dict(x) for x in attrs]
                    new_attrs[i], new_attrs[j] = new_attrs[j], new_attrs[i]
                    out.append(build_scene(new_attrs, rels))
        # 2. Swap a single attribute between i and j
        for i in range(n):
            for j in range(i + 1, n):
                for a in ATTRIBUTES:
                    if attrs[i][a] != attrs[j][a]:
                        new_attrs = [dict(x) for x in attrs]
                        new_attrs[i][a], new_attrs[j][a] = new_attrs[j][a], new_attrs[i][a]
                        out.append(build_scene(new_attrs, rels))
        return out

    def single_edits(self, scene: Scene, allowed_kinds: Sequence[str] | None = None) -> list[Scene]:
        """All scenes one minimal edit away, strictly preserving object count and relation topology."""
        attrs = [{a: o.value(a) for a in ATTRIBUTES} for o in scene.objects]
        rels = [(r.name, r.subject, r.object) for r in scene.relations]
        out: list[Scene] = list(self.binding_edits(scene))
        kinds = list(allowed_kinds) if allowed_kinds is not None else list(self.ontology.kinds)
        for i, obj_attrs in enumerate(attrs):
            for a in ATTRIBUTES:
                vals = kinds if a == "kind" else self.ontology.values(a)
                for v in vals:
                    if v != obj_attrs[a]:
                        new_attrs = [dict(x) for x in attrs]
                        new_attrs[i][a] = v
                        out.append(build_scene(new_attrs, rels))
        for j, (name, s, o) in enumerate(rels):
            base = rels[:j] + rels[j + 1:]
            # Change relation type on the exact same endpoints (preserves topology)
            for other in self.ontology.relation_names:
                if other != name and (other, s, o) not in rels:
                    out.append(build_scene(attrs, base + [(other, s, o)]))
        return out

    def value_swap_quad_scenes(
        self,
        rng: random.Random,
        m1: Meaning,
        allowed_kinds: Sequence[str] | None = None,
        accept_m2: Callable[[Meaning], bool] | None = None,
    ) -> tuple[Meaning, Scene, Scene] | None:
        """Build a symmetric 2x2 value-swap quad (m2, s1, s2) for m1.

        Returns ``(m2, s1, s2)`` such that:
        - ``m1`` is True in ``s1`` and False in ``s2``,
        - ``m2`` is True in ``s2`` and False in ``s1``,
        - ``(s1, s2)`` and ``(m1, m2)`` form an exact 2x2 crossover with zero marginal
          feature or token difference between True and False examples.
        """
        kinds = list(allowed_kinds) if allowed_kinds is not None else list(self.ontology.kinds)
        lf1 = self.grammar.to_lf(m1)

        # Case 1: Bare NP (1 NP, 0 adjectives, no relation)
        if len(m1.nps) == 1 and len(m1.nps[0][1].adjectives) == 0 and m1.relation is None:
            k1 = m1.nps[0][1].kind
            cand_k2 = [k for k in kinds if k != k1]
            rng.shuffle(cand_k2)
            for k2 in cand_k2:
                m2 = Meaning(m1.construction, (("np", NPSpec(k2, ())),), None)
                if accept_m2 is not None and not accept_m2(m2):
                    continue
                lf2 = self.grammar.to_lf(m2)
                distractor_kinds = [k for k in kinds if k != k2]
                for _ in range(20):
                    s1 = self.true_scene(rng, m1, allowed_kinds=distractor_kinds)
                    attrs1 = [{a: o.value(a) for a in ATTRIBUTES} for o in s1.objects]
                    if any(d["kind"] == k2 for d in attrs1):
                        continue
                    rels = [(r.name, r.subject, r.object) for r in s1.relations]
                    attrs2 = [dict(x) for x in attrs1]
                    for d in attrs2:
                        if d["kind"] == k1:
                            d["kind"] = k2
                    s2 = build_scene(attrs2, rels)
                    if (
                        evaluate(lf1, s1, self.symmetric)
                        and not evaluate(lf1, s2, self.symmetric)
                        and evaluate(lf2, s2, self.symmetric)
                        and not evaluate(lf2, s1, self.symmetric)
                    ):
                        return m2, s1, s2
            return None

        # Case 2: 1-adjective single-NP
        if len(m1.nps) == 1 and len(m1.nps[0][1].adjectives) == 1 and m1.relation is None:
            k1 = m1.nps[0][1].kind
            adj_attr, v1 = m1.nps[0][1].adjectives[0]
            cand_v2 = [v for v in self.ontology.values(adj_attr) if v != v1]
            rng.shuffle(cand_v2)
            for v2 in cand_v2:
                m2 = Meaning(m1.construction, (("np", NPSpec(k1, ((adj_attr, v2),))),), None)
                if accept_m2 is not None and not accept_m2(m2):
                    continue
                lf2 = self.grammar.to_lf(m2)
                for _ in range(20):
                    s1 = self.true_scene(rng, m1, allowed_kinds=kinds)
                    attrs1 = [{a: o.value(a) for a in ATTRIBUTES} for o in s1.objects]
                    rels = [(r.name, r.subject, r.object) for r in s1.relations]
                    attrs2 = [dict(x) for x in attrs1]
                    for d in attrs2:
                        if d["kind"] == k1 and d[adj_attr] == v1:
                            d[adj_attr] = v2
                    s2 = build_scene(attrs2, rels)
                    if (
                        evaluate(lf1, s1, self.symmetric)
                        and not evaluate(lf1, s2, self.symmetric)
                        and evaluate(lf2, s2, self.symmetric)
                        and not evaluate(lf2, s1, self.symmetric)
                    ):
                        return m2, s1, s2
            return None

        # Case 3: Transitive relation type swap
        if m1.relation is not None:
            r1 = m1.relation
            cand_r2 = [r for r in self.ontology.relation_names if r != r1]
            rng.shuffle(cand_r2)
            for r2 in cand_r2:
                m2 = Meaning(m1.construction, m1.nps, r2)
                if accept_m2 is not None and not accept_m2(m2):
                    continue
                lf2 = self.grammar.to_lf(m2)
                for _ in range(20):
                    s1 = self.true_scene(rng, m1, allowed_kinds=kinds)
                    attrs1 = [{a: o.value(a) for a in ATTRIBUTES} for o in s1.objects]
                    rels1 = [(r.name, r.subject, r.object) for r in s1.relations]
                    for idx, (r_name, s_idx, o_idx) in enumerate(rels1):
                        if r_name == r1 and (r2, s_idx, o_idx) not in rels1:
                            rels2 = rels1[:idx] + [(r2, s_idx, o_idx)] + rels1[idx + 1:]
                            s2 = build_scene(attrs1, rels2)
                            if (
                                evaluate(lf1, s1, self.symmetric)
                                and not evaluate(lf1, s2, self.symmetric)
                                and evaluate(lf2, s2, self.symmetric)
                                and not evaluate(lf2, s1, self.symmetric)
                            ):
                                return m2, s1, s2
            return None

        return None

    def _falsify_scene(
        self,
        rng: random.Random,
        ts: Scene,
        meaning: Meaning,
        allowed_kinds: Sequence[str] | None = None,
        allow_value_edit: bool = False,
    ) -> Scene:
        """Transform a true scene ``ts`` into a false scene without distorting topology or feature balance."""
        lf = self.grammar.to_lf(meaning)
        kinds = list(allowed_kinds) if allowed_kinds is not None else list(self.ontology.kinds)
        attrs = [{a: o.value(a) for a in ATTRIBUTES} for o in ts.objects]
        rels = [(r.name, r.subject, r.object) for r in ts.relations]

        # Case 1: Bare NP (1 NP, 0 adjectives, no relation) -> must replace the witness kind
        if len(meaning.nps) == 1 and len(meaning.nps[0][1].adjectives) == 0 and meaning.relation is None:
            target_kind = meaning.nps[0][1].kind
            other_kinds = [k for k in kinds if k != target_kind]
            new_attrs = [dict(x) for x in attrs]
            for obj_d in new_attrs:
                if obj_d["kind"] == target_kind:
                    obj_d["kind"] = rng.choice(other_kinds)
            s = build_scene(new_attrs, rels)
            if not evaluate(lf, s, self.symmetric):
                return s

        # Case 2: Value-replacement edit in training (for direct lexical contrast on adjectives and verbs)
        if allow_value_edit:
            if len(meaning.nps) == 1 and len(meaning.nps[0][1].adjectives) == 1 and meaning.relation is None:
                target_kind = meaning.nps[0][1].kind
                adj_attr, adj_val = meaning.nps[0][1].adjectives[0]
                other_vals = [v for v in self.ontology.values(adj_attr) if v != adj_val]
                new_attrs = [dict(x) for x in attrs]
                for obj_d in new_attrs:
                    if obj_d["kind"] == target_kind and obj_d[adj_attr] == adj_val:
                        obj_d[adj_attr] = rng.choice(other_vals)
                s = build_scene(new_attrs, rels)
                if not evaluate(lf, s, self.symmetric):
                    return s
            elif meaning.relation is not None:
                other_rels = [r for r in self.ontology.relation_names if r != meaning.relation]
                if other_rels:
                    new_r = rng.choice(other_rels)
                    # Replace the witness relation's type on the exact same endpoints
                    for idx, (r_name, s_idx, o_idx) in enumerate(rels):
                        if r_name == meaning.relation and (new_r, s_idx, o_idx) not in rels:
                            cand_rels = rels[:idx] + [(new_r, s_idx, o_idx)] + rels[idx + 1:]
                            s = build_scene(attrs, cand_rels)
                            if not evaluate(lf, s, self.symmetric):
                                return s

        # Case 3: Feature-preserving binding / role swap (preserves 100% of feature and relation counts)
        b_cands = [s for s in self.binding_edits(ts) if not evaluate(lf, s, self.symmetric)]
        if b_cands:
            return rng.choice(b_cands)

        if allow_value_edit:
            all_cands = [s for s in self.single_edits(ts, allowed_kinds) if not evaluate(lf, s, self.symmetric)]
            if all_cands:
                return rng.choice(all_cands)
        raise GenerationError(f"could not falsify scene for {lf.canonical()}")

    def false_scene(
        self,
        rng: random.Random,
        meaning: Meaning,
        allowed_kinds: Sequence[str] | None = None,
        allow_value_edit: bool = False,
    ) -> Scene:
        """A false scene derived from a true scene to match structure and feature balance."""
        for _ in range(self.config.generation.max_rejection_tries):
            ts = self.true_scene(rng, meaning, allowed_kinds)
            try:
                return self._falsify_scene(rng, ts, meaning, allowed_kinds, allow_value_edit=allow_value_edit)
            except GenerationError:
                continue
        raise GenerationError(f"could not build false_scene for {meaning}")

    def contrast_pair_scenes(
        self,
        rng: random.Random,
        meaning: Meaning,
        allowed_kinds: Sequence[str] | None = None,
        allow_value_edit: bool = False,
    ) -> tuple[Scene, Scene]:
        """A true scene and a minimally edited false scene for ``meaning``."""
        for _ in range(self.config.generation.max_rejection_tries):
            true_scene = self.true_scene(rng, meaning, allowed_kinds)
            try:
                false_scene = self._falsify_scene(
                    rng, true_scene, meaning, allowed_kinds, allow_value_edit=allow_value_edit
                )
                return true_scene, false_scene
            except GenerationError:
                continue
        raise GenerationError(f"could not build contrast_pair_scenes for {meaning}")

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
        allow_value_edit: bool = False,
    ) -> Example:
        """One example of ``meaning`` whose truth value is ``want``."""
        scene = (
            self.true_scene(rng, meaning, allowed_kinds)
            if want
            else self.false_scene(rng, meaning, allowed_kinds, allow_value_edit=allow_value_edit)
        )
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
        allow_value_edit: bool = False,
    ) -> list[Example]:
        """Same sentence in a true scene and in a near-miss false scene."""
        true_scene, false_scene = self.contrast_pair_scenes(
            rng, meaning, allowed_kinds, allow_value_edit=allow_value_edit
        )
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
        want_single: bool | None = None,
    ) -> list[Example]:
        """One or two examples for ``meaning`` (training-stream generation)."""
        allow_val = rng.random() < 0.5
        if rng.random() < self.config.generation.p_contrast_pair:
            return self.contrast_pair(rng, meaning, next_id, tags, allowed_kinds, allow_value_edit=allow_val)
        want = (rng.random() < 0.5) if want_single is None else want_single
        return [
            self.single_example(rng, meaning, next_id(), want, tags, allowed_kinds, allow_value_edit=allow_val)
        ]

