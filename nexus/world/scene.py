"""Scenes: small sets of attributed objects plus binary relations.

A scene is what the learner *perceives*. Objects carry symbolic attribute
values (kind, color, size) drawn from the ontology; relations are directed
tuples. The symbols are perceptual primitives given to the learner; the words
that name them are not.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from nexus.world.config import ATTRIBUTES, Ontology, SceneConfig


class SceneError(ValueError):
    """Raised for malformed scenes."""


@dataclass(frozen=True)
class SceneObject:
    """One object in a scene."""

    id: int
    kind: str
    color: str
    size: str

    def value(self, attribute: str) -> str:
        """Return the value of ``attribute`` for this object."""
        if attribute not in ATTRIBUTES:
            raise SceneError(f"unknown attribute {attribute!r}")
        return getattr(self, attribute)

    def features(self) -> tuple[str, ...]:
        """Return predicate symbols true of this object, e.g. ``kind:cube``."""
        return tuple(f"{a}:{self.value(a)}" for a in ATTRIBUTES)


@dataclass(frozen=True)
class SceneRelation:
    """A directed relation instance ``name(subject, object)``."""

    name: str
    subject: int
    object: int


@dataclass(frozen=True)
class Scene:
    """An immutable scene."""

    objects: tuple[SceneObject, ...]
    relations: tuple[SceneRelation, ...]

    def __post_init__(self) -> None:
        ids = [o.id for o in self.objects]
        if ids != list(range(len(ids))):
            raise SceneError(f"object ids must be 0..n-1 in order, got {ids}")
        seen = set()
        for r in self.relations:
            if r.subject == r.object:
                raise SceneError(f"reflexive relation {r}")
            if not (0 <= r.subject < len(ids) and 0 <= r.object < len(ids)):
                raise SceneError(f"relation {r} references a missing object")
            key = (r.name, r.subject, r.object)
            if key in seen:
                raise SceneError(f"duplicate relation {r}")
            seen.add(key)

    def holds(self, relation: str, a: int, b: int, symmetric: bool) -> bool:
        """Whether ``relation(a, b)`` holds (both orders if symmetric)."""
        for r in self.relations:
            if r.name != relation:
                continue
            if (r.subject, r.object) == (a, b):
                return True
            if symmetric and (r.subject, r.object) == (b, a):
                return True
        return False

    def to_json(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "objects": [
                {"id": o.id, "kind": o.kind, "color": o.color, "size": o.size} for o in self.objects
            ],
            "relations": [
                {"name": r.name, "subject": r.subject, "object": r.object} for r in self.relations
            ],
        }

    @staticmethod
    def from_json(obj: Mapping[str, Any], ontology: Ontology | None = None) -> "Scene":
        """Deserialize; validates against ``ontology`` if given. Fails loudly."""
        if not isinstance(obj, Mapping) or set(obj) != {"objects", "relations"}:
            raise SceneError("scene JSON must have exactly 'objects' and 'relations'")
        objects = []
        for o in obj["objects"]:
            if not isinstance(o, Mapping) or set(o) != {"id", "kind", "color", "size"}:
                raise SceneError(f"malformed object {o!r}")
            objects.append(SceneObject(int(o["id"]), str(o["kind"]), str(o["color"]), str(o["size"])))
        relations = []
        for r in obj["relations"]:
            if not isinstance(r, Mapping) or set(r) != {"name", "subject", "object"}:
                raise SceneError(f"malformed relation {r!r}")
            relations.append(SceneRelation(str(r["name"]), int(r["subject"]), int(r["object"])))
        scene = Scene(tuple(objects), tuple(relations))
        if ontology is not None:
            validate_scene(scene, ontology)
        return scene

    def canonical_key(self) -> str:
        """A string identical for scenes equal up to object permutation.

        Objects are sorted by their attributes (ties broken by the sorted
        relation signature), relations are re-indexed accordingly.
        """
        return min(self._key_for_order(order) for order in self._candidate_orders())

    def _candidate_orders(self) -> Iterable[tuple[int, ...]]:
        from itertools import permutations

        attrs = {o.id: (o.kind, o.color, o.size) for o in self.objects}
        base = sorted(attrs, key=lambda i: attrs[i])
        # Only permute within groups of attribute-identical objects.
        groups: list[list[int]] = []
        for i in base:
            if groups and attrs[groups[-1][0]] == attrs[i]:
                groups[-1].append(i)
            else:
                groups.append([i])
        orders: list[tuple[int, ...]] = [()]
        for g in groups:
            orders = [o + p for o in orders for p in permutations(g)]
        return orders

    def _key_for_order(self, order: tuple[int, ...]) -> str:
        remap = {old: new for new, old in enumerate(order)}
        objs = [[self.objects[i].kind, self.objects[i].color, self.objects[i].size] for i in order]
        rels = sorted([r.name, remap[r.subject], remap[r.object]] for r in self.relations)
        return json.dumps({"o": objs, "r": rels}, separators=(",", ":"))


def validate_scene(scene: Scene, ontology: Ontology) -> None:
    """Raise :class:`SceneError` if the scene uses values outside the ontology."""
    for o in scene.objects:
        for a in ATTRIBUTES:
            if o.value(a) not in ontology.values(a):
                raise SceneError(f"object {o.id}: {a}={o.value(a)!r} not in ontology")
    for r in scene.relations:
        if r.name not in ontology.relation_names:
            raise SceneError(f"unknown relation {r.name!r}")


def random_object_attrs(rng: random.Random, ontology: Ontology, fixed: Mapping[str, str] | None = None) -> dict[str, str]:
    """Sample attribute values, keeping any values given in ``fixed``."""
    fixed = dict(fixed or {})
    unknown = set(fixed) - set(ATTRIBUTES)
    if unknown:
        raise SceneError(f"unknown attributes {sorted(unknown)}")
    return {a: fixed[a] if a in fixed else rng.choice(ontology.values(a)) for a in ATTRIBUTES}


def build_scene(attrs: list[dict[str, str]], relations: list[tuple[str, int, int]]) -> Scene:
    """Build a scene from attribute dicts (index = id) and relation triples."""
    objects = tuple(SceneObject(i, a["kind"], a["color"], a["size"]) for i, a in enumerate(attrs))
    rels = tuple(SceneRelation(n, s, o) for n, s, o in relations)
    return Scene(objects, rels)


def add_random_relations(
    rng: random.Random,
    relations: list[tuple[str, int, int]],
    n_objects: int,
    ontology: Ontology,
    config: SceneConfig,
) -> list[tuple[str, int, int]]:
    """Add at most one random relation with probability ``p_relation``."""
    out = list(relations)
    if len(out) >= config.max_relations or rng.random() >= config.p_relation:
        return out
    name = rng.choice(ontology.relation_names)
    a, b = rng.sample(range(n_objects), 2)
    if (name, a, b) not in out:
        out.append((name, a, b))
    return out


def sample_scene(rng: random.Random, ontology: Ontology, config: SceneConfig) -> Scene:
    """Sample a random scene with no constraints."""
    n = rng.randint(config.min_objects, config.max_objects)
    attrs = [random_object_attrs(rng, ontology) for _ in range(n)]
    return build_scene(attrs, add_random_relations(rng, [], n, ontology, config))
