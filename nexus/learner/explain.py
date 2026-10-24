"""Structured explanations, provenance change records, and faithfulness interventions.

Every prediction made by :class:`~nexus.learner.nexus_learner.NEXUSLearner` can
be inspected via :meth:`explain` and :meth:`what_changed`, and tested for
causal faithfulness by intervening on the scene witness or lexical entry cited
in the explanation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from nexus.learner.categorial import LexicalEntry
from nexus.world.logic import LogicalForm
from nexus.world.scene import Scene, SceneObject, SceneRelation


@dataclass(frozen=True)
class Explanation:
    """Inspectable derivation, provenance, and grounding witness for one judgment."""

    tokens: tuple[str, ...]
    parse_tree: dict[str, Any] | None
    lexical_entries: tuple[LexicalEntry, ...]
    derived_lf: LogicalForm | None
    is_rel_symmetric: bool
    truth: bool
    p_true: float
    posterior_weight: float
    witness: dict[str, int] | None
    failure_reason: str

    def to_json(self) -> dict[str, Any]:
        return {
            "tokens": list(self.tokens),
            "parse_tree": self.parse_tree,
            "lexical_entries": [e.to_json() for e in self.lexical_entries],
            "derived_lf": None if self.derived_lf is None else self.derived_lf.canonical(),
            "is_rel_symmetric": self.is_rel_symmetric,
            "truth": self.truth,
            "p_true": self.p_true,
            "posterior_weight": self.posterior_weight,
            "witness": self.witness,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class ChangeRecord:
    """Provenance record of how a single observation or correction changed the beam."""

    example_id: str
    event_kind: str  # "observe" | "correct"
    promoted: tuple[str, ...]
    demoted: tuple[str, ...]
    created: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "event_kind": self.event_kind,
            "promoted": list(self.promoted),
            "demoted": list(self.demoted),
            "created": list(self.created),
        }


def intervene_on_scene(scene: Scene, explanation: Explanation) -> Scene:
    """Counterfactually alter ``scene`` on the witness/atoms cited in ``explanation`` to flip truth."""
    lf = explanation.derived_lf
    if lf is None:
        return scene

    if explanation.truth:
        # Break the first unary or relational atom cited in the derived LF across the scene
        rel_atoms = [a for a in lf.atoms if a.family == "rel"]
        if rel_atoms:
            target_rel = rel_atoms[0].value
            new_rels = tuple(r for r in scene.relations if r.name != target_rel)
            return Scene(scene.objects, new_rels)

        unary_atoms = [a for a in lf.atoms if a.family in ("kind", "color", "size")]
        if unary_atoms:
            target = unary_atoms[0]
            fam, val = target.family, target.value
            new_objs = []
            for o in scene.objects:
                if o.value(fam) == val:
                    attrs = {"kind": o.kind, "color": o.color, "size": o.size}
                    attrs[fam] = f"counterfactual_{val}"
                    new_objs.append(SceneObject(o.id, attrs["kind"], attrs["color"], attrs["size"]))
                else:
                    new_objs.append(o)
            return Scene(tuple(new_objs), scene.relations)
        return scene

    # explanation.truth is False: satisfy all atoms of derived_lf on objects 0 (and 1)
    var_to_idx = {v: idx for idx, v in enumerate(lf.variables)}
    new_attrs = [
        {"kind": o.kind, "color": o.color, "size": o.size}
        for o in scene.objects
    ]
    while len(new_attrs) < len(lf.variables):
        new_attrs.append({"kind": "entity_a", "color": "shade_a", "size": "scale_a"})

    new_rels = list(scene.relations)
    for atom in lf.atoms:
        if atom.family in ("kind", "color", "size"):
            idx = var_to_idx[atom.args[0]]
            new_attrs[idx][atom.family] = atom.value
        elif atom.family == "rel":
            s_idx = var_to_idx[atom.args[0]]
            o_idx = var_to_idx[atom.args[1]]
            if not any(r.name == atom.value and r.subject == s_idx and r.object == o_idx for r in new_rels):
                new_rels.append(SceneRelation(atom.value, s_idx, o_idx))

    objs = tuple(
        SceneObject(i, a["kind"], a["color"], a["size"])
        for i, a in enumerate(new_attrs)
    )
    return Scene(objs, tuple(new_rels))


def intervene_on_lexicon(
    entries: Mapping[str, LexicalEntry],
    explanation: Explanation,
    scene: Scene,
) -> dict[str, LexicalEntry]:
    """Counterfactually alter one lexical entry cited in ``explanation`` so the judgment flips."""
    updated = dict(entries)
    content_entries = [e for e in explanation.lexical_entries if e.atom_count > 0]
    if not content_entries:
        return updated

    if explanation.truth:
        # Replace the first content word's predicate with a predicate absent from the scene
        target = content_entries[0]
        if target.category == "N":
            new_pred = "kind:absent_concept"
        elif target.category in ("N/N", "N\\N"):
            fam = target.family
            new_pred = f"{fam}:absent_value"
        else:
            new_pred = "rel:absent_relation(x0,x1)"
        updated[target.word] = LexicalEntry(
            word=target.word,
            category=target.category,
            predicate=new_pred,
            supporting_examples=target.supporting_examples,
        )
        return updated

    # explanation.truth is False: align the failing content entries with scene objects
    nouns = [e for e in content_entries if e.category == "N"]
    adjs = [e for e in content_entries if e.category in ("N/N", "N\\N")]
    verbs = [e for e in content_entries if e.category in ("(S\\NP)\\NP", "(S\\NP)/NP", "(S/NP)/NP")]

    if len(nouns) <= 1 or not verbs:
        obj0 = scene.objects[0]
        for n_entry in nouns:
            updated[n_entry.word] = LexicalEntry(
                word=n_entry.word,
                category="N",
                predicate=f"kind:{obj0.kind}",
                supporting_examples=n_entry.supporting_examples,
            )
        for a_entry in adjs:
            fam = a_entry.family if a_entry.family in ("color", "size") else "color"
            updated[a_entry.word] = LexicalEntry(
                word=a_entry.word,
                category=a_entry.category,
                predicate=f"{fam}:{obj0.value(fam)}",
                supporting_examples=a_entry.supporting_examples,
            )
        return updated

    # Transitive sentence with 2 nouns and a verb
    tok_list = list(explanation.tokens)
    repeated_adjs = {a.word for a in adjs if tok_list.count(a.word) > 1}

    chosen_rel = scene.relations[0] if scene.relations else None
    if repeated_adjs and len(scene.relations) > 1:
        # Prefer a relation edge whose endpoints share color or size so a repeated modifier can be satisfied
        for rel_cand in scene.relations:
            sa = scene.objects[rel_cand.subject]
            oa = scene.objects[rel_cand.object]
            if sa.kind != oa.kind and (sa.color == oa.color or sa.size == oa.size):
                chosen_rel = rel_cand
                break

    if chosen_rel is not None:
        s_obj = scene.objects[chosen_rel.subject]
        o_obj = scene.objects[chosen_rel.object]
        r_name = chosen_rel.name
    else:
        s_obj = scene.objects[0]
        o_obj = scene.objects[1 if len(scene.objects) > 1 else 0]
        r_name = "chases"

    # Ensure s_obj and o_obj have distinct kinds so _apply_forward/_apply_backward succeeds
    if s_obj.kind == o_obj.kind:
        for alt_o in scene.objects:
            if alt_o.kind != s_obj.kind:
                o_obj = alt_o
                break

    updated[nouns[0].word] = LexicalEntry(
        word=nouns[0].word,
        category="N",
        predicate=f"kind:{s_obj.kind}",
        supporting_examples=nouns[0].supporting_examples,
    )
    updated[nouns[1].word] = LexicalEntry(
        word=nouns[1].word,
        category="N",
        predicate=f"kind:{o_obj.kind}",
        supporting_examples=nouns[1].supporting_examples,
    )
    n0_pos = tok_list.index(nouns[0].word)
    n1_pos = tok_list.index(nouns[1].word)
    for adj in adjs:
        fam = adj.family if adj.family in ("color", "size") else "color"
        if adj.word in repeated_adjs:
            if s_obj.color == o_obj.color:
                fam = "color"
            elif s_obj.size == o_obj.size:
                fam = "size"
        adj_pos = tok_list.index(adj.word)
        if adj.category == "N/N":
            ref_obj = s_obj if adj_pos < n0_pos else o_obj
        else:
            ref_obj = o_obj if (adj_pos > n1_pos or abs(adj_pos - n1_pos) < abs(adj_pos - n0_pos)) else s_obj
        updated[adj.word] = LexicalEntry(
            word=adj.word,
            category=adj.category,
            predicate=f"{fam}:{ref_obj.value(fam)}",
            supporting_examples=adj.supporting_examples,
        )
    for v_entry in verbs:
        updated[v_entry.word] = LexicalEntry(
            word=v_entry.word,
            category=v_entry.category,
            predicate=f"rel:{r_name}(x0,x1)",
            supporting_examples=v_entry.supporting_examples,
        )
    return updated
