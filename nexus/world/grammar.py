"""The hidden grammar: maps structured meanings to tokens and logical forms.

This module is part of the *world*, not the learner. It knows the lexicon and
word orders from the grammar config. Learners must never import it.
"""

from __future__ import annotations

from dataclasses import dataclass

from nexus.world.config import ADJECTIVE_ATTRIBUTES, ConfigError, GrammarConfig
from nexus.world.logic import Atom, LogicalForm, make_lf


@dataclass(frozen=True)
class NPSpec:
    """A noun phrase meaning: a kind plus zero or more adjectives."""

    kind: str
    adjectives: tuple[tuple[str, str], ...]  # sorted (attribute, value) pairs

    def __post_init__(self) -> None:
        attrs = [a for a, _ in self.adjectives]
        if any(a not in ADJECTIVE_ATTRIBUTES for a in attrs):
            raise ConfigError(f"adjective attributes must be in {ADJECTIVE_ATTRIBUTES}: {attrs}")
        if len(set(attrs)) != len(attrs):
            raise ConfigError(f"at most one adjective per attribute: {self.adjectives}")
        if tuple(sorted(self.adjectives)) != self.adjectives:
            raise ConfigError("adjectives must be sorted")

    @property
    def kind_predicate(self) -> str:
        """E.g. ``kind:cube``."""
        return f"kind:{self.kind}"

    @property
    def adjective_predicates(self) -> tuple[str, ...]:
        """E.g. ``("color:red",)``."""
        return tuple(f"{a}:{v}" for a, v in self.adjectives)

    def fixed_attributes(self) -> dict[str, str]:
        """Attribute values a witness object must have."""
        return {"kind": self.kind, **dict(self.adjectives)}


@dataclass(frozen=True)
class Meaning:
    """A structured meaning prior to realization."""

    construction: str
    nps: tuple[tuple[str, NPSpec], ...]  # (role, spec) in template order
    relation: str | None

    def np(self, role: str) -> NPSpec:
        """Return the NP for ``role``."""
        for r, spec in self.nps:
            if r == role:
                return spec
        raise KeyError(role)

    def with_np(self, role: str, spec: NPSpec) -> "Meaning":
        """Return a copy with the NP for ``role`` replaced."""
        return Meaning(self.construction, tuple((r, spec if r == role else s) for r, s in self.nps), self.relation)


class Grammar:
    """Realizes meanings as token sequences and logical forms."""

    def __init__(self, config: GrammarConfig) -> None:
        self.config = config

    def realize_np(self, spec: NPSpec) -> tuple[str, ...]:
        """Realize an NP using the configured NP word order."""
        adjectives = dict(spec.adjectives)
        out: list[str] = []
        for slot in self.config.np_order:
            if slot == "noun":
                out.append(self.config.word_for(spec.kind_predicate))
            elif slot in adjectives:
                out.append(self.config.word_for(f"{slot}:{adjectives[slot]}"))
        return tuple(out)

    def realize(self, meaning: Meaning) -> tuple[str, ...]:
        """Realize a full sentence."""
        construction = self._construction(meaning)
        tokens: list[str] = []
        for symbol in construction.template:
            if symbol == "NP":
                tokens.extend(self.realize_np(meaning.np("np")))
            elif symbol == "NP_subj":
                tokens.extend(self.realize_np(meaning.np("subj")))
            elif symbol == "NP_obj":
                tokens.extend(self.realize_np(meaning.np("obj")))
            elif symbol == "VERB":
                tokens.append(self.config.word_for(f"rel:{meaning.relation}"))
            elif symbol == "Q":
                tokens.append(self.config.question_particle)
            else:  # pragma: no cover - config validation forbids this
                raise ConfigError(f"unknown template symbol {symbol!r}")
        return tuple(tokens)

    def to_lf(self, meaning: Meaning) -> LogicalForm:
        """Translate a meaning into its canonical logical form."""
        construction = self._construction(meaning)
        variables = {role: f"x{i}" for i, (role, _) in enumerate(meaning.nps)}
        atoms: list[Atom] = []
        for role, spec in meaning.nps:
            var = variables[role]
            atoms.append(Atom(spec.kind_predicate, (var,)))
            atoms.extend(Atom(p, (var,)) for p in spec.adjective_predicates)
        if construction.has_relation:
            atoms.append(Atom(f"rel:{meaning.relation}", (variables["subj"], variables["obj"])))
        return make_lf(construction.act, tuple(variables.values()), atoms)

    def _construction(self, meaning: Meaning):
        if meaning.construction not in self.config.constructions:
            raise ConfigError(f"unknown construction {meaning.construction!r}")
        construction = self.config.constructions[meaning.construction]
        roles = tuple(r for r, _ in meaning.nps)
        if roles != construction.roles:
            raise ConfigError(f"{meaning.construction}: roles {roles} != {construction.roles}")
        if construction.has_relation != (meaning.relation is not None):
            raise ConfigError(f"{meaning.construction}: relation presence mismatch")
        return construction
