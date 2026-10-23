"""Logical forms: existentially quantified conjunctions of atoms.

Every meaning in the Phase 0 world has the shape::

    ACT( EXISTS x0 .. xn . atom_1 & ... & atom_k )

where ``ACT`` is ``assert`` or ``ask`` and each atom is a unary attribute
predicate (``color:red(x0)``) or a binary relation (``rel:chases(x0,x1)``).
Logical forms are ground truth for evaluation only; learners never see them.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from typing import Any, Iterable, Mapping

from nexus.world.config import ACTS, ATTRIBUTES
from nexus.world.scene import Scene


class LogicError(ValueError):
    """Raised for malformed logical forms."""


@dataclass(frozen=True, order=True)
class Atom:
    """A predicate applied to variables."""

    predicate: str
    args: tuple[str, ...]

    def __post_init__(self) -> None:
        prefix, sep, value = self.predicate.partition(":")
        if not sep or not value:
            raise LogicError(f"predicate must look like 'attr:value', got {self.predicate!r}")
        arity = 2 if prefix == "rel" else 1
        if prefix != "rel" and prefix not in ATTRIBUTES:
            raise LogicError(f"unknown predicate family {prefix!r}")
        if len(self.args) != arity:
            raise LogicError(f"{self.predicate} expects {arity} args, got {self.args}")

    @property
    def family(self) -> str:
        """``kind``, ``color``, ``size`` or ``rel``."""
        return self.predicate.split(":", 1)[0]

    @property
    def value(self) -> str:
        """The value part, e.g. ``red`` for ``color:red``."""
        return self.predicate.split(":", 1)[1]

    def __str__(self) -> str:
        return f"{self.predicate}({','.join(self.args)})"


@dataclass(frozen=True)
class LogicalForm:
    """A canonical logical form. Build with :func:`make_lf`."""

    act: str
    variables: tuple[str, ...]
    atoms: tuple[Atom, ...]

    def __post_init__(self) -> None:
        if self.act not in ACTS:
            raise LogicError(f"unknown act {self.act!r}")
        if tuple(sorted(set(self.atoms))) != self.atoms:
            raise LogicError("atoms must be sorted and unique; use make_lf()")
        used = {v for a in self.atoms for v in a.args}
        if used != set(self.variables):
            raise LogicError(f"variables {self.variables} do not match atoms' vars {sorted(used)}")

    def canonical(self) -> str:
        """Human-readable canonical string."""
        body = " & ".join(str(a) for a in self.atoms)
        return f"{self.act.upper()}(EXISTS {' '.join(self.variables)} . {body})"

    def to_json(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "act": self.act,
            "variables": list(self.variables),
            "atoms": [[a.predicate, list(a.args)] for a in self.atoms],
        }

    @staticmethod
    def from_json(obj: Mapping[str, Any]) -> "LogicalForm":
        """Deserialize; fails loudly on malformed input."""
        if not isinstance(obj, Mapping) or set(obj) != {"act", "variables", "atoms"}:
            raise LogicError(f"malformed logical form {obj!r}")
        atoms = [Atom(str(p), tuple(str(x) for x in args)) for p, args in obj["atoms"]]
        return make_lf(str(obj["act"]), tuple(obj["variables"]), atoms)


def make_lf(act: str, variables: tuple[str, ...], atoms: Iterable[Atom]) -> LogicalForm:
    """Construct a canonical logical form (sorted, deduplicated atoms)."""
    return LogicalForm(act, tuple(variables), tuple(sorted(set(atoms))))


def atom_holds(atom: Atom, assignment: Mapping[str, int], scene: Scene, symmetric: frozenset[str]) -> bool:
    """Whether ``atom`` is true under a variable assignment."""
    if atom.family == "rel":
        a, b = (assignment[v] for v in atom.args)
        return scene.holds(atom.value, a, b, atom.value in symmetric)
    (obj_id,) = (assignment[v] for v in atom.args)
    return scene.objects[obj_id].value(atom.family) == atom.value


def evaluate(lf: LogicalForm, scene: Scene, symmetric: frozenset[str]) -> bool:
    """Model-check ``lf`` against ``scene`` by brute-force over assignments.

    Variables range over all objects independently (standard first-order
    semantics; scene relations are irreflexive so relational atoms force
    distinct objects anyway).
    """
    ids = range(len(scene.objects))
    for values in product(ids, repeat=len(lf.variables)):
        assignment = dict(zip(lf.variables, values))
        if all(atom_holds(a, assignment, scene, symmetric) for a in lf.atoms):
            return True
    return False


def naive_evaluate(lf: LogicalForm, scene: Scene, symmetric: frozenset[str]) -> bool:
    """Second, independent naive model checker using ``itertools.permutations``.

    Shares no helper functions (``atom_holds``, ``scene.holds``, ``obj.value``)
    with :func:`evaluate`, inspecting raw ``SceneObject`` attributes and
    ``Relation`` tuples directly.
    """
    n_objs = len(scene.objects)
    n_vars = len(lf.variables)
    has_rel = any(a.predicate.startswith("rel:") for a in lf.atoms)

    candidate_tuples: list[tuple[int, ...]] = list(permutations(range(n_objs), n_vars))
    if not has_rel and n_vars == 2:
        candidate_tuples.extend((i, i) for (i,) in permutations(range(n_objs), 1))

    for tup in candidate_tuples:
        var_map = {lf.variables[idx]: tup[idx] for idx in range(n_vars)}
        ok = True
        for atom in lf.atoms:
            prefix, _, val = atom.predicate.partition(":")
            if prefix == "kind":
                obj = scene.objects[var_map[atom.args[0]]]
                if obj.kind != val:
                    ok = False
                    break
            elif prefix == "color":
                obj = scene.objects[var_map[atom.args[0]]]
                if obj.color != val:
                    ok = False
                    break
            elif prefix == "size":
                obj = scene.objects[var_map[atom.args[0]]]
                if obj.size != val:
                    ok = False
                    break
            elif prefix == "rel":
                u = var_map[atom.args[0]]
                v = var_map[atom.args[1]]
                found = False
                for r in scene.relations:
                    if r.name == val:
                        if r.subject == u and r.object == v:
                            found = True
                            break
                        if val in symmetric and r.subject == v and r.object == u:
                            found = True
                            break
                if not found:
                    ok = False
                    break
            else:
                raise LogicError(f"unsupported prefix {prefix!r}")
        if ok:
            return True
    return False

