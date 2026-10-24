"""Categorial Grammar over base categories N, NP, S with slash depth <= 2.

Supports only Forward Application (>) and Backward Application (<):
- Forward Application (>):  X/Y + Y -> X
- Backward Application (<): Y + X\\Y -> X

Word meanings in the lexical hypothesis space are:
- Unary predicates: ``kind:<val>``, ``color:<val>``, ``size:<val>``
- Binary relations: ``rel:<name>(x0,x1)``, ``rel:<name>(x1,x0)``, or ``rel:<name>:sym(x0,x1)``
- Speech-act marker: ``Q`` (question particle)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping, Sequence

from nexus.world.logic import Atom, LogicalForm, make_lf
from nexus.world.scene import Scene

BASE_CATEGORIES: frozenset[str] = frozenset({"N", "NP", "S"})

CATEGORY_DEPTHS: Mapping[str, int] = {
    "N": 0,
    "NP": 0,
    "S": 0,
    "N/N": 1,
    "N\\N": 1,
    "S/S": 1,
    "S\\S": 1,
    "S\\NP": 1,
    "S/NP": 1,
    "(S\\NP)\\NP": 2,
    "(S\\NP)/NP": 2,
    "(S/NP)/NP": 2,
}


def category_depth(category: str) -> int:
    """Return slash depth (0, 1, or 2) of ``category``."""
    if category not in CATEGORY_DEPTHS:
        raise ValueError(f"Unsupported category or depth > 2: {category!r}")
    return CATEGORY_DEPTHS[category]


def predicate_atom_count(predicate: str) -> int:
    """Number of logical atoms contributed by ``predicate`` (0 for Q, 1 otherwise)."""
    return 0 if predicate == "Q" else 1


@dataclass(frozen=True)
class LexicalEntry:
    """A single word's hypothesized syntactic category, semantic predicate, and provenance."""

    word: str
    category: str
    predicate: str
    supporting_examples: tuple[str, ...] = ()

    @property
    def depth(self) -> int:
        return category_depth(self.category)

    @property
    def atom_count(self) -> int:
        return predicate_atom_count(self.predicate)

    @property
    def family(self) -> str:
        if self.predicate == "Q":
            return "Q"
        return self.predicate.split(":", 1)[0]

    def to_json(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "category": self.category,
            "predicate": self.predicate,
            "depth": self.depth,
            "atom_count": self.atom_count,
            "supporting_examples": list(self.supporting_examples),
        }


@dataclass(frozen=True)
class ParseNode:
    """A node in a binary categorial derivation tree."""

    cat: str
    span: tuple[int, int]
    rule: str  # "LEX", ">", or "<"
    act: str = "assert"
    head_kind: str | None = None
    adjectives: tuple[str, ...] = ()
    subj_np: tuple[str, tuple[str, ...]] | None = None
    obj_np: tuple[str, tuple[str, ...]] | None = None
    rel_pred: str | None = None
    lexical_entry: LexicalEntry | None = None
    left: ParseNode | None = None
    right: ParseNode | None = None

    def to_tree_dict(self) -> dict[str, Any]:
        """Serialize derivation tree for explain()."""
        out: dict[str, Any] = {
            "category": self.cat,
            "span": list(self.span),
            "rule": self.rule,
        }
        if self.lexical_entry is not None:
            out["word"] = self.lexical_entry.word
            out["predicate"] = self.lexical_entry.predicate
        if self.left is not None and self.right is not None:
            out["left"] = self.left.to_tree_dict()
            out["right"] = self.right.to_tree_dict()
        return out


def _apply_forward(left: ParseNode, right: ParseNode, total_len: int) -> ParseNode | None:
    """Forward Application (>): X/Y + Y -> X."""
    span = (left.span[0], right.span[1])
    # 1. Pre-nominal modifier: N/N + N -> N
    if left.cat == "N/N" and right.cat == "N" and right.head_kind is not None:
        assert left.lexical_entry is not None
        mod_pred = left.lexical_entry.predicate
        mod_fam = mod_pred.split(":", 1)[0]
        if any(a.split(":", 1)[0] == mod_fam for a in right.adjectives):
            return None
        return ParseNode(
            cat="N",
            span=span,
            rule=">",
            act=right.act,
            head_kind=right.head_kind,
            adjectives=tuple(sorted((*right.adjectives, mod_pred))),
            left=left,
            right=right,
        )

    # 2. SVO transitive verb step 1: (S\NP)/NP + N -> S\NP
    if left.cat == "(S\\NP)/NP" and right.cat == "N" and right.head_kind is not None:
        return ParseNode(
            cat="S\\NP",
            span=span,
            rule=">",
            act="assert",
            obj_np=(right.head_kind, right.adjectives),
            rel_pred=left.rel_pred,
            left=left,
            right=right,
        )

    # 3. VSO transitive verb step 1: (S/NP)/NP + N -> S/NP
    if left.cat == "(S/NP)/NP" and right.cat == "N" and right.head_kind is not None:
        return ParseNode(
            cat="S/NP",
            span=span,
            rule=">",
            act="assert",
            subj_np=(right.head_kind, right.adjectives),
            rel_pred=left.rel_pred,
            left=left,
            right=right,
        )

    # 4. VSO transitive verb step 2: S/NP + N -> S
    if left.cat == "S/NP" and right.cat == "N" and right.head_kind is not None and left.subj_np is not None:
        if left.subj_np[0] == right.head_kind:
            return None
        return ParseNode(
            cat="S",
            span=span,
            rule=">",
            act="assert",
            subj_np=left.subj_np,
            obj_np=(right.head_kind, right.adjectives),
            rel_pred=left.rel_pred,
            left=left,
            right=right,
        )

    # 5. Initial question marker: S/S + S -> S(ask) (at root span)
    if left.cat == "S/S" and span == (0, total_len) and right.act == "assert":
        if right.cat == "S" and right.subj_np is not None and right.obj_np is not None:
            return ParseNode(
                cat="S",
                span=span,
                rule=">",
                act="ask",
                subj_np=right.subj_np,
                obj_np=right.obj_np,
                rel_pred=right.rel_pred,
                left=left,
                right=right,
            )
        if right.cat == "N" and right.head_kind is not None:
            return ParseNode(
                cat="S",
                span=span,
                rule=">",
                act="ask",
                head_kind=right.head_kind,
                adjectives=right.adjectives,
                left=left,
                right=right,
            )

    return None


def _apply_backward(left: ParseNode, right: ParseNode, total_len: int) -> ParseNode | None:
    """Backward Application (<): Y + X\\Y -> X."""
    span = (left.span[0], right.span[1])
    # 1. Post-nominal modifier: N + N\N -> N
    if left.cat == "N" and left.head_kind is not None and right.cat == "N\\N":
        assert right.lexical_entry is not None
        mod_pred = right.lexical_entry.predicate
        mod_fam = mod_pred.split(":", 1)[0]
        if any(a.split(":", 1)[0] == mod_fam for a in left.adjectives):
            return None
        return ParseNode(
            cat="N",
            span=span,
            rule="<",
            act=left.act,
            head_kind=left.head_kind,
            adjectives=tuple(sorted((*left.adjectives, mod_pred))),
            left=left,
            right=right,
        )

    # 2. SOV transitive verb step 1: N + (S\NP)\NP -> S\NP
    if left.cat == "N" and left.head_kind is not None and right.cat == "(S\\NP)\\NP":
        return ParseNode(
            cat="S\\NP",
            span=span,
            rule="<",
            act="assert",
            obj_np=(left.head_kind, left.adjectives),
            rel_pred=right.rel_pred,
            left=left,
            right=right,
        )

    # 3. SOV and SVO transitive verb step 2: N + S\NP -> S
    if left.cat == "N" and left.head_kind is not None and right.cat == "S\\NP" and right.obj_np is not None:
        if left.head_kind == right.obj_np[0]:
            return None
        return ParseNode(
            cat="S",
            span=span,
            rule="<",
            act="assert",
            subj_np=(left.head_kind, left.adjectives),
            obj_np=right.obj_np,
            rel_pred=right.rel_pred,
            left=left,
            right=right,
        )

    # 4. Final question marker: S + S\S -> S(ask) (at root span)
    if right.cat == "S\\S" and span == (0, total_len) and left.act == "assert":
        if left.cat == "S" and left.subj_np is not None and left.obj_np is not None:
            return ParseNode(
                cat="S",
                span=span,
                rule="<",
                act="ask",
                subj_np=left.subj_np,
                obj_np=left.obj_np,
                rel_pred=left.rel_pred,
                left=left,
                right=right,
            )
        if left.cat == "N" and left.head_kind is not None:
            return ParseNode(
                cat="S",
                span=span,
                rule="<",
                act="ask",
                head_kind=left.head_kind,
                adjectives=left.adjectives,
                left=left,
                right=right,
            )

    return None


def node_to_lf(node: ParseNode) -> tuple[LogicalForm, bool, bool] | None:
    """Convert a root ParseNode into ``(LogicalForm, is_rel_symmetric, is_rel_reversed)``."""
    if node.cat == "N" and node.head_kind is not None:
        atoms = [Atom(node.head_kind, ("x0",))]
        atoms.extend(Atom(a, ("x0",)) for a in node.adjectives)
        return make_lf(node.act, ("x0",), atoms), False, False

    if node.cat == "S":
        if node.head_kind is not None and node.subj_np is None:
            atoms = [Atom(node.head_kind, ("x0",))]
            atoms.extend(Atom(a, ("x0",)) for a in node.adjectives)
            return make_lf(node.act, ("x0",), atoms), False, False

        if node.subj_np is not None and node.obj_np is not None and node.rel_pred is not None:
            subj_kind, subj_adjs = node.subj_np
            obj_kind, obj_adjs = node.obj_np
            # Parse rel_pred format: "rel:<name>(x0,x1)", "rel:<name>(x1,x0)", or "rel:<name>:sym(x0,x1)"
            raw = node.rel_pred
            is_sym = ":sym(" in raw
            is_rev = raw.endswith("(x1,x0)")
            base_rel = raw.split("(", 1)[0].replace(":sym", "")
            if is_rev:
                # Reversed argument structure: first NP is object x1, second NP is subject x0
                atoms = [
                    Atom(obj_kind, ("x0",)),
                    Atom(subj_kind, ("x1",)),
                    Atom(base_rel, ("x0", "x1")),
                ]
                atoms.extend(Atom(a, ("x1",)) for a in subj_adjs)
                atoms.extend(Atom(a, ("x0",)) for a in obj_adjs)
            else:
                atoms = [
                    Atom(subj_kind, ("x0",)),
                    Atom(obj_kind, ("x1",)),
                    Atom(base_rel, ("x0", "x1")),
                ]
                atoms.extend(Atom(a, ("x0",)) for a in subj_adjs)
                atoms.extend(Atom(a, ("x1",)) for a in obj_adjs)
            return make_lf(node.act, ("x0", "x1"), atoms), is_sym, is_rev

    return None


def parse_tokens(
    tokens: Sequence[str],
    entries: Mapping[str, LexicalEntry],
) -> tuple[ParseNode, LogicalForm, bool] | None:
    """Parse ``tokens`` using CKY chart over Forward (>) and Backward (<) Application.

    Returns ``(root_node, lf, is_rel_symmetric)`` if a valid spanning derivation exists,
    or ``None`` if the sentence cannot be derived.
    """
    n = len(tokens)
    if n == 0 or any(t not in entries for t in tokens):
        return None

    chart: list[list[list[ParseNode]]] = [[[] for _ in range(n + 1)] for _ in range(n)]
    for i, tok in enumerate(tokens):
        entry = entries[tok]
        if entry.category == "N":
            node = ParseNode(
                cat="N",
                span=(i, i + 1),
                rule="LEX",
                head_kind=entry.predicate,
                lexical_entry=entry,
            )
        elif entry.category in ("(S\\NP)\\NP", "(S\\NP)/NP", "(S/NP)/NP"):
            node = ParseNode(
                cat=entry.category,
                span=(i, i + 1),
                rule="LEX",
                rel_pred=entry.predicate,
                lexical_entry=entry,
            )
        else:
            node = ParseNode(
                cat=entry.category,
                span=(i, i + 1),
                rule="LEX",
                lexical_entry=entry,
            )
        chart[i][i + 1].append(node)

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length
            cell = chart[i][j]
            seen_sigs: set[tuple[Any, ...]] = set()
            for k in range(i + 1, j):
                for left_node in chart[i][k]:
                    for right_node in chart[k][j]:
                        for comb in (
                            _apply_forward(left_node, right_node, n),
                            _apply_backward(left_node, right_node, n),
                        ):
                            if comb is not None:
                                sig = (
                                    comb.cat,
                                    comb.act,
                                    comb.head_kind,
                                    comb.adjectives,
                                    comb.subj_np,
                                    comb.obj_np,
                                    comb.rel_pred,
                                )
                                if sig not in seen_sigs:
                                    seen_sigs.add(sig)
                                    cell.append(comb)

    for root in chart[0][n]:
        if root.cat in ("S", "N"):
            conv = node_to_lf(root)
            if conv is not None:
                lf, is_sym, _ = conv
                return root, lf, is_sym
    return None


def evaluate_parsed_lf(lf: LogicalForm, scene: Scene, is_rel_symmetric: bool) -> bool:
    """Evaluate ``lf`` against ``scene`` using the induced symmetry of its relation."""
    sym_set: frozenset[str] = frozenset()
    if is_rel_symmetric:
        for atom in lf.atoms:
            if atom.family == "rel":
                sym_set = frozenset({atom.value})
    ids = range(len(scene.objects))
    for values in product(ids, repeat=len(lf.variables)):
        assignment = dict(zip(lf.variables, values))
        ok = True
        for atom in lf.atoms:
            if atom.family == "rel":
                a, b = assignment[atom.args[0]], assignment[atom.args[1]]
                if not scene.holds(atom.value, a, b, atom.value in sym_set):
                    ok = False
                    break
            else:
                obj_id = assignment[atom.args[0]]
                if scene.objects[obj_id].value(atom.family) != atom.value:
                    ok = False
                    break
        if ok:
            return True
    return False


def find_witness_or_failure(
    lf: LogicalForm, scene: Scene, is_rel_symmetric: bool
) -> tuple[dict[str, int] | None, str]:
    """Find a satisfying variable->object assignment or a human-readable failure reason."""
    sym_set: frozenset[str] = frozenset()
    if is_rel_symmetric:
        for atom in lf.atoms:
            if atom.family == "rel":
                sym_set = frozenset({atom.value})

    ids = range(len(scene.objects))
    best_failed_count = -1
    best_failure_msg = "No objects in scene."

    for values in product(ids, repeat=len(lf.variables)):
        assignment = dict(zip(lf.variables, values))
        failed_atoms: list[str] = []
        matched_count = 0
        for atom in lf.atoms:
            if atom.family == "rel":
                a, b = assignment[atom.args[0]], assignment[atom.args[1]]
                if scene.holds(atom.value, a, b, atom.value in sym_set):
                    matched_count += 1
                else:
                    failed_atoms.append(f"{atom.predicate}(obj_{a}, obj_{b}) does not hold")
            else:
                obj_id = assignment[atom.args[0]]
                actual = scene.objects[obj_id].value(atom.family)
                if actual == atom.value:
                    matched_count += 1
                else:
                    failed_atoms.append(
                        f"{atom.predicate}(obj_{obj_id}) failed (obj_{obj_id}.{atom.family}={actual})"
                    )
        if not failed_atoms:
            return assignment, "All atoms satisfied."
        if matched_count > best_failed_count:
            best_failed_count = matched_count
            assign_str = ", ".join(f"{v}->obj_{idx}" for v, idx in assignment.items())
            best_failure_msg = f"Closest assignment ({assign_str}) failed on: {'; '.join(failed_atoms)}"

    return None, best_failure_msg
