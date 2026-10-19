"""Loading and validation of world and grammar configuration files.

All configuration is plain JSON. Validation is strict: any missing key, unknown
key, or inconsistent value raises :class:`ConfigError`. There are no silent
defaults inside this module; defaults live in the JSON files themselves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

#: Perceptual attributes every scene object carries. This is part of the
#: (documented) inductive bias: the learner is given these dimensions.
ATTRIBUTES: tuple[str, ...] = ("kind", "color", "size")
#: Attributes that can be expressed by adjectives.
ADJECTIVE_ATTRIBUTES: tuple[str, ...] = ("color", "size")
#: Template symbols the sentence realizer understands.
TEMPLATE_SYMBOLS: frozenset[str] = frozenset({"NP", "NP_subj", "NP_obj", "VERB", "Q"})
#: Speech acts.
ACTS: frozenset[str] = frozenset({"assert", "ask"})

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_WORLD_CONFIG = CONFIG_DIR / "world_default.json"


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or inconsistent."""


@dataclass(frozen=True)
class RelationSpec:
    """A binary relation that may hold between two scene objects."""

    name: str
    symmetric: bool


@dataclass(frozen=True)
class Ontology:
    """The perceptual vocabulary of the world (attribute values, relations)."""

    kinds: tuple[str, ...]
    colors: tuple[str, ...]
    sizes: tuple[str, ...]
    relations: tuple[RelationSpec, ...]

    def values(self, attribute: str) -> tuple[str, ...]:
        """Return the possible values of ``attribute``."""
        table = {"kind": self.kinds, "color": self.colors, "size": self.sizes}
        if attribute not in table:
            raise ConfigError(f"unknown attribute {attribute!r}")
        return table[attribute]

    @property
    def relation_names(self) -> tuple[str, ...]:
        """Names of all relations."""
        return tuple(r.name for r in self.relations)

    @property
    def symmetric_relations(self) -> frozenset[str]:
        """Names of symmetric relations."""
        return frozenset(r.name for r in self.relations if r.symmetric)

    def all_predicates(self) -> tuple[str, ...]:
        """Every predicate symbol of the world, e.g. ``color:red``, ``rel:chases``."""
        preds = [f"{a}:{v}" for a in ATTRIBUTES for v in self.values(a)]
        preds.extend(f"rel:{r}" for r in self.relation_names)
        return tuple(preds)


@dataclass(frozen=True)
class ConstructionSpec:
    """One sentence construction of the hidden grammar."""

    name: str
    act: str
    template: tuple[str, ...]
    weight: float

    @property
    def roles(self) -> tuple[str, ...]:
        """NP roles in template order (``np``, ``subj``, ``obj``)."""
        mapping = {"NP": "np", "NP_subj": "subj", "NP_obj": "obj"}
        return tuple(mapping[s] for s in self.template if s in mapping)

    @property
    def has_relation(self) -> bool:
        """Whether the construction expresses a relation (contains a verb)."""
        return "VERB" in self.template


@dataclass(frozen=True)
class GrammarConfig:
    """The hidden grammar: lexicon, word orders, constructions."""

    name: str
    lexicon: Mapping[str, str]
    question_particle: str
    np_order: tuple[str, ...]
    constructions: Mapping[str, ConstructionSpec]

    def word_for(self, predicate: str) -> str:
        """Return the surface word for a predicate symbol."""
        if predicate not in self.lexicon:
            raise ConfigError(f"no word for predicate {predicate!r}")
        return self.lexicon[predicate]

    @property
    def vocabulary(self) -> frozenset[str]:
        """All surface words, including the question particle."""
        return frozenset(self.lexicon.values()) | {self.question_particle}


@dataclass(frozen=True)
class SceneConfig:
    """Parameters of the scene sampler."""

    min_objects: int
    max_objects: int
    p_relation: float
    max_relations: int


@dataclass(frozen=True)
class GenerationConfig:
    """Parameters of example generation."""

    max_adjectives_train: int
    p_adjective: float
    max_adjectives_longer: int
    p_adjective_longer: float
    p_contrast_pair: float
    p_near_miss_false: float
    max_rejection_tries: int


@dataclass(frozen=True)
class SplitConfig:
    """Sizes and structure of the dataset splits."""

    n_train: int
    n_dev_per_construction: int
    n_test_novel_combination: int
    n_test_novel_word_queries: int
    n_novel_word_support_pairs: int
    n_test_longer: int
    n_contradiction_misleading: int
    n_contradiction_filler: int
    n_contradiction_probes: int
    n_heldout_adj_noun_pairs: int
    n_heldout_noun_roles: int
    curriculum: tuple[tuple[str, ...], ...]
    shuffle_within_stage: bool


@dataclass(frozen=True)
class EvaluationConfig:
    """Parameters of the evaluation harness."""

    checkpoint_every: int
    criterion: float
    calibration_bins: int


@dataclass(frozen=True)
class WorldConfig:
    """Complete configuration of a world + dataset + evaluation."""

    name: str
    ontology: Ontology
    grammar: GrammarConfig
    scene: SceneConfig
    generation: GenerationConfig
    splits: SplitConfig
    evaluation: EvaluationConfig
    raw: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        """Return the raw JSON this config was built from (for result records)."""
        return json.loads(json.dumps(self.raw))


# --------------------------------------------------------------------------
# parsing helpers
# --------------------------------------------------------------------------


def _require_keys(obj: Mapping[str, Any], keys: set[str], where: str) -> None:
    if not isinstance(obj, Mapping):
        raise ConfigError(f"{where}: expected an object, got {type(obj).__name__}")
    missing = keys - set(obj)
    extra = set(obj) - keys
    if missing:
        raise ConfigError(f"{where}: missing keys {sorted(missing)}")
    if extra:
        raise ConfigError(f"{where}: unknown keys {sorted(extra)}")


def _str_tuple(value: Any, where: str, min_len: int = 1) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise ConfigError(f"{where}: expected a list of non-empty strings")
    if len(value) < min_len:
        raise ConfigError(f"{where}: expected at least {min_len} entries")
    if len(set(value)) != len(value):
        raise ConfigError(f"{where}: duplicate entries")
    return tuple(value)


def _prob(value: Any, where: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= value <= 1.0:
        raise ConfigError(f"{where}: expected a probability in [0, 1], got {value!r}")
    return float(value)


def _int(value: Any, where: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ConfigError(f"{where}: expected an integer >= {minimum}, got {value!r}")
    return value


def parse_ontology(obj: Mapping[str, Any]) -> Ontology:
    """Parse and validate the ``ontology`` section."""
    _require_keys(obj, {"kinds", "colors", "sizes", "relations"}, "ontology")
    rels_raw = obj["relations"]
    if not isinstance(rels_raw, list) or not rels_raw:
        raise ConfigError("ontology.relations: expected a non-empty list")
    relations = []
    for i, r in enumerate(rels_raw):
        _require_keys(r, {"name", "symmetric"}, f"ontology.relations[{i}]")
        if not isinstance(r["name"], str) or not r["name"]:
            raise ConfigError(f"ontology.relations[{i}].name must be a non-empty string")
        if not isinstance(r["symmetric"], bool):
            raise ConfigError(f"ontology.relations[{i}].symmetric must be a boolean")
        relations.append(RelationSpec(r["name"], r["symmetric"]))
    if len({r.name for r in relations}) != len(relations):
        raise ConfigError("ontology.relations: duplicate relation names")
    return Ontology(
        kinds=_str_tuple(obj["kinds"], "ontology.kinds", 2),
        colors=_str_tuple(obj["colors"], "ontology.colors", 2),
        sizes=_str_tuple(obj["sizes"], "ontology.sizes", 2),
        relations=tuple(relations),
    )


def parse_grammar(obj: Mapping[str, Any], ontology: Ontology) -> GrammarConfig:
    """Parse and validate a grammar file against an ontology."""
    _require_keys(
        obj,
        {"name", "description", "lexicon", "question_particle", "np_order", "constructions"},
        "grammar",
    )
    lexicon = obj["lexicon"]
    if not isinstance(lexicon, Mapping):
        raise ConfigError("grammar.lexicon must be an object")
    expected = set(ontology.all_predicates())
    if set(lexicon) != expected:
        missing = sorted(expected - set(lexicon))
        extra = sorted(set(lexicon) - expected)
        raise ConfigError(f"grammar.lexicon: missing {missing}, unknown {extra}")
    words = list(lexicon.values())
    if not all(isinstance(w, str) and w and " " not in w for w in words):
        raise ConfigError("grammar.lexicon: words must be non-empty strings without spaces")
    particle = obj["question_particle"]
    if not isinstance(particle, str) or not particle:
        raise ConfigError("grammar.question_particle must be a non-empty string")
    if len(set(words) | {particle}) != len(words) + 1:
        raise ConfigError("grammar: every word (and the particle) must be unique")
    np_order = _str_tuple(obj["np_order"], "grammar.np_order")
    if set(np_order) != {"noun", *ADJECTIVE_ATTRIBUTES}:
        raise ConfigError(f"grammar.np_order must be a permutation of noun + {ADJECTIVE_ATTRIBUTES}")
    constructions = _parse_constructions(obj["constructions"])
    return GrammarConfig(
        name=str(obj["name"]),
        lexicon=dict(lexicon),
        question_particle=particle,
        np_order=np_order,
        constructions=constructions,
    )


def _parse_constructions(obj: Any) -> dict[str, ConstructionSpec]:
    if not isinstance(obj, Mapping) or not obj:
        raise ConfigError("grammar.constructions must be a non-empty object")
    out: dict[str, ConstructionSpec] = {}
    for name, spec in obj.items():
        where = f"grammar.constructions.{name}"
        _require_keys(spec, {"act", "template", "weight"}, where)
        if spec["act"] not in ACTS:
            raise ConfigError(f"{where}.act must be one of {sorted(ACTS)}")
        template = _template(spec["template"], where)
        weight = spec["weight"]
        if not isinstance(weight, (int, float)) or weight <= 0:
            raise ConfigError(f"{where}.weight must be positive")
        out[name] = ConstructionSpec(name, spec["act"], template, float(weight))
    return out


def _template(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{where}.template must be a non-empty list")
    if any(s not in TEMPLATE_SYMBOLS for s in value):
        raise ConfigError(f"{where}.template: symbols must be in {sorted(TEMPLATE_SYMBOLS)}")
    template = tuple(value)
    has_verb = "VERB" in template
    nps = [s for s in template if s.startswith("NP")]
    if has_verb and sorted(nps) != ["NP_obj", "NP_subj"]:
        raise ConfigError(f"{where}: a VERB template needs exactly NP_subj and NP_obj")
    if not has_verb and nps != ["NP"]:
        raise ConfigError(f"{where}: a template without VERB needs exactly one NP")
    if template.count("Q") > 1:
        raise ConfigError(f"{where}: at most one Q")
    return template


def parse_world(obj: Mapping[str, Any], grammar_obj: Mapping[str, Any]) -> WorldConfig:
    """Parse a world config object plus its grammar object."""
    _require_keys(
        obj,
        {"name", "grammar_file", "ontology", "scene", "generation", "splits", "evaluation"},
        "world",
    )
    ontology = parse_ontology(obj["ontology"])
    grammar = parse_grammar(grammar_obj, ontology)
    scene = _parse_scene(obj["scene"])
    generation = _parse_generation(obj["generation"])
    splits = _parse_splits(obj["splits"], grammar)
    evaluation = _parse_evaluation(obj["evaluation"])
    raw = {"world": dict(obj), "grammar": dict(grammar_obj)}
    return WorldConfig(str(obj["name"]), ontology, grammar, scene, generation, splits, evaluation, raw)


def _parse_scene(obj: Mapping[str, Any]) -> SceneConfig:
    _require_keys(obj, {"min_objects", "max_objects", "p_relation", "max_relations"}, "scene")
    lo = _int(obj["min_objects"], "scene.min_objects", 2)
    hi = _int(obj["max_objects"], "scene.max_objects", lo)
    return SceneConfig(lo, hi, _prob(obj["p_relation"], "scene.p_relation"),
                       _int(obj["max_relations"], "scene.max_relations", 1))


def _parse_generation(obj: Mapping[str, Any]) -> GenerationConfig:
    keys = {f for f in GenerationConfig.__dataclass_fields__}
    _require_keys(obj, keys, "generation")
    max_train = _int(obj["max_adjectives_train"], "generation.max_adjectives_train", 0)
    max_longer = _int(obj["max_adjectives_longer"], "generation.max_adjectives_longer", max_train + 1)
    if max_longer > len(ADJECTIVE_ATTRIBUTES):
        raise ConfigError("generation.max_adjectives_longer exceeds number of adjective attributes")
    return GenerationConfig(
        max_adjectives_train=max_train,
        p_adjective=_prob(obj["p_adjective"], "generation.p_adjective"),
        max_adjectives_longer=max_longer,
        p_adjective_longer=_prob(obj["p_adjective_longer"], "generation.p_adjective_longer"),
        p_contrast_pair=_prob(obj["p_contrast_pair"], "generation.p_contrast_pair"),
        p_near_miss_false=_prob(obj["p_near_miss_false"], "generation.p_near_miss_false"),
        max_rejection_tries=_int(obj["max_rejection_tries"], "generation.max_rejection_tries", 1),
    )


def _parse_splits(obj: Mapping[str, Any], grammar: GrammarConfig) -> SplitConfig:
    keys = {f for f in SplitConfig.__dataclass_fields__}
    _require_keys(obj, keys, "splits")
    ints = {k: _int(obj[k], f"splits.{k}", 0) for k in keys - {"curriculum", "shuffle_within_stage"}}
    curriculum_raw = obj["curriculum"]
    if not isinstance(curriculum_raw, list) or not curriculum_raw:
        raise ConfigError("splits.curriculum must be a non-empty list of stages")
    curriculum = tuple(_str_tuple(stage, f"splits.curriculum[{i}]") for i, stage in enumerate(curriculum_raw))
    flat = [c for stage in curriculum for c in stage]
    if sorted(flat) != sorted(grammar.constructions):
        raise ConfigError("splits.curriculum must mention every construction exactly once")
    if not isinstance(obj["shuffle_within_stage"], bool):
        raise ConfigError("splits.shuffle_within_stage must be a boolean")
    return SplitConfig(curriculum=curriculum, shuffle_within_stage=obj["shuffle_within_stage"], **ints)


def _parse_evaluation(obj: Mapping[str, Any]) -> EvaluationConfig:
    _require_keys(obj, {"checkpoint_every", "criterion", "calibration_bins"}, "evaluation")
    return EvaluationConfig(
        checkpoint_every=_int(obj["checkpoint_every"], "evaluation.checkpoint_every", 1),
        criterion=_prob(obj["criterion"], "evaluation.criterion"),
        calibration_bins=_int(obj["calibration_bins"], "evaluation.calibration_bins", 1),
    )


def load_world_config(path: str | Path = DEFAULT_WORLD_CONFIG) -> WorldConfig:
    """Load a world config JSON file and the grammar file it references."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        world_obj = json.load(fh)
    if not isinstance(world_obj, dict) or "grammar_file" not in world_obj:
        raise ConfigError(f"{path}: missing grammar_file")
    grammar_path = path.parent / world_obj["grammar_file"]
    with grammar_path.open("r", encoding="utf-8") as fh:
        grammar_obj = json.load(fh)
    return parse_world(world_obj, grammar_obj)


def randomize_grammar(base_grammar: GrammarConfig, ontology: Ontology, rng: Any) -> GrammarConfig:
    """Derive a randomized grammar for a seed.

    1. Permutes word-to-meaning assignment within each category
       (kinds, colors, sizes, relations).
    2. Randomizes within-NP word order among permutations of noun + adjectives.
    3. Randomizes clause order among typological variants (SOV, SVO, VSO).
    4. Randomizes question particle position between initial and final.
    """
    new_lexicon: dict[str, str] = {}
    for attr, vals in [
        ("kind", ontology.kinds),
        ("color", ontology.colors),
        ("size", ontology.sizes),
    ]:
        words = [base_grammar.lexicon[f"{attr}:{v}"] for v in vals]
        rng.shuffle(words)
        for v, w in zip(vals, words):
            new_lexicon[f"{attr}:{v}"] = w

    rel_names = ontology.relation_names
    rel_words = [base_grammar.lexicon[f"rel:{r}"] for r in rel_names]
    rng.shuffle(rel_words)
    for r, w in zip(rel_names, rel_words):
        new_lexicon[f"rel:{r}"] = w

    # NP order: permutation of ("noun", "color", "size")
    np_order_candidates = [
        ("noun", "size", "color"),
        ("noun", "color", "size"),
        ("color", "noun", "size"),
        ("size", "color", "noun"),
        ("color", "size", "noun"),
        ("size", "noun", "color"),
    ]
    np_order = tuple(rng.choice(np_order_candidates))

    # Clause order: typological variants for transitive sentences
    clause_variants = [
        ("NP_subj", "NP_obj", "VERB"),  # SOV
        ("NP_subj", "VERB", "NP_obj"),  # SVO
        ("VERB", "NP_subj", "NP_obj"),  # VSO
    ]
    trans_clause = rng.choice(clause_variants)

    # Question particle position: initial or final
    q_pos = rng.choice(["initial", "final"])
    if q_pos == "initial":
        np_ask_template = ("Q", "NP")
        trans_ask_template = ("Q", *trans_clause)
    else:
        np_ask_template = ("NP", "Q")
        trans_ask_template = (*trans_clause, "Q")

    constructions = {
        "np_assert": ConstructionSpec("np_assert", "assert", ("NP",), base_grammar.constructions["np_assert"].weight),
        "np_ask": ConstructionSpec("np_ask", "ask", np_ask_template, base_grammar.constructions["np_ask"].weight),
        "trans_assert": ConstructionSpec(
            "trans_assert", "assert", trans_clause, base_grammar.constructions["trans_assert"].weight
        ),
        "trans_ask": ConstructionSpec(
            "trans_ask", "ask", trans_ask_template, base_grammar.constructions["trans_ask"].weight
        ),
    }

    return GrammarConfig(
        name=f"{base_grammar.name}_rand_{rng.randint(0, 999999)}",
        lexicon=new_lexicon,
        question_particle=base_grammar.question_particle,
        np_order=np_order,
        constructions=constructions,
    )

