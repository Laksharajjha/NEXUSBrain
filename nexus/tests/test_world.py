"""Tests for config loading, scenes, logical forms and the hidden grammar."""

from __future__ import annotations

import copy
import json

import pytest

from nexus.world.config import CONFIG_DIR, ConfigError, parse_world
from nexus.world.grammar import Grammar, Meaning, NPSpec
from nexus.world.logic import Atom, LogicError, LogicalForm, evaluate, make_lf
from nexus.world.scene import Scene, SceneError, build_scene, validate_scene

ENGLISH = {"the", "a", "red", "blue", "green", "yellow", "big", "small", "cube", "ball", "is", "chases"}


def _raw():
    world = json.loads((CONFIG_DIR / "world_default.json").read_text())
    grammar = json.loads((CONFIG_DIR / "grammar_default.json").read_text())
    return world, grammar


# ---------------------------------------------------------------- config


def test_default_config_loads(config):
    assert len(config.ontology.kinds) == 6
    assert len(config.ontology.colors) == 4
    assert len(config.ontology.sizes) == 3
    assert set(config.grammar.constructions) == {"np_assert", "np_ask", "trans_assert", "trans_ask"}


def test_words_are_invented_and_unique(config):
    vocab = config.grammar.vocabulary
    assert len(vocab) == len(config.grammar.lexicon) + 1
    assert not vocab & ENGLISH


def test_word_order_is_not_english(config):
    assert config.grammar.np_order[0] == "noun"  # adjectives follow the noun
    trans = config.grammar.constructions["trans_assert"].template
    assert trans.index("VERB") > trans.index("NP_obj")  # SOV


@pytest.mark.parametrize("mutate,match", [
    (lambda w, g: g["lexicon"].pop("kind:cube"), "missing"),
    (lambda w, g: g["lexicon"].__setitem__("kind:ball", "dax"), "unique"),
    (lambda w, g: g["constructions"]["np_ask"].__setitem__("template", ["NP", "VERB"]), "VERB"),
    (lambda w, g: g.__setitem__("np_order", ["noun", "color"]), "np_order"),
    (lambda w, g: w["scene"].__setitem__("p_relation", 1.5), "probability"),
    (lambda w, g: w["splits"].__setitem__("bogus", 1), "unknown keys"),
    (lambda w, g: w["splits"]["curriculum"].pop(), "curriculum"),
])
def test_malformed_config_fails_loudly(mutate, match):
    world, grammar = _raw()
    world, grammar = copy.deepcopy(world), copy.deepcopy(grammar)
    mutate(world, grammar)
    with pytest.raises(ConfigError, match=match):
        parse_world(world, grammar)


# ---------------------------------------------------------------- scenes


def _scene():
    attrs = [
        {"kind": "cube", "color": "red", "size": "small"},
        {"kind": "ball", "color": "blue", "size": "large"},
        {"kind": "cone", "color": "red", "size": "medium"},
    ]
    return build_scene(attrs, [("chases", 0, 1), ("next_to", 2, 1)])


def test_scene_json_roundtrip(config):
    s = _scene()
    assert Scene.from_json(json.loads(json.dumps(s.to_json())), config.ontology) == s


def test_canonical_key_is_permutation_invariant():
    a = _scene()
    attrs = [
        {"kind": "cone", "color": "red", "size": "medium"},
        {"kind": "ball", "color": "blue", "size": "large"},
        {"kind": "cube", "color": "red", "size": "small"},
    ]
    b = build_scene(attrs, [("chases", 2, 1), ("next_to", 0, 1)])
    c = build_scene(attrs, [("chases", 1, 2), ("next_to", 0, 1)])
    assert a.canonical_key() == b.canonical_key()
    assert a.canonical_key() != c.canonical_key()


@pytest.mark.parametrize("bad", [
    {"objects": [{"id": 1, "kind": "cube", "color": "red", "size": "small"}], "relations": []},
    {"objects": [{"id": 0, "kind": "cube", "color": "red"}], "relations": []},
    {"objects": [{"id": 0, "kind": "cube", "color": "red", "size": "small"}],
     "relations": [{"name": "chases", "subject": 0, "object": 0}]},
    {"objects": []},
])
def test_malformed_scene_fails_loudly(bad):
    with pytest.raises(SceneError):
        Scene.from_json(bad)


def test_scene_outside_ontology_fails(config):
    s = build_scene([{"kind": "unicorn", "color": "red", "size": "small"},
                     {"kind": "cube", "color": "red", "size": "small"}], [])
    with pytest.raises(SceneError):
        validate_scene(s, config.ontology)


# ---------------------------------------------------------------- logic


def test_evaluate_attributes_and_relations(config):
    s, sym = _scene(), config.ontology.symmetric_relations
    red_cube = make_lf("assert", ("x0",), [Atom("kind:cube", ("x0",)), Atom("color:red", ("x0",))])
    blue_cube = make_lf("assert", ("x0",), [Atom("kind:cube", ("x0",)), Atom("color:blue", ("x0",))])
    assert evaluate(red_cube, s, sym) and not evaluate(blue_cube, s, sym)
    chase = [Atom("kind:cube", ("x0",)), Atom("kind:ball", ("x1",)), Atom("rel:chases", ("x0", "x1"))]
    assert evaluate(make_lf("ask", ("x0", "x1"), chase), s, sym)
    reverse = [Atom("kind:ball", ("x0",)), Atom("kind:cube", ("x1",)), Atom("rel:chases", ("x0", "x1"))]
    assert not evaluate(make_lf("ask", ("x0", "x1"), reverse), s, sym)  # chases is asymmetric
    nxt = [Atom("kind:ball", ("x0",)), Atom("kind:cone", ("x1",)), Atom("rel:next_to", ("x0", "x1"))]
    assert evaluate(make_lf("ask", ("x0", "x1"), nxt), s, sym)  # next_to is symmetric


def test_lf_canonical_and_roundtrip():
    a = make_lf("ask", ("x0",), [Atom("kind:cube", ("x0",)), Atom("color:red", ("x0",))])
    b = make_lf("ask", ("x0",), [Atom("color:red", ("x0",)), Atom("kind:cube", ("x0",))])
    assert a == b and a.canonical() == b.canonical()
    assert LogicalForm.from_json(json.loads(json.dumps(a.to_json()))) == a


def test_malformed_lf_fails():
    with pytest.raises(LogicError):
        Atom("rel:chases", ("x0",))
    with pytest.raises(LogicError):
        Atom("smell:bad", ("x0",))
    with pytest.raises(LogicError):
        LogicalForm("assert", ("x0", "x1"), (Atom("kind:cube", ("x0",)),))


# ---------------------------------------------------------------- grammar


def test_realization_follows_hidden_order(config):
    g = Grammar(config.grammar)
    lex = config.grammar.lexicon
    subj = NPSpec("cube", (("color", "red"), ("size", "large")))
    obj = NPSpec("ball", ())
    m = Meaning("trans_ask", (("subj", subj), ("obj", obj)), "chases")
    assert g.realize(m) == (lex["kind:cube"], lex["size:large"], lex["color:red"], lex["kind:ball"],
                            lex["rel:chases"], config.grammar.question_particle)
    lf = g.to_lf(m)
    assert lf.act == "ask" and Atom("rel:chases", ("x0", "x1")) in lf.atoms


def test_meaning_construction_mismatch_fails(config):
    g = Grammar(config.grammar)
    with pytest.raises(ConfigError):
        g.realize(Meaning("np_assert", (("subj", NPSpec("cube", ())),), None))
    with pytest.raises(ConfigError):
        g.realize(Meaning("np_assert", (("np", NPSpec("cube", ())),), "chases"))
