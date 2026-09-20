import json
from collections import Counter
from dataclasses import replace

import pytest

from ragqa_core import db
from ragqa_core.builder import INDUCE_BATCH, Builder
from ragqa_core.db import connect
from ragqa_core.fakes import fake_ai, fake_models
from ragqa_core.indexer import Indexer, ModelMismatch
from ragqa_core.store import Store
from ragqa_core.types import Item, Question
from ragqa_core.vocabulary import terms_version


def worked(n: int, explained: bool = True) -> list[Item]:
    return [
        Item(
            f"i{k}",
            f"problem {k} asks for a speed",
            "10 m/s",
            f"solution {k} uses energy" if explained else None,
        )
        for k in range(n)
    ]


@pytest.fixture
def env(tmp_path):
    """A Builder on a worked-examples collection; set env.reply to answer prompts."""
    conn = connect(tmp_path / "r.db")
    store, models = Store(None, None), fake_models()
    prompts: list[str] = []
    state = {"reply": lambda p: '{"terms": []}'}
    ai = fake_ai(conn, lambda p: prompts.append(p) or state["reply"](p))
    ix = Indexer(conn, store, models, ai)
    spec = ix.create("w", "science", False, recipe="worked_examples")
    return Builder(conn, store, models, ai, ix), spec, prompts, state


def purposes(conn) -> list[str]:
    return [r["purpose"] for r in conn.execute("select purpose from ai_ledger")]


def test_induce_uses_explanations_and_merges(env):
    builder, spec, prompts, state = env

    def reply(prompt: str) -> str:
        if "solution 0 uses" in prompt:
            return json.dumps({"terms": ["Law of conservation of energy", "ideal gas law"]})
        return json.dumps(
            {"terms": ["conservation of energy law", "law of conservation of energy"]}
        )

    state["reply"] = reply
    seen: list[float] = []
    terms = builder.induce(spec, worked(INDUCE_BATCH + 5), seen.append)
    assert terms == ["law of conservation of energy", "ideal gas law"]
    assert len(prompts) == 2 and seen[-1] == 1.0
    assert all(f"solution {k} uses energy" in "".join(prompts) for k in range(INDUCE_BATCH + 5))
    assert not any("asks for a speed" in p for p in prompts)  # the solutions, not the problems
    assert "skills or principles" in prompts[0]
    assert set(purposes(builder.conn)) == {"induce"}


def test_induce_falls_back_to_text_without_explanations(env):
    builder, spec, prompts, state = env
    state["reply"] = lambda p: '{"terms": ["kinematics"]}'
    assert builder.induce(spec, worked(3, explained=False), lambda p: None) == ["kinematics"]
    assert "problem 2 asks for a speed" in prompts[0]


def test_induce_skips_unparseable_batches(env):
    builder, spec, prompts, state = env

    def reply(prompt: str) -> str:
        if "solution 0 uses" in prompt:
            return "nope"
        if f"solution {INDUCE_BATCH} uses" in prompt:
            return '{"other": ["ignored"]}'
        return '{"terms": ["dimensional analysis"]}'

    state["reply"] = reply
    terms = builder.induce(spec, worked(2 * INDUCE_BATCH + 1), lambda p: None)
    assert len(prompts) == 3 and terms == ["dimensional analysis"]


def test_estimate_induce_prices_every_batch_without_spending(env):
    builder, spec, prompts, _ = env
    one = builder.estimate_induce_usd(spec, worked(INDUCE_BATCH))
    two = builder.estimate_induce_usd(spec, worked(INDUCE_BATCH + 1))
    assert 0 < one < two and prompts == [] and purposes(builder.conn) == []


def test_save_vocabulary_cleans_and_does_not_switch(env):
    builder, spec, *_ = env
    version = builder.save_vocabulary(spec, ["  Ohm's  Law ", "ohm's law", "ideal gas law"])
    assert version == terms_version(["ohm's law", "ideal gas law"])
    assert db.get_vocabulary(builder.conn, "w", version) == ["ohm's law", "ideal gas law"]
    assert db.get_collection(builder.conn, "w").vocabulary_version is None
    with pytest.raises(ValueError, match="empty"):
        builder.save_vocabulary(spec, ["  ", "x" * 61])


TERMS = ["enzyme kinetics", "stock market", "volcano lava", "optics light"]


def with_build(builder: Builder, spec, **build):
    """The spec with some build settings changed, saved so a job reading it sees them."""
    spec = replace(spec, build=replace(spec.build, **build))
    db.save_collection(builder.conn, spec)
    return spec


def test_similarity_tags_top_k(env):
    builder, spec, prompts, _ = env
    text = ["enzyme kinetics in a volcano"]
    two = with_build(builder, spec, tagging="similarity", tag_k=2)
    assert builder.tags_for(two, TERMS, text, lambda p: None) == [
        ["enzyme kinetics", "volcano lava"]
    ]
    one = with_build(builder, spec, tagging="similarity", tag_k=1)
    assert builder.tags_for(one, TERMS, text, lambda p: None) == [["enzyme kinetics"]]
    assert prompts == []  # similarity never asks the AI
    assert builder.term_vectors(TERMS) is builder.term_vectors(TERMS)  # cached by version
    # The same version in another order: rows still follow the terms given.
    assert (builder.term_vectors(TERMS[::-1]) == builder.term_vectors(TERMS)[::-1]).all()


def test_cutoff_tags_at_least_one(env):
    builder, spec, *_ = env
    text = ["enzyme kinetics in a volcano"]  # cosine 0.63 to the first term, 0.32 to lava

    def tags(cutoff: float, terms=TERMS, texts=text) -> list[list[str]]:
        cut = with_build(builder, spec, tagging="similarity_cutoff", tag_cutoff=cutoff)
        return builder.tags_for(cut, terms, texts, lambda p: None)

    assert tags(0.3) == [["enzyme kinetics", "volcano lava"]]
    assert tags(0.5) == [["enzyme kinetics"]]
    assert tags(0.99) == [["enzyme kinetics"]]  # none reach it: still the single best
    many = [f"common w{n}" for n in range(12)]
    picked = tags(0.5, many, ["common"])[0]
    assert len(picked) == 10 and set(picked) <= set(many)  # at most 10


def test_ai_tagging_keeps_only_vocabulary_terms(env):
    builder, spec, prompts, state = env
    state["reply"] = lambda p: json.dumps({"terms": ["Ohm's Law", "made-up"]})
    terms = ["ohm's law", "ideal gas law"]
    got = builder.tags_for(spec, terms, ["a current through a resistor"], lambda p: None)
    assert got == [["ohm's law"]]
    assert "1. ohm's law" in prompts[0] and "2. ideal gas law" in prompts[0]
    assert purposes(builder.conn) == ["tag"]
    state["reply"] = lambda p: json.dumps({"terms": ["optics light", "stock market", TERMS[0]]})
    two = with_build(builder, spec, tag_k=2)
    assert builder.tags_for(two, TERMS, ["x"], lambda p: None) == [["optics light", "stock market"]]


def test_ai_tagging_falls_back_to_similarity(env):
    builder, spec, _, state = env
    state["reply"] = lambda p: "garbage" if "volcano" in p else json.dumps({"terms": ["made-up"]})
    one = with_build(builder, spec, tag_k=1)
    texts = ["enzyme kinetics in a volcano", "a stock market crash"]
    assert builder.tags_for(one, TERMS, texts, lambda p: None) == [
        ["enzyme kinetics"],
        ["stock market"],
    ]


def test_estimate_tag_prices_only_ai_tagging(env):
    builder, spec, prompts, _ = env
    assert builder.estimate_tag_usd(spec, TERMS, worked(3)) > 0
    similar = with_build(builder, spec, tagging="similarity")
    assert builder.estimate_tag_usd(similar, TERMS, worked(3)) == 0
    assert prompts == [] and purposes(builder.conn) == []


def switch_to(builder: Builder, spec, terms: list[str]) -> str:
    version = builder.save_vocabulary(spec, terms)
    db.set_vocabulary_version(builder.conn, spec.name, version)
    return version


def test_tagging_describes_text_when_no_explanation(env):
    builder, spec, prompts, state = env
    items = [
        Item("a", "problem about a volcano", "1", "uses volcano lava flow"),
        Item("b", "problem on enzyme kinetics", "2", None),
    ]
    builder.indexer.add(spec, items, "x")
    switch_to(builder, spec, TERMS)

    def reply(prompt: str) -> str:
        term = "volcano lava" if "lava flow" in prompt else "enzyme kinetics"
        return json.dumps({"terms": [term]})

    state["reply"] = reply
    builder.tag(spec, items, lambda p: None)
    assert any("uses volcano lava flow" in p for p in prompts)
    assert not any("problem about a volcano" in p for p in prompts)  # the solution instead
    assert any("problem on enzyme kinetics" in p for p in prompts)  # no solution: the text
    assert {h.id: h.tags for h in builder.store.get("w", ["a", "b"])} == {
        "a": ["volcano lava"],
        "b": ["enzyme kinetics"],
    }
    assert db.get_collection(builder.conn, "w").profile == {"enzyme kinetics": 1, "volcano lava": 1}


def stored(builder: Builder, spec) -> None:
    texts = ["enzyme kinetics rate", "stock market price", "volcano lava flow", "enzyme assay"]
    builder.indexer.add(spec, [Item(f"i{n}", t) for n, t in enumerate(texts)], "x")


def test_retag_switches_version_and_profile(env):
    builder, spec, prompts, _ = env
    spec = with_build(builder, spec, tagging="similarity", tag_k=1)
    stored(builder, spec)
    switch_to(builder, spec, ["kinematics"])
    new = builder.save_vocabulary(spec, TERMS)
    seen: list[float] = []
    builder.retag("w", new, seen.append)
    after = db.get_collection(builder.conn, "w")
    assert after.vocabulary_version == new
    hits = [h for batch in builder.store.scroll("w") for h in batch]
    assert {h.id: h.tags for h in hits} == {
        "i0": ["enzyme kinetics"],
        "i1": ["stock market"],
        "i2": ["volcano lava"],
        "i3": ["enzyme kinetics"],
    }
    assert after.profile == dict(Counter(t for h in hits for t in h.tags))
    assert list(after.profile) == ["enzyme kinetics", "stock market", "volcano lava"]
    found = builder.store.search("w", "kw_tags", builder.models.embed(["stock market"])[0], 1)
    assert found[0].id == "i1"
    assert seen and seen[-1] == 1.0 and seen == sorted(seen)
    assert prompts == []


def test_retag_failure_keeps_old_version(env):
    builder, spec, *_ = env
    spec = with_build(builder, spec, tagging="similarity")
    stored(builder, spec)
    old = switch_to(builder, spec, ["kinematics"])
    new = builder.save_vocabulary(spec, TERMS)

    def boom(*args, **kwargs):
        raise RuntimeError("store down")

    builder.store.set_tags = boom
    with pytest.raises(RuntimeError):
        builder.retag("w", new, lambda p: None)
    assert db.get_collection(builder.conn, "w").vocabulary_version == old
    del builder.store.set_tags
    # Built with another model: refused before any term is embedded.
    db.save_collection(
        builder.conn, replace(db.get_collection(builder.conn, "w"), dense_model="x/y")
    )
    embedded: list[list[str]] = []
    builder.models.embed = lambda texts: embedded.append(texts)
    with pytest.raises(ModelMismatch):
        builder.retag("w", new, lambda p: None)
    assert embedded == [] and db.get_collection(builder.conn, "w").vocabulary_version == old
    with pytest.raises(ValueError, match="nope"):
        builder.retag("w", "nope", lambda p: None)


def test_refresh_profile_saves_only_when_the_build_keeps_one(env):
    builder, spec, *_ = env
    spec = with_build(builder, spec, tagging="similarity", tag_k=1)
    stored(builder, spec)
    builder.retag("w", builder.save_vocabulary(spec, TERMS), lambda p: None)
    counts = {"enzyme kinetics": 2, "stock market": 1, "volcano lava": 1}
    db.set_profile(builder.conn, "w", {})
    no_profile = replace(spec, build=replace(spec.build, profile=False))
    assert builder.refresh_profile(no_profile) == counts
    assert db.get_collection(builder.conn, "w").profile == {}
    assert builder.refresh_profile(spec) == counts
    assert db.get_collection(builder.conn, "w").profile == counts


SECRET = {"skill": "SECRET-SKILL", "topic": "SECRET-TOPIC", "lecture": "SECRET-LECTURE"}
# Every build step that writes a prompt from items; later steps join this table.
BUILD_STEPS = {
    "induce": lambda builder, spec, items: builder.induce(spec, items, lambda p: None),
    "propose": lambda builder, spec, items: builder.indexer.propose_lists(spec, items),
    "tag": lambda builder, spec, items: (
        builder.indexer.add(spec, items, "x"),
        switch_to(builder, spec, ["energy"]),
        builder.tag(spec, items, lambda p: None),
    ),
    "retag": lambda builder, spec, items: (
        builder.indexer.add(spec, items, "x"),
        builder.retag(spec.name, builder.save_vocabulary(spec, ["energy"]), lambda p: None),
    ),
}


@pytest.mark.parametrize("step", list(BUILD_STEPS))
def test_build_prompts_never_see_meta(env, step):
    builder, spec, prompts, state = env
    state["reply"] = lambda p: json.dumps(
        {"terms": ["energy"], "topic": ["motion"], "method": ["energy"], "common": ["speed"]}
    )
    items = [Item(i.id, i.text, i.answer, i.explanation, dict(SECRET)) for i in worked(3)]
    db.save_questions(
        builder.conn, "w", [Question("q1", "how fast?", None, ["10"], [], dict(SECRET))]
    )
    BUILD_STEPS[step](builder, spec, items)
    assert prompts, "the step sent no prompt"
    assert not any("SECRET" in p for p in prompts)
