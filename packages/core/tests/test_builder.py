import json
from collections import Counter
from dataclasses import replace

import pytest

from ragqa_core import db
from ragqa_core.builder import GUESSED_TERMS, INDUCE_BATCH, Builder
from ragqa_core.db import connect
from ragqa_core.fakes import fake_ai, fake_models
from ragqa_core.indexer import Indexer, ModelMismatch
from ragqa_core.store import Store
from ragqa_core.types import Item, Question
from ragqa_core.vocabulary import MAX_TERM_CHARS, MAX_TERMS, terms_version


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
    dense = "fake/dense"
    assert builder.term_vectors(dense, TERMS) is builder.term_vectors(dense, TERMS)  # cached
    # The same version in another order: rows still follow the terms given.
    reverse = builder.term_vectors(dense, TERMS[::-1])
    assert (reverse == builder.term_vectors(dense, TERMS)[::-1]).all()


def test_term_vector_caches_are_per_model(tmp_path):
    conn = connect(tmp_path / "r.db")
    store, models = Store(None, None), fake_models(dims={"fake/small": 64})
    ai = fake_ai(conn, lambda p: "{}")
    builder = Builder(conn, store, models, ai, Indexer(conn, store, models, ai))
    assert len(builder.similar("fake/dense", TERMS, ["x"], 1)[0]) == 1
    assert len(builder.similar("fake/small", TERMS, ["x"], 1)[0]) == 1  # not 256 columns
    assert builder.term_vectors("fake/small", TERMS).shape[1] == 64


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
    # Built with a model this service can't load: refused before any term is embedded.
    db.save_collection(
        builder.conn, replace(db.get_collection(builder.conn, "w"), dense_model="x/y")
    )
    real = builder.models.dense_dim

    def dense_dim(model=None):
        if model == "x/y":
            raise ValueError("x/y is not in fastembed's supported models")
        return real(model)

    builder.models.dense_dim = dense_dim  # type: ignore[method-assign]
    embedded: list[list[str]] = []
    builder.models.embed = lambda texts, model=None: embedded.append(texts)
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
    "abstract": lambda builder, spec, items: (
        builder.indexer.add(spec, items, "x"),
        builder.abstract(spec, items, lambda p: None),
    ),
    "build": lambda builder, spec, items: (
        builder.indexer.add(spec, items, "x"),
        builder.build(spec, items, lambda p: None),
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


ABSTRACTED = json.dumps(
    {"principle": "Energy is conserved.", "method": "Equate the energies.", "answer": "10 m/s"}
)


def test_abstract_stores_principle_method_answer(env):
    builder, spec, prompts, state = env
    items = worked(3)
    builder.indexer.add(spec, items, "x")

    def reply(prompt: str) -> str:
        if "solution 0 uses" in prompt:
            return ABSTRACTED
        if "solution 1 uses" in prompt:  # steps as a list are numbered
            return json.dumps({"principle": "P.", "method": ["Draw it.", "Solve."], "answer": "3"})
        return '{"principle": "P.", "answer": "no method"}'  # unusable: stores nothing

    state["reply"] = reply
    seen: list[float] = []
    assert builder.abstract(spec, items, seen.append) == 2
    got = {h.id: h.abstracted for h in builder.store.get("w", ["i0", "i1", "i2"])}
    assert got == {
        "i0": "Principle: Energy is conserved.\nMethod: Equate the energies.\nAnswer: 10 m/s",
        "i1": "Principle: P.\nMethod: 1. Draw it. 2. Solve.\nAnswer: 3",
        "i2": None,
    }
    assert "problem 0 asks for a speed" in prompts[0] and "solution 0 uses energy" in prompts[0]
    assert "10 m/s" in prompts[0] and '"principle"' in prompts[0]
    assert purposes(builder.conn) == ["abstract"] * 3
    assert seen and seen[-1] == 1.0 and seen == sorted(seen)


def test_abstract_skips_items_without_explanation(env):
    builder, spec, prompts, state = env
    state["reply"] = lambda p: ABSTRACTED
    items = [*worked(2, explained=False), Item("blank", "a problem", "1", "   ")]
    builder.indexer.add(spec, items, "x")
    seen: list[float] = []
    assert builder.abstract(spec, items, seen.append) == 0
    assert prompts == [] and seen == [1.0]
    assert all(h.abstracted is None for h in builder.store.get("w", ["i0", "i1", "blank"]))
    # A reference corpus has no worked examples to abstract, explained or not.
    ref = builder.indexer.create("r", "multihop", False, recipe="reference")
    builder.indexer.add(ref, worked(2), "x")
    assert builder.abstract(ref, worked(2), lambda p: None) == 0 and prompts == []


def test_build_induces_switches_then_tags_and_abstracts(env):
    builder, spec, prompts, state = env

    def reply(prompt: str) -> str:
        if "skills or principles" in prompt:
            return '{"terms": ["energy conservation", "kinematics"]}'
        if "Pick 1 to" in prompt:
            return '{"terms": ["kinematics"]}'
        return ABSTRACTED

    state["reply"] = reply
    items = worked(3)
    builder.indexer.add(spec, items, "x")
    seen: list[float] = []
    result = builder.build(spec, items, seen.append)
    version = db.get_collection(builder.conn, "w").vocabulary_version
    assert version == terms_version(["energy conservation", "kinematics"])
    assert result == {"vocabulary_version": version, "tagged": 3, "abstracted": 3}
    assert purposes(builder.conn) == ["induce"] + ["tag"] * 3 + ["abstract"] * 3
    assert all(h.tags == ["kinematics"] for h in builder.store.get("w", ["i0", "i1", "i2"]))
    assert db.get_collection(builder.conn, "w").profile == {"kinematics": 3}
    assert seen and seen[-1] == 1.0 and seen == sorted(seen)
    # With a vocabulary in force, the next build only tags and abstracts.
    more = [Item("j0", "problem j asks for a speed", "1", "solution j uses energy")]
    builder.indexer.add(spec, more, "x")
    assert builder.build(spec, more, lambda p: None)["vocabulary_version"] == version
    assert purposes(builder.conn).count("induce") == 1
    assert db.get_collection(builder.conn, "w").profile == {"kinematics": 4}


def test_build_with_no_usable_terms_keeps_the_items_untagged(env):
    builder, spec, prompts, state = env
    state["reply"] = lambda p: ABSTRACTED if '"principle"' in p else "no terms here"
    items = worked(2)
    builder.indexer.add(spec, items, "x")
    result = builder.build(spec, items, lambda p: None)
    assert result == {"vocabulary_version": None, "tagged": 0, "abstracted": 2}
    assert "tag" not in purposes(builder.conn)


def test_paper_2024_build_does_nothing(env):
    builder, _, prompts, _ = env
    paper = builder.indexer.create("p", "science", False)
    builder.indexer.add(paper, worked(2), "x")
    assert builder.build(paper, worked(2), lambda p: None) == {}
    assert builder.estimate_build_usd(paper, worked(2)) == 0
    assert prompts == [] and purposes(builder.conn) == []


def test_estimate_build_prices_induction_only_without_a_vocabulary(env):
    builder, spec, prompts, _ = env
    items = worked(3)
    first = builder.estimate_build_usd(spec, items)
    assert first == pytest.approx(
        builder.estimate_induce_usd(spec, items)
        + builder.estimate_tag_usd(spec, GUESSED_TERMS, items)
        + builder.estimate_abstract_usd(spec, items)
    )
    switch_to(builder, spec, ["kinematics"])
    later = builder.estimate_build_usd(spec, items)
    assert later == pytest.approx(
        builder.estimate_tag_usd(spec, ["kinematics"], items)
        + builder.estimate_abstract_usd(spec, items)
    )
    assert 0 < later < first
    assert builder.build_calls(spec, items) == 3 + 3  # tags and abstracts; no induction
    assert builder.estimate_abstract_usd(spec, worked(3, explained=False)) == 0
    assert prompts == [] and purposes(builder.conn) == []
    builder.ai.set_role("default", None)  # no model for keywords: nothing can price it
    assert builder.estimate_build_usd(spec, items) is None


def test_guessed_terms_bound_any_induced_vocabulary():
    assert len(set(GUESSED_TERMS)) == MAX_TERMS
    assert all(len(t) == MAX_TERM_CHARS for t in GUESSED_TERMS)


# -- authored outputs ---------------------------------------------------------------------


def authored(n: int) -> dict[str, dict]:
    return {
        f"i{k}": {
            "tags": ["Unitary Method"] if k % 2 else ["relative speed", "unitary method"],
            "principle": "Scale one unit to the quantity asked.",
            "method": ["Find the rate for one unit.", "Multiply by the quantity asked."],
            "answer": "10 m/s",
        }
        for k in range(n)
    }


def test_authored_outputs_build_the_collection_with_no_ai_call(env):
    builder, spec, prompts, _ = env
    builder.indexer.add(spec, worked(3), "x")
    got = builder.apply_outputs(spec, ["unitary method", "relative speed"], authored(3))
    assert got == {
        "vocabulary_version": terms_version(["unitary method", "relative speed"]),
        "tagged": 3,
        "abstracted": 3,
    }
    assert prompts == [] and purposes(builder.conn) == []
    fresh = db.get_collection(builder.conn, "w")
    assert fresh is not None and fresh.vocabulary_version == got["vocabulary_version"]
    hits = {h.id: h for h in builder.store.get("w", ["i0", "i1"])}
    assert hits["i0"].tags == ["relative speed", "unitary method"] and hits["i1"].tags == [
        "unitary method"
    ]
    assert hits["i1"].abstracted == (
        "Principle: Scale one unit to the quantity asked.\n"
        "Method: 1. Find the rate for one unit. 2. Multiply by the quantity asked.\n"
        "Answer: 10 m/s"
    )
    assert fresh.profile == {"unitary method": 3, "relative speed": 2}
    # Nothing is left to build, so a later load of the same items needs no model.
    assert not any(builder.unbuilt(spec, []).values()) and builder.build_calls(spec, []) == 0


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda o: o["i0"].update(tags=["pigeonhole principle"]), "outside the vocabulary"),
        (lambda o: o["i0"].update(tags=[]), "without tags"),
        (lambda o: o.pop("i2"), "no output for"),
        (lambda o: o.update(nope=dict(o["i0"])), "not in collection"),
        (lambda o: o["i1"].update(method=""), "no usable abstracted form"),
    ],
)
def test_authored_outputs_are_refused_whole_before_anything_is_written(env, change, message):
    builder, spec, _, _ = env
    builder.indexer.add(spec, worked(3), "x")
    outputs = authored(3)
    change(outputs)
    with pytest.raises(ValueError, match=message):
        builder.apply_outputs(spec, ["unitary method", "relative speed"], outputs)
    assert db.get_collection(builder.conn, "w").vocabulary_version is None
    assert all(not h.tags for h in builder.store.get("w", ["i0", "i1", "i2"]))


def test_apply_outputs_command_reads_the_vocabulary_and_outputs_files(env, tmp_path, monkeypatch):
    from ragqa_core import apply_outputs

    builder, spec, _, _ = env
    builder.indexer.add(spec, worked(3), "x")
    vocabulary = tmp_path / "vocabulary.json"
    vocabulary.write_text(
        json.dumps(
            {
                "terms": [
                    {"term": "unitary method", "definition": "scale one unit"},
                    {"term": "relative speed", "definition": "speeds add or subtract"},
                ]
            }
        )
    )
    outputs = tmp_path / "outputs.jsonl"
    outputs.write_text("".join(json.dumps({"id": k, **v}) + "\n" for k, v in authored(3).items()))
    monkeypatch.setattr(apply_outputs, "make_builder", lambda: builder)
    assert apply_outputs.main(["w", str(vocabulary), str(outputs)]) == 0
    assert db.get_collection(builder.conn, "w").vocabulary_version is not None


def test_plan_tag_build_prices_the_tagged_build_of_the_stored_passages(tmp_path):
    conn = connect(tmp_path / "r.db")
    store, models = Store(None, None), fake_models()
    ai = fake_ai(conn, lambda p: '{"terms": []}')
    ix = Indexer(conn, store, models, ai)
    spec = ix.create("d", "documents", False, recipe="documents")
    ix.add(spec, [Item(f"p{k}", f"passage {k} about tides") for k in range(3)], "x")
    builder = Builder(conn, store, models, ai, ix)
    assert builder.plan_build(spec, []) == (0.0, 0)  # documents builds nothing
    usd, calls = builder.plan_tag_build(spec)
    assert usd is not None and usd > 0
    assert calls == 1  # one induction batch; tagging by similarity sends nothing
    assert purposes(conn) == [] and db.get_collection(conn, "d").recipe == "documents"
    ai.set_role("default", None)
    assert builder.plan_tag_build(spec)[0] is None  # no model for keywords: unpriced
