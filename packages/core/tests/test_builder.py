import json

import pytest

from ragqa_core import db
from ragqa_core.builder import INDUCE_BATCH, Builder
from ragqa_core.db import connect
from ragqa_core.fakes import fake_ai, fake_models
from ragqa_core.indexer import Indexer
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


SECRET = {"skill": "SECRET-SKILL", "topic": "SECRET-TOPIC", "lecture": "SECRET-LECTURE"}
# Every build step that writes a prompt from items; later steps join this table.
BUILD_STEPS = {
    "induce": lambda builder, spec, items: builder.induce(spec, items, lambda p: None),
    "propose": lambda builder, spec, items: builder.indexer.propose_lists(spec, items),
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
