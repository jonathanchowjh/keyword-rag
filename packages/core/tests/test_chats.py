import re
from dataclasses import replace

import pytest

from aiconnect.ai import UnknownModel, UnknownPrice
from ragqa_core import db
from ragqa_core.builder import Builder
from ragqa_core.chats import EVIDENCE_SIZE, RRF_K, AlreadyAnswered, Chats, UnknownChat
from ragqa_core.db import connect, replace_questions
from ragqa_core.evaluator import Evaluator
from ragqa_core.fakes import fake_ai, fake_models
from ragqa_core.indexer import Indexer
from ragqa_core.prompts import CHAT_TURNS
from ragqa_core.retriever import MethodUnavailable, Retriever, UnknownCollection
from ragqa_core.router import BAR_MIN, Router
from ragqa_core.store import Store
from ragqa_core.types import Item, Question

WORDS = {
    "a": "enzyme kinetics catalyst assay substrate",
    "b": "volcano magma tectonics seismograph lava",
}
ENZYME = "enzyme kinetics catalyst assay"  # clears a's bar, not b's
VOLCANO = "volcano magma tectonics seismograph"
BOTH = f"{ENZYME} {VOLCANO}"
NOTHING = "zzz qqq"  # clears no bar
REWRITTEN = "what do enzyme kinetics catalyst assays measure"
PASSAGE = re.compile(r"^\[(\d+)\] (.*)$", re.M)


def items(store: str, n: int = 10) -> list[Item]:
    return [Item(f"{store}-{i}", f"{WORDS[store]} note{i}", "A", "e") for i in range(n)]


def questions(store: str, n: int = BAR_MIN) -> list[Question]:
    words = " ".join(WORDS[store].split()[:2])
    return [Question(f"q{i:02d}", f"{words} q{i}", None, ["A"], [], {}) for i in range(n)]


def reply(prompts: list[str]):
    def answer(prompt: str) -> str:
        prompts.append(prompt)
        if prompt.startswith("Earlier messages:"):
            return f"  {REWRITTEN}  \n"
        if "Passages found so far" in prompt:
            return '{"done": true}'
        question = prompt.rsplit("Question: ", 1)[-1]
        return f"On {question}: {len(PASSAGE.findall(prompt))} passages, see [1] and [3]."

    return answer


def build(tmp_path):
    """Stores a (enzymes) and b (volcanoes), each with BAR_MIN questions of its own, and a
    Chats on them. Returns the Chats and every prompt the fake AI was sent."""
    conn = connect(tmp_path / "r.db")
    store, models = Store(None, None), fake_models()
    prompts: list[str] = []
    ai = fake_ai(conn, reply(prompts))
    ix = Indexer(conn, store, models, ai)
    for name in WORDS:
        ix.add(ix.create(name, "multihop", False), items(name), "x")
        replace_questions(conn, name, questions(name), None)
    retriever = Retriever(conn, store, models, ai, ix, Builder(conn, store, models, ai, ix))
    router = Router(conn, retriever, Evaluator(conn, retriever, models, ai))
    return Chats(conn, retriever, router, ai), prompts


def copy_store(chats: Chats, name: str, source: str) -> None:
    ix = chats.retriever.indexer
    ix.add(ix.create(name, "multihop", False), items(source), "x")


def drop_store(chats: Chats, name: str) -> None:
    chats.retriever.store.drop(name)
    chats.retriever.forget(name)
    db.delete_collection_rows(chats.conn, name)


def attempt(message: dict) -> dict:
    return message["attempts"][-1]


def decision(attempt: dict, store: str) -> dict:
    return next(d for d in attempt["decisions"] if d["store"] == store)


def chat_prompts(prompts: list[str]) -> list[str]:
    return [p for p in prompts if not p.startswith("Earlier messages:") and "found so far" not in p]


def ledger(chats: Chats, purpose: str) -> float:
    rows = chats.ai.sql("select sum(usd) as usd from ai_ledger where purpose = ?", (purpose,))
    return rows[0]["usd"] or 0.0


def unpriced(chats: Chats) -> str:
    chats.ai.add_model("anthropic", "unpriced-model", price_in=None, price_out=None)
    return "anthropic/unpriced-model"


def test_create_list_get_update_delete_a_chat(tmp_path):
    chats, _ = build(tmp_path)
    with pytest.raises(UnknownCollection):
        chats.create(["a", "nope"])
    with pytest.raises(ValueError):
        chats.create([])
    with pytest.raises(UnknownChat, match="^no chat nope$"):
        chats.get("nope")

    first = chats.create(["a", "b"])
    assert first["title"] == "New chat"
    assert first["settings"] == {"evidence_size": EVIDENCE_SIZE, "rewrite": False, "routing": "all"}
    second = chats.create(["b"], title="Volcanoes")  # made in the same second
    assert [c["id"] for c in chats.list()] == [second["id"], first["id"]]

    chats.ask(first["id"], "x" * 70 + " enzyme")
    listed = {c["id"]: c for c in chats.list()}
    assert listed[first["id"]]["title"] == "x" * 60
    assert listed[first["id"]]["messages"] == 1
    assert listed[second["id"]]["title"] == "Volcanoes"
    assert listed[second["id"]]["messages"] == 0
    chats.ask(first["id"], "a second message")
    assert chats.get(first["id"])["title"] == "x" * 60

    updated = chats.update(first["id"], title="Enzymes", evidence_size=3, rewrite=True)
    assert updated["title"] == "Enzymes"
    assert updated["settings"] == {"evidence_size": 3, "rewrite": True, "routing": "all"}
    with pytest.raises(ValueError):
        chats.update(first["id"], evidence_size=21)
    with pytest.raises(UnknownCollection):
        chats.update(first["id"], stores=["a", "nope"])

    drop_store(chats, "b")
    got = chats.get(first["id"])
    assert got["deleted_stores"] == ["b"]
    assert [m["n"] for m in got["messages"]] == [1, 2]
    assert got["messages"][0]["attempts"][0]["attempt"] == 1
    # A list that still names the deleted store can be saved; only added stores must exist.
    assert chats.update(first["id"], stores=["b", "a"])["stores"] == ["b", "a"]

    chats.rerun(first["id"], 1, {"a": {"k": 2}})
    chats.delete(first["id"])
    assert first["id"] not in chats.overrides
    with pytest.raises(UnknownChat):
        chats.get(first["id"])
    assert [c["id"] for c in chats.list()] == [second["id"]]
    assert chats.conn.execute("select count(*) as n from chat_attempts")[0]["n"] == 0


def test_follow_up_search_text_joins_the_previous_two_user_messages(tmp_path):
    chats, _ = build(tmp_path)
    chat = chats.create(["a"])
    for text in ["m1 enzyme", "m2 enzyme", "m3 enzyme"]:
        assert attempt(chats.ask(chat["id"], text))["answer"]
    fourth = attempt(chats.ask(chat["id"], "m4 enzyme"))
    assert fourth["search_text"] == "m2 enzyme\nm3 enzyme\nm4 enzyme"
    assert "On " not in fourth["search_text"]  # an assistant reply is never searched
    assert chats.search_text(chat["id"], 1, "m1 enzyme") == ("m1 enzyme", 0.0, None)


def test_rewrite_on_uses_the_ai_rewrite_and_falls_back_on_error(tmp_path):
    chats, prompts = build(tmp_path)
    chat = chats.update(chats.create(["a"])["id"], rewrite=True)
    first = attempt(chats.ask(chat["id"], ENZYME))
    assert first["search_text"] == ENZYME  # the first message is never rewritten
    assert not [p for p in prompts if p.startswith("Earlier messages:")]

    second = attempt(chats.ask(chat["id"], "and its rate?"))
    rewrites = [p for p in prompts if p.startswith("Earlier messages:")]
    assert len(rewrites) == 1 and ENZYME in rewrites[0]
    assert second["search_text"] == REWRITTEN
    assert second["usd"] > 0 and ledger(chats, "chat_rewrite") > 0

    chats.ai.set_role("rewrite", unpriced(chats))
    third = attempt(chats.ask(chat["id"], "and its cost?"))
    assert third["search_text"] == f"{ENZYME}\nand its rate?\nand its cost?"
    assert third["error"] and "no price" in third["error"]
    assert third["answer"]

    before = len(prompts)
    rerun = chats.rerun(chat["id"], 2, {"a": {"k": 2}})
    assert rerun["search_text"] == REWRITTEN
    assert prompts[before:] == []  # no rewrite, no answer: it spent nothing
    assert rerun["usd"] == 0.0


def test_ask_keeps_stores_that_clear_their_bar_and_merges_by_rrf(tmp_path):
    chats, _ = build(tmp_path)
    chat = chats.create(["a", "b"])
    one = attempt(chats.ask(chat["id"], ENZYME))
    a, b = decision(one, "a"), decision(one, "b")
    assert (a["kept"], a["reason"], a["bar_source"]) == (True, "clears its bar", "set")
    assert (b["kept"], b["reason"], b["bar_source"]) == (False, "below its bar", "set")
    assert a["best"] >= a["bar"] and b["best"] < b["bar"]
    assert (a["method"], a["k"]) == ("hybrid", EVIDENCE_SIZE)
    assert a["search"]["weights"] and a["error"] is None
    evidence = one["evidence"]
    assert evidence and len(evidence) <= EVIDENCE_SIZE
    assert {e["store"] for e in evidence} == {"a"}
    assert [e["rank"] for e in evidence] == list(range(1, len(evidence) + 1))
    assert [e["score"] for e in evidence] == pytest.approx(
        [1 / (RRF_K + e["rank"]) for e in evidence]
    )
    assert sorted(e["n"] for e in evidence) == list(range(1, len(evidence) + 1))
    assert one["answer"] and one["error"] is None

    # The message alone is scored too: joined to a long first message, it scores lower.
    other = chats.create(["a"])
    junk = attempt(chats.ask(other["id"], " ".join(f"junk{i}" for i in range(20))))
    assert decision(junk, "a")["reason"] == "below its bar"
    follow = attempt(chats.ask(other["id"], ENZYME))
    assert follow["search_text"].endswith(f"\n{ENZYME}")
    assert decision(follow, "a")["reason"] == "clears its bar"


def test_merge_keeps_passages_of_stores_that_share_ids(tmp_path):
    chats, _ = build(tmp_path)
    copy_store(chats, "a_copy", "a")  # the same ids, no eval set: no bar
    chat = chats.create(["a", "a_copy"])
    one = attempt(chats.ask(chat["id"], ENZYME))
    assert decision(one, "a_copy")["reason"] == "no bar"
    evidence = one["evidence"]
    assert len(evidence) == EVIDENCE_SIZE
    assert {e["store"] for e in evidence} == {"a", "a_copy"}
    assert len({(e["store"], e["id"]) for e in evidence}) == EVIDENCE_SIZE
    assert len({e["id"] for e in evidence}) < EVIDENCE_SIZE  # the same ids from both


def test_follow_up_keeps_last_messages_stores(tmp_path):
    chats, _ = build(tmp_path)
    chat = chats.create(["a", "b"])
    one = attempt(chats.ask(chat["id"], ENZYME))
    assert decision(one, "a")["reason"] == "clears its bar"
    db.set_chat_bar(chats.conn, "a", 0.99)  # no message reaches it now

    two = attempt(chats.ask(chat["id"], "and its rate?"))
    a = decision(two, "a")
    assert a["best"] < a["bar"] == 0.99
    assert (a["kept"], a["reason"]) == (True, "kept from the last message")
    assert a["bar_source"] == "manual"
    assert decision(two, "b")["reason"] == "below its bar"
    assert {e["store"] for e in two["evidence"]} == {"a"}

    # An unanswered re-run that searched a anyway is not the last *answered* attempt.
    rerun = chats.rerun(chat["id"], 2, {"a": {"keep": True}})
    assert decision(rerun, "a")["reason"] == "search anyway" and rerun["answer"] is None
    chats = Chats(chats.conn, chats.retriever, chats.router, chats.ai)  # drops the override

    three = attempt(chats.ask(chat["id"], "and its cost?"))
    a = decision(three, "a")
    assert (a["kept"], a["reason"]) == (False, "below its bar")
    assert three["evidence"] == []


def test_rerun_adds_an_attempt_without_answering_and_answer_answers_it(tmp_path):
    chats, prompts = build(tmp_path)
    chat = chats.create(["a", "b"])
    chats.ask(chat["id"], ENZYME)
    before = len(prompts)
    rerun = chats.rerun(chat["id"], 1, {"a": {"k": 2}, "b": {"keep": True, "method": "vector"}})
    assert rerun["attempt"] == 2 and rerun["answer"] is None and rerun["citations"] == []
    assert rerun["model"] is None and rerun["usd"] == 0.0 and prompts[before:] == []
    assert decision(rerun, "a")["k"] == 2
    b = decision(rerun, "b")
    assert (b["kept"], b["reason"], b["method"]) == (True, "search anyway", "vector")
    assert {e["store"] for e in rerun["evidence"]} == {"a", "b"}
    assert len(chats.get(chat["id"])["messages"][0]["attempts"]) == 2

    answered = chats.answer(chat["id"], 1, 2)
    assert answered["attempt"] == 2 and answered["answer"].startswith(f"On {ENZYME}")
    assert answered["evidence"] == rerun["evidence"]
    assert [c["n"] for c in answered["citations"]] == [1, 3]
    assert answered["model"] and answered["usd"] > 0
    assert chats.get(chat["id"])["messages"][0]["attempts"][1] == answered
    with pytest.raises(UnknownChat):
        chats.answer(chat["id"], 1, 3)
    with pytest.raises(UnknownChat):
        chats.rerun(chat["id"], 2, {})


def test_answer_refuses_an_answered_attempt(tmp_path):
    chats, prompts = build(tmp_path)
    chat = chats.create(["a", "b"])
    chats.ask(chat["id"], ENZYME)
    before, spent = len(prompts), ledger(chats, "chat")
    with pytest.raises(AlreadyAnswered):
        chats.answer(chat["id"], 1, 1)
    assert len(prompts) == before and ledger(chats, "chat") == spent


def test_rerun_rejects_a_bad_override_without_keeping_it(tmp_path):
    chats, _ = build(tmp_path)
    chat = chats.create(["a", "b"])
    chats.ask(chat["id"], ENZYME)
    chats.rerun(chat["id"], 1, {"a": {"k": 3}})
    kept = {"a": {"k": 3}}
    assert chats.overrides[chat["id"]] == kept
    bad = [
        ({"a": {"k": 4}, "b": {"method": "keyword_rag"}}, MethodUnavailable),
        ({"a": {"search": {"weights": {"bogus": 1.0}}}}, ValueError),
        ({"a": {"search": {"query_tags": "nope"}}}, ValueError),
        ({"a": {"k": 0}}, ValueError),
        ({"a": {"k": 21}}, ValueError),
        ({"a": {"method": "nope"}}, ValueError),
        ({"a": {"colour": "red"}}, ValueError),
        ({"c": {"k": 2}}, UnknownCollection),
    ]
    for overrides, error in bad:
        with pytest.raises(error):
            chats.rerun(chat["id"], 1, overrides)
        assert chats.overrides[chat["id"]] == kept
    assert len(chats.get(chat["id"])["messages"][0]["attempts"]) == 2


def test_set_default_writes_only_the_named_fields_and_drops_the_override(tmp_path):
    chats, _ = build(tmp_path)
    conn = chats.conn
    chat = chats.create(["a", "b"])
    chats.ask(chat["id"], ENZYME)
    db.set_chat_bar(conn, "a", 0.5)
    search = chats.retriever.spec("a").search

    assert chats.get(chat["id"])["overrides"] == {}
    chats.rerun(chat["id"], 1, {"a": {"method": "vector"}, "b": {"keep": True}})
    held = chats.get(chat["id"])["overrides"]
    assert held == {"a": {"method": "vector"}, "b": {"keep": True}}
    held["a"]["method"] = "keyword"  # a copy: the chat's own is untouched
    assert chats.overrides[chat["id"]]["a"] == {"method": "vector"}
    saved = chats.set_default(chat["id"], "a")
    assert saved["chat_method"] == "vector" and saved["chat_bar"] == 0.5
    assert saved["chat_k"] is None and chats.retriever.spec("a").search == search
    assert "a" not in chats.overrides[chat["id"]]
    assert chats.get(chat["id"])["overrides"] == {"b": {"keep": True}}

    chats.rerun(chat["id"], 1, {"a": {"bar": None, "k": 3, "keep": True}})
    chats.rerun(chat["id"], 1, {"a": {"search": {"weights": {"words": 0.0}}}})
    chats.rerun(chat["id"], 1, {"a": {"search": {"weights": {"keywords": 0.5}}}})
    saved = chats.set_default(chat["id"], "a")
    assert (saved["chat_method"], saved["chat_k"], saved["chat_bar"]) == ("vector", 3, None)
    assert saved["search"]["weights"] == {"meaning": 1.0, "words": 0.0, "keywords": 0.5}
    assert chats.retriever.spec("a").search.weights == saved["search"]["weights"]
    assert db.get_chat_bar(conn, "a") is None
    assert "a" not in chats.overrides[chat["id"]]
    after = chats.rerun(chat["id"], 1, {})
    assert decision(after, "a")["reason"] == "clears its bar"  # keep was not saved

    chats.rerun(chat["id"], 1, {"a": {"method": None}})
    assert chats.set_default(chat["id"], "a")["chat_method"] is None
    with pytest.raises(UnknownCollection):
        chats.set_default(chat["id"], "nope")


def test_no_store_kept_answers_without_passages_and_says_so(tmp_path):
    chats, prompts = build(tmp_path)
    chat = chats.create(["a", "b"])
    one = attempt(chats.ask(chat["id"], NOTHING))
    assert [d["reason"] for d in one["decisions"]] == ["below its bar", "below its bar"]
    assert one["evidence"] == [] and one["citations"] == []
    assert one["answer"] == f"On {NOTHING}: 0 passages, see [1] and [3]."
    assert one["error"] is None
    assert "Passages" not in chat_prompts(prompts)[-1]


def test_budget_stop_keeps_the_attempts_evidence_and_error(tmp_path):
    chats, _ = build(tmp_path)
    db.set_chat_method(chats.conn, "a", "iterative")  # a paid method: one hop call
    chat = chats.update(chats.create(["a", "b"])["id"], rewrite=True)
    chats.ask(chat["id"], ENZYME)
    real = chats.ai.complete

    def complete(prompt, **kw):
        if kw.get("purpose") == "chat":  # the cap is reached just before the answer
            chats.ai.set_cap(chats.ai.month_spend())
        return real(prompt, **kw)

    chats.ai.complete = complete  # type: ignore[method-assign]
    rewrite_before = ledger(chats, "chat_rewrite")
    two = attempt(chats.ask(chat["id"], "and its rate?"))
    assert two["error"] and "monthly cap" in two["error"]
    assert two["answer"] is None and two["citations"] == []
    assert two["evidence"] and {e["store"] for e in two["evidence"]} == {"a"}
    a = decision(two, "a")
    assert a["method"] == "iterative" and a["usd"] > 0
    rewrite = ledger(chats, "chat_rewrite") - rewrite_before
    assert rewrite > 0
    assert two["usd"] == pytest.approx(rewrite + sum(d["usd"] for d in two["decisions"]))


def test_unpriced_answer_model_records_an_attempt_error(tmp_path):
    chats, _ = build(tmp_path)
    chat = chats.create(["a"])
    with pytest.raises(UnknownPrice):
        chats.ask(chat["id"], ENZYME, model=unpriced(chats))
    with pytest.raises(UnknownModel):
        chats.ask(chat["id"], ENZYME, model="anthropic/no-such-model")
    chats.ai.set_role("answer", "anthropic/unpriced-model")
    with pytest.raises(UnknownPrice):
        chats.ask(chat["id"], ENZYME)
    assert chats.get(chat["id"])["messages"] == []
    assert chats.conn.execute("select count(*) as n from chat_messages")[0]["n"] == 0

    chats.ai.set_role("answer", None)  # the default model answers again
    assert attempt(chats.ask(chat["id"], ENZYME))["answer"]
    assert [m["n"] for m in chats.get(chat["id"])["messages"]] == [1]


def test_a_store_error_becomes_its_decisions_error(tmp_path):
    chats, _ = build(tmp_path)
    spec = chats.retriever.spec("b")
    db.save_collection(chats.conn, replace(spec, dense_model="other/model"))
    chat = chats.create(["a", "b"])
    one = attempt(chats.ask(chat["id"], BOTH))
    b = decision(one, "b")
    assert (b["kept"], b["reason"]) == (False, "error")
    assert b["error"] and "other/model" in b["error"]
    assert decision(one, "a")["kept"]
    assert {e["store"] for e in one["evidence"]} == {"a"}
    assert one["answer"] and one["error"] is None


def test_evidence_numbers_match_the_cited_numbers(tmp_path):
    chats, prompts = build(tmp_path)
    chat = chats.create(["a", "b"])
    one = attempt(chats.ask(chat["id"], BOTH))
    shown = dict(PASSAGE.findall(chat_prompts(prompts)[-1]))
    assert len(shown) == len(one["evidence"]) == EVIDENCE_SIZE
    for e in one["evidence"]:
        assert shown[str(e["n"])] == e["text"]
    by_n = {e["n"]: e for e in one["evidence"]}
    assert [c["n"] for c in one["citations"]] == [1, 3]
    for c in one["citations"]:
        e = by_n[c["n"]]
        assert (c["id"], c["store"], c["text"], c["source"]) == (
            e["id"],
            e["store"],
            e["text"],
            e["source"],
        )
    # Evidence is kept in merged order; n follows the prompt's edge order (1, 3, 5, …, 4, 2).
    assert [e["n"] for e in one["evidence"]][:3] == [1, EVIDENCE_SIZE, 2]


def test_history_is_the_last_chat_turns_answered_attempts(tmp_path):
    chats, prompts = build(tmp_path)
    chat = chats.create(["a"])
    for i in range(1, 6):
        chats.ask(chat["id"], f"m{i} {ENZYME}")
    # Message 5's latest answered attempt is a re-run with fewer passages; a later
    # unanswered one is skipped.
    chats.rerun(chat["id"], 5, {"a": {"k": 2}})
    chats.answer(chat["id"], 5, 2)
    chats.rerun(chat["id"], 5, {"a": {"k": 3}})
    chats.ask(chat["id"], f"m6 {ENZYME}")
    last = chat_prompts(prompts)[-1]
    turns = last.split("\n\n")[0].split("\n")
    assert len(turns) == CHAT_TURNS
    assert turns[0] == f"User: m3 {ENZYME}"
    assert turns[-2] == f"User: m5 {ENZYME}"
    assert turns[-1] == f"Assistant: On m5 {ENZYME}: 2 passages, see [1] and [3]."
    assert "m2" not in last.split("Passages:")[0]


def test_overrides_are_dropped_on_restart_and_defaults_kept(tmp_path):
    chats, _ = build(tmp_path)
    chat = chats.create(["a", "b"])
    chats.ask(chat["id"], ENZYME)
    chats.rerun(chat["id"], 1, {"a": {"method": "vector", "k": 3}, "b": {"keep": True}})
    chats.set_default(chat["id"], "a")
    assert chats.overrides[chat["id"]] == {"b": {"keep": True}}

    restarted = Chats(chats.conn, chats.retriever, chats.router, chats.ai)
    assert restarted.overrides == {}
    two = attempt(restarted.ask(chat["id"], ENZYME))
    assert (decision(two, "a")["method"], decision(two, "a")["k"]) == ("vector", 3)
    assert decision(two, "b")["reason"] == "below its bar"


def test_a_deleted_store_is_skipped_and_old_messages_keep_their_evidence(tmp_path):
    chats, _ = build(tmp_path)
    chat = chats.create(["a", "b"])
    one = attempt(chats.ask(chat["id"], BOTH))
    assert {e["store"] for e in one["evidence"]} == {"a", "b"}

    drop_store(chats, "a")
    got = chats.get(chat["id"])
    assert got["deleted_stores"] == ["a"]
    assert got["messages"][0]["attempts"][0]["evidence"] == one["evidence"]

    two = attempt(chats.ask(chat["id"], BOTH))
    a = decision(two, "a")
    assert (a["kept"], a["reason"], a["error"]) == (False, "deleted", None)
    assert {e["store"] for e in two["evidence"]} == {"b"}
    assert two["answer"] and two["error"] is None
    with pytest.raises(UnknownCollection):
        chats.rerun(chat["id"], 2, {"a": {"k": 2}})
