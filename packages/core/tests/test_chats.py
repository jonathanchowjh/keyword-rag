import re
import threading
import time
from dataclasses import replace

import pytest

from aiconnect.ai import UnknownModel, UnknownPrice
from ragqa_core import db, search_defaults
from ragqa_core.builder import Builder
from ragqa_core.chats import AlreadyAnswered, Chats, UnknownChat, round_robin
from ragqa_core.db import connect
from ragqa_core.fakes import fake_ai, fake_models
from ragqa_core.indexer import Indexer
from ragqa_core.prompts import CHAT_TURNS, chat_prompt
from ragqa_core.retriever import MethodUnavailable, Retriever, UnknownCollection
from ragqa_core.search_defaults import RECOMMENDED
from ragqa_core.store import Store
from ragqa_core.types import Hit, Item, KeywordLists

WORDS = {
    "a": "enzyme kinetics catalyst assay substrate",
    "b": "volcano magma tectonics seismograph lava",
}
ENZYME = "enzyme kinetics catalyst assay"  # a's words, none of b's
VOLCANO = "volcano magma tectonics seismograph"
BOTH = f"{ENZYME} {VOLCANO}"
NOTHING = "zzz qqq"  # no store's words
SIZE = RECOMMENDED["evidence_size"]  # a new chat's evidence size: Settings saves none here
CANDIDATES = RECOMMENDED["candidates"]
LISTS = KeywordLists(["enzyme", "volcano"], ["assay", "seismograph"], ["substrate", "lava"])
REWRITTEN = "what do enzyme kinetics catalyst assays measure"
PASSAGE = re.compile(r"^\[(\d+)\] (.*)$", re.M)
TIDES = "tides moon orbit gravity ocean"  # a third store's words
TURBO = "jinaai/jina-reranker-v1-turbo-en"


def items(store: str, n: int = 10, words: str | None = None, start: int = 0) -> list[Item]:
    words = words or WORDS[store]
    return [Item(f"{store}-{i}", f"{words} note{i}", "A", "e") for i in range(start, n)]


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


def build(tmp_path, b_model: str | None = None):
    """Stores a (enzymes) and b (volcanoes, embedded with b_model; None is the service's),
    and a Chats on them. Returns the Chats and every prompt the fake AI was sent."""
    conn = connect(tmp_path / "r.db")
    search_defaults.save(conn, {"rewrite": False})  # follow-ups search the joined text
    store, models = Store(None, None), fake_models(dims={"fake/small": 64})
    prompts: list[str] = []
    ai = fake_ai(conn, reply(prompts))
    ix = Indexer(conn, store, models, ai)
    for name in WORDS:
        model = b_model if name == "b" else None
        ix.add(ix.create(name, "multihop", False, dense_model=model), items(name), "x")
    retriever = Retriever(conn, store, models, ai, ix, Builder(conn, store, models, ai, ix))
    return Chats(conn, retriever, ai), prompts


def copy_store(chats: Chats, name: str, source: str) -> None:
    ix = chats.retriever.indexer
    ix.add(ix.create(name, "multihop", False), items(source), "x")


def summarized(chats: Chats, name: str = "s") -> None:
    """A summarize store with a's items and keyword lists: a keyword method on it pays to
    summarise each query."""
    ix = chats.retriever.indexer
    ix.add(ix.create(name, "multihop", True), items("a"), "x")
    ix.set_lists(chats.retriever.spec(name), LISTS)


def unloadable(chats: Chats, name: str) -> None:
    """Saves the store as built with gone/model, which these models can't load."""
    models = chats.retriever.models
    real = models.dense_dim

    def dense_dim(model=None):
        if model == "gone/model":
            raise ValueError("gone/model is not in fastembed's supported models")
        return real(model)

    models.dense_dim = dense_dim  # type: ignore[method-assign]
    spec = chats.retriever.spec(name)
    db.save_collection(chats.conn, replace(spec, dense_model="gone/model"))


def drop_store(chats: Chats, name: str) -> None:
    chats.retriever.store.drop(name)
    chats.retriever.forget(name)
    db.delete_collection_rows(chats.conn, name)


def found_counts(chats: Chats) -> dict[str, int]:
    """Patches the retriever to record how many hits each store's last search found."""
    counts: dict[str, int] = {}
    real = chats.retriever.retrieve

    def retrieve(collection, *args, **kw):
        hits = real(collection, *args, **kw)
        counts[collection] = len(hits)
        return hits

    chats.retriever.retrieve = retrieve  # type: ignore[method-assign]
    return counts


def attempt(message: dict) -> dict:
    return message["attempts"][-1]


def decision(attempt: dict, store: str) -> dict:
    return next(d for d in attempt["decisions"] if d["store"] == store)


def rewrites(prompts: list[str]) -> list[str]:
    return [p for p in prompts if p.startswith("Earlier messages:")]


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
    new = {"evidence_size": None, "rewrite": None, "rerank": None, "stores": {}}
    assert first["settings"] == new  # None: inherited
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

    updated = chats.update(first["id"], {"title": "Enzymes", "evidence_size": 3, "rewrite": True})
    assert updated["title"] == "Enzymes"
    assert updated["settings"] == {
        "evidence_size": 3,
        "rewrite": True,
        "rerank": None,
        "stores": {},
    }
    with pytest.raises(ValueError):
        chats.update(first["id"], {"evidence_size": 21})
    with pytest.raises(UnknownCollection):
        chats.update(first["id"], {"stores": ["a", "nope"]})

    drop_store(chats, "b")
    got = chats.get(first["id"])
    assert got["deleted_stores"] == ["b"]
    assert [m["n"] for m in got["messages"]] == [1, 2]
    assert got["messages"][0]["attempts"][0]["attempt"] == 1
    # A list that still names the deleted store can be saved; only added stores must exist.
    assert chats.update(first["id"], {"stores": ["b", "a"]})["stores"] == ["b", "a"]

    chats.rerun(first["id"], 1, stores={"a": {"candidates": 2}})
    chats.delete(first["id"])
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
    first = chats.search_text(chat["id"], 1, "m1 enzyme", True, 2)
    assert first == {
        "search_text": "m1 enzyme",
        "rerank_text": "m1 enzyme",
        "from": "message",
        "usd": 0.0,
        "error": None,
    }


def test_rewrite_on_uses_the_ai_rewrite_and_falls_back_on_error(tmp_path):
    chats, prompts = build(tmp_path)
    chat = chats.update(chats.create(["a"])["id"], {"rewrite": True})
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
    rerun = chats.rerun(chat["id"], 2, stores={"a": {"candidates": 2}})
    assert rerun["search_text"] == REWRITTEN
    assert prompts[before:] == []  # no rewrite, no answer: it spent nothing
    assert rerun["usd"] == 0.0


def test_ask_searches_every_store_on_the_chat(tmp_path):
    chats, _ = build(tmp_path)
    chat = chats.create(["a", "b"])
    one = attempt(chats.ask(chat["id"], ENZYME))  # only a holds its words
    assert [d["store"] for d in one["decisions"]] == ["a", "b"]
    gone = {"reason", "best", "bar", "bar_source", "route", "route_bar"}
    assert all(not gone & set(d) for d in one["decisions"])
    assert decision(one, "b")["found"] > 0  # b was searched too
    evidence = one["evidence"]
    assert len(evidence) == SIZE  # the evidence size Settings has in force
    assert sorted(e["n"] for e in evidence) == list(range(1, len(evidence) + 1))
    assert one["answer"] and one["error"] is None


def test_every_store_adds_its_candidates_to_one_reranked_pool(tmp_path):
    chats, _ = build(tmp_path)
    counts = found_counts(chats)
    chat = chats.create(["a", "b"])
    one = attempt(chats.ask(chat["id"], BOTH))
    assert one["pipeline"]["merged_by"] == "rerank"
    evidence = one["evidence"]
    scores = [e["score"] for e in evidence]
    assert scores == sorted(scores, reverse=True)
    for e in evidence:
        assert e["store"] in ("a", "b") and e["rank"] >= 1 and 0 < e["relevance"] < 1
    decisions = one["decisions"]
    assert sum(d["used"] for d in decisions) == len(evidence) <= SIZE
    assert {d["store"]: d["found"] for d in decisions} == counts
    assert one["pipeline"]["pool"] == sum(counts.values())


def test_rerank_scores_against_the_message_alone_without_a_rewrite_and_the_rewrite_with_one(
    tmp_path,
):
    chats, _ = build(tmp_path)
    search_defaults.save(chats.conn, {"rewrite": None})  # on
    models, retriever = chats.retriever.models, chats.retriever
    reranked, retrieved = [], []
    real_rerank, real_retrieve = models.rerank, retriever.retrieve

    def rerank(query, texts, model=None):
        reranked.append(query)
        return real_rerank(query, texts, model)

    def retrieve(collection, question, *args, **kw):
        retrieved.append(question)
        return real_retrieve(collection, question, *args, **kw)

    models.rerank = rerank  # type: ignore[method-assign]
    retriever.retrieve = retrieve  # type: ignore[method-assign]
    chat = chats.create(["a"])
    model = chats.ai.model_for("answer")
    chats.ask(chat["id"], ENZYME)
    chats.ai.set_role("default", None)  # no rewrite model
    two = attempt(chats.ask(chat["id"], "and its rate?", model=model))
    assert (reranked[-1], retrieved[-1]) == ("and its rate?", f"{ENZYME}\nand its rate?")
    assert two["pipeline"]["search_from"] == "joined"
    assert two["pipeline"]["rerank_text"] == "and its rate?"

    chats.ai.set_role("default", model)
    three = attempt(chats.ask(chat["id"], "and its cost?"))
    assert (reranked[-1], retrieved[-1]) == (REWRITTEN, REWRITTEN)
    assert three["pipeline"]["search_from"] == "rewrite"


def test_the_pool_is_capped_round_robin_by_rank(tmp_path):
    chats, _ = build(tmp_path)
    ix = chats.retriever.indexer
    for name in WORDS:
        ix.add(chats.retriever.spec(name), items(name, 30, start=10), "x")
    ix.add(ix.create("c", "multihop", False), items("c", 30, words=TIDES), "x")
    search_defaults.save(chats.conn, {"candidates": 30})
    texts: list[list[str]] = []
    firsts: dict[str, str] = {}
    real_rerank, real_retrieve = chats.retriever.models.rerank, chats.retriever.retrieve

    def rerank(query, sent, model=None):
        texts.append(sent)
        return real_rerank(query, sent, model)

    def retrieve(collection, *args, **kw):
        hits = real_retrieve(collection, *args, **kw)
        firsts[collection] = hits[0].text
        return hits

    chats.retriever.models.rerank = rerank  # type: ignore[method-assign]
    chats.retriever.retrieve = retrieve  # type: ignore[method-assign]
    chat = chats.create(["a", "b", "c"])
    one = attempt(chats.ask(chat["id"], f"{BOTH} {TIDES}"))
    assert len(texts[-1]) == RECOMMENDED["rerank_pool"] == one["pipeline"]["pool"]
    assert all(d["found"] == 30 for d in one["decisions"])

    search_defaults.save(chats.conn, {"rerank_pool": 45})
    two = attempt(chats.ask(chat["id"], f"{BOTH} {TIDES}"))
    assert len(texts[-1]) == 45 == two["pipeline"]["pool"]
    assert texts[-1][:3] == [firsts["a"], firsts["b"], firsts["c"]]
    assert two["pipeline"]["rerank_pool"] == 45


def test_round_robin_by_id_keeps_the_first_rank_of_each_id():
    def hit(i: str, text: str | None = None) -> Hit:
        return Hit(i, text or f"text {i}", 0.0, {}, [], None, None, {})

    a, b, c, d = (hit(i) for i in "abcd")
    lists = {"vector": [a, b], "keyword": [b, c], "hybrid": [a, d]}
    pool = round_robin(lists, None, key=lambda h: h.id)
    assert [h.id for _, h, _ in pool] == ["a", "b", "c", "d"]
    assert [rank for _, _, rank in pool] == [1, 1, 2, 2]
    assert [m for m, _, _ in pool] == ["vector", "keyword", "keyword", "hybrid"]
    same = {"vector": [hit("e", "one text"), hit("f", "one text")]}
    assert [h.id for _, h, _ in round_robin(same, None, key=lambda h: h.id)] == ["e", "f"]
    assert [h.id for _, h, _ in round_robin(same, None)] == ["e"]  # chat's key: by text
    assert len(round_robin(lists, 3)) == 3


def test_a_reranker_chosen_in_settings_is_used_and_noted_while_it_loads(tmp_path):
    chats, _ = build(tmp_path)
    models = chats.retriever.models
    search_defaults.save(chats.conn, {"rerank_model": TURBO})
    status = {TURBO: "loading"}
    used: list[str | None] = []
    real = models.rerank

    def rerank(query, texts, model=None):
        used.append(model)
        return real(query, texts, model)

    models.rerank_status = lambda model=None: status.get(model or "", "ready")  # type: ignore[method-assign]
    models.rerank = rerank  # type: ignore[method-assign]
    chat = chats.create(["a", "b"])
    one = attempt(chats.ask(chat["id"], BOTH))
    assert one["pipeline"]["merged_by"] == "rrf"
    assert one["pipeline"]["notes"] == ["reranker still loading: merged by rank"]
    assert used == [] and one["answer"]

    status[TURBO] = "ready"
    two = attempt(chats.ask(chat["id"], BOTH))
    assert used == [TURBO]
    assert two["pipeline"]["merged_by"] == "rerank" and two["pipeline"]["notes"] == []
    assert two["pipeline"]["rerank_model"] == TURBO


def test_a_passage_in_two_stores_takes_one_slot(tmp_path):
    chats, _ = build(tmp_path)
    copy_store(chats, "a_copy", "a")  # the same passages
    chat = chats.create(["a", "a_copy"])
    one = attempt(chats.ask(chat["id"], ENZYME))
    texts = [e["text"] for e in one["evidence"]]
    assert len(texts) == SIZE and len(set(texts)) == len(texts)
    assert sum(d["used"] for d in one["decisions"]) == SIZE


def test_rerank_off_merges_by_rrf_with_no_note(tmp_path):
    chats, _ = build(tmp_path)
    chat = chats.update(chats.create(["a", "b"])["id"], {"rerank": False})
    one = attempt(chats.ask(chat["id"], BOTH))
    assert one["pipeline"]["merged_by"] == "rrf" and one["pipeline"]["notes"] == []
    assert one["pipeline"]["rerank"] is False
    evidence = one["evidence"]
    assert [e["score"] for e in evidence] == pytest.approx(
        [1 / (RECOMMENDED["rrf_k"] + e["rank"]) for e in evidence]
    )
    assert all(e["relevance"] is None for e in evidence)
    order = [(e["store"], e["rank"]) for e in evidence[:4]]
    assert order == [("a", 1), ("b", 1), ("a", 2), ("b", 2)]  # round-robin by rank


def test_reranker_loading_or_failing_merges_by_rank_with_a_note(tmp_path):
    chats, _ = build(tmp_path)
    models = chats.retriever.models
    chat = chats.create(["a", "b"])
    warmed: list[str | None] = []
    models.warm_rerank_in_background = warmed.append  # type: ignore[method-assign]

    def broken(query, texts, model=None):
        raise RuntimeError("no such file")

    cases = [
        ("loading", models.rerank, "reranker still loading: merged by rank"),
        ("failed", models.rerank, "reranker unavailable: merged by rank"),
        ("ready", broken, "reranker unavailable: merged by rank"),
    ]
    for status, rerank, note in cases:
        models.rerank_status = lambda model=None, s=status: s  # type: ignore[method-assign]
        models.rerank = rerank  # type: ignore[method-assign]
        one = attempt(chats.ask(chat["id"], BOTH))
        assert one["pipeline"]["merged_by"] == "rrf", status
        assert one["pipeline"]["notes"] == [note], status
        assert one["answer"] and one["error"] is None, status
        assert all(e["relevance"] is None for e in one["evidence"])
    assert warmed == [models.rerank_model]  # once, for the failed one


def test_stores_built_with_different_models_pool_and_rerank_together(tmp_path):
    chats, _ = build(tmp_path, b_model="fake/small")
    models = chats.retriever.models
    asked: set[str | None] = set()
    real = models.embed_query

    def embed_query(texts, task="", model=None):
        asked.add(model)
        return real(texts, task, model)

    models.embed_query = embed_query  # type: ignore[method-assign]
    chat = chats.create(["a", "b"])
    one = attempt(chats.ask(chat["id"], BOTH))
    assert one["pipeline"]["merged_by"] == "rerank"
    assert {e["store"] for e in one["evidence"]} == {"a", "b"}
    assert all(d["error"] is None for d in one["decisions"])
    assert asked == {"fake/dense", "fake/small"}


def test_an_empty_store_or_one_whose_model_cannot_load_is_left_out_with_a_note(tmp_path):
    chats, _ = build(tmp_path)
    ix = chats.retriever.indexer
    for name in ("e1", "e2"):
        ix.create(name, "multihop", False)  # no items: nothing to find
    unloadable(chats, "b")
    chat = chats.create(["a", "e1", "b"])
    one = attempt(chats.ask(chat["id"], BOTH))
    e1, b = decision(one, "e1"), decision(one, "b")
    assert (e1["found"], e1["used"], e1["error"]) == (0, 0, None)
    assert b["error"] and "gone/model" in b["error"] and "can't load" in b["error"]
    assert b["found"] == 0
    assert {e["store"] for e in one["evidence"]} == {"a"}
    assert one["answer"] and one["error"] is None

    def broken(query, texts, model=None):
        raise AssertionError("nothing to rerank")

    chats.retriever.models.rerank = broken  # type: ignore[method-assign]
    empty = chats.create(["e1", "e2"])
    two = attempt(chats.ask(empty["id"], BOTH))
    assert two["evidence"] == [] and two["pipeline"]["pool"] == 0
    assert two["pipeline"]["notes"] == [] and two["answer"] and two["error"] is None


def test_no_answer_model_saves_a_passages_only_attempt_without_an_error(tmp_path):
    chats, prompts = build(tmp_path)
    search_defaults.save(chats.conn, {"rewrite": None})  # on
    model = chats.ai.model_for("answer")
    chats.ai.set_role("default", None)
    chat = chats.create(["a", "b"])
    for text in (ENZYME, "and its rate?"):
        one = attempt(chats.ask(chat["id"], text))
        assert one["answer"] is None and one["error"] is None and one["model"] is None
        assert one["pipeline"]["passages_only"] is True
        assert one["evidence"]
    assert chats.in_force(chat["id"])["rewrite_available"] is False
    rerun = chats.rerun(chat["id"], 1)  # a re-run with no answer model is passages only too
    assert rerun["pipeline"]["passages_only"] is True and rerun["answer"] is None
    assert prompts == []  # no rewrite and no answer was sent

    chats.ai.set_role("default", model)
    answered = chats.answer(chat["id"], 1, 1)
    assert answered["answer"] and answered["error"] is None
    assert attempt(chats.ask(chat["id"], VOLCANO))["pipeline"]["passages_only"] is False
    assert chats.rerun(chat["id"], 1)["pipeline"]["passages_only"] is False  # one resolves


def test_rerun_recomputes_the_search_text_only_when_rewrite_or_follow_up_changed(tmp_path):
    chats, _ = build(tmp_path)
    asked: list[str] = []  # every rewrite asked for, cached or not
    real = chats.ai.complete

    def complete(prompt, **kw):
        if kw.get("purpose") == "chat_rewrite":
            asked.append(prompt)
        return real(prompt, **kw)

    chats.ai.complete = complete  # type: ignore[method-assign]
    chat = chats.create(["a"])
    chats.ask(chat["id"], ENZYME)
    two = attempt(chats.ask(chat["id"], "and its rate?"))
    joined = f"{ENZYME}\nand its rate?"
    assert two["search_text"] == joined

    again = chats.rerun(chat["id"], 2)
    assert again["search_text"] == joined and asked == []

    rewritten = chats.rerun(chat["id"], 2, {"rewrite": True})
    assert rewritten["search_text"] == REWRITTEN and len(asked) == 1
    assert rewritten["pipeline"]["rerank_text"] == REWRITTEN

    search_defaults.save(chats.conn, {"follow_up_turns": 0})
    alone = chats.rerun(chat["id"], 2, {"rewrite": False})
    assert alone["search_text"] == "and its rate?" and len(asked) == 1

    chats.rerun(chat["id"], 2, {"rewrite": True})
    assert len(asked) == 2
    search_defaults.save(chats.conn, {"follow_up_turns": 3})
    kept = chats.rerun(chat["id"], 2)
    assert kept["search_text"] == REWRITTEN and len(asked) == 2


def test_rerun_with_changed_turns_and_rewrite_off_joins_again_with_no_ai_call(tmp_path):
    chats, _ = build(tmp_path)
    search_defaults.save(chats.conn, {"rewrite": False})
    chat = chats.create(["a"])
    chats.ask(chat["id"], ENZYME)
    two = attempt(chats.ask(chat["id"], "and its rate?"))
    assert two["search_text"] == f"{ENZYME}\nand its rate?"
    asked: list[str] = []
    real = chats.ai.complete

    def complete(prompt, **kw):
        asked.append(kw.get("purpose"))
        return real(prompt, **kw)

    chats.ai.complete = complete  # type: ignore[method-assign]
    search_defaults.save(chats.conn, {"follow_up_turns": 0})
    again = chats.rerun(chat["id"], 2)
    assert again["search_text"] == "and its rate?"
    assert again["pipeline"]["search_from"] == "joined" and asked == []


def test_old_attempts_with_bar_reasons_still_read(tmp_path):
    chats, _ = build(tmp_path)
    chat = chats.create(["a"])
    n = db.add_chat_message(chats.conn, chat["id"], ENZYME)
    old = {"store": "a", "method": "hybrid", "k": 5, "kept": False, "best": 0.1, "bar": 0.4}
    old.update(reason="below its bar", usd=0.0, error=None, note=None)
    row = {"search_text": "an old text", "decisions": [old], "evidence": [], "answer": "old"}
    row.update(citations=[], usd=0.0, search_error=None, error=None, model="m")
    db.add_chat_attempt(chats.conn, chat["id"], n, row)
    got = chats.get(chat["id"])["messages"][0]["attempts"][0]
    assert got["pipeline"] is None and got["decisions"] == [old]
    rerun = chats.rerun(chat["id"], n)
    assert rerun["search_text"] == "an old text"
    assert rerun["pipeline"]["rerank_text"] == ENZYME and rerun["evidence"]


def test_old_follow_ups_re_run_twice_without_a_rewrite(tmp_path):
    chats, _ = build(tmp_path)
    search_defaults.save(chats.conn, {"rewrite": None})  # on, with a model
    asked: list[str] = []
    real = chats.ai.complete

    def complete(prompt, **kw):
        if kw.get("purpose") == "chat_rewrite":
            asked.append(prompt)
        return real(prompt, **kw)

    chats.ai.complete = complete  # type: ignore[method-assign]
    chat = chats.create(["a"])
    chats.ask(chat["id"], ENZYME)
    n = db.add_chat_message(chats.conn, chat["id"], "and its rate?")
    row = {"search_text": "an old text", "decisions": [], "evidence": [], "answer": "old"}
    row.update(citations=[], usd=0.0, search_error=None, error=None, model="m")
    db.add_chat_attempt(chats.conn, chat["id"], n, row)  # from before the pipeline
    for _ in range(2):
        rerun = chats.rerun(chat["id"], n)
        assert rerun["search_text"] == "an old text"
        assert rerun["pipeline"]["search_from"] == "legacy"
    assert asked == []


def test_the_pool_cap_never_shrinks_the_evidence(tmp_path):
    chats, _ = build(tmp_path)
    search_defaults.save(chats.conn, {"rerank_pool": 5})
    chat = chats.update(chats.create(["a", "b"])["id"], {"rerank": False})
    one = attempt(chats.ask(chat["id"], BOTH))
    assert len(one["evidence"]) == SIZE == 8 and one["pipeline"]["pool"] == SIZE


def test_rerun_adds_an_attempt_without_answering_and_answer_answers_it(tmp_path):
    chats, prompts = build(tmp_path)
    chat = chats.create(["a", "b"])
    chats.ask(chat["id"], ENZYME)
    before = len(prompts)
    rerun = chats.rerun(chat["id"], 1, stores={"a": {"candidates": 2}, "b": {"method": "vector"}})
    assert rerun["attempt"] == 2 and rerun["answer"] is None and rerun["citations"] == []
    assert rerun["model"] is None and rerun["usd"] == 0.0 and prompts[before:] == []
    assert decision(rerun, "a")["candidates"] == 2
    assert decision(rerun, "b")["method"] == "vector"
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


def test_two_answers_of_one_attempt_at_once_pay_once(tmp_path):
    chats, _ = build(tmp_path)
    chat = chats.create(["a", "b"])
    chats.ask(chat["id"], ENZYME)
    chats.rerun(chat["id"], 1, stores={"a": {"candidates": 2}})
    real = chats.ai.complete

    def complete(prompt, **kw):
        time.sleep(0.3)  # both answers are on their way at once
        return real(prompt, **kw)

    chats.ai.complete = complete  # type: ignore[method-assign]
    before = ledger(chats, "chat")
    results: list = []

    def answer():
        try:
            results.append(chats.answer(chat["id"], 1, 2))
        except AlreadyAnswered as e:
            results.append(e)

    threads = [threading.Thread(target=answer) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(type(r).__name__ for r in results) == ["AlreadyAnswered", "dict"]
    saved = chats.get(chat["id"])["messages"][0]["attempts"][1]
    assert saved["answer"] and saved["usd"] > 0
    assert saved["usd"] == pytest.approx(ledger(chats, "chat") - before)


def test_answer_keeps_the_attempts_rewrite_note(tmp_path):
    chats, _ = build(tmp_path)
    chat = chats.update(chats.create(["a"])["id"], {"rewrite": True})
    chats.ask(chat["id"], ENZYME)
    chats.ai.set_role("rewrite", unpriced(chats))  # the rewrite fails: the join is searched
    real = chats.ai.complete
    capped = {"on": True}

    def complete(prompt, **kw):
        if kw.get("purpose") == "chat" and capped["on"]:  # the answer fails too
            chats.ai.set_cap(chats.ai.month_spend())
        return real(prompt, **kw)

    chats.ai.complete = complete  # type: ignore[method-assign]
    note = "the rewrite failed, so the joined text was searched: "
    two = attempt(chats.ask(chat["id"], "and its rate?"))
    assert two["answer"] is None
    assert two["error"].startswith(note) and "no price" in two["error"]
    assert two["error"].count("monthly cap") == 1

    again = chats.answer(chat["id"], 2, 1)  # fails again: the note stays, its error once
    assert again["answer"] is None
    assert again["error"].startswith(note) and again["error"].count("monthly cap") == 1

    capped["on"] = False
    chats.ai.set_cap(20.0)
    done = chats.answer(chat["id"], 2, 1)
    assert done["answer"]
    assert done["error"].startswith(note) and "no price" in done["error"]
    assert "monthly cap" not in done["error"]


def test_a_keyword_rag_decision_records_its_query_tags(tmp_path):
    chats, _ = build(tmp_path)
    spec = chats.retriever.spec("a")
    builder = chats.retriever.builder
    db.set_vocabulary_version(chats.conn, "a", builder.save_vocabulary(spec, WORDS["a"].split()))
    builder.tag(chats.retriever.spec("a"), builder.stored_items("a"))
    db.set_chat_method(chats.conn, "a", "keyword_rag")  # the recommended method is hybrid
    chat = chats.create(["a", "b"])
    one = attempt(chats.ask(chat["id"], BOTH))
    a, b = decision(one, "a"), decision(one, "b")
    assert a["method"] == "keyword_rag" and a["error"] is None
    assert a["tags"] and set(a["tags"]) <= set(WORDS["a"].split())
    assert b["method"] == "hybrid" and b["tags"] is None


def test_an_answered_attempts_usd_is_its_rewrite_stores_and_answer(tmp_path):
    chats, _ = build(tmp_path)
    summarized(chats)
    db.set_chat_method(chats.conn, "s", "kw_individual")  # paid: it summarises the query
    chat = chats.update(chats.create(["s", "b"])["id"], {"rewrite": True})
    chats.ask(chat["id"], ENZYME)
    rewrite, answer = ledger(chats, "chat_rewrite"), ledger(chats, "chat")
    two = attempt(chats.ask(chat["id"], "and its rate?"))
    rewrite, answer = ledger(chats, "chat_rewrite") - rewrite, ledger(chats, "chat") - answer
    stores = sum(d["usd"] for d in two["decisions"])
    assert rewrite > 0 and answer > 0 and stores > 0 and two["answer"]
    assert decision(two, "s")["usd"] > 0 and decision(two, "b")["usd"] == 0
    assert all(d["found"] > 0 for d in two["decisions"])
    assert two["usd"] == pytest.approx(rewrite + stores + answer)


def test_rerun_of_a_message_with_no_attempt_is_unknown(tmp_path):
    chats, _ = build(tmp_path)
    chat = chats.create(["a"])
    n = db.add_chat_message(chats.conn, chat["id"], ENZYME)  # as while its attempt is made
    with pytest.raises(UnknownChat, match=f"^no attempt 1 of message {n}"):
        chats.rerun(chat["id"], n, {})


def test_rewrite_sees_at_most_chat_turns_earlier_messages(tmp_path):
    chats, prompts = build(tmp_path)
    chat = chats.update(chats.create(["a"])["id"], {"rewrite": True})
    for i in range(1, CHAT_TURNS + 3):
        chats.ask(chat["id"], f"m{i} {ENZYME}")
    last = [p for p in prompts if p.startswith("Earlier messages:")][-1]
    earlier = last.split("\n\nLast message:")[0].split("\n")[1:]
    assert len(earlier) == CHAT_TURNS
    assert earlier[0] == f"User: m2 {ENZYME}"


def test_every_store_searched_and_nothing_found_answers_without_passages(tmp_path):
    chats, prompts = build(tmp_path)
    ix = chats.retriever.indexer
    for name in ("e1", "e2"):
        ix.create(name, "multihop", False)  # no items: nothing to find
    systems: list[str] = []
    real = chats.ai.complete

    def complete(prompt, **kw):
        if kw.get("purpose") == "chat":
            systems.append(kw["system"])
        return real(prompt, **kw)

    chats.ai.complete = complete  # type: ignore[method-assign]
    chat = chats.create(["e1", "e2"])
    one = attempt(chats.ask(chat["id"], NOTHING))
    assert [d["store"] for d in one["decisions"]] == ["e1", "e2"]
    assert all((d["error"], d["note"]) == (None, None) for d in one["decisions"])
    assert one["evidence"] == [] and one["citations"] == []
    assert one["answer"] == f"On {NOTHING}: 0 passages, see [1] and [3]."
    assert one["error"] is None
    assert "Passages" not in chat_prompts(prompts)[-1]
    assert "no store passages were used" in systems[-1]


def test_budget_stop_keeps_the_attempts_evidence_and_error(tmp_path):
    chats, _ = build(tmp_path)
    summarized(chats)
    db.set_chat_method(chats.conn, "s", "kw_individual")  # paid: it summarises the query
    chat = chats.update(chats.create(["s", "b"])["id"], {"rewrite": True})
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
    assert two["evidence"] and two["pipeline"]["merged_by"] == "rerank"
    assert sum(d["used"] for d in two["decisions"]) == len(two["evidence"])
    s = decision(two, "s")
    assert s["method"] == "kw_individual" and s["usd"] > 0 and s["used"] > 0
    rewrite = ledger(chats, "chat_rewrite") - rewrite_before
    assert rewrite > 0
    assert two["usd"] == pytest.approx(rewrite + sum(d["usd"] for d in two["decisions"]))


def test_unpriced_answer_model_is_refused_before_saving(tmp_path):
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
    unloadable(chats, "b")
    chat = chats.create(["a", "b"])
    one = attempt(chats.ask(chat["id"], BOTH))
    b = decision(one, "b")
    assert b["error"] and "gone/model" in b["error"]
    assert decision(one, "a")["error"] is None
    assert {e["store"] for e in one["evidence"]} == {"a"}
    assert one["answer"] and one["error"] is None


def test_evidence_numbers_match_the_cited_numbers(tmp_path):
    chats, prompts = build(tmp_path)
    chat = chats.create(["a", "b"])
    one = attempt(chats.ask(chat["id"], BOTH))
    shown = dict(PASSAGE.findall(chat_prompts(prompts)[-1]))
    assert len(shown) == len(one["evidence"]) == SIZE
    assert sum(d["used"] for d in one["decisions"]) == SIZE
    for e in one["evidence"]:
        assert shown[str(e["n"])] == e["text"]
    by_n = {e["n"]: e for e in one["evidence"]}
    assert [c["n"] for c in one["citations"]] == [1, 3]
    for c in one["citations"]:
        e = by_n[c["n"]]
        assert (c["id"], c["store"], c["text"], c["file"], c["source"]) == (
            e["id"],
            e["store"],
            e["text"],
            e["file"],
            e["source"],
        )
    # Evidence saved before it named its file still answers, its citations with no file.
    old = [{k: v for k, v in e.items() if k != "file"} for e in one["evidence"]]
    assert [c["file"] for c in chats.answer_from(BOTH, [], old, None)["citations"]] == [None] * 2
    # Evidence is kept in merged order; n follows the prompt's edge order (1, 3, 5, …, 4, 2).
    assert [e["n"] for e in one["evidence"]][:3] == [1, SIZE, 2]


def test_history_is_the_last_chat_turns_answered_attempts(tmp_path):
    chats, prompts = build(tmp_path)
    chat = chats.create(["a"])
    for i in range(1, 6):
        chats.ask(chat["id"], f"m{i} {ENZYME}")
    # Message 5's latest answered attempt is a re-run with fewer passages; a later
    # unanswered one is skipped.
    chats.rerun(chat["id"], 5, stores={"a": {"candidates": 2}})
    chats.answer(chat["id"], 5, 2)
    chats.rerun(chat["id"], 5, stores={"a": {"candidates": 3}})
    chats.ask(chat["id"], f"m6 {ENZYME}")
    last = chat_prompts(prompts)[-1]
    turns = last.split("\n\n")[0].split("\n")
    assert len(turns) == CHAT_TURNS
    assert turns[0] == f"User: m3 {ENZYME}"
    assert turns[-2] == f"User: m5 {ENZYME}"
    assert turns[-1] == f"Assistant: On m5 {ENZYME}: 2 passages, see [1] and [3]."
    assert "m2" not in last.split("Passages:")[0]


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
    assert (a["note"], a["error"], a["found"], a["used"]) == ("deleted", None, 0, 0)
    assert decision(two, "b")["used"] == len(two["evidence"])
    assert {e["store"] for e in two["evidence"]} == {"b"}
    assert two["answer"] and two["error"] is None
    with pytest.raises(UnknownCollection):
        chats.rerun(chat["id"], 2, stores={"a": {"candidates": 2}})


def test_a_store_taken_off_the_chat_then_deleted_is_marked_on_old_rows(tmp_path):
    chats, _ = build(tmp_path)
    chat = chats.create(["a", "b"])
    chats.ask(chat["id"], BOTH)
    chats.update(chat["id"], {"stores": ["b"]})
    drop_store(chats, "a")
    got = chats.get(chat["id"])
    assert got["stores"] == ["b"]
    assert got["deleted_stores"] == ["a"]  # its old decisions and passages still name it


def test_a_new_chat_inherits_everything(tmp_path):
    chats, _ = build(tmp_path)
    chat = chats.create(["a", "b"])
    force = chats.in_force(chat["id"])
    assert force["evidence_size"] == {"value": SIZE, "from": "recommended"}
    assert force["rerank"] == {"value": True, "from": "recommended"}
    assert force["rewrite"] == {"value": False, "from": "settings"}  # build turns it off
    assert force["stores"]["a"] == {
        "method": {"value": "hybrid", "from": "recommended"},
        "candidates": {"value": CANDIDATES, "from": "recommended"},
    }
    assert list(force["stores"]) == ["a", "b"]
    assert (force["follow_up_turns"], force["rrf_k"], force["rerank_pool"]) == (2, 60, 20)
    assert force["rerank_model"] == chats.retriever.models.rerank_model
    assert force["rewrite_available"] and force["answer_available"]
    assert chats.get(chat["id"])["in_force"] == force
    assert "overrides" not in chats.get(chat["id"])


def test_precedence_is_chat_then_store_then_settings_then_recommended(tmp_path):
    chats, _ = build(tmp_path)
    conn = chats.conn
    chat = chats.create(["a", "b"])
    chats.ask(chat["id"], ENZYME)

    def candidates(store: str) -> dict:
        return chats.in_force(chat["id"])["stores"][store]["candidates"]

    search_defaults.save(conn, {"candidates": 30})
    assert candidates("a") == candidates("b") == {"value": 30, "from": "settings"}
    db.set_chat_k(conn, "a", 25)
    assert candidates("a") == {"value": 25, "from": "store"}
    one = chats.rerun(chat["id"], 1, stores={"a": {"candidates": 12}})
    assert candidates("a") == {"value": 12, "from": "chat"}
    assert (decision(one, "a")["candidates"], decision(one, "b")["candidates"]) == (12, 30)
    chats.rerun(chat["id"], 1, stores={"a": {"candidates": None}})
    assert candidates("a") == {"value": 25, "from": "store"}
    db.set_chat_k(conn, "a", None)
    assert candidates("a") == {"value": 30, "from": "settings"}
    search_defaults.save(conn, {"candidates": 40})  # no other write
    assert candidates("b") == {"value": 40, "from": "settings"}


def test_a_null_evidence_size_uses_the_settings_value(tmp_path):
    chats, _ = build(tmp_path)
    search_defaults.save(chats.conn, {"evidence_size": 3})
    chat = chats.create(["a", "b"])
    assert chats.in_force(chat["id"])["evidence_size"] == {"value": 3, "from": "settings"}
    one = attempt(chats.ask(chat["id"], BOTH))
    assert len(one["evidence"]) == 3


def test_chat_values_survive_a_restart(tmp_path):
    chats, _ = build(tmp_path)
    chat = chats.create(["a", "b"])
    chats.ask(chat["id"], ENZYME)
    chats.update(chat["id"], {"evidence_size": 5, "rerank": False})
    stores = {"a": {"method": "vector", "candidates": 7}}
    chats.rerun(chat["id"], 1, {"rewrite": True}, stores)
    force = chats.in_force(chat["id"])
    assert force["stores"]["a"]["method"] == {"value": "vector", "from": "chat"}
    assert force["rewrite"] == {"value": True, "from": "chat"}

    restarted = Chats(chats.conn, chats.retriever, chats.ai)
    assert restarted.in_force(chat["id"]) == force
    two = decision(attempt(restarted.ask(chat["id"], ENZYME)), "a")
    assert (two["method"], two["candidates"]) == ("vector", 7)


def test_rerun_checks_every_value_before_saving_any(tmp_path):
    chats, _ = build(tmp_path)
    chat = chats.create(["a", "b"])
    chats.ask(chat["id"], ENZYME)
    chats.rerun(chat["id"], 1, {"evidence_size": 5}, {"a": {"candidates": 3}})
    kept = chats.get(chat["id"])["settings"]
    assert kept["stores"] == {"a": {"method": None, "candidates": 3}}
    bad = [
        ({}, {"a": {"candidates": 4}, "b": {"method": "keyword_rag"}}, MethodUnavailable),
        ({"evidence_size": 21}, {"a": {"candidates": 4}}, ValueError),
        ({"evidence_size": 4}, {"a": {"method": "hybrid_rerank"}}, ValueError),
        ({}, {"a": {"method": "nope"}}, ValueError),
        ({}, {"a": {"candidates": 0}}, ValueError),
        ({}, {"a": {"candidates": 61}}, ValueError),
        ({"evidence_size": 4}, {"c": {"candidates": 2}}, UnknownCollection),
        ({"colour": "red"}, {}, ValueError),
        ({"rewrite": "yes"}, {}, ValueError),
        ({}, {"a": {"k": 2}}, ValueError),  # the old override fields are gone
        ({}, {"a": {"search": {"example": "off"}}}, ValueError),
    ]
    for settings, stores, error in bad:
        with pytest.raises(error):
            chats.rerun(chat["id"], 1, settings, stores)
        assert chats.get(chat["id"])["settings"] == kept
    assert len(chats.get(chat["id"])["messages"][0]["attempts"]) == 2


def test_a_saved_method_outside_the_chat_list_reads_as_inherited(tmp_path):
    chats, _ = build(tmp_path)
    db.set_chat_method(chats.conn, "a", "hybrid_rerank")  # saved before it left the list
    chat = chats.create(["a"])
    hybrid = {"value": "hybrid", "from": "recommended"}
    assert chats.in_force(chat["id"])["stores"]["a"]["method"] == hybrid
    assert chats.store_in_force("a")["method"] == hybrid
    assert decision(attempt(chats.ask(chat["id"], ENZYME)), "a")["method"] == "hybrid"


def test_an_inherited_method_the_store_cannot_run_falls_back_to_hybrid_with_a_note(tmp_path):
    chats, _ = build(tmp_path)
    search_defaults.save(chats.conn, {"method": "keyword_rag"})
    chat = chats.create(["a"])
    a = decision(attempt(chats.ask(chat["id"], ENZYME)), "a")
    assert a["method"] == "hybrid" and a["error"] is None
    assert a["note"] == "keyword_rag needs a vocabulary, so hybrid was used"

    summarized(chats)
    search_defaults.save(chats.conn, {"method": "kw_individual"})
    model = chats.ai.model_for("answer")
    chat = chats.create(["s"])
    chats.ask(chat["id"], ENZYME, model=model)
    chats.ai.set_role("default", None)  # no AI resolves: it can't summarise the query
    s = decision(attempt(chats.ask(chat["id"], VOLCANO, model=model)), "s")
    assert s["method"] == "hybrid" and s["error"] is None
    assert s["note"] == "kw_individual needs AI, so hybrid was used"
    # A chat's own value is refused the same way when it is set.
    with pytest.raises(MethodUnavailable, match="needs AI"):
        chats.rerun(chat["id"], 1, stores={"s": {"method": "kw_individual"}})


def test_rewrite_is_on_by_default_when_a_model_resolves(tmp_path):
    chats, prompts = build(tmp_path)
    search_defaults.save(chats.conn, {"rewrite": None})
    chat = chats.create(["a"])
    assert chats.in_force(chat["id"])["rewrite"] == {"value": True, "from": "recommended"}
    chats.ask(chat["id"], ENZYME)
    assert attempt(chats.ask(chat["id"], "and its rate?"))["search_text"] == REWRITTEN
    assert len(rewrites(prompts)) == 1

    model = chats.ai.model_for("answer")  # priced
    chats.ai.set_role("default", None)
    force = chats.in_force(chat["id"])
    assert not force["rewrite_available"] and not force["answer_available"]
    three = attempt(chats.ask(chat["id"], "and its cost?", model=model))
    assert len(rewrites(prompts)) == 1  # nothing is tried
    assert three["search_text"] == f"{ENZYME}\nand its rate?\nand its cost?"
    assert three["error"] is None and three["answer"]


def test_candidates_per_store_is_the_k_each_store_is_searched_with(tmp_path):
    chats, _ = build(tmp_path)
    seen: list[tuple[str, int]] = []
    real = chats.retriever.retrieve

    def retrieve(collection, question, method, k=5, *args, **kw):
        seen.append((collection, k))
        return real(collection, question, method, k, *args, **kw)

    chats.retriever.retrieve = retrieve  # type: ignore[method-assign]
    chat = chats.create(["a", "b"])
    chats.ask(chat["id"], ENZYME)
    assert [k for _, k in seen] == [CANDIDATES, CANDIDATES]
    seen.clear()
    chats.rerun(chat["id"], 1, stores={"a": {"candidates": 5}})
    assert seen == [("a", 5), ("b", CANDIDATES)]


def test_set_default_saves_the_chats_store_values_and_clears_them(tmp_path):
    chats, _ = build(tmp_path)
    conn = chats.conn
    chat = chats.create(["a", "b"])
    chats.ask(chat["id"], ENZYME)
    search = chats.retriever.spec("a").search
    stores = {"a": {"method": "vector", "candidates": 12}, "b": {"candidates": 4}}
    chats.rerun(chat["id"], 1, stores=stores)

    saved = chats.set_default(chat["id"], "a")
    assert (db.get_chat_method(conn, "a"), db.get_chat_k(conn, "a")) == ("vector", 12)
    from_store = {
        "method": {"value": "vector", "from": "store"},
        "candidates": {"value": 12, "from": "store"},
    }
    assert saved == {"chat_method": "vector", "chat_k": 12, "in_force": from_store}
    force = chats.in_force(chat["id"])["stores"]
    assert force["a"] == from_store
    assert chats.retriever.spec("a").search == search  # the chat never edits a store's search
    # b is untouched
    assert force["b"]["candidates"] == {"value": 4, "from": "chat"}
    assert (db.get_chat_method(conn, "b"), db.get_chat_k(conn, "b")) == (None, None)

    before = chats.get(chat["id"])
    assert before["settings"]["stores"]["a"] == {"method": None, "candidates": None}
    assert chats.set_default(chat["id"], "a") == saved  # nothing set: nothing changes
    after = chats.get(chat["id"])
    assert (after["settings"], after["updated_at"]) == (before["settings"], before["updated_at"])
    with pytest.raises(UnknownCollection):
        chats.set_default(chat["id"], "nope")


def test_update_takes_null_as_inherit(tmp_path):
    chats, _ = build(tmp_path)
    chat = chats.create(["a"])
    chats.update(chat["id"], {"evidence_size": 5, "rerank": False})
    force = chats.in_force(chat["id"])
    assert force["evidence_size"] == {"value": 5, "from": "chat"}
    assert force["rerank"] == {"value": False, "from": "chat"}
    chats.update(chat["id"], {"evidence_size": None, "rerank": None})
    force = chats.in_force(chat["id"])
    assert force["evidence_size"] == {"value": SIZE, "from": "recommended"}
    assert force["rerank"] == {"value": True, "from": "recommended"}
    bad: list[dict[str, object]] = [
        {"title": None},
        {"stores": None},
        {"title": " "},
        {"stores": []},
        {"evidence_size": 0},
        {"rerank": "yes"},
        {"colour": "red"},
    ]
    for changes in bad:
        with pytest.raises(ValueError):
            chats.update(chat["id"], changes)


def test_removing_a_store_drops_its_chat_values(tmp_path):
    chats, _ = build(tmp_path)
    chat = chats.create(["a", "b"])
    chats.ask(chat["id"], ENZYME)
    chats.rerun(chat["id"], 1, stores={"a": {"method": "vector"}, "b": {"candidates": 3}})
    chats.update(chat["id"], {"stores": ["b"]})
    assert chats.get(chat["id"])["settings"]["stores"] == {"b": {"method": None, "candidates": 3}}
    assert list(chats.in_force(chat["id"])["stores"]) == ["b"]
    chats.update(chat["id"], {"stores": ["a", "b"]})
    method = chats.in_force(chat["id"])["stores"]["a"]["method"]
    assert method == {"value": "hybrid", "from": "recommended"}


# -- chat from your own app --------------------------------------------------------------


def template(chats: Chats) -> dict:
    chat = chats.create(["a", "b"])
    settings = {
        "evidence_size": 3,
        "rerank": False,
        "rewrite": None,
        "stores": {"a": {"method": "keyword", "candidates": 5}},
    }
    db.update_chat(chats.conn, chat["id"], settings=settings)
    chats.ask(chat["id"], ENZYME)
    return chats.get(chat["id"])


def test_create_like_copies_stores_and_saved_settings(tmp_path):
    chats, _ = build(tmp_path)
    first = template(chats)
    copy = chats.create_like(first["id"])
    assert copy["id"] != first["id"] and copy["stores"] == ["a", "b"]
    assert copy["settings"] == first["settings"] and copy["title"] == "New chat"
    assert chats.get(copy["id"])["messages"] == []
    assert chats.create_like(first["id"], "Bot")["title"] == "Bot"
    with pytest.raises(UnknownChat):
        chats.create_like("nope")


def test_a_copied_chat_inherits_live_but_not_from_its_template(tmp_path):
    chats, _ = build(tmp_path)
    first = template(chats)
    copy = chats.create_like(first["id"])
    chats.update(first["id"], {"evidence_size": 7})
    assert chats.get(copy["id"])["settings"]["evidence_size"] == 3
    other = chats.create(["b"])
    chats.ask(other["id"], VOLCANO)
    chats.rerun(other["id"], 1, stores={"b": {"method": "vector", "candidates": 12}})
    before = chats.in_force(copy["id"])["stores"]["b"]
    chats.set_default(other["id"], "b")
    after = chats.in_force(copy["id"])["stores"]["b"]
    assert after != before and after["method"] == {"value": "vector", "from": "store"}


def test_create_like_leaves_out_deleted_stores(tmp_path):
    chats, _ = build(tmp_path)
    first = template(chats)
    drop_store(chats, "b")
    copy = chats.create_like(first["id"])
    assert copy["stores"] == ["a"] and set(copy["settings"]["stores"]) <= {"a"}
    assert copy["settings"]["stores"]["a"] == {"method": "keyword", "candidates": 5}
    drop_store(chats, "a")
    with pytest.raises(ValueError, match="a chat needs at least one store"):
        chats.create_like(first["id"])


def test_a_message_sent_without_an_answer_pays_for_none(tmp_path):
    chats, _ = build(tmp_path)
    asked = []
    complete = chats.ai.complete

    def counting(*args, **kwargs):
        asked.append(kwargs.get("role"))
        return complete(*args, **kwargs)

    chats.ai.complete = counting  # type: ignore[method-assign]
    chat = chats.create(["a"])
    one = attempt(chats.ask(chat["id"], ENZYME, answer=False))
    assert "answer" not in asked
    assert one["answer"] is None and one["error"] is None and one["evidence"]
    assert one["pipeline"]["answer_skipped"] is True
    answered = chats.answer(chat["id"], 1, 1)
    assert answered["answer"] and "answer" in asked
    two = attempt(chats.ask(chat["id"], ENZYME))
    assert two["pipeline"]["answer_skipped"] is False and two["answer"]


def test_reference_prompt_is_chats_prompt_over_the_y_passages(tmp_path):
    chats, _ = build(tmp_path)
    p1, p2 = "a-1", "a-2"
    got = chats.reference_prompt("a", "what?", [p2, "gone", p1, p2])
    hits = chats.retriever.store.get("a", [p2, p1])
    assert [h.id for h in hits] == [p2, p1]
    assert got == chat_prompt("what?", [], hits)
    with pytest.raises(ValueError, match="mark at least one passage Right that is in a first"):
        chats.reference_prompt("a", "what?", ["gone"])
    with pytest.raises(UnknownCollection):
        chats.reference_prompt("nope", "what?", [p1])


def test_a_repeated_question_marks_its_answer_cached(tmp_path):
    chats, prompts = build(tmp_path)
    one = attempt(chats.ask(chats.create(["a", "b"])["id"], ENZYME))
    before = len(prompts)
    two = attempt(chats.ask(chats.create(["a", "b"])["id"], ENZYME))  # the same evidence
    assert two["evidence"] == one["evidence"] and prompts[before:] == []
    assert one["pipeline"]["answer_cached"] is False and one["usd"] > 0
    assert two["pipeline"]["answer_cached"] is True and two["usd"] == 0
    assert two["answer"] == one["answer"]


def test_answer_from_this_attempt_marks_a_cached_answer(tmp_path):
    chats, _ = build(tmp_path)
    chat = chats.create(["a", "b"])
    chats.ask(chat["id"], ENZYME)
    rerun = chats.rerun(chat["id"], 1)
    assert rerun["pipeline"]["answer_cached"] is False and rerun["answer"] is None
    answered = chats.answer(chat["id"], 1, 2)  # the same evidence as attempt 1's answer
    assert answered["answer"] and answered["pipeline"]["answer_cached"] is True
    assert {k: v for k, v in answered["pipeline"].items() if k != "answer_cached"} == {
        k: v for k, v in rerun["pipeline"].items() if k != "answer_cached"
    }


def test_answering_a_legacy_attempt_keeps_its_pipeline_none(tmp_path):
    chats, _ = build(tmp_path)
    chat = chats.create(["a"])
    n = db.add_chat_message(chats.conn, chat["id"], ENZYME)
    row = {"search_text": "an old text", "decisions": [], "evidence": [], "answer": None}
    row.update(citations=[], usd=0.0, search_error=None, error=None, model=None)
    db.add_chat_attempt(chats.conn, chat["id"], n, row)  # from before the pipeline
    answered = chats.answer(chat["id"], n, 1)
    assert answered["answer"] and answered["pipeline"] is None
    rerun = chats.rerun(chat["id"], n)
    assert rerun["search_text"] == "an old text" and rerun["pipeline"]["search_from"] == "legacy"


def test_a_cached_rewrite_alone_is_not_marked(tmp_path):
    chats, prompts = build(tmp_path)
    first = chats.update(chats.create(["a"])["id"], {"rewrite": True})
    chats.ask(first["id"], ENZYME)
    chats.ask(first["id"], "and its rate?")
    assert len(rewrites(prompts)) == 1
    # The same messages, so the same rewrite; fewer passages, so a fresh answer.
    second = chats.update(chats.create(["a"])["id"], {"rewrite": True, "evidence_size": 3})
    chats.ask(second["id"], ENZYME)
    before = len(chat_prompts(prompts))
    two = attempt(chats.ask(second["id"], "and its rate?"))
    assert len(rewrites(prompts)) == 1  # served from the cache
    assert two["pipeline"]["search_from"] == "rewrite"
    assert len(chat_prompts(prompts)) == before + 1 and two["answer"]
    assert two["pipeline"]["answer_cached"] is False


def test_store_in_force_matches_search_defaults(tmp_path):
    chats, _ = build(tmp_path)
    db.set_chat_k(chats.conn, "a", 7)
    search_defaults.save(chats.conn, {"method": "vector"})
    chosen = {"method": "keyword"}
    for picked in (None, chosen):
        assert chats.store_in_force("a", picked) == search_defaults.store_in_force(
            chats.conn, "a", picked
        )
    assert chats.store_in_force("a", chosen)["method"] == {"value": "keyword", "from": "chat"}
    assert chats.store_in_force("a")["candidates"] == {"value": 7, "from": "store"}
