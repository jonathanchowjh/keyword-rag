"""A connection answered by hand: nothing is sent; its prompts wait until replies come in."""

import pytest

from aiconnect.ai import AI, AwaitingReplies, Conflict, UnknownPrice, cache_key


def by_hand(tmp_path, price: float | None = 0.0) -> AI:
    ai = AI(tmp_path / "ai.db")
    ai.add_connection("hand", "manual", "Answered by hand")
    ai.add_model("hand", "m", price_in=price, price_out=price)
    ai.set_role("answer", "hand/m")
    ai._send = lambda *a: pytest.fail("a connection answered by hand never sends")
    return ai


def ask(ai: AI, prompts: list[str]):
    return ai.complete_many(prompts, role="answer", system="sys", max_tokens=50, purpose="answer")


def test_a_miss_waits_and_a_reply_is_then_cached(tmp_path):
    ai = by_hand(tmp_path)
    with (
        pytest.raises(AwaitingReplies, match="2 prompts wait"),
        ai.for_job("j1", lambda: None, print),
    ):
        ask(ai, ["one", "two"])
    waiting = ai.waiting()
    assert [(w["prompt"], w["model"], w["system"], w["job_id"]) for w in waiting] == [
        ("one", "hand/m", "sys", "j1"),
        ("two", "hand/m", "sys", "j1"),
    ]
    assert waiting[0]["key"] == cache_key("m", "sys", "one", 50)
    assert ai.answer({w["key"]: f"re {w['prompt']}" for w in waiting}) == 2
    assert [c.text for c in ask(ai, ["one", "two"])] == ["re one", "re two"]
    assert ai.waiting() == []
    ledger = ai.sql("select job_id, usd, purpose from ai_ledger")
    assert [tuple(r) for r in ledger] == [("j1", 0.0, "answer")] * 2


def test_a_partial_reply_keeps_the_rest_waiting(tmp_path):
    ai = by_hand(tmp_path)
    with pytest.raises(AwaitingReplies):
        ask(ai, ["one", "two"])
    first = ai.waiting()[0]
    ai.answer({first["key"]: "re one"})
    with pytest.raises(AwaitingReplies, match="1 prompt wait"):
        ask(ai, ["one", "two"])
    assert [w["prompt"] for w in ai.waiting()] == ["two"]


def test_a_prompt_two_jobs_ask_waits_for_each_and_one_reply_answers_both(tmp_path):
    # Runs queued together share prompts (X4's arms share most answers): each run's export
    # must list them, and the first job to ask is the one charged.
    ai = by_hand(tmp_path, price=1.0)
    for job, prompts in (("j1", ["one", "two"]), ("j2", ["two", "three"])):
        with pytest.raises(AwaitingReplies), ai.for_job(job, lambda: None, lambda m: None):
            ask(ai, prompts)
    assert [w["prompt"] for w in ai.waiting("j2")] == ["two", "three"]
    assert [(w["prompt"], w["job_id"]) for w in ai.waiting()] == [
        ("one", "j1"),
        ("two", "j1"),
        ("three", "j2"),
    ]
    two = ai.waiting("j2")[0]["key"]
    ai.answer({two: "re two"})
    assert [w["prompt"] for w in ai.waiting("j1")] == ["one"]
    assert [w["prompt"] for w in ai.waiting("j2")] == ["three"]
    assert [r[0] for r in ai.sql("select job_id from ai_ledger")] == ["j1"]


def test_replies_to_nothing_waiting_or_empty_are_refused_whole(tmp_path):
    ai = by_hand(tmp_path)
    with pytest.raises(AwaitingReplies):
        ask(ai, ["one"])
    key = ai.waiting()[0]["key"]
    with pytest.raises(ValueError, match="answer no waiting prompt"):
        ai.answer({key: "re one", "nope": "x"})
    with pytest.raises(ValueError, match="empty"):
        ai.answer({key: "  "})
    assert len(ai.waiting()) == 1 and ai.sql("select count(*) from ai_cache")[0][0] == 0


def test_a_reply_is_ledgered_at_the_models_price(tmp_path):
    ai = by_hand(tmp_path, price=1.0)
    with pytest.raises(AwaitingReplies):
        ask(ai, ["one"])
    ai.answer({ai.waiting()[0]["key"]: "re one"})
    assert ai.month_spend() > 0


def test_an_unpriced_model_is_refused_before_anything_waits(tmp_path):
    ai = by_hand(tmp_path, price=None)
    with pytest.raises(UnknownPrice):
        ask(ai, ["one"])
    assert ai.waiting() == []


def test_it_needs_no_key_has_nothing_to_test_and_no_model_list(tmp_path):
    ai = by_hand(tmp_path)
    assert ai.ready("answer")
    assert ai.test_connection("hand")["ok"] is False
    with pytest.raises(Conflict, match="by hand"):
        ai.fetch_models("hand")


def test_removing_the_connection_drops_its_waiting_prompts(tmp_path):
    ai = by_hand(tmp_path)
    with pytest.raises(AwaitingReplies):
        ask(ai, ["one"])
    ai.set_role("answer", None)
    ai.remove_connection("hand")
    assert ai.waiting() == []
