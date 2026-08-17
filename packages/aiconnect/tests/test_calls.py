import hashlib
import json
import threading

import pytest

import aiconnect.ai as core
from aiconnect import (
    BudgetExceeded,
    Cancelled,
    NoConnection,
    ProviderError,
    UnknownModel,
    UnknownPrice,
    fakes,
)
from aiconnect.fakes import fake_ai

NEW_BATCH = "insert into ai_batches (id, model, purpose, created_at, usd, connection_id) values "


def rows(ai, query):
    return ai.sql(query)


def ledger_rows(ai):
    return rows(ai, "select count(*) from ai_ledger")[0][0]


def cache_rows(ai):
    return rows(ai, "select count(*) from ai_cache")[0][0]


def test_complete_records_cost_and_caches(tmp_path):
    c = fake_ai(tmp_path / "ai.db", lambda p: "Answer: B")
    first = c.complete("q" * 400, purpose="answer")
    again = c.complete("q" * 400, purpose="answer")
    assert first.text == "Answer: B" and not first.cached and first.usd > 0
    assert again.cached and again.usd == 0
    assert c.month_spend() == pytest.approx(first.usd)
    assert rows(c, "select connection_id from ai_ledger")[0][0] == "anthropic"


def test_cost_uses_haiku_prices(tmp_path):
    c = fake_ai(tmp_path / "ai.db", lambda p: "x" * 4000)
    r = c.complete("y" * 4000, purpose="t")
    assert r.usd == pytest.approx(1000 * 0.10 / 1e6 + 1000 * 0.50 / 1e6)


def test_sonnet_prices(tmp_path):
    c = fake_ai(tmp_path / "ai.db", lambda p: "x" * 4000)
    r = c.complete("y" * 4000, model="claude-sonnet-5-5", purpose="t")
    assert r.usd == pytest.approx(1000 * 2.00 / 1e6 + 1000 * 10.00 / 1e6)


def test_cap_refuses_before_sending(tmp_path):
    sent = []
    c = fake_ai(tmp_path / "ai.db", lambda p: sent.append(p) or "ok", cap_usd=0.0)
    with pytest.raises(BudgetExceeded):
        c.complete("hello", purpose="t")
    assert sent == []


def test_cache_hit_skips_cap(tmp_path):
    fake_ai(tmp_path / "ai.db", lambda p: "ok").complete("hello", purpose="t")
    capped = fake_ai(tmp_path / "ai.db", lambda p: "ok", cap_usd=0.0)
    assert capped.complete("hello", purpose="t").cached


def test_no_key_raises(tmp_path):
    c = fake_ai(tmp_path / "ai.db", lambda p: "ok", api_key=None)
    with pytest.raises(NoConnection):
        c.complete("hello", purpose="t")


def test_unknown_model(tmp_path):
    c = fake_ai(tmp_path / "ai.db", lambda p: "ok")
    with pytest.raises(UnknownModel):
        c.complete("hello", model="nope/x", purpose="t")


def test_many_uses_batch_above_50_and_halves_price(tmp_path):
    c = fake_ai(tmp_path / "ai.db", lambda p: "x" * 4000)
    out = c.complete_many(["y" * 4000 + str(i) for i in range(51)], purpose="t")
    assert [o.text for o in out] == ["x" * 4000] * 51
    found = rows(c, "select batch, usd, connection_id from ai_ledger")
    assert {r["batch"] for r in found} == {1}
    assert {r["connection_id"] for r in found} == {"anthropic"}
    assert found[0]["usd"] == pytest.approx((1001 * 0.10e-6 + 1000 * 0.50e-6) / 2)


def test_many_at_50_is_realtime(tmp_path):
    c = fake_ai(tmp_path / "ai.db", lambda p: "ok")
    c.complete_many([f"p{i}" for i in range(50)], purpose="t")
    assert {r["batch"] for r in rows(c, "select batch from ai_ledger")} == {0}


def test_many_keeps_order_with_cache_hits(tmp_path):
    c = fake_ai(tmp_path / "ai.db", lambda p: p.upper())
    c.complete("b", purpose="t")
    out = c.complete_many(["a", "b", "c"], purpose="t")
    assert [o.text for o in out] == ["A", "B", "C"] and [o.cached for o in out] == [
        False,
        True,
        False,
    ]


def test_many_checks_cap_once_before_sending(tmp_path):
    sent = []
    c = fake_ai(tmp_path / "ai.db", lambda p: sent.append(p) or "ok", cap_usd=0.0)
    with pytest.raises(BudgetExceeded):
        c.complete_many(["a", "b"], purpose="t")
    assert sent == []


def test_estimate(tmp_path):
    c = fake_ai(tmp_path / "ai.db", lambda p: "")
    full = 1300 * 0.10e-6 + 520 * 0.50e-6
    assert c.estimate_usd(4000, 400) == pytest.approx(full)
    assert c.estimate_usd(4000, 400, batch=True) == pytest.approx(full / 2)


def test_realtime_failure_keeps_completed_calls(tmp_path):
    def reply(p):
        if p == "p2":
            raise RuntimeError("529 overloaded")
        return "ok"

    c = fake_ai(tmp_path / "ai.db", reply)
    with pytest.raises(ProviderError, match="overloaded"):
        c.complete_many(["p0", "p1", "p2", "p3"], purpose="t")
    assert ledger_rows(c) == 2 and cache_rows(c) == 2


def test_batch_with_failed_entry_ledgers_the_rest(tmp_path):
    c = fake_ai(tmp_path / "ai.db", lambda p: "FAIL" if p == "p7" else "ok")
    with pytest.raises(RuntimeError, match="1 of 51 batch requests failed"):
        c.complete_many([f"p{i}" for i in range(51)], purpose="t")
    assert ledger_rows(c) == 50 and cache_rows(c) == 50


def test_batch_interrupted_while_waiting_is_collected_after_restart(tmp_path):
    path = tmp_path / "ai.db"
    prompts = [f"p{i}" for i in range(51)]
    first = fake_ai(path, lambda p: "ok")

    def die(connection, batch_id):
        raise KeyboardInterrupt  # the process is killed while polling

    first._batch_done = die
    with pytest.raises(KeyboardInterrupt):
        first.complete_many(prompts, purpose="t")
    assert ledger_rows(first) == 0
    assert first.pending_batches()[0]["connection_id"] == "anthropic"

    submitted: list[list[dict]] = []
    second = fake_ai(path, lambda p: "ok")
    real_submit = second._submit_batch
    second._submit_batch = lambda connection, reqs: (
        submitted.append(reqs) or real_submit(connection, reqs)
    )
    out = second.complete_many(prompts, purpose="t")
    assert all(o.cached for o in out) and submitted == []
    assert ledger_rows(second) == 51


def test_cap_counts_spend_still_in_flight(tmp_path):
    path = tmp_path / "ai.db"
    seen: list[Exception] = []
    holder: dict = {}

    def reply(p):
        if p == "outer":
            try:
                holder["c"].complete("inner", purpose="t")
            except BudgetExceeded as e:
                seen.append(e)
        return "ok"

    probe = fake_ai(path, lambda p: "")
    one_call = probe.estimate_usd(len("outer"), 1024)
    holder["c"] = fake_ai(path, reply, cap_usd=one_call * 1.5)
    holder["c"].complete("outer", purpose="t")
    assert len(seen) == 1


def batch_estimate(c, prompts):
    return sum(c.estimate_usd(len(p), 1024, batch=True) for p in prompts)


def test_batch_being_waited_on_is_ledgered_once(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "BATCH_POLL_SECONDS", 0)  # every call checks pending batches
    c = fake_ai(tmp_path / "ai.db", lambda p: "ok")
    polls = []

    def done(connection, batch_id):
        polls.append(batch_id)
        if len(polls) == 1:
            c.complete("meanwhile", purpose="t")  # another request misses the cache mid-wait
        return True

    c._batch_done = done
    c.complete_many([f"p{i}" for i in range(51)], purpose="t")
    assert ledger_rows(c) == 52


def test_unreadable_pending_batch_does_not_block_calls(tmp_path):
    c = fake_ai(tmp_path / "ai.db", lambda p: "ok")
    c.sql(
        NEW_BATCH + "('gone', 'claude-haiku-5-5', 't', '2026-10-01T00:00:00+00:00', 0, 'anthropic')"
    )

    def done(connection, batch_id):
        raise RuntimeError("404 batch not found")

    c._batch_done = done
    assert c.complete("a", purpose="t").text == "ok"


def test_pending_batches_are_checked_at_most_once_per_poll_interval(tmp_path):
    c = fake_ai(tmp_path / "ai.db", lambda p: "ok")
    c.sql(
        NEW_BATCH + "('slow', 'claude-haiku-5-5', 't', '2026-10-01T00:00:00+00:00', 0, 'anthropic')"
    )
    checks = []
    c._batch_done = lambda connection, batch_id: checks.append(batch_id) or False
    for i in range(3):
        c.complete(f"q{i}", purpose="t")
    assert checks == ["slow"]


def test_cap_counts_batches_submitted_before_a_restart(tmp_path):
    path = tmp_path / "ai.db"
    prompts = [f"p{i}" for i in range(51)]
    first = fake_ai(path, lambda p: "ok")

    def die(connection, batch_id):
        raise KeyboardInterrupt

    first._batch_done = die
    with pytest.raises(KeyboardInterrupt):
        first.complete_many(prompts, purpose="t")

    one_call = first.estimate_usd(len("next"), 1024)
    second = fake_ai(path, lambda p: "ok", cap_usd=batch_estimate(first, prompts) + one_call / 2)
    second._batch_done = lambda connection, batch_id: False  # still running at the provider
    with pytest.raises(BudgetExceeded):
        second.complete("next", purpose="t")


def test_batch_in_flight_counts_once_against_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "BATCH_POLL_SECONDS", 0)
    path = tmp_path / "ai.db"
    prompts = [f"p{i}" for i in range(51)]
    probe = fake_ai(path, lambda p: "")
    cap = batch_estimate(probe, prompts) + 2 * probe.estimate_usd(len("inner"), 1024)
    c = fake_ai(path, lambda p: "ok", cap_usd=cap)
    polls = []

    def done(connection, batch_id):
        polls.append(batch_id)
        if len(polls) == 1:
            c.complete("inner", purpose="t")  # must fit: the batch is reserved, not doubled
        return True

    c._batch_done = done
    c.complete_many(prompts, purpose="t")
    assert ledger_rows(c) == 52


def test_ledger_rows_carry_the_job(tmp_path):
    ai = fake_ai(tmp_path / "ai.db", lambda p: "ok")
    with ai.for_job("j1", check=lambda: None, log=lambda m: None):
        ai.complete("hi", purpose="answer")
    ai.complete("other", purpose="query")
    assert ai.job_spend("j1") > 0
    assert rows(ai, "select count(*) from ai_ledger where job_id is null")[0][0] == 1


def test_cancel_while_waiting_on_a_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "BATCH_POLL_SECONDS", 1)
    ai = fake_ai(tmp_path / "ai.db", lambda p: "ok")
    ai._batch_done = lambda connection, batch_id: False  # type: ignore[method-assign]
    logs: list[str] = []

    def check():
        raise Cancelled()

    with pytest.raises(Cancelled), ai.for_job("j1", check=check, log=logs.append):
        ai.complete_many([f"p{n}" for n in range(51)], purpose="answer")
    assert fakes.CANCELLED
    assert rows(ai, "select job_id from ai_batches")[0][0] == "j1"
    assert logs and "submitted, 51 requests" in logs[0] and logs[0].startswith("AI batch")
    assert ai._reserved == 0 and not ai._waiting


def test_job_context_is_per_thread(tmp_path):
    ai = fake_ai(tmp_path / "ai.db", lambda p: "ok")
    with ai.for_job("j1", check=lambda: None, log=lambda m: None):
        t = threading.Thread(target=lambda: ai.complete("elsewhere", purpose="query"))
        t.start()
        t.join()
    assert rows(ai, "select count(*) from ai_ledger where job_id is null")[0][0] == 1


def test_failed_cancel_still_reports_cancelled(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "BATCH_POLL_SECONDS", 1)
    ai = fake_ai(tmp_path / "ai.db", lambda p: "ok")
    ai._batch_done = lambda connection, batch_id: False  # type: ignore[method-assign]

    def broken(connection, batch_id):
        raise RuntimeError("boom")

    ai._cancel_batch = broken  # type: ignore[method-assign]
    logs: list[str] = []

    def check():
        raise Cancelled()

    with pytest.raises(Cancelled), ai.for_job("j1", check=check, log=logs.append):
        ai.complete_many([f"p{n}" for n in range(51)], purpose="answer")
    assert any("could not cancel AI batch" in m for m in logs)
    assert ai._reserved == 0
    assert rows(ai, "select count(*) from ai_batches")[0][0] == 1


def test_collected_batch_keeps_its_own_job(tmp_path):
    ai = fake_ai(tmp_path / "ai.db", lambda p: "ok")
    connection = ai._connection("anthropic")
    batch_id = ai._submit_batch(
        connection,
        [
            {
                "custom_id": "k",
                "model": "claude-haiku-5-5",
                "system": "",
                "prompt": "p",
                "max_tokens": 9,
            }
        ],
    )
    ai.sql(
        NEW_BATCH + "(?, 'claude-haiku-5-5', 'answer', '2026-01-01', 0.0, 'anthropic')", (batch_id,)
    )
    with ai.for_job("j2", check=lambda: None, log=lambda m: None):
        ai._collect(batch_id, "anthropic", "claude-haiku-5-5", "answer")
    assert rows(ai, "select count(*) from ai_ledger where job_id is null")[0][0] == 1
    assert ai.job_spend("j2") == 0


def test_tally_adds_up_this_threads_spend_only(tmp_path):
    c = fake_ai(tmp_path / "ai.db", lambda p: "ok")
    with c.tally() as spent:
        mine = c.complete("mine", purpose="t").usd + c.complete("mine", purpose="t").usd
        other = threading.Thread(target=lambda: c.complete("the worker's", purpose="t"))
        other.start()
        other.join()
        # "a" twice is one call, paid once
        many = sum(d.usd for d in c.complete_many(["a", "b", "a"], purpose="t")[:2])
    assert spent["usd"] == pytest.approx(mine + many) and spent["usd"] > 0
    assert c.month_spend() > spent["usd"]  # the other thread's call was not counted
    c.complete("after", purpose="t")
    assert spent["usd"] == pytest.approx(mine + many)


def test_cache_key_formula_is_unchanged(tmp_path):
    ai = fake_ai(tmp_path / "ai.db", lambda p: "ok")
    expected = hashlib.sha256(
        json.dumps(["claude-haiku-5-5", "", "hello", 1024]).encode()
    ).hexdigest()
    ai.complete("hello", purpose="t")
    assert ai._db.execute("select key from ai_cache").fetchall()[0][0] == expected


def test_unpriced_model_refused_before_sending(tmp_path):
    sent = []
    ai = fake_ai(tmp_path / "ai.db", lambda p: sent.append(p) or "ok")
    ai.add_model("anthropic", "claude-new")
    with pytest.raises(UnknownPrice):
        ai.complete("hi", model="anthropic/claude-new", purpose="t")
    assert sent == []


def test_role_picks_model_and_explicit_model_wins(tmp_path):
    ai = fake_ai(tmp_path / "ai.db", lambda p: "ok")
    ai.set_role("answer", "anthropic/claude-sonnet-5-5")
    ai.complete("a", role="answer", purpose="t")
    ai.complete("b", purpose="t")  # no role of its own: default
    ai.complete("c", role="answer", model="anthropic/claude-haiku-5-5", purpose="t")
    ai.complete("d", role="nobody", purpose="t")  # unknown role falls back to default
    used = [r["model"] for r in rows(ai, "select model from ai_ledger order by rowid")]
    assert used == [
        "claude-sonnet-5-5",
        "claude-haiku-5-5",
        "claude-haiku-5-5",
        "claude-haiku-5-5",
    ]
    assert ai.model_for("answer") == "anthropic/claude-sonnet-5-5"


def test_model_for_without_a_model_raises(tmp_path):
    ai = core.AI(tmp_path / "ai.db")
    with pytest.raises(NoConnection) as e:
        ai.model_for("answer")
    assert e.value.role == "answer"
    with pytest.raises(NoConnection):
        ai.complete("hi", role="answer", purpose="t")


def test_job_spend_sums_one_job(tmp_path):
    ai = fake_ai(tmp_path / "ai.db", lambda p: "x" * 400)
    with ai.for_job("j1", check=lambda: None, log=lambda m: None):
        a = ai.complete("one", purpose="t").usd
        b = ai.complete("two", purpose="t").usd
    with ai.for_job("j2", check=lambda: None, log=lambda m: None):
        ai.complete("three", purpose="t")
    assert ai.job_spend("j1") == pytest.approx(a + b)
    assert ai.job_spend("nobody") == 0


def test_cache_false_always_sends(tmp_path):
    sent = []
    ai = fake_ai(tmp_path / "ai.db", lambda p: sent.append(p) or "ok")
    ai.complete("same", purpose="t", cache=False)
    again = ai.complete("same", purpose="t", cache=False)
    assert sent == ["same", "same"] and not again.cached
    assert ai.complete("same", purpose="t").cached  # what was paid for is still cached


def test_ready_is_false_without_key_for_anthropic(tmp_path):
    ai = fake_ai(tmp_path / "ai.db", lambda p: "ok", api_key=None)
    assert not ai.ready()
    ai.update_connection("anthropic", api_key="sk-ant-api03-abcdWXYZ")
    assert ai.ready()
    ai.update_model("anthropic", "claude-haiku-5-5", enabled=False)
    assert not ai.ready()
    ai.update_model("anthropic", "claude-haiku-5-5", enabled=True, price_in=None)
    assert not ai.ready()


def test_ready_for_openai_kind_needs_no_key(tmp_path):
    ai = core.AI(tmp_path / "ai.db")
    ai.add_connection("local", "openai", "Local", base_url="http://x/v1", local=True)
    ai.add_model("local", "llama", price_in=0, price_out=0)
    ai.set_role("default", "local/llama")
    assert ai.ready()
    assert not core.AI(tmp_path / "other.db").ready()


def test_provider_error_names_connection_and_hides_key(tmp_path):
    ai = fake_ai(tmp_path / "ai.db", lambda p: "ok", api_key="sk-ant-api03-abcdWXYZ")

    def boom(*args):
        raise RuntimeError("401 bad key sk-ant-api03-abcdWXYZ")

    from aiconnect import wire

    wire.SEND["anthropic"] = boom
    try:
        ai = core.AI(tmp_path / "ai.db")  # unpatched seams: the real _send
        with pytest.raises(ProviderError) as e:
            ai.complete("hi", purpose="t")
    finally:
        wire.SEND["anthropic"] = wire.anthropic_send
    assert str(e.value).startswith("anthropic: ")
    assert "abcdWXYZ" not in str(e.value) and "sk-…WXYZ" in str(e.value)
