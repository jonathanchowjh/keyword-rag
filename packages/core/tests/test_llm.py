import pytest

from ragqa_core import fakes, llm
from ragqa_core.db import connect, job_spend
from ragqa_core.fakes import fake_claude
from ragqa_core.llm import BudgetExceeded, Cancelled, NoApiKey, UnknownModel


def test_complete_records_cost_and_caches(tmp_path):
    c = fake_claude(connect(tmp_path / "r.db"), lambda p: "Answer: B")
    first = c.complete("q" * 400, purpose="answer")
    again = c.complete("q" * 400, purpose="answer")
    assert first.text == "Answer: B" and not first.cached and first.usd > 0
    assert again.cached and again.usd == 0
    assert c.month_spend() == pytest.approx(first.usd)


def test_cost_uses_haiku_prices(tmp_path):
    c = fake_claude(connect(tmp_path / "r.db"), lambda p: "x" * 4000)
    r = c.complete("y" * 4000, purpose="t")
    assert r.usd == pytest.approx(1000 * 0.10 / 1e6 + 1000 * 0.50 / 1e6)


def test_sonnet_prices(tmp_path):
    c = fake_claude(connect(tmp_path / "r.db"), lambda p: "x" * 4000)
    r = c.complete("y" * 4000, model="claude-sonnet-5-5", purpose="t")
    assert r.usd == pytest.approx(1000 * 2.00 / 1e6 + 1000 * 10.00 / 1e6)


def test_cap_refuses_before_sending(tmp_path):
    sent = []
    c = fake_claude(connect(tmp_path / "r.db"), lambda p: sent.append(p) or "ok", cap_usd=0.0)
    with pytest.raises(BudgetExceeded):
        c.complete("hello", purpose="t")
    assert sent == []


def test_cache_hit_skips_cap(tmp_path):
    conn = connect(tmp_path / "r.db")
    fake_claude(conn, lambda p: "ok").complete("hello", purpose="t")
    capped = fake_claude(conn, lambda p: "ok", cap_usd=0.0)
    assert capped.complete("hello", purpose="t").cached


def test_no_key_raises(tmp_path):
    c = fake_claude(connect(tmp_path / "r.db"), lambda p: "ok", api_key=None)
    with pytest.raises(NoApiKey):
        c.complete("hello", purpose="t")


def test_unknown_model(tmp_path):
    c = fake_claude(connect(tmp_path / "r.db"), lambda p: "ok")
    with pytest.raises(UnknownModel):
        c.complete("hello", model="claude-nope", purpose="t")


def test_many_uses_batch_above_50_and_halves_price(tmp_path):
    c = fake_claude(connect(tmp_path / "r.db"), lambda p: "x" * 4000)
    out = c.complete_many(["y" * 4000 + str(i) for i in range(51)], purpose="t")
    assert [o.text for o in out] == ["x" * 4000] * 51
    rows = c.conn.execute("select batch, usd from ledger")
    assert {r["batch"] for r in rows} == {1}
    assert rows[0]["usd"] == pytest.approx((1001 * 0.10e-6 + 1000 * 0.50e-6) / 2)


def test_many_at_50_is_realtime(tmp_path):
    c = fake_claude(connect(tmp_path / "r.db"), lambda p: "ok")
    c.complete_many([f"p{i}" for i in range(50)], purpose="t")
    assert {r["batch"] for r in c.conn.execute("select batch from ledger")} == {0}


def test_many_keeps_order_with_cache_hits(tmp_path):
    c = fake_claude(connect(tmp_path / "r.db"), lambda p: p.upper())
    c.complete("b", purpose="t")
    out = c.complete_many(["a", "b", "c"], purpose="t")
    assert [o.text for o in out] == ["A", "B", "C"] and [o.cached for o in out] == [
        False,
        True,
        False,
    ]


def test_many_checks_cap_once_before_sending(tmp_path):
    sent = []
    c = fake_claude(connect(tmp_path / "r.db"), lambda p: sent.append(p) or "ok", cap_usd=0.0)
    with pytest.raises(BudgetExceeded):
        c.complete_many(["a", "b"], purpose="t")
    assert sent == []


def test_estimate(tmp_path):
    c = fake_claude(connect(tmp_path / "r.db"), lambda p: "")
    full = 1300 * 0.10e-6 + 520 * 0.50e-6
    assert c.estimate_usd(4000, 400) == pytest.approx(full)
    assert c.estimate_usd(4000, 400, batch=True) == pytest.approx(full / 2)


def ledger_rows(c):
    return c.conn.execute("select count(*) from ledger")[0][0]


def cache_rows(c):
    return c.conn.execute("select count(*) from cache")[0][0]


def test_realtime_failure_keeps_completed_calls(tmp_path):
    def reply(p):
        if p == "p2":
            raise RuntimeError("529 overloaded")
        return "ok"

    c = fake_claude(connect(tmp_path / "r.db"), reply)
    with pytest.raises(RuntimeError, match="overloaded"):
        c.complete_many(["p0", "p1", "p2", "p3"], purpose="t")
    assert ledger_rows(c) == 2 and cache_rows(c) == 2


def test_batch_with_failed_entry_ledgers_the_rest(tmp_path):
    c = fake_claude(connect(tmp_path / "r.db"), lambda p: "FAIL" if p == "p7" else "ok")
    with pytest.raises(RuntimeError, match="1 of 51 batch requests failed"):
        c.complete_many([f"p{i}" for i in range(51)], purpose="t")
    assert ledger_rows(c) == 50 and cache_rows(c) == 50


def test_batch_interrupted_while_waiting_is_collected_after_restart(tmp_path):
    conn = connect(tmp_path / "r.db")
    prompts = [f"p{i}" for i in range(51)]
    first = fake_claude(conn, lambda p: "ok")

    def die(batch_id):
        raise KeyboardInterrupt  # the process is killed while polling

    first._batch_done = die
    with pytest.raises(KeyboardInterrupt):
        first.complete_many(prompts, purpose="t")
    assert ledger_rows(first) == 0

    submitted: list[list[dict]] = []
    second = fake_claude(conn, lambda p: "ok")
    real_submit = second._submit_batch
    second._submit_batch = lambda reqs: submitted.append(reqs) or real_submit(reqs)
    out = second.complete_many(prompts, purpose="t")
    assert all(o.cached for o in out) and submitted == []
    assert ledger_rows(second) == 51


def test_cap_counts_spend_still_in_flight(tmp_path):
    conn = connect(tmp_path / "r.db")
    seen: list[Exception] = []
    holder: dict = {}

    def reply(p):
        if p == "outer":
            try:
                holder["c"].complete("inner", purpose="t")
            except BudgetExceeded as e:
                seen.append(e)
        return "ok"

    probe = fake_claude(conn, lambda p: "")
    one_call = probe.estimate_usd(len("outer"), 1024)
    holder["c"] = fake_claude(conn, reply, cap_usd=one_call * 1.5)
    holder["c"].complete("outer", purpose="t")
    assert len(seen) == 1


def batch_estimate(c, prompts):
    return sum(c.estimate_usd(len(p), 1024, batch=True) for p in prompts)


def test_batch_being_waited_on_is_ledgered_once(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "BATCH_POLL_SECONDS", 0)  # every call checks pending batches
    c = fake_claude(connect(tmp_path / "r.db"), lambda p: "ok")
    polls = []

    def done(batch_id):
        polls.append(batch_id)
        if len(polls) == 1:
            c.complete("meanwhile", purpose="t")  # another request misses the cache mid-wait
        return True

    c._batch_done = done
    c.complete_many([f"p{i}" for i in range(51)], purpose="t")
    assert ledger_rows(c) == 52


def test_unreadable_pending_batch_does_not_block_calls(tmp_path):
    conn = connect(tmp_path / "r.db")
    conn.execute(
        "insert into pending_batches (id, model, purpose, created_at)"
        " values ('gone', 'claude-haiku-5-5', 't', '2026-10-01T00:00:00+00:00')"
    )
    c = fake_claude(conn, lambda p: "ok")

    def done(batch_id):
        raise RuntimeError("404 batch not found")

    c._batch_done = done
    assert c.complete("a", purpose="t").text == "ok"


def test_pending_batches_are_checked_at_most_once_per_poll_interval(tmp_path):
    conn = connect(tmp_path / "r.db")
    conn.execute(
        "insert into pending_batches (id, model, purpose, created_at)"
        " values ('slow', 'claude-haiku-5-5', 't', '2026-10-01T00:00:00+00:00')"
    )
    c = fake_claude(conn, lambda p: "ok")
    checks = []
    c._batch_done = lambda batch_id: checks.append(batch_id) or False
    for i in range(3):
        c.complete(f"q{i}", purpose="t")
    assert checks == ["slow"]


def test_cap_counts_batches_submitted_before_a_restart(tmp_path):
    conn = connect(tmp_path / "r.db")
    prompts = [f"p{i}" for i in range(51)]
    first = fake_claude(conn, lambda p: "ok")

    def die(batch_id):
        raise KeyboardInterrupt

    first._batch_done = die
    with pytest.raises(KeyboardInterrupt):
        first.complete_many(prompts, purpose="t")

    one_call = first.estimate_usd(len("next"), 1024)
    second = fake_claude(
        conn, lambda p: "ok", cap_usd=batch_estimate(first, prompts) + one_call / 2
    )
    second._batch_done = lambda batch_id: False  # still running at the provider
    with pytest.raises(BudgetExceeded):
        second.complete("next", purpose="t")


def test_batch_in_flight_counts_once_against_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "BATCH_POLL_SECONDS", 0)
    conn = connect(tmp_path / "r.db")
    prompts = [f"p{i}" for i in range(51)]
    probe = fake_claude(conn, lambda p: "")
    cap = batch_estimate(probe, prompts) + 2 * probe.estimate_usd(len("inner"), 1024)
    c = fake_claude(conn, lambda p: "ok", cap_usd=cap)
    polls = []

    def done(batch_id):
        polls.append(batch_id)
        if len(polls) == 1:
            c.complete("inner", purpose="t")  # must fit: the batch is reserved, not doubled
        return True

    c._batch_done = done
    c.complete_many(prompts, purpose="t")
    assert ledger_rows(c) == 52


def test_ledger_rows_carry_the_job(tmp_path):
    conn = connect(tmp_path / "r.db")
    claude = fake_claude(conn, lambda p: "ok")
    with claude.for_job("j1", check=lambda: None, log=lambda m: None):
        claude.complete("hi", purpose="answer")
    claude.complete("other", purpose="query")
    assert job_spend(conn, "j1") > 0
    assert conn.execute("select count(*) from ledger where job_id is null")[0][0] == 1


def test_cancel_while_waiting_on_a_batch(tmp_path, monkeypatch):
    conn = connect(tmp_path / "r.db")
    monkeypatch.setattr(llm, "BATCH_POLL_SECONDS", 1)
    claude = fake_claude(conn, lambda p: "ok")
    claude._batch_done = lambda batch_id: False  # type: ignore[method-assign]
    logs: list[str] = []

    def check():
        raise Cancelled()

    with pytest.raises(Cancelled), claude.for_job("j1", check=check, log=logs.append):
        claude.complete_many([f"p{n}" for n in range(51)], purpose="answer")
    assert fakes.CANCELLED
    assert conn.execute("select job_id from pending_batches")[0][0] == "j1"
    assert logs and "submitted, 51 requests" in logs[0]
    assert claude._reserved == 0 and not claude._waiting


def test_job_context_is_per_thread(tmp_path):
    import threading

    conn = connect(tmp_path / "r.db")
    claude = fake_claude(conn, lambda p: "ok")
    with claude.for_job("j1", check=lambda: None, log=lambda m: None):
        t = threading.Thread(target=lambda: claude.complete("elsewhere", purpose="query"))
        t.start()
        t.join()
    assert conn.execute("select count(*) from ledger where job_id is null")[0][0] == 1


def test_failed_cancel_still_reports_cancelled(tmp_path, monkeypatch):
    conn = connect(tmp_path / "r.db")
    monkeypatch.setattr(llm, "BATCH_POLL_SECONDS", 1)
    claude = fake_claude(conn, lambda p: "ok")
    claude._batch_done = lambda batch_id: False  # type: ignore[method-assign]

    def broken(batch_id):
        raise RuntimeError("boom")

    claude._cancel_batch = broken  # type: ignore[method-assign]
    logs: list[str] = []

    def check():
        raise Cancelled()

    with pytest.raises(Cancelled), claude.for_job("j1", check=check, log=logs.append):
        claude.complete_many([f"p{n}" for n in range(51)], purpose="answer")
    assert any("could not cancel Claude batch" in m for m in logs)
    assert claude._reserved == 0
    assert conn.execute("select count(*) from pending_batches")[0][0] == 1


def test_collected_batch_keeps_its_own_job(tmp_path):
    conn = connect(tmp_path / "r.db")
    claude = fake_claude(conn, lambda p: "ok")
    batch_id = claude._submit_batch(
        [
            {
                "custom_id": "k",
                "model": "claude-haiku-5-5",
                "system": "",
                "prompt": "p",
                "max_tokens": 9,
            }
        ]
    )
    conn.execute(
        "insert into pending_batches (id, model, purpose, created_at, usd) values (?, ?, ?, ?, ?)",
        (batch_id, "claude-haiku-5-5", "answer", "2026-01-01", 0.0),
    )
    with claude.for_job("j2", check=lambda: None, log=lambda m: None):
        claude._collect(batch_id, "claude-haiku-5-5", "answer")
    assert conn.execute("select count(*) from ledger where job_id is null")[0][0] == 1
    assert job_spend(conn, "j2") == 0
