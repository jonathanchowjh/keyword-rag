import pytest

from ragqa_core.db import connect
from ragqa_core.fakes import fake_claude
from ragqa_core.llm import BudgetExceeded, NoApiKey, UnknownModel


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
    rows = c.conn.execute("select batch, usd from ledger").fetchall()
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
    return c.conn.execute("select count(*) from ledger").fetchone()[0]


def cache_rows(c):
    return c.conn.execute("select count(*) from cache").fetchone()[0]


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
