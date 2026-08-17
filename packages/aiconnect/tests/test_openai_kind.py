from types import SimpleNamespace

import pytest

from aiconnect import Conflict, ProviderError, wire
from aiconnect.ai import estimate_tokens
from aiconnect.fakes import fake_ai
from aiconnect.wire import openai_client, openai_send


def fake_client(usage=(12, 3), text="hi"):
    """A chat.completions client that records the arguments it was called with."""
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        counts = None
        if usage:
            counts = SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1])
        choice = SimpleNamespace(message=SimpleNamespace(content=text))
        return SimpleNamespace(choices=[choice], usage=counts)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.seen = seen  # type: ignore[attr-defined]
    return client


def test_openai_send_reads_text_and_usage():
    text, tin, tout = openai_send(fake_client(usage=(12, 3), text="hi"), "m", "sys", "q", 50)
    assert (text, tin, tout) == ("hi", 12, 3)


def test_openai_send_without_usage_estimates_tokens():
    out = openai_send(fake_client(usage=None, text="answer"), "m", "sys", "q", 50)
    assert out == ("answer", estimate_tokens(len("sysq")), estimate_tokens(len("answer")))


def test_openai_send_omits_empty_system():
    client = fake_client()
    openai_send(client, "m", "", "q", 50)
    assert client.seen["messages"] == [{"role": "user", "content": "q"}]
    assert client.seen["max_tokens"] == 50 and client.seen["model"] == "m"


def test_openai_send_keeps_system_first():
    client = fake_client()
    openai_send(client, "m", "sys", "q", 50)
    assert client.seen["messages"][0] == {"role": "system", "content": "sys"}


def test_openai_send_tolerates_empty_content():
    assert openai_send(fake_client(text=None), "m", "", "q", 5)[0] == ""


def test_openai_client_needs_no_key():
    assert openai_client("http://x:11434/v1", None).api_key == "unused"


def local_ai(tmp_path, local=True):
    ai = fake_ai(tmp_path / "ai.db", lambda p: "ok")
    ai.add_connection("ollama", "openai", "Ollama", base_url="http://x/v1", local=local)
    return ai


def test_openai_kind_never_batches(tmp_path):
    ai = local_ai(tmp_path)
    ai.add_model("ollama", "llama", price_in=0, price_out=0)
    sent = []

    def send(connection, model_id, system, prompt, max_tokens):
        sent.append(prompt)
        return "ok", 1, 1

    ai._send = send
    ai._submit_batch = lambda *a: pytest.fail("batched")
    out = ai.complete_many([f"p{i}" for i in range(60)], model="ollama/llama", purpose="t")
    assert len(sent) == 60 and len(out) == 60


def test_fetch_models_prices_local_at_zero(tmp_path):
    ai = local_ai(tmp_path)
    ai._list_models = lambda connection: ["a", "b"]
    assert ai.fetch_models("ollama") == ["a", "b"]
    prices = {m["model_id"]: m["price_in"] for m in ai.models() if m["connection_id"] == "ollama"}
    assert prices == {"a": 0, "b": 0}


def test_fetch_models_leaves_remote_unpriced_and_adds_only_new(tmp_path):
    ai = local_ai(tmp_path, local=False)
    ai._list_models = lambda connection: ["a", "b"]
    assert ai.fetch_models("ollama") == ["a", "b"]
    prices = [m["price_in"] for m in ai.models() if m["connection_id"] == "ollama"]
    assert prices == [None, None]
    assert ai.fetch_models("ollama") == []


def test_fetch_models_unknown_connection(tmp_path):
    with pytest.raises(ValueError):
        local_ai(tmp_path).fetch_models("nope")


def test_list_models_error_hides_key(tmp_path, monkeypatch):
    ai = fake_ai(tmp_path / "ai.db", lambda p: "ok", api_key="sk-ant-api03-SECRETxyz9")

    def boom(client):
        raise RuntimeError("bad key sk-ant-api03-SECRETxyz9")

    ai._client = lambda connection: None
    monkeypatch.setitem(wire.LIST_MODELS, "anthropic", boom)
    with pytest.raises(ProviderError) as e:
        ai.fetch_models("anthropic")
    assert "SECRET" not in str(e.value)


def test_test_connection_ok_is_ledgered_under_test(tmp_path):
    ai = local_ai(tmp_path)
    ai.add_model("ollama", "llama", price_in=1, price_out=1)
    out = ai.test_connection("ollama")
    assert out["ok"] is True and out["model"] == "llama" and out["error"] is None
    assert out["usd"] > 0
    assert ai.sql("select purpose from ai_ledger")[0][0] == "test"
    assert ai.test_connection("ollama")["usd"] > 0  # cache=False: paid again


def test_test_connection_reports_unreachable_server(tmp_path):
    ai = fake_ai(tmp_path / "ai.db", lambda p: "ok")
    ai.add_connection("ollama", "openai", "Ollama", base_url="http://x:11434/v1", local=True)
    ai.add_model("ollama", "llama3.1:8b", price_in=0, price_out=0)
    ai._send = lambda *a: (_ for _ in ()).throw(ConnectionError("refused"))
    out = ai.test_connection("ollama")
    assert out["ok"] is False and "refused" in out["error"]


def test_provider_error_hides_key(tmp_path):
    ai = fake_ai(tmp_path / "ai.db", lambda p: "ok", api_key="sk-ant-api03-SECRETxyz9")
    ai._send = lambda *a: (_ for _ in ()).throw(RuntimeError("bad key sk-ant-api03-SECRETxyz9"))
    with pytest.raises(ProviderError) as e:
        ai.complete("hi", purpose="t")
    assert "SECRET" not in str(e.value)


def test_test_connection_without_priced_model_says_so(tmp_path):
    ai = local_ai(tmp_path)
    ai.add_model("ollama", "llama")
    out = ai.test_connection("ollama")
    assert out["ok"] is False and "price" in out["error"] and out["model"] is None


def test_test_connection_without_models_says_so(tmp_path):
    out = local_ai(tmp_path).test_connection("ollama")
    assert out["ok"] is False and "model" in out["error"]


def test_test_connection_is_refused_by_the_cap(tmp_path):
    ai = local_ai(tmp_path)
    ai.add_model("ollama", "llama", price_in=1e9, price_out=1e9)
    out = ai.test_connection("ollama")
    assert out["ok"] is False and "cap" in out["error"]


# carried from Task 2's review


def test_paid_call_is_ledgered_even_if_price_cleared_mid_run(tmp_path):
    holder = {}

    def reply(prompt):
        holder["ai"].update_model("anthropic", "claude-haiku-5-5", price_in=None, price_out=None)
        return "x" * 400

    ai = holder["ai"] = fake_ai(tmp_path / "ai.db", reply)
    out = ai.complete("hello", purpose="t")
    assert out.usd > 0
    assert ai.sql("select count(*) from ai_ledger")[0][0] == 1
    assert ai.month_spend() == pytest.approx(out.usd)


def pending_batch(ai, connection_id="anthropic", model="claude-haiku-5-5"):
    ai.sql(
        "insert into ai_batches (id, model, purpose, created_at, usd, connection_id) values"
        " ('b1', ?, 't', '2026-10-01', 1.0, ?)",
        (model, connection_id),
    )


def test_remove_connection_refused_while_a_batch_is_pending(tmp_path):
    ai = local_ai(tmp_path)
    ai.add_model("ollama", "llama", price_in=0, price_out=0)
    pending_batch(ai, "ollama", "llama")
    with pytest.raises(Conflict, match="batch"):
        ai.remove_connection("ollama")


def test_clearing_a_price_refused_while_a_batch_is_pending(tmp_path):
    ai = fake_ai(tmp_path / "ai.db", lambda p: "ok")
    pending_batch(ai)
    with pytest.raises(Conflict, match="batch"):
        ai.update_model("anthropic", "claude-haiku-5-5", price_in=None)
    ai.update_model("anthropic", "claude-haiku-5-5", price_in=2.0)  # changing it is fine


def test_batch_with_null_connection_blocks_by_model(tmp_path):
    ai = fake_ai(tmp_path / "ai.db", lambda p: "ok")
    ai.sql(
        "insert into ai_batches (id, model, purpose, created_at, usd) values"
        " ('b1', 'claude-haiku-5-5', 't', '2026-10-01', 1.0)"
    )
    ai.set_role("default", None)
    with pytest.raises(Conflict, match="batch"):
        ai.remove_connection("anthropic")
