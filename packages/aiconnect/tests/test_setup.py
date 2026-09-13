import json
import os
import sqlite3

import pytest

from aiconnect import AI, Conflict, UnknownModel


def test_new_ai_has_no_connections_and_default_cap(tmp_path):
    ai = AI(tmp_path / "ai.db")
    assert ai.connections() == [] and ai.cap_usd == 20.0


def test_key_is_masked_and_file_private(tmp_path):
    ai = AI(tmp_path / "ai.db")
    ai.add_connection("anthropic", "anthropic", "Anthropic", api_key="sk-ant-api03-abcdWXYZ")
    assert ai.connections()[0]["key"] == "sk-…WXYZ"
    assert "abcd" not in json.dumps(ai.connections())
    assert (tmp_path / "ai.db").stat().st_mode & 0o777 == 0o600


def test_short_key_shows_only_set(tmp_path):
    ai = AI(tmp_path / "ai.db")
    ai.add_connection("a", "openai", "A", api_key="abc")
    ai.add_connection("b", "openai", "B")
    assert [c["key"] for c in ai.connections()] == ["set", None]


def test_model_ref_splits_on_first_slash(tmp_path):
    ai = AI(tmp_path / "ai.db")
    ai.add_connection("or", "openai", "OpenRouter", base_url="https://openrouter.ai/api/v1")
    ai.add_model("or", "meta-llama/llama-3.1-8b", price_in=0.02, price_out=0.05)
    assert ai.resolve("or/meta-llama/llama-3.1-8b") == ("or", "meta-llama/llama-3.1-8b")
    assert ai.resolve("meta-llama/llama-3.1-8b") == ("or", "meta-llama/llama-3.1-8b")


def test_bare_id_on_two_connections_is_ambiguous(tmp_path):
    ai = AI(tmp_path / "ai.db")
    for cid in ("one", "two"):
        ai.add_connection(cid, "openai", cid)
        ai.add_model(cid, "gpt-x")
    with pytest.raises(UnknownModel, match="one/gpt-x.*two/gpt-x"):
        ai.resolve("gpt-x")
    assert ai.resolve("two/gpt-x") == ("two", "gpt-x")


def test_unknown_kind_and_bad_id_are_value_errors(tmp_path):
    ai = AI(tmp_path / "ai.db")
    with pytest.raises(ValueError):
        ai.add_connection("ok", "grpc", "X")
    with pytest.raises(ValueError):
        ai.add_connection("Has Space", "openai", "X")


def test_duplicate_connection_is_conflict(tmp_path):
    ai = AI(tmp_path / "ai.db")
    ai.add_connection("a", "openai", "A")
    with pytest.raises(Conflict):
        ai.add_connection("a", "openai", "A")


def test_role_falls_back_to_default(tmp_path):
    ai = AI(tmp_path / "ai.db")
    assert ai.role_model("answer") is None
    ai.add_connection("a", "openai", "A")
    ai.add_model("a", "m1")
    ai.add_model("a", "m2")
    ai.set_role("default", "a/m1")
    assert ai.role_model("answer") == "a/m1"
    ai.set_role("answer", "m2")
    assert ai.role_model("answer") == "a/m2" and ai.role_model("summary") == "a/m1"
    ai.set_role("answer", None)
    assert ai.roles() == {"default": "a/m1"}


def test_remove_connection_used_by_role_is_conflict(tmp_path):
    ai = AI(tmp_path / "ai.db")
    ai.add_connection("a", "openai", "A")
    ai.add_model("a", "m")
    ai.set_role("answer", "a/m")
    with pytest.raises(Conflict, match="answer"):
        ai.remove_connection("a")
    ai.set_role("answer", None)
    ai.remove_connection("a")
    assert ai.connections() == [] and ai.models() == []


def test_negative_cap_and_price_are_value_errors(tmp_path):
    ai = AI(tmp_path / "ai.db")
    with pytest.raises(ValueError):
        ai.set_cap(-1)
    ai.add_connection("a", "openai", "A")
    with pytest.raises(ValueError):
        ai.add_model("a", "m", price_in=-0.1)


def test_nan_inf_and_empty_label_are_value_errors(tmp_path):
    ai = AI(tmp_path / "ai.db")
    ai.add_connection("a", "openai", "A")
    ai.add_model("a", "m")
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            ai.set_cap(bad)
        with pytest.raises(ValueError):
            ai.update_model("a", "m", price_in=bad)
    with pytest.raises(ValueError):
        ai.update_connection("a", label="")
    assert ai.connections()[0]["label"] == "A"


def test_env_values_win_and_are_read_only(tmp_path):
    ai = AI(tmp_path / "ai.db")
    ai.apply_env(api_key="k-1234567890ab", model="claude-sonnet-5-5", cap_usd=5.0)
    assert {m["ref"] for m in ai.models()} >= {
        "anthropic/claude-haiku-5-5",
        "anthropic/claude-sonnet-5-5",
    }
    assert ai.role_model("answer") == "anthropic/claude-sonnet-5-5" and ai.cap_usd == 5.0
    assert set(ai.connections()[0]["from_env"]) == {"api_key", "base_url"}
    with pytest.raises(Conflict):
        ai.update_connection("anthropic", api_key="other")
    with pytest.raises(Conflict):
        ai.set_cap(9)


def test_env_absent_keeps_stored_values(tmp_path):
    ai = AI(tmp_path / "ai.db")
    ai.set_cap(7)
    ai.apply_env(api_key=None, model=None, cap_usd=None)
    assert ai.cap_usd == 7.0 and ai.connections() == []


def test_env_key_without_model_uses_preset_default(tmp_path):
    ai = AI(tmp_path / "ai.db")
    ai.apply_env(api_key="k-1234567890ab", model=None, cap_usd=None)
    assert ai.role_model("default") == "anthropic/claude-haiku-5-5"
    ai.set_role("default", "claude-sonnet-5-5")  # not locked, and a restart keeps the choice
    ai.apply_env(api_key="k-1234567890ab", model=None, cap_usd=None)
    assert ai.role_model("default") == "anthropic/claude-sonnet-5-5"


def test_env_model_names_a_stored_connections_model(tmp_path):
    ai = AI(tmp_path / "ai.db")
    ai.add_connection("ollama", "openai", "Ollama", local=True)
    ai.add_model("ollama", "llama3.1:8b", price_in=0, price_out=0)
    ai.apply_env(api_key="k-1234567890ab", model="ollama/llama3.1:8b", cap_usd=None)
    assert ai.role_model("default") == "ollama/llama3.1:8b" and ai.ready()
    assert "anthropic/ollama/llama3.1:8b" not in {m["ref"] for m in ai.models()}


def test_env_bare_model_id_resolves_to_its_connection(tmp_path):
    ai = AI(tmp_path / "ai.db")
    ai.add_connection("ollama", "openai", "Ollama", local=True)
    ai.add_model("ollama", "llama3.1:8b", price_in=0, price_out=0)
    ai.apply_env(api_key="k-1234567890ab", model="llama3.1:8b", cap_usd=None)
    assert ai.role_model("default") == "ollama/llama3.1:8b" and ai.ready()
    assert all(
        m["model_id"] != "llama3.1:8b" for m in ai.models() if m["connection_id"] == "anthropic"
    )


@pytest.mark.parametrize("value", ["claude-opus-9", "anthropic/claude-opus-9"])
def test_env_model_named_nowhere_is_added_to_the_anthropic_connection(tmp_path, value):
    ai = AI(tmp_path / "ai.db")
    ai.apply_env(api_key="k-1234567890ab", model=value, cap_usd=None)
    assert ai.role_model("default") == "anthropic/claude-opus-9"
    added = next(m for m in ai.models() if m["ref"] == "anthropic/claude-opus-9")
    assert added["price_in"] is None  # needs a price before use
    assert "anthropic/anthropic/claude-opus-9" not in {m["ref"] for m in ai.models()}


def test_env_model_without_any_connection_is_still_the_default(tmp_path):
    ai = AI(tmp_path / "ai.db")
    ai.apply_env(api_key=None, model="claude-haiku-5-5", cap_usd=None)
    assert ai.role_model("default") == "anthropic/claude-haiku-5-5" and not ai.ready()
    assert ai.connections() == [] and ai.models() == []
    with pytest.raises(Conflict):
        ai.set_role("default", None)


def test_env_model_is_added_to_a_stored_anthropic_connection(tmp_path):
    ai = AI(tmp_path / "ai.db")
    ai.add_connection("anthropic", "anthropic", "A", api_key="sk-stored-123456")
    ai.apply_env(api_key=None, model="claude-opus-9", cap_usd=None)
    assert ai.role_model("default") == "anthropic/claude-opus-9"
    assert "anthropic/claude-opus-9" in {m["ref"] for m in ai.models()}


def test_env_model_locks_the_default_role(tmp_path):
    ai = AI(tmp_path / "ai.db")
    ai.apply_env(api_key="k-1234567890ab", model="claude-sonnet-5-5", cap_usd=None)
    assert ai.env_default
    with pytest.raises(Conflict, match="environment"):
        ai.set_role("default", "anthropic/claude-haiku-5-5")
    ai.set_role("answer", "anthropic/claude-haiku-5-5")  # other jobs stay editable


def test_wal_files_made_by_another_connection_become_private(tmp_path):
    path = tmp_path / "ai.db"
    other = sqlite3.connect(path)  # the app's own connection opens the file first
    other.execute("pragma journal_mode=wal")
    other.execute("create table t (x)")
    files = [path, tmp_path / "ai.db-wal", tmp_path / "ai.db-shm"]
    for file in files:
        os.chmod(file, 0o644)
    AI(path)
    assert [file.stat().st_mode & 0o777 for file in files] == [0o600] * 3
    other.close()


def test_new_base_url_without_a_new_key_drops_the_stored_key(tmp_path):
    ai = AI(tmp_path / "ai.db")
    ai.add_connection("c", "openai", "C", base_url="https://a/v1", api_key="sk-stored-123456")
    ai.update_connection("c", label="C2")
    ai.update_connection("c", base_url="https://a/v1")  # the same address keeps it
    assert ai.connections()[0]["key"] == "sk-…3456"
    ai.update_connection("c", base_url="https://b/v1")
    assert ai.connections()[0]["key"] is None
    ai.update_connection("c", base_url="https://c/v1", api_key="sk-other-654321")
    assert ai.connections()[0]["key"] == "sk-…4321"
    assert ai.connections()[0]["base_url"] == "https://c/v1"


def test_env_keyed_connection_keeps_its_address(tmp_path):
    ai = AI(tmp_path / "ai.db")
    ai.add_connection("anthropic", "anthropic", "A", base_url="http://proxy")
    ai.apply_env(api_key="k-1234567890ab", model=None, cap_usd=None)
    assert ai.connections()[0]["base_url"] is None  # the env key goes to the preset's address
    with pytest.raises(Conflict, match="environment"):
        ai.update_connection("anthropic", base_url="http://elsewhere")
    assert ai.connections()[0]["key"] == "k-1…90ab"


def test_env_connection_cannot_be_removed(tmp_path):
    ai = AI(tmp_path / "ai.db")
    ai.apply_env(api_key="k-1234567890ab", model=None, cap_usd=None)
    ai.set_role("default", None)  # nothing uses it now
    with pytest.raises(Conflict, match="environment"):
        ai.remove_connection("anthropic")
    assert [c["id"] for c in ai.connections()] == ["anthropic"]
