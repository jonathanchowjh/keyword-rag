from pathlib import Path

import pytest

from ragqa_core.settings import load_settings


def test_defaults():
    s = load_settings({})
    assert s.qdrant_url is None
    assert s.home == Path("~/.ragqa").expanduser()
    assert s.judge_model == "BAAI/bge-small-en-v1.5"


def test_the_default_embedding_model_is_bge_small():
    assert load_settings({}).dense_model == "BAAI/bge-small-en-v1.5"
    assert load_settings({"RAGQA_DENSE_MODEL": "a/x"}).dense_model == "a/x"


def test_the_default_reranker_is_jina_tiny():
    assert load_settings({}).rerank_model == "jinaai/jina-reranker-v1-tiny-en"
    assert load_settings({"RAGQA_RERANK_MODEL": "a/x"}).rerank_model == "a/x"


def test_reads_every_variable():
    s = load_settings(
        {
            "ANTHROPIC_API_KEY": "k",
            "RAGQA_MONTHLY_CAP_USD": "5.5",
            "RAGQA_MODEL": "claude-sonnet-5-5",
            "RAGQA_QDRANT_URL": "http://qdrant:6333",
            "RAGQA_HOME": "/data",
            "RAGQA_USERNAME": "u",
            "RAGQA_PASSWORD": "p",
            "RAGQA_DENSE_MODEL": "d",
            "RAGQA_JUDGE_MODEL": "j",
            "RAGQA_RERANK_MODEL": "r",
        }
    )
    assert (s.api_key, s.cap_usd, s.model, s.qdrant_url) == (
        "k",
        5.5,
        "claude-sonnet-5-5",
        "http://qdrant:6333",
    )
    assert (s.home, s.username, s.password) == (Path("/data"), "u", "p")
    assert (s.dense_model, s.judge_model, s.rerank_model) == ("d", "j", "r")


def test_empty_strings_are_unset():
    s = load_settings({"ANTHROPIC_API_KEY": "", "RAGQA_QDRANT_URL": ""})
    assert s.api_key is None and s.qdrant_url is None


def test_unset_ai_settings_are_none():
    # Unset, the values stored through the Settings page stand.
    s = load_settings({})
    assert (s.api_key, s.model, s.cap_usd) == (None, None, None)
    s = load_settings({"ANTHROPIC_API_KEY": "", "RAGQA_MODEL": "", "RAGQA_MONTHLY_CAP_USD": ""})
    assert (s.api_key, s.model, s.cap_usd) == (None, None, None)


@pytest.mark.parametrize("cap", ["-1", "x"])
def test_set_cap_still_validated(cap):
    with pytest.raises(ValueError):
        load_settings({"RAGQA_MONTHLY_CAP_USD": cap})


def test_the_build_hub_is_on_by_default_and_empty_turns_it_off():
    from ragqa_core.settings import HUB_REPO

    assert load_settings({}).build_hub == HUB_REPO
    assert load_settings({"RAGQA_BUILD_HUB": "me/builds"}).build_hub == "me/builds"
    assert load_settings({"RAGQA_BUILD_HUB": ""}).build_hub == ""
