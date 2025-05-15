from pathlib import Path

import pytest

from ragqa_api.config import load_settings


def test_defaults():
    s = load_settings({})
    assert s.cap_usd == 20.0 and s.model == "claude-haiku-5-5" and s.port == 8080
    assert s.api_key is None and s.qdrant_url is None
    assert s.home == Path("~/.ragqa").expanduser()
    assert s.dense_model == "Qwen/Qwen3-Embedding-0.6B-Q"
    assert s.judge_model == "BAAI/bge-small-en-v1.5"
    assert s.rerank_model == "BAAI/bge-reranker-v2-m3-int8"


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
            "PORT": "9000",
        }
    )
    assert (s.api_key, s.cap_usd, s.model, s.qdrant_url) == (
        "k",
        5.5,
        "claude-sonnet-5-5",
        "http://qdrant:6333",
    )
    assert (s.home, s.username, s.password, s.port) == (Path("/data"), "u", "p", 9000)
    assert (s.dense_model, s.judge_model, s.rerank_model) == ("d", "j", "r")


def test_empty_strings_are_unset():
    s = load_settings({"ANTHROPIC_API_KEY": "", "RAGQA_QDRANT_URL": ""})
    assert s.api_key is None and s.qdrant_url is None


@pytest.mark.parametrize("cap", ["-1", "lots"])
def test_bad_cap(cap):
    with pytest.raises(ValueError):
        load_settings({"RAGQA_MONTHLY_CAP_USD": cap})
