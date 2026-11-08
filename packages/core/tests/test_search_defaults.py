import pytest

from ragqa_core import db, search_defaults
from ragqa_core.search_defaults import (
    RECOMMENDED,
    chat_method,
    check,
    in_force,
    save,
    saved,
    source,
)


@pytest.fixture
def conn(tmp_path):
    return db.connect(tmp_path / "ragqa.db")


def test_in_force_is_recommended_until_saved_and_null_resets(conn):
    assert in_force(conn) == RECOMMENDED
    save(conn, {"candidates": 30})
    assert in_force(conn)["candidates"] == 30 and source(conn, "candidates") == "settings"
    save(conn, {"candidates": None})
    assert in_force(conn)["candidates"] == 20 and source(conn, "candidates") == "recommended"


def test_save_checks_every_change_before_writing_any(conn):
    with pytest.raises(ValueError):
        save(conn, {"candidates": 30, "rrf_k": -1})
    assert saved(conn)["candidates"] is None


REFUSED = [
    ("candidates", 0),
    ("candidates", 61),
    ("evidence_size", 21),
    ("follow_up_turns", 11),
    ("rrf_k", 1001),
    ("method", "hybrid_rerank"),
    ("method", "iterative"),
    ("method", "question_only"),
    ("rerank", "yes"),
    ("nope", 1),
]


def test_bounds_and_methods():
    for field, value in REFUSED:
        with pytest.raises(ValueError, match=field):
            check(field, value)
    with pytest.raises(ValueError, match="is not a chat method"):
        check("method", "hybrid_rerank")
    check("candidates", 60)
    check("follow_up_turns", 0)
    check("rrf_k", 0)


def test_rerank_model_and_pool_are_checked(conn, monkeypatch):
    monkeypatch.setattr(search_defaults, "supported_rerankers", lambda: {"a/x": 0.1})
    save(conn, {"rerank_model": "a/x"})
    assert saved(conn)["rerank_model"] == "a/x"
    save(conn, {"rerank_model": None})
    assert saved(conn)["rerank_model"] is None
    with pytest.raises(ValueError, match="unknown reranker nope/y"):
        save(conn, {"rerank_model": "nope/y"})
    for pool in (0, 61):
        with pytest.raises(ValueError, match="rerank_pool"):
            check("rerank_pool", pool)
    check("rerank_pool", 60)


def test_dense_model_is_checked_and_null_is_the_service_model(conn, monkeypatch):
    assert RECOMMENDED["dense_model"] is None
    monkeypatch.setattr(
        search_defaults, "supported_embedders", lambda: {"a/x": {"dim": 8, "size_gb": 0.1}}
    )
    save(conn, {"dense_model": "a/x"})
    assert saved(conn)["dense_model"] == "a/x"
    save(conn, {"dense_model": None})
    assert saved(conn)["dense_model"] is None
    with pytest.raises(ValueError, match="unknown embedding model nope/y"):
        save(conn, {"dense_model": "nope/y"})


def test_the_short_list_is_supported_by_fastembed():
    supported = search_defaults.supported_embedders()
    assert all(name in supported for name in search_defaults.EMBEDDERS)
    assert supported["BAAI/bge-small-en-v1.5"]["dim"] == 384
    assert all(search_defaults.EMBEDDER_NOTES.get(name) for name in search_defaults.EMBEDDERS)


def test_chat_method_reads_an_old_saved_value_as_null(conn):
    assert chat_method("hybrid_rerank") is None
    assert chat_method("keyword_rag") == "keyword_rag"
    assert chat_method(None) is None
    db.set_app_setting(conn, "search.method", "hybrid_rerank")  # saved before it was refused
    assert saved(conn)["method"] is None and in_force(conn)["method"] == "hybrid"


def test_saved_drops_a_stored_value_that_no_longer_checks(conn):
    db.set_app_setting(conn, "search.rerank_model", "gone/model")
    db.set_app_setting(conn, "search.evidence_size", 999)
    got = saved(conn)
    assert got["rerank_model"] is None and got["evidence_size"] is None
    force = in_force(conn)
    assert force["rerank_model"] == RECOMMENDED["rerank_model"]
    assert force["evidence_size"] == RECOMMENDED["evidence_size"]


def test_the_supported_rerankers_are_fastembeds_with_their_sizes():
    rerankers = search_defaults.supported_rerankers()
    assert rerankers["jinaai/jina-reranker-v1-tiny-en"] > 0
    assert all(isinstance(size, float) for size in rerankers.values())


def test_app_settings_are_read_by_prefix_and_null_deletes(conn):
    db.set_app_setting(conn, "search.candidates", 30)
    db.set_app_setting(conn, "other.candidates", 5)
    assert db.get_app_settings(conn, "search.") == {"candidates": 30}
    db.set_app_setting(conn, "search.candidates", None)
    assert db.get_app_settings(conn, "search.") == {}
