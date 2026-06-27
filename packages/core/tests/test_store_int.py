import os
import uuid

import pytest

from ragqa_core.fakes import fake_models
from ragqa_core.store import Store
from ragqa_core.types import Item

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("RAGQA_TEST_QDRANT_URL"), reason="RAGQA_TEST_QDRANT_URL unset"
    ),
]


@pytest.fixture
def store_and_name():
    store = Store(os.environ["RAGQA_TEST_QDRANT_URL"], None)
    name = f"test_{uuid.uuid4().hex[:8]}"
    yield store, name
    store.drop(name)


def test_server_dense_sparse_and_keywords(store_and_name):
    s, name = store_and_name
    m = fake_models()
    items = [Item("1", "enzyme kinetics rate"), Item("2", "stock market prices")]
    texts = [i.text for i in items]
    s.create(name, m.dense_dim())
    s.upsert_items(name, items, m.embed(texts), m.sparse(texts), [None, None])
    assert s.search(name, "text_dense", m.embed_query(["enzyme rate"])[0], 1)[0].id == "1"
    assert s.search_sparse(name, m.sparse_query(["market"])[0], 1)[0].id == "2"
    s.set_keywords(name, ["1", "2"], [{"topic": ["b"]}, {}], {"kw_topic": m.embed(texts)})
    assert s.search(name, "kw_topic", m.embed(["stock"])[0], 1)[0].id == "2"
    s.upsert_chunks(name, ["1"], ["enzyme kinetics"], m.embed(["enzyme kinetics"]), [None])
    assert s.search_chunks(name, m.embed(["enzyme"])[0], 1)[0][0] == "1"


def test_source_counts_after_ensure_indexes_on_a_collection_made_without_them(store_and_name):
    s, name = store_and_name
    m = fake_models()
    s.create(name, m.dense_dim())
    for key in ("meta.source",):  # as a collection made before the index existed
        s.client.delete_payload_index(name, key)
    s.upsert_items(name, [Item("1", "a b c")], m.embed(["a b c"]), m.sparse(["a b c"]), [None])
    s.set_payloads(name, {"1": {"meta": {"source": "hand"}}})
    with pytest.raises(Exception, match="index"):
        s.source_counts(name)
    s.ensure_indexes(name)
    assert s.source_counts(name) == {"hand": 1}


def test_source_counts_work_on_a_fresh_collection_and_an_empty_one(store_and_name):
    s, name = store_and_name
    s.create(name, fake_models().dense_dim())
    assert s.source_counts(name) == {}
