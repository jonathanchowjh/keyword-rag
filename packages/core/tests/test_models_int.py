import numpy as np
import pytest

from ragqa_core.models import Models

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    return Models(
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-small-en-v1.5",
        "Xenova/ms-marco-MiniLM-L-6-v2",
        tmp_path_factory.mktemp("models"),
    )


def test_dense(models):
    assert models.dense_dim() == 384
    vecs = models.embed(["a", "b"])
    assert vecs.shape == (2, 384)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    assert models.embed_query(["a"]).shape == (1, 384)


def test_rerank_prefers_the_relevant_text(models):
    scores = models.rerank("cat", ["the cat sat on the mat", "stock market prices"])
    assert scores[0] > scores[1]


def test_sparse(models):
    assert models.sparse(["hello world"])[0].indices
    assert models.sparse_query(["hello"])[0].indices
