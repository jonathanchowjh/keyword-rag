from pathlib import Path

import numpy as np
import pytest

from ragqa_core.fakes import fake_models
from ragqa_core.models import Models


class Recorder:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.batch_sizes: list[int | None] = []

    def query_embed(self, texts, batch_size=None):
        self.texts.extend(texts)
        self.batch_sizes.append(batch_size)
        return [[1.0, 0.0] for _ in texts]

    def embed(self, texts, batch_size=None):
        return self.query_embed(texts, batch_size)


def test_every_dense_call_names_a_small_batch():
    # fastembed's default batch is 256: one call of 64 long documents became one padded
    # ONNX batch of ~6 GB, which killed the 7.6 GB container on a BRIGHT load.
    from ragqa_core.models import EMBED_BATCH

    models, recorder = models_with("Qwen/Qwen3-Embedding-0.6B-Q")
    models.embed(["a"] * 64)
    models.embed_query(["q"])
    models.judge(["b"])
    assert recorder.batch_sizes == [EMBED_BATCH] * 3 and EMBED_BATCH <= 8


def models_with(dense: str) -> tuple[Models, Recorder]:
    models, recorder = Models(dense, "j", "r", Path("/nonexistent")), Recorder()
    models._load = lambda kind, name: recorder  # type: ignore[method-assign]
    return models, recorder


def test_qwen3_queries_get_the_instruction_prefix():
    models, recorder = models_with("Qwen/Qwen3-Embedding-0.6B-Q")
    models.embed_query(["what is rate?"], task="Find similar questions")
    assert recorder.texts == ["Instruct: Find similar questions\nQuery:what is rate?"]


def test_models_trained_without_instructions_get_none():
    models, recorder = models_with("BAAI/bge-small-en-v1.5")
    models.embed_query(["what is rate?"], task="Find similar questions")
    assert recorder.texts == ["what is rate?"]
    assert models.query_instruction("Find similar questions") == ""


def test_no_task_means_no_prefix():
    models, recorder = models_with("Qwen/Qwen3-Embedding-0.6B-Q")
    models.embed_query(["q"])
    assert recorder.texts == ["q"]


def test_dense_calls_load_the_model_asked_for():
    models, loaded = Models("d", "j", "r", Path("/nonexistent")), []

    class Stub:
        def embed(self, texts, batch_size=None):
            return [np.ones(4) for _ in texts]

        def query_embed(self, texts, batch_size=None):
            return [np.ones(4) for _ in texts]

    def load(kind, name):
        loaded.append((kind, name))
        return Stub()

    models._load = load  # type: ignore[method-assign]
    models.embed(["a"], model="m2")
    models.embed(["a"])
    models.embed_query(["a"], "t", "m2")
    assert loaded == [("text", "m2"), ("text", models.dense_model), ("text", "m2")]


def test_query_instruction_follows_the_named_model():
    models = Models("BAAI/bge-small-en-v1.5", "j", "r", Path("/nonexistent"))
    assert models.query_instruction("Find it") == ""
    qwen = "Qwen/Qwen3-Embedding-0.6B-Q"
    assert models.query_instruction("Find it", qwen) == "Instruct: Find it\nQuery:"


def test_dense_dim_of_a_named_model():
    # fastembed's static list: nothing downloads.
    models = Models("d", "j", "r", Path("/nonexistent"))
    assert models.dense_dim("BAAI/bge-small-en-v1.5") == 384
    assert models.dense_dim("Qwen/Qwen3-Embedding-0.6B-Q") == 1024
    with pytest.raises(ValueError):
        models.dense_dim("nope/x")


def test_dense_dim_reads_fastembeds_list_once(monkeypatch):
    from ragqa_core import models as models_module

    models_module.supported_dims.cache_clear()
    calls = []
    real = models_module.TextEmbedding.list_supported_models
    monkeypatch.setattr(
        models_module.TextEmbedding,
        "list_supported_models",
        staticmethod(lambda: calls.append(1) or real()),
    )
    models = Models("d", "j", "r", Path("/nonexistent"))
    for _ in range(3):
        assert models.dense_dim("BAAI/bge-small-en-v1.5") == 384
    assert len(calls) == 1


def test_fake_embeddings_give_each_model_its_dimension():
    m = fake_models(dims={"fake/small": 64})
    assert m.embed(["a b"], "fake/small").shape == (1, 64)
    assert m.embed(["a b"]).shape == (1, 256)
    assert m.dense_dim("fake/small") == 64


def test_a_model_is_built_once_when_two_threads_ask_at_once(monkeypatch):
    import threading
    import time

    from ragqa_core import models as models_module

    built = []

    class Slow:
        def __init__(self, model_name, cache_dir):
            built.append(model_name)
            time.sleep(0.05)  # long enough for the other thread to miss the cache too

    monkeypatch.setattr(models_module, "TextEmbedding", Slow)
    models = Models("d", "j", "r", Path("/nonexistent"))
    got = []
    threads = [
        threading.Thread(target=lambda: got.append(models._load("text", "d"))) for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert built == ["d"] and got[0] is got[1]


class Kwargs:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    def rerank(self, query, texts, **kwargs):
        self.kwargs = kwargs
        return [0.0 for _ in texts]


def test_rerank_passes_an_explicit_batch_size():
    from ragqa_core.models import RERANK_BATCH

    models, recorder = Models("d", "j", "r", Path("/nonexistent")), Kwargs()
    models._load = lambda kind, name: recorder  # type: ignore[method-assign]
    models.rerank("q", ["a", "b"])
    assert recorder.kwargs["batch_size"] == RERANK_BATCH


def test_rerank_uses_the_model_asked_for():
    models, recorder, loaded = Models("d", "j", "r", Path("/nonexistent")), Kwargs(), []

    def load(kind, name):
        loaded.append((kind, name))
        return recorder

    models._load = load  # type: ignore[method-assign]
    models.rerank("q", ["a"], model="m2")
    models.rerank("q", ["a"])
    assert loaded == [("rerank", "m2"), ("rerank", models.rerank_model)]


def test_warm_up_state_is_per_model():
    models = Models("d", "j", "r", Path("/nonexistent"))

    def build(kind, name):
        if name == "a":
            raise RuntimeError("hub down")
        return object()

    models._build = build  # type: ignore[method-assign]
    models.warm_rerank("a")
    assert models.rerank_status("a") == "failed" and models.rerank_status("b") == "idle"
    models.warm_rerank("b")
    assert models.rerank_status("b") == "ready" and models.rerank_status("a") == "failed"
    assert set(models.rerank_errors) == {"a"} and set(models.rerank_failed_at) == {"a"}
    assert models.rerank_status() == "idle"  # the service's own reranker, "r"


def test_a_failed_warm_up_is_failed_not_raised_and_is_retried():
    models = Models("d", "j", "r", Path("/nonexistent"))

    def down(kind, name):
        raise RuntimeError("hub down")

    models._build = down  # type: ignore[method-assign]
    models.warm_rerank()
    assert models.rerank_status() == "failed"
    assert "hub down" in models.rerank_errors["r"]
    models.warm_rerank_in_background()
    assert "r" not in models._warm_threads  # failed under 5 minutes ago: no retry yet
    models.warm_rerank_in_background("other")  # the guard is per model
    models._warm_threads["other"].join(timeout=5)
    assert models.rerank_status("other") == "failed"

    models._build = lambda kind, name: object()  # type: ignore[method-assign]
    models.rerank_failed_at["r"] -= 301
    models.warm_rerank_in_background()
    models._warm_threads["r"].join(timeout=5)
    assert models.rerank_status() == "ready" and "r" not in models.rerank_errors


def test_retry_rerank_skips_the_wait_after_a_failure():
    models = Models("d", "j", "r", Path("/nonexistent"))

    def down(kind, name):
        raise RuntimeError("hub down")

    models._build = down  # type: ignore[method-assign]
    models.warm_rerank()
    models.warm_rerank_in_background()
    assert "r" not in models._warm_threads  # the guard holds a background warm-up back
    models._build = lambda kind, name: object()  # type: ignore[method-assign]
    models.retry_rerank()  # someone picked it again: no 5-minute wait
    models._warm_threads["r"].join(timeout=5)
    assert models.rerank_status() == "ready" and "r" not in models.rerank_errors


def test_a_warm_up_failing_without_a_message_is_still_failed():
    models = Models("d", "j", "r", Path("/nonexistent"))

    def down(kind, name):
        raise RuntimeError()

    models._build = down  # type: ignore[method-assign]
    models.warm_rerank()
    assert models.rerank_status() == "failed" and models.rerank_errors["r"] == "RuntimeError"


def test_rerank_status_is_loading_while_the_warm_up_runs():
    import threading

    models, release = Models("d", "j", "r", Path("/nonexistent")), threading.Event()

    def slow(kind, name):
        release.wait(5)
        return object()

    models._build = slow  # type: ignore[method-assign]
    assert models.rerank_status() == "idle"
    models.warm_rerank_in_background()
    assert models.rerank_status() == "loading" and models.rerank_status("other") == "idle"
    release.set()
    models._warm_threads["r"].join(timeout=5)
    assert models.rerank_status() == "ready"


def test_a_reranker_warm_up_does_not_block_loading_another_model():
    import threading

    models, release, stub = Models("d", "j", "r", Path("/nonexistent")), threading.Event(), object()

    def build(kind, name):
        if kind == "rerank":
            release.wait(5)
        return stub

    models._build = build  # type: ignore[method-assign]
    models.warm_rerank_in_background()
    assert models.rerank_status() == "loading"
    got = []
    loader = threading.Thread(target=lambda: got.append(models._load("text", "fake/dense")))
    loader.start()
    loader.join(timeout=1)
    assert got == [stub] and models.rerank_status() == "loading"
    release.set()
    models._warm_threads["r"].join(timeout=5)
    assert models.rerank_status() == "ready"
