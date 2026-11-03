from pathlib import Path

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
