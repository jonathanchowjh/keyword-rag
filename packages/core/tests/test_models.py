from pathlib import Path

from ragqa_core.models import Models


class Recorder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def query_embed(self, texts):
        self.texts.extend(texts)
        return [[1.0, 0.0] for _ in texts]


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
