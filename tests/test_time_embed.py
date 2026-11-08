"""scripts/time_embed.py with a stub model: no model is downloaded or run."""

import importlib.util
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "time_embed.py"
spec = importlib.util.spec_from_file_location("time_embed", SCRIPT)
assert spec and spec.loader
time_embed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(time_embed)
passages = time_embed.passages


def test_passages_are_1500_characters_and_distinct():
    assert all(len(p) == 1500 for p in passages(1000))
    assert len(set(passages(1000))) == 1000


class SleepyModel:
    """fastembed's embed is a generator: nothing happens until it is consumed."""

    def __init__(self, name):
        self.model_name = name

    def embed(self, texts, batch_size=256):
        for _ in texts:
            time.sleep(0.0001)
            yield [0.0] * 4

    def query_embed(self, queries):
        for _ in queries:
            time.sleep(0.0001)
            yield [0.0] * 4


def test_main_times_the_consumed_generator_for_each_model(monkeypatch, capsys):
    monkeypatch.setattr(time_embed, "load_model", lambda name, settings: SleepyModel(name))
    assert time_embed.main(["a/x", "b/y"]) == 0
    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    rows = [row for row in rows if row and row[0] in ("a/x", "b/y")]
    assert [row[0] for row in rows] == ["a/x", "b/y"]
    assert all(float(row[3]) >= 0.1 for row in rows)
