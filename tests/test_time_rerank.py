"""scripts/time_rerank.py with a stub encoder: no model is downloaded or run."""

import importlib.util
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "time_rerank.py"
spec = importlib.util.spec_from_file_location("time_rerank", SCRIPT)
assert spec and spec.loader
time_rerank = importlib.util.module_from_spec(spec)
spec.loader.exec_module(time_rerank)
choose_cap = time_rerank.choose_cap
passages = time_rerank.passages


def test_choose_cap_takes_the_faster_batch_then_the_largest_size_at_or_under_two_seconds():
    assert choose_cap({8: {20: 0.8, 40: 1.7, 60: 2.6}, 16: {20: 0.9, 40: 1.9, 60: 2.9}}) == (8, 40)
    assert choose_cap({8: {20: 0.5, 40: 1.1, 60: 2.1}, 16: {20: 0.5, 40: 1.0, 60: 1.8}}) == (
        16,
        60,
    )
    assert choose_cap({8: {20: 2.1, 40: 4.0, 60: 6.0}, 16: {20: 2.3, 40: 4.4, 60: 6.5}}) == (
        8,
        None,
    )


def test_choose_cap_says_which_pool_size_is_missing():
    with pytest.raises(ValueError, match="batch 16 has no timing for pool size 40"):
        choose_cap({8: {20: 0.8, 40: 1.7, 60: 2.6}, 16: {20: 0.9, 60: 2.9}})


class SleepyEncoder:
    """fastembed's rerank is a generator: nothing happens until it is consumed."""

    def rerank(self, query, texts, batch_size=64):
        for _ in texts:
            time.sleep(0.001)
            yield 0.0


def test_main_times_the_consumed_generator(monkeypatch, capsys):
    monkeypatch.setattr(time_rerank, "load_encoder", lambda settings: SleepyEncoder())
    assert time_rerank.main([]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    rows = [line.split() for line in lines if line.split() and line.split()[0].isdigit()]
    assert len(rows) == 6
    assert all(float(row[2]) >= 0.02 for row in rows)
    assert lines[-1].startswith("batch ")


def test_passages_are_1500_characters_and_distinct():
    assert all(len(p) == 1500 for p in passages(60))
    assert len(set(passages(60))) == 60
