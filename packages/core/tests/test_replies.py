"""python -m ragqa_core.replies: waiting prompts out, replies back in."""

import json

import pytest

from aiconnect.ai import AI, AwaitingReplies
from ragqa_core import replies


@pytest.fixture
def ai(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGQA_HOME", str(tmp_path))
    ai = AI(tmp_path / "ragqa.db")
    ai.add_connection("hand", "manual", "Answered by hand")
    ai.add_model("hand", "m", price_in=0.0, price_out=0.0)
    ai.set_role("query", "hand/m")
    for job, prompt in (("j1", "one"), ("j2", "two")):
        with pytest.raises(AwaitingReplies), ai.for_job(job, lambda: None, lambda m: None):
            ai.complete(prompt, role="query", max_tokens=40, purpose="query_tags")
    return ai


def lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_export_answer_import_then_the_call_is_cached(ai, tmp_path, capsys):
    out = tmp_path / "out.jsonl"
    assert replies.main(["export", str(out), "--job", "j1"]) == 0
    [row] = lines(out)
    assert (row["prompt"], row["purpose"], row["max_tokens"], row["job_id"]) == (
        "one",
        "query_tags",
        40,
        "j1",
    )
    back = tmp_path / "replies.jsonl"
    back.write_text(json.dumps({"key": row["key"], "reply": "alpha"}) + "\n\n")
    assert replies.main(["import", str(back)]) == 0
    assert '"still_waiting": 1' in capsys.readouterr().out
    assert ai.complete("one", role="query", max_tokens=40, purpose="query_tags").text == "alpha"


def test_an_import_with_a_stray_or_repeated_key_records_nothing(ai, tmp_path, capsys):
    key = ai.waiting()[0]["key"]
    back = tmp_path / "replies.jsonl"
    back.write_text(
        json.dumps({"key": key, "reply": "a"}) + "\n" + json.dumps({"key": "x", "reply": "b"})
    )
    assert replies.main(["import", str(back)]) == 2
    back.write_text(
        json.dumps({"key": key, "reply": "a"}) + "\n" + json.dumps({"key": key, "reply": "b"})
    )
    assert replies.main(["import", str(back)]) == 2
    assert "refused" in capsys.readouterr().err and len(ai.waiting()) == 2
