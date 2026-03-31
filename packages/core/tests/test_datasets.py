import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ragqa_core.datasets import (
    DATASETS,
    dev_slice,
    load_musique,
    load_scienceqa,
    load_tatqa,
    mask_template,
)
from ragqa_core.types import Question

DATA = Path(__file__).parent / "data"
TABLE_UID = "3ffd9053-a45d-491c-957a-1b2fa0af0570"
IMAGE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def sqa_rows(split: str) -> list[dict]:
    base = {
        "question": "Which force changes the cart's speed by 5 meters?",
        "choices": ["friction", "gravity"],
        "answer": 1,
        "hint": "A cart rolls downhill.",
        "image": None,
        "subject": "natural science",
        "grade": "grade8",
        "topic": "physics",
        "skill": "s1",
        "solution": "Because.",
    }
    return [
        base,
        {**base, "question": "What is in the picture?", "image": {"bytes": b"x", "path": "a.png"}},
        {**base, "question": "Is a rock a living thing?", "grade": "grade3"},
        {**base, "question": "Which is a capital city?", "subject": "social science"},
        {**base, "question": "Which force changes the cart's speed by 9 meters?"},
        {**base, "question": f"A {split} question about magnets?", "skill": "s2", "answer": 0},
    ]


def write_mini_scienceqa(tmp_path: Path) -> dict[str, Path]:
    files = {}
    for split in ("train", "test"):
        rows = sqa_rows(split)
        table = pa.table(
            {
                **{
                    k: [r[k] for r in rows]
                    for k in rows[0]
                    if k not in ("image", "answer", "choices")
                },
                "choices": pa.array([r["choices"] for r in rows], pa.list_(pa.string())),
                "answer": pa.array([r["answer"] for r in rows], pa.int8()),
                "image": pa.array([r["image"] for r in rows], IMAGE),
            }
        )
        files[split] = tmp_path / f"{split}.parquet"
        pq.write_table(table, files[split])
    return files


def test_datasets_map_to_qa_types():
    assert DATASETS == {"scienceqa": "mcq", "musique": "multihop", "tatqa": "table"}


def test_scienceqa_filters_and_dedupes(tmp_path):
    items, questions = load_scienceqa(tmp_path, write_mini_scienceqa(tmp_path))
    assert [i.id for i in items] == ["sqa-train-0", "sqa-train-5"]
    first = items[0]
    assert first.answer == "B" and first.explanation == "Because."
    assert first.meta == {"skill": "s1", "topic": "physics", "grade": "grade8"}
    assert "Context: A cart rolls downhill." in first.text
    assert "Choices: A. friction; B. gravity" in first.text
    assert [q.id for q in questions] == ["sqa-test-0", "sqa-test-5"]
    q = questions[0]
    assert q.gold_answers == ["B"] and q.gold_ids == [] and q.choices == ["friction", "gravity"]
    assert q.meta == {"skill": "s1", "topic": "physics", "solution": "Because."}


def test_musique_gold_ids_point_into_corpus(tmp_path):
    files = {"questions": DATA / "musique_mini.json", "corpus": DATA / "musique_corpus_mini.json"}
    items, questions = load_musique(tmp_path, files)
    ids = {i.id for i in items}
    assert len(ids) == len(items) and all(i.id.startswith("mq-") for i in items)
    assert len(questions[0].gold_ids) == 2 and set(questions[0].gold_ids) <= ids
    assert questions[0].gold_answers[0] == "June 1982"
    first = json.loads(files["corpus"].read_text())[0]
    assert items[0].text == f"{first['title']}\n{first['text']}"


def test_tatqa_renders_table_and_links_gold(tmp_path):
    files = {s: DATA / "tatqa_mini.json" for s in ("train", "dev", "test")}
    items, questions = load_tatqa(tmp_path, files)
    assert len(items) == 2
    assert " | " in items[0].text and "Fixed Price | $  1,452.4" in items[0].text
    assert "Sales by Contract Type" in items[0].text
    assert questions[0].gold_ids == ["tq-" + TABLE_UID]
    assert questions[1].gold_answers == ["$1,496.5"] and questions[1].meta == {"scale": "million"}
    assert questions[3].gold_answers == ["2019, 2018, 2017"]


def test_dev_slice_is_order_independent():
    qs = [Question(f"id{i}", "q", None, [], [], {}) for i in range(500)]
    forward = [x.id for x in dev_slice(qs)]
    assert forward == [x.id for x in dev_slice(list(reversed(qs)))]
    assert len(forward) == 200 and len(dev_slice(qs[:50])) == 50
    assert [x.id for x in dev_slice(qs[:3])] == ["id0", "id1", "id2"]


def test_dev_slice_is_stable_when_questions_are_added():
    qs = [Question(f"id{i}", "q", None, [], [], {}) for i in range(500)]
    before = {q.id for q in dev_slice(qs)}
    after = {q.id for q in dev_slice(qs + [Question("new", "q", None, [], [], {})])}
    assert len(before - after) <= 1  # one new question displaces at most one


def test_mask_template():
    a = mask_template("Which is heavier, 5 kg or 7 kg? Ask Maria.")
    assert a == mask_template("Which is heavier, 2 kg or 9 kg? Ask Pedro.")
    assert a != mask_template("Which is lighter, 5 kg or 7 kg? Ask Maria.")


def test_tatqa_fixture_is_real_format():
    doc = json.loads((DATA / "tatqa_mini.json").read_text())[0]
    assert {"table", "paragraphs", "questions"} <= set(doc)
