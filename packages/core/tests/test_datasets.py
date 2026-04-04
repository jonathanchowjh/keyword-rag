import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ragqa_core.datasets import (
    DATASETS,
    dev_slice,
    load_musique,
    load_scibench,
    load_scienceqa,
    load_tatqa,
    mask_template,
    split_unit,
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
    assert DATASETS == {
        "scienceqa": "mcq",
        "musique": "multihop",
        "tatqa": "table",
        "scibench": "science",
    }


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


SCIBENCH = DATA / "scibench_mini"


def scibench_files() -> dict[str, Path]:
    return {p.name: p for p in SCIBENCH.glob("*.json")}


def test_split_unit_folds_the_power_of_ten():
    assert split_unit(" $10^{-35} \\mathrm{~J} \\mathrm{~s}$") == (-35, "J s")
    assert split_unit(" $10^6$ m") == (6, "m")
    assert split_unit("$10^{42}$") == (42, "")
    assert split_unit(" 10^{-5} \\mathrm{~m}") == (-5, "m")
    assert split_unit(" $\\mathrm{~kJ} \\mathrm{~mol}^{-1}$") == (0, "kJ mol^{-1}")
    assert split_unit(" ") == (0, "")


def test_load_scibench_splits_solved_problems_into_items(tmp_path):
    items, questions = load_scibench(tmp_path, scibench_files())
    solved = {r["problem_text"] for s in ("fund", "atkins") for r in read(f"{s}_sol.json")}
    texts = {r["problem_text"]: r for s in ("fund", "atkins") for r in read(f"{s}.json")}
    assert len(items) + len(questions) == len(solved) + len(texts)
    assert sum(i.explanation is not None for i in items) == 2
    assert not {i.id for i in items} & {q.id for q in questions}
    assert questions and all(q.choices is None and q.gold_ids == [] for q in questions)
    for q in questions:
        row = texts[q.text.split("\nGive the answer in")[0]]
        exp, plain = split_unit(row["unit"])
        assert q.gold_answers == [format(float(row["answer_number"]) * 10**exp, ".6g")]
        assert q.text.endswith(f"\nGive the answer in {plain}.") == bool(plain)
        assert q.meta["topic"] == row["source"]


def read(name: str) -> list[dict]:
    return json.loads((SCIBENCH / name).read_text())


def test_scibench_skill_is_textbook_and_chapter(tmp_path):
    cases = [
        ("class", " Problem 3.40", "class:3"),
        ("atkins", " e1.17(a)(a)", "atkins:1"),
        ("atkins", "p2.3(a)", "atkins:2"),
        ("fund", " Question 21.75", "fund:21"),
        ("chemmc", "D-7", None),
    ]
    rows: dict[str, list[dict]] = {}
    for n, (source, problemid, _) in enumerate(cases):
        rows.setdefault(source, []).append(
            {
                "problem_text": f"problem {n}",
                "answer_latex": "1",
                "answer_number": "1",
                "unit": " ",
                "source": source,
                "problemid": problemid,
                "comment": " ",
                "solution": "",
            }
        )
    files = {}
    for source, source_rows in rows.items():
        files[f"{source}_sol.json"] = tmp_path / f"{source}_sol.json"
        files[f"{source}_sol.json"].write_text(json.dumps(source_rows))
    items, _ = load_scibench(tmp_path, files)
    by_text = {i.text: i for i in items}
    for n, (_, _, skill) in enumerate(cases):
        meta = by_text[f"problem {n}"].meta
        assert meta.get("skill") == skill
        assert ("skill" in meta) == (skill is not None)


def test_scibench_power_of_ten_unit_is_folded_into_the_gold(tmp_path):
    items, questions = load_scibench(tmp_path, scibench_files())
    assert "2e+06 m" in {i.answer for i in items}
    assert ["4.16e+42"] in [q.gold_answers for q in questions]


def test_scibench_meta_and_split_are_stable(tmp_path):
    first = load_scibench(tmp_path, scibench_files())
    second = load_scibench(tmp_path, scibench_files())
    assert [q.id for q in first[1]] == [q.id for q in second[1]]
    assert all(i.id.startswith("sb-") and len(i.id) == 15 for i in first[0])
    subjects = {i.meta["subject"] for i in first[0]}
    assert subjects <= {"physics", "chemistry"}
