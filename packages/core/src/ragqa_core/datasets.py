"""The four datasets, downloaded at runtime and never committed.

- ScienceQA (CC BY-SA 4.0): exam multiple choice; train is the example store,
  test the queries. Text-only natural science, grades 7–12, templates deduped.
- MuSiQue (CC BY 4.0), HippoRAG-2 protocol: 1,000 dev questions, 11,656 passages.
- TAT-QA (CC BY 4.0): every table+text context pooled; dev questions.
- SciBench (MIT, xw27/scibench): university physics and chemistry problems with
  a numeric answer. Fields: problem_text, answer_latex, answer_number, unit,
  source, problemid, comment, solution. 416 unsolved problems and 76 solved
  ones, across 7 sources; 105 units carry a power of ten and 67 are blank. SciBench scores
  with math.isclose(rel_tol=0.05) (eval/eval_zero.py, equiv) and folds a
  power-of-ten unit into the gold (eval/post_process.py, remove_not, cal_not),
  so split_unit does the same. Facts checked 2026-10-27.
"""

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path

import httpx
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from ragqa_core.prompts import LETTERS
from ragqa_core.types import Item, QaType, Question

DATASETS: dict[str, QaType] = {
    "scienceqa": "mcq",
    "musique": "multihop",
    "tatqa": "table",
    "scibench": "science",
}
FILTER_NOTES: dict[QaType, str] = {
    "mcq": "ScienceQA text-only, natural science, grades 7-12, one item per masked template",
    "science": (
        "SciBench physics and chemistry textbook problems (7 sources); "
        "the solved ones are the worked examples"
    ),
    "multihop": "MuSiQue, HippoRAG-2 protocol: 1,000 dev questions over 11,656 passages",
    "table": "TAT-QA dev questions over every train/dev/test context",
}
SCIENCEQA_REPO = "derek-thomas/ScienceQA"
MUSIQUE_REPO = "osunlp/HippoRAG_2"
TATQA_URL = (
    "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/"
    "tatqa_dataset_{split}.json"
)
SCIBENCH_REPO = "xw27/scibench"
SCIBENCH_SOURCES = ("atkins", "chemmc", "quan", "matter", "fund", "class", "thermo")
CHEMISTRY = {"atkins", "chemmc", "quan", "matter"}
GRADES = {f"grade{n}" for n in range(7, 13)}
SCIENCEQA_COLUMNS = [
    "question",
    "choices",
    "answer",
    "hint",
    "image",
    "subject",
    "grade",
    "topic",
    "skill",
    "solution",
]
DEV_SLICE, DEV_SEED = 200, 472

NUMBER = re.compile(r"\d+(\.\d+)?")
WORDS = re.compile(r"\S+")
UNIT_EXPONENT = re.compile(r"^\s*\$?\s*10\s*\^\s*\{?\s*([-+]?\d+)\s*\}?")
DEGREE = re.compile(r"\^\s*\{?\s*\\circ\s*\}?\s*")
UNIT_WRAPPER = re.compile(r"\\(?:mathrm|text)\s*\{([^{}]*)\}")
CHAPTER = re.compile(r"^\s*(?:problem|question|[ep])?\s*(\d+)", re.I)


def mask_template(text: str) -> str:
    """Normalise away the parts that vary between instances of one template:
    numbers, and capitalised words that are not at the start of a sentence."""
    out, sentence_start = [], True
    for word in WORDS.findall(text):
        if not sentence_start and word[:1].isupper():
            out.append("X")
        else:
            out.append(NUMBER.sub("#", word.lower()))
        sentence_start = word[-1:] in ".?!"
    return " ".join(out)


def dev_slice(
    questions: list[Question], n: int = DEV_SLICE, seed: int = DEV_SEED
) -> list[Question]:
    """Keep the n questions whose sha1 of "seed:id" ranks lowest, sorted by id.
    A question's place does not depend on the others, so adding one moves at
    most one question out of the slice."""
    ranked = sorted(questions, key=lambda q: hashlib.sha1(f"{seed}:{q.id}".encode()).hexdigest())
    return sorted(ranked[:n], key=lambda q: q.id)


def load(name: str, cache: Path) -> tuple[list[Item], list[Question]]:
    loaders: dict[str, Callable[[Path], tuple[list[Item], list[Question]]]] = {
        "scienceqa": load_scienceqa,
        "musique": load_musique,
        "tatqa": load_tatqa,
        "scibench": load_scibench,
    }
    if name not in loaders:
        raise ValueError(f"unknown dataset {name!r}; known: {', '.join(loaders)}")
    return loaders[name](cache)


# -- ScienceQA -----------------------------------------------------------------


def _scienceqa_files(cache: Path) -> dict[str, Path]:
    files = HfApi().list_repo_files(SCIENCEQA_REPO, repo_type="dataset")
    out = {}
    for split in ("train", "test"):
        name = next(f for f in files if f.startswith(f"data/{split}-") and f.endswith(".parquet"))
        out[split] = Path(
            hf_hub_download(SCIENCEQA_REPO, name, repo_type="dataset", cache_dir=cache / "datasets")
        )
    return out


def _scienceqa_rows(path: Path) -> list[tuple[int, dict]]:
    table = pq.read_table(path, columns=SCIENCEQA_COLUMNS)
    images = table.column("image").is_null().to_pylist()
    rows = table.drop(["image"]).to_pylist()
    kept, seen = [], set()
    for n, (row, no_image) in enumerate(zip(rows, images, strict=True)):
        if not no_image or row["subject"] != "natural science" or row["grade"] not in GRADES:
            continue
        key = mask_template(row["question"] + " " + " ".join(row["choices"]))
        if key in seen:
            continue
        seen.add(key)
        kept.append((n, row))
    return kept


def _scienceqa_text(row: dict) -> str:
    text = row["question"]
    if row["hint"]:
        text += "\nContext: " + row["hint"]
    return text


def load_scienceqa(
    cache: Path, files: dict[str, Path] | None = None
) -> tuple[list[Item], list[Question]]:
    files = files or _scienceqa_files(cache)
    items = []
    for n, row in _scienceqa_rows(files["train"]):
        choices = "; ".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(row["choices"]))
        items.append(
            Item(
                id=f"sqa-train-{n}",
                text=_scienceqa_text(row) + "\nChoices: " + choices,
                answer=LETTERS[row["answer"]],
                explanation=row["solution"],
                meta={"skill": row["skill"], "topic": row["topic"], "grade": row["grade"]},
            )
        )
    questions = [
        Question(
            id=f"sqa-test-{n}",
            text=_scienceqa_text(row),
            choices=list(row["choices"]),
            gold_answers=[LETTERS[row["answer"]]],
            gold_ids=[],
            meta={"skill": row["skill"], "topic": row["topic"], "solution": row["solution"]},
        )
        for n, row in _scienceqa_rows(files["test"])
    ]
    return items, questions


# -- MuSiQue -------------------------------------------------------------------


def musique_id(title: str, text: str) -> str:
    return "mq-" + hashlib.sha1(f"{title}\n{text}".encode()).hexdigest()[:16]


def load_musique(
    cache: Path, files: dict[str, Path] | None = None
) -> tuple[list[Item], list[Question]]:
    if files is None:
        files = {
            key: Path(
                hf_hub_download(
                    MUSIQUE_REPO, name, repo_type="dataset", cache_dir=cache / "datasets"
                )
            )
            for key, name in (("questions", "musique.json"), ("corpus", "musique_corpus.json"))
        }
    items: dict[str, Item] = {}
    for p in json.loads(files["corpus"].read_text()):
        item_id = musique_id(p["title"], p["text"])
        items.setdefault(item_id, Item(item_id, f"{p['title']}\n{p['text']}"))
    questions = [
        Question(
            id=q["id"],
            text=q["question"],
            choices=None,
            gold_answers=[q["answer"], *q.get("answer_aliases", [])],
            gold_ids=[
                musique_id(p["title"], p["paragraph_text"])
                for p in q["paragraphs"]
                if p["is_supporting"]
            ],
            meta={},
        )
        for q in json.loads(files["questions"].read_text())
    ]
    return list(items.values()), questions


# -- TAT-QA --------------------------------------------------------------------


def _tatqa_file(cache: Path, split: str) -> Path:
    path = cache / "datasets" / f"tatqa_dataset_{split}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = httpx.get(TATQA_URL.format(split=split), follow_redirects=True, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def _tatqa_text(doc: dict) -> str:
    rows = "\n".join(" | ".join(str(cell) for cell in row) for row in doc["table"]["table"])
    paragraphs = "\n".join(p["text"] for p in sorted(doc["paragraphs"], key=lambda p: p["order"]))
    return f"{rows}\n\n{paragraphs}"


def _flatten(answer) -> list[str]:
    if isinstance(answer, list):
        return [", ".join(str(a) for a in answer)]
    return [str(answer)]


def load_tatqa(
    cache: Path, files: dict[str, Path] | None = None
) -> tuple[list[Item], list[Question]]:
    files = files or {s: _tatqa_file(cache, s) for s in ("train", "dev", "test")}
    items: dict[str, Item] = {}
    for split in ("train", "dev", "test"):
        for doc in json.loads(files[split].read_text()):
            item_id = "tq-" + doc["table"]["uid"]
            items.setdefault(item_id, Item(item_id, _tatqa_text(doc)))
    questions = [
        Question(
            id=q["uid"],
            text=q["question"],
            choices=None,
            gold_answers=_flatten(q["answer"]),
            gold_ids=["tq-" + doc["table"]["uid"]],
            meta={"scale": q.get("scale", "")},
        )
        for doc in json.loads(files["dev"].read_text())
        for q in doc["questions"]
    ]
    return list(items.values()), questions


# -- SciBench ------------------------------------------------------------------


def split_unit(unit: str) -> tuple[int, str]:
    """The leading power of ten (0 if none) and the rest as plain text:
    " $10^{-35} \\mathrm{~J} \\mathrm{~s}$" -> (-35, "J s")."""
    match = UNIT_EXPONENT.match(unit)
    exponent = int(match.group(1)) if match else 0
    rest = unit[match.end() :] if match else unit
    rest = DEGREE.sub("°", rest)
    rest = UNIT_WRAPPER.sub(r"\1", rest).replace("\\cdot", "·").replace("$", "").replace("~", " ")
    return exponent, " ".join(rest.split())


def _scibench_files(cache: Path) -> dict[str, Path]:
    return {
        name: Path(
            hf_hub_download(SCIBENCH_REPO, name, repo_type="dataset", cache_dir=cache / "datasets")
        )
        for source in SCIBENCH_SOURCES
        for name in (f"{source}.json", f"{source}_sol.json")
    }


def load_scibench(
    cache: Path, files: dict[str, Path] | None = None
) -> tuple[list[Item], list[Question]]:
    """Every solved problem is a worked example. Of the rest, those whose id
    hashes even are questions and the others examples without a solution."""
    files = files or _scibench_files(cache)
    items: dict[str, Item] = {}
    questions: dict[str, Question] = {}
    for source in SCIBENCH_SOURCES:
        for solved in (True, False):
            name = f"{source}_sol.json" if solved else f"{source}.json"
            if name not in files:
                continue
            for row in json.loads(files[name].read_text()):
                digest = hashlib.sha1(f"{source}\n{row['problem_text']}".encode()).hexdigest()[:12]
                item_id = "sb-" + digest
                if item_id in items or item_id in questions:
                    continue
                exponent, plain = split_unit(row["unit"])
                gold = format(float(row["answer_number"]) * 10**exponent, ".6g")
                text = row["problem_text"] + (f"\nGive the answer in {plain}." if plain else "")
                meta = {
                    "topic": source,
                    "subject": "chemistry" if source in CHEMISTRY else "physics",
                }
                chapter = CHAPTER.match(row["problemid"])
                if chapter:
                    meta["skill"] = f"{source}:{chapter.group(1)}"
                if solved or int(digest, 16) % 2:
                    items[item_id] = Item(
                        item_id,
                        text,
                        answer=f"{gold} {plain}".strip(),
                        explanation=row["solution"] if solved else None,
                        meta=meta,
                    )
                else:
                    questions[item_id] = Question(item_id, text, None, [gold], [], meta)
    return list(items.values()), list(questions.values())
