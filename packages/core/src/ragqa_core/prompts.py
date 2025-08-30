"""Prompt builders and answer parsing. The mcq format is the 2024 paper's:
"Example question, Example answer, Question" (§3.0), one-shot chain of thought."""

import json
import re

from ragqa_core.types import Hit, QaType

LETTERS = "ABCDEFGH"
ANSWER_LINE = re.compile(r"^\s*Answer:\s*(.+?)\s*$", re.M)
MARKDOWN = re.compile(r"[*`]")
LEADING_UPPER = re.compile(r"^\(?([A-H])\)?(?=[.:),;\s]|$)")
LEADING_MARKED = re.compile(r"^(?:\(([a-h])\)|([a-h])(?=[.)]))")
NAMED_CHOICE = re.compile(
    r"\b(?:answer|option|choice)\s+(?:is\s+)?\(?([A-Ha-h])\)?(?![A-Za-z])", re.I
)
SUMMARY_TOKENS = 120

METHOD_HINT = {
    "mcq": "experiment or analysis types (e.g. tabular scrutiny, controlled experiment)",
    "table": "operations (ratio, difference, % change, average, sum, comparison)",
    "multihop": "relation or question types (e.g. birthplace, founding date, membership)",
}
# What a dense query is looking for, for embedding models trained with a task
# instruction on the query side (Models.query_instruction).
QUERY_TASK: dict[QaType, str] = {
    "mcq": "Given a science question, retrieve a solved question that needs the same reasoning",
    "multihop": "Given a multi-hop question, retrieve passages with the facts needed to answer it",
    "table": "Given a financial question, retrieve the table and text needed to compute the answer",
}


def summary_prompt(text: str) -> str:
    return (
        "Summarise the following in at most 2 sentences. Name the subject area and"
        " what is measured or asked. Reply with the summary only.\n\n" + text
    )


def propose_prompt(samples: list[str], qa_type: QaType) -> str:
    joined = "\n---\n".join(s[:800] for s in samples)
    return (
        "Below are sample items from one question-answering collection. Propose two"
        " keyword lists that describe what such items are about.\n"
        '- "topic": about 50 short subject phrases.\n'
        f'- "method": about 15 phrases naming {METHOD_HINT[qa_type]}.\n'
        'Reply with JSON only: {"topic": [...], "method": [...]}.\n\n' + joined
    )


def answer_prompt(
    qa_type: QaType, question: str, choices: list[str] | None, hits: list[Hit]
) -> tuple[str, str]:
    if qa_type == "mcq":
        system = "You answer multiple-choice science questions."
        parts = []
        for hit in hits:
            parts.append(
                f"Example question:\n{hit.text}\n"
                f"Example answer: {hit.answer}\n"
                f"Example explanation:\n{hit.explanation or ''}\n"
            )
        lines = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices or []))
        parts.append(f"Question:\n{question}\nChoices:\n{lines}\n")
        parts.append("Reason step by step, then finish with a line `Answer: <letter>`.")
        return system, "\n".join(parts)
    system = "You answer questions using the passages given."
    passages = "\n".join(f"[{n}] {hit.text}" for n, hit in enumerate(hits, start=1))
    context = f"Passages:\n{passages}\n\n" if hits else ""
    return system, (
        f"{context}Question: {question}\n"
        "Reason briefly, then finish with a line `Answer: <short answer>`."
    )


def hop_prompt(question: str, passages: list[str]) -> str:
    joined = "\n".join(f"[{n}] {p}" for n, p in enumerate(passages, start=1))
    return (
        f"Question: {question}\n\nPassages found so far:\n{joined}\n\n"
        "Do these passages contain everything needed to answer? If not, give the next"
        " search query that would find the missing fact. Reply with JSON only:"
        ' {"done": true|false, "next_query": "..."}'
    )


def answer_line(text: str) -> str:
    """The content of the last `Answer:` line, or "" if there is none. Markdown
    emphasis and code marks are dropped first: "**Answer: B**" is an answer."""
    found = ANSWER_LINE.findall(MARKDOWN.sub("", text))
    return found[-1].strip() if found else ""


def parse_answer(qa_type: QaType, text: str) -> str:
    answer = answer_line(text)
    if qa_type != "mcq" or not answer:
        return answer
    # A bare letter: "B", "B.", "(c)", "c) sulfate". Lower case only when it is
    # marked as a choice, so the article in "a dog" is not read as A.
    for pattern in (LEADING_UPPER, LEADING_MARKED, NAMED_CHOICE):
        match = pattern.search(answer)
        if match:
            return next(g for g in match.groups() if g).upper()
    return ""


def parse_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}
