"""Prompt builders and answer parsing. The mcq format is the 2024 paper's:
"Example question, Example answer, Question" (§3.0), one-shot chain of thought."""

import json
import re

from ragqa_core.types import CorpusType, Hit, QaType

LETTERS = "ABCDEFGH"
ANSWER_LINE = re.compile(r"^\s*Answer:\s*(.+?)\s*$", re.M)
MARKDOWN = re.compile(r"[*`]")
LEADING_UPPER = re.compile(r"^\(?([A-H])\)?(?=[.:),;\s]|$)")
LEADING_MARKED = re.compile(r"^(?:\(([a-h])\)|([a-h])(?=[.)]))")
NAMED_CHOICE = re.compile(
    r"\b(?:answer|option|choice)\s+(?:is\s+)?\(?([A-Ha-h])\)?(?![A-Za-z])", re.I
)
SUMMARY_TOKENS = 120
CHAT_TOKENS = 1024
CHAT_TURNS = 6  # the history a chat reply sees
CHAT_PASSAGE_CHARS = 1500  # imports.CHUNK_CHARS: a text upload's chunk is never cut

METHOD_HINT = {
    "mcq": "experiment or analysis types (e.g. tabular scrutiny, controlled experiment)",
    "science": "physical laws or solution methods (e.g. conservation of energy, ideal gas law,"
    " dimensional analysis)",
    "table": "operations (ratio, difference, % change, average, sum, comparison)",
    "multihop": "relation or question types (e.g. birthplace, founding date, membership)",
}
# What a dense query is looking for, for embedding models trained with a task
# instruction on the query side (Models.query_instruction).
QUERY_TASK: dict[QaType, str] = {
    "mcq": "Given a science question, retrieve a solved question that needs the same reasoning",
    "science": "Given a science problem, retrieve a solved problem that needs the same reasoning",
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
        "Below are sample items from one question-answering collection. Propose three"
        " keyword lists that describe what such items are about.\n"
        '- "topic": about 50 short subject phrases.\n'
        f'- "method": about 15 phrases naming {METHOD_HINT[qa_type]}.\n'
        '- "common": about 30 terms of the domain that such items use often.\n'
        'Reply with JSON only: {"topic": [...], "method": [...], "common": [...]}.\n\n' + joined
    )


INDUCE_ASKS: dict[CorpusType, str] = {
    "worked": "the skills or principles these solutions use",
    "reference": "the topics and concepts these passages cover",
}


def induce_prompt(texts: list[str], corpus_type: CorpusType) -> str:
    joined = "\n---\n".join(t[:800] for t in texts)
    return (
        f"Below are texts from one collection. Name {INDUCE_ASKS[corpus_type]}, as up to"
        " 10 short terms in total (a few words each), general enough to apply to other"
        ' texts like these. Reply with JSON only: {"terms": [...]}.\n\n' + joined
    )


def tag_prompt(terms: list[str], text: str, k: int) -> str:
    numbered = "\n".join(f"{n}. {t}" for n, t in enumerate(terms, start=1))
    return (
        f"Pick 1 to {k} terms that describe the text below, chosen only from the numbered"
        ' list; copy each term exactly. Reply with JSON only: {"terms": [...]}.\n\n'
        f"Terms:\n{numbered}\n\nText:\n{text[:1500]}"
    )


QUERY_TAG_ASKS: dict[CorpusType, str] = {
    "worked": "name the skills or principles a solution to the question below would use",
    "reference": "name the topics a passage that answers the question below would cover",
}


def query_tag_prompt(terms: list[str], question: str, k: int, corpus_type: CorpusType) -> str:
    """A question's tags, asked of the AI: the same list and reply as tag_prompt, but asked
    about what would answer the question, since the items were tagged by what they hold."""
    numbered = "\n".join(f"{n}. {t}" for n, t in enumerate(terms, start=1))
    return (
        f"Pick 1 to {k} terms that {QUERY_TAG_ASKS[corpus_type]}, chosen only from the"
        ' numbered list; copy each term exactly. Reply with JSON only: {"terms": [...]}.\n\n'
        f"Terms:\n{numbered}\n\nQuestion:\n{question[:1500]}"
    )


def abstract_prompt(text: str, explanation: str, answer: str | None) -> str:
    """A worked example rewritten as what to follow (spec D4), from its text, solution
    and answer only."""
    given = f"\n\nAnswer: {answer}" if answer else ""
    return (
        "Below is a solved problem. Rewrite its solution as the way to solve problems like"
        ' it: "principle" is the idea it rests on, in one sentence; "method" is at most 5'
        ' short steps; "answer" is what the solution found. Copy no numbers from the problem'
        " into the principle or the method. Reply with JSON only:"
        ' {"principle": "...", "method": "...", "answer": "..."}.\n\n'
        f"Problem:\n{text[:1500]}\n\nSolution:\n{explanation[:3000]}{given}"
    )


def edge_order(hits: list[Hit]) -> list[Hit]:
    """Best first and last (spec D12): ranks 1, 3, 5, … then …, 6, 4, 2, so the weakest
    evidence sits in the middle, where a long prompt is read least."""
    return hits[0::2] + hits[1::2][::-1]


def example_block(hit: Hit, example: str, asked: str, explained: str) -> str:
    """One worked example: its abstracted form (principle, method, answer) when asked for
    and written, else the full solution as the paper showed it."""
    if example == "abstracted" and hit.abstracted:
        return f"Example problem:\n{hit.text}\n{hit.abstracted}\n"
    return (
        f"Example {asked}:\n{hit.text}\n"
        f"Example answer: {hit.answer}\n"
        f"Example {explained}:\n{hit.explanation or ''}\n"
    )


def finish(final: str, program: bool) -> str:
    if program:
        return (
            "Write a short plan, then one ```python block that prints only the final answer"
            f" (a number in the unit asked), then finish with a line {final}."
        )
    return f"Reason step by step, then finish with a line {final}."


def answer_prompt(
    qa_type: QaType,
    question: str,
    choices: list[str] | None,
    hits: list[Hit],
    example: str = "full",
    program: bool = False,
) -> tuple[str, str]:
    """`example` is how worked examples are shown: abstracted, full or off. `program` asks
    for a program to run (spec D14), only when the question has no choices."""
    program = program and not choices
    shown = [] if example == "off" else hits
    if qa_type == "mcq":
        system = "You answer multiple-choice science questions."
        parts = [example_block(hit, example, "question", "explanation") for hit in shown]
        lines = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices or []))
        parts.append(f"Question:\n{question}\nChoices:\n{lines}\n")
        parts.append(finish("`Answer: <letter>`", program))
        return system, "\n".join(parts)
    if qa_type == "science":
        parts = [example_block(hit, example, "problem", "solution") for hit in shown]
        parts.append(f"Problem:\n{question}\n")
        if choices:
            lines = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
            parts.append(f"Choices:\n{lines}\n")
            final = "`Answer: <letter>`"
        else:
            final = "`Answer: <number in the unit asked, or a short phrase>`"
        parts.append(finish(final, program))
        return "You solve science problems.", "\n".join(parts)
    system = "You answer questions using the passages given."
    ordered = edge_order(hits)
    passages = "\n".join(f"[{n}] {hit.text}" for n, hit in enumerate(ordered, start=1))
    context = f"Passages:\n{passages}\n\n" if hits else ""
    cite = "Cite the passages you use as [n].\n" if hits else ""
    final = "`Answer: <short answer>`"
    ask = finish(final, True) if program else f"Reason briefly, then finish with a line {final}."
    return system, f"{context}Question: {question}\n{cite}{ask}"


def chat_prompt(question: str, history: list[dict], hits: list[Hit]) -> tuple[str, str]:
    """A chat reply's prompt: the last CHAT_TURNS of the history, the passages numbered from 1
    in edge order (each cut short), then the question."""
    system = (
        "You answer questions using the passages given. Cite the passages you use as [n]."
        " If they don't hold the answer, say so."
    )
    parts = []
    if history[-CHAT_TURNS:]:
        turns = (
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history[-CHAT_TURNS:]
        )
        parts.append("\n".join(turns))
    if hits:
        lines = (
            f"[{n}] {hit.text[:CHAT_PASSAGE_CHARS]}"
            for n, hit in enumerate(edge_order(hits), start=1)
        )
        parts.append("Passages:\n" + "\n".join(lines))
    parts.append(f"Question: {question}")
    return system, "\n\n".join(parts)


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


def parse_answer(qa_type: QaType, text: str, has_choices: bool = True) -> str:
    answer = answer_line(text)
    letters = qa_type == "mcq" or (qa_type == "science" and has_choices)
    if not letters or not answer:
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
