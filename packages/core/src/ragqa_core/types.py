"""Plain data shared by every module. No logic lives here."""

from dataclasses import dataclass, field
from typing import Literal

QaType = Literal["science", "mcq", "multihop", "table"]
# The types that retrieve worked examples.
EXAMPLE_TYPES: tuple[QaType, ...] = ("mcq", "science")
FACETS = ("topic", "method", "common")


@dataclass
class CollectionSpec:
    name: str
    qa_type: QaType
    summarize: bool
    dense_model: str
    dense_dim: int
    keyword_version: str | None = None


@dataclass
class Item:
    id: str
    text: str
    answer: str | None = None
    explanation: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Question:
    """A query with its gold. For mcq, meta carries skill, topic and solution;
    for table, meta carries scale. gold_ids are supporting item ids (empty for mcq)."""

    id: str
    text: str
    choices: list[str] | None
    gold_answers: list[str]
    gold_ids: list[str]
    meta: dict = field(default_factory=dict)


@dataclass
class KeywordLists:
    topic: list[str]
    method: list[str]
    common: list[str]


@dataclass
class Hit:
    id: str
    text: str
    score: float
    keywords: dict[str, list[str]]
    matched: list[str]
    answer: str | None
    explanation: str | None
    meta: dict
    summary: str | None = None


@dataclass
class Completion:
    text: str
    usd: float
    cached: bool
    input_tokens: int
    output_tokens: int
