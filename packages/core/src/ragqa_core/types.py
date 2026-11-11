"""Plain data shared by every module, the errors core raises for a missing or taken name,
and `slug`, the rule that turns a name into a valid one."""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

QaType = Literal["documents", "science", "mcq", "multihop", "table"]
# The types that retrieve worked examples.
EXAMPLE_TYPES: tuple[QaType, ...] = ("mcq", "science")
FACETS = ("topic", "method", "common")
CorpusType = Literal["worked", "reference"]


class NotFound(Exception):
    """What was asked for is not there; the message says what. The API answers 404."""


class AlreadyExists(Exception):
    """The name is taken; the message says which. The API answers 409."""


class UnknownCollection(Exception):
    pass


class Stale(Exception):
    """The thing changed since the caller read it. The API answers 409."""


class EvalSetMismatch(Exception):
    """An eval set and a store of different question types: a run cannot ask one of the other."""


def type_clash(eval_set: str, set_type: str, store: str, store_type: str) -> EvalSetMismatch:
    return EvalSetMismatch(f"eval set '{eval_set}' is {set_type}; store '{store}' is {store_type}")


def unknown_set(name: str) -> str:
    return f"no eval set {name!r}"


def unknown_store(name: str) -> str:
    return f"no store named {name!r}"


def slug(name: str) -> str:
    """A name a store or eval set may carry: ASCII letters folded, other runs become '-'."""
    plain = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9_-]+", "-", plain.lower()).strip("-")[:64] or "store"


@dataclass
class BuildSettings:
    """How a collection is built (spec §15). A recipe fills every field."""

    units: Literal["whole", "chunks"]
    added_text: Literal["none", "title_prefix", "summary"]
    vocabulary: Literal["lists", "induced"]
    describes: Literal["text", "solution"]
    tagging: Literal["similarity", "similarity_cutoff", "ai"]
    abstracted: bool
    classifier: bool
    profile: bool
    chunk_words: int = 100
    chunk_overlap: int = 25
    tag_k: int = 3
    tag_cutoff: float = 0.5


@dataclass
class SearchSettings:
    """How a collection is searched. Stored as the default; a request may override."""

    weights: dict[str, float] = field(
        default_factory=lambda: {"meaning": 1.0, "words": 1.0, "keywords": 1.0}
    )
    query_tags: Literal["classifier", "similarity", "oracle", "model"] = "similarity"
    example: Literal["abstracted", "full", "off"] = "full"
    program: bool = False


@dataclass(frozen=True)
class Skip:
    """What one search leaves out: items of disabled sources, and items by id (a BRIGHT
    question's excluded ids). Every channel leaves them out before it fills its pool."""

    sources: tuple[str, ...] = ()
    items: frozenset[str] = frozenset()


@dataclass
class CollectionSpec:
    name: str
    qa_type: QaType
    summarize: bool
    dense_model: str
    dense_dim: int
    keyword_version: str | None = None
    corpus_type: CorpusType = "worked"
    recipe: str = "paper_2024"
    build: BuildSettings | None = None
    search: SearchSettings | None = None
    vocabulary_version: str | None = None


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
    for table, meta carries scale. gold_ids are supporting item ids (empty for mcq). judged_ids is
    a superset of gold_ids once anyone has labelled the question."""

    id: str
    text: str
    choices: list[str] | None
    gold_answers: list[str]
    gold_ids: list[str]
    meta: dict = field(default_factory=dict)
    judged_ids: list[str] | None = None  # every passage marked right or wrong; None: unknown


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
    tags: list[str] = field(default_factory=list)
    abstracted: str | None = None
    channels: list[str] = field(default_factory=list)
