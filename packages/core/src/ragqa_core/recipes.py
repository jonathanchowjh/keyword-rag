"""Build recipes (spec §15) and search-setting validation. Imports only types."""

import math
from dataclasses import replace

from ragqa_core.types import BuildSettings, CorpusType, QaType, SearchSettings

RECIPES: dict[str, BuildSettings] = {
    "worked_examples": BuildSettings(
        units="whole",
        added_text="none",
        vocabulary="induced",
        describes="solution",
        tagging="ai",
        abstracted=True,
        classifier=True,
        profile=True,
    ),
    "reference": BuildSettings(
        units="chunks",
        chunk_words=100,
        chunk_overlap=10,
        added_text="title_prefix",
        vocabulary="induced",
        describes="text",
        tagging="similarity",
        abstracted=False,
        classifier=True,
        profile=True,
    ),
    "paper_2024": BuildSettings(
        units="chunks",
        chunk_words=100,
        chunk_overlap=75,
        added_text="summary",
        vocabulary="lists",
        describes="text",
        tagging="similarity",
        abstracted=False,
        classifier=False,
        profile=True,
    ),
}

RECIPE_LABELS: dict[str, str] = {
    "worked_examples": "worked examples",
    "reference": "reference",
    "paper_2024": "paper 2024",
}

CHANNELS = ("meaning", "words", "keywords")
QUERY_TAGS = ("classifier", "similarity", "oracle")
EXAMPLES = ("abstracted", "full", "off")


def corpus_type_for(qa_type: QaType) -> CorpusType:
    return "worked" if qa_type in ("science", "mcq") else "reference"


def build_for(recipe: str, summarize: bool = False) -> BuildSettings:
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe {recipe!r}; known: {list(RECIPES)}")
    build = replace(RECIPES[recipe])
    if recipe == "paper_2024":
        build.added_text = "summary" if summarize else "none"
    return build


def default_search(qa_type: QaType, build: BuildSettings) -> SearchSettings:
    return SearchSettings(
        query_tags="classifier" if build.classifier else "similarity",
        example="abstracted" if build.abstracted else "full",
        program=qa_type in ("science", "table"),
    )


def search_settings(base: SearchSettings, override: dict | None) -> SearchSettings:
    """The base settings with a partial override merged in. Raises ValueError on bad input."""
    override = override or {}
    unknown = [k for k in override if k not in ("weights", "query_tags", "example", "program")]
    if unknown:
        raise ValueError(f"unknown search settings {unknown}")
    weights = dict(base.weights)
    given = override.get("weights", {})
    if not isinstance(given, dict):
        raise ValueError("weights must be an object")
    for channel, weight in given.items():
        if channel not in CHANNELS:
            raise ValueError(f"unknown channel {channel!r}; known: {list(CHANNELS)}")
        is_number = isinstance(weight, int | float) and not isinstance(weight, bool)
        if not is_number or not math.isfinite(weight) or weight < 0:
            raise ValueError(f"weight for {channel!r} must be a finite number, 0 or more")
        weights[channel] = float(weight)
    if not any(weights.values()):
        raise ValueError("at least one channel weight must be above 0")
    query_tags = override.get("query_tags", base.query_tags)
    if query_tags not in QUERY_TAGS:
        raise ValueError(f"query_tags must be one of {list(QUERY_TAGS)}")
    example = override.get("example", base.example)
    if example not in EXAMPLES:
        raise ValueError(f"example must be one of {list(EXAMPLES)}")
    program = override.get("program", base.program)
    if not isinstance(program, bool):
        raise ValueError("program must be true or false")
    return SearchSettings(weights, query_tags, example, program)
