"""Build recipes (spec §15) and build- and search-setting validation. Imports only types."""

import math
from dataclasses import fields, replace

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
        chunk_overlap=25,
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
QUERY_TAGS = ("classifier", "similarity", "oracle", "model")
EXAMPLES = ("abstracted", "full", "off")
# The values each worded build setting may take; flags and numbers are checked on their own.
BUILD_CHOICES: dict[str, tuple[str, ...]] = {
    "units": ("whole", "chunks"),
    "added_text": ("none", "title_prefix", "summary"),
    "vocabulary": ("lists", "induced"),
    "describes": ("text", "solution"),
    "tagging": ("similarity", "similarity_cutoff", "ai"),
}
BUILD_FLAGS = ("abstracted", "classifier", "profile")


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


def _whole_number(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def build_settings(base: BuildSettings, override: dict | None) -> BuildSettings:
    """The base build (a recipe's) with a partial override merged in. Raises ValueError on
    bad input, and on an override that would make lists and run the build both."""
    override = override or {}
    known = [f.name for f in fields(BuildSettings)]
    unknown = [k for k in override if k not in known]
    if unknown:
        raise ValueError(f"unknown build settings {unknown}; known: {known}")
    for key, value in override.items():
        if key in BUILD_CHOICES and value not in BUILD_CHOICES[key]:
            raise ValueError(f"{key} must be one of {list(BUILD_CHOICES[key])}")
        if key in BUILD_FLAGS and not isinstance(value, bool):
            raise ValueError(f"{key} must be true or false")
    build = replace(base, **override)
    if not _whole_number(build.chunk_words) or build.chunk_words < 1:
        raise ValueError("chunk_words must be a whole number, 1 or more")
    if not _whole_number(build.chunk_overlap) or build.chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a whole number, 0 or more")
    if build.units == "chunks" and build.chunk_overlap >= build.chunk_words:
        raise ValueError("chunk_overlap must be below chunk_words, or the windows never move")
    if not _whole_number(build.tag_k) or build.tag_k < 1:
        raise ValueError("tag_k must be a whole number, 1 or more")
    cutoff = build.tag_cutoff
    if not isinstance(cutoff, int | float) or isinstance(cutoff, bool) or not 0 <= cutoff <= 1:
        raise ValueError("tag_cutoff must be a number from 0 to 1")
    # THE RECIPE DECIDES THE VOCABULARY: paper 2024 makes the three lists and the others
    # run the build, and nothing makes both. Summaries only feed the lists' vectors.
    if build.vocabulary != base.vocabulary:
        raise ValueError(
            f"the recipe sets vocabulary to {base.vocabulary!r}; choose another recipe instead"
        )
    if build.added_text == "summary" and build.vocabulary != "lists":
        raise ValueError("added_text 'summary' needs a recipe that uses keyword lists")
    return build
