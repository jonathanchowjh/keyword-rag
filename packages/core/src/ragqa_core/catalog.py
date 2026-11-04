"""Plain-language names for methods, metrics and QA types (spec §7.1, §7.2, §8.4).
Served by GET /catalog so the API and every page agree."""

from ragqa_core.recipes import RECIPE_LABELS
from ragqa_core.types import QaType

ANSWER_PAIR = ("kw_individual", "hybrid")  # the candidate, then the reference
CONTROLS = ("random_example", "fixed_example")


def _method(label: str, group: str, origin: str, says: str, paid: str = "never") -> dict:
    return {
        "label": label,
        "group": group,
        "origin": origin,
        "says": says,
        "paid": paid,
        "retrieves": True,
    }


METHODS: dict[str, dict] = {
    "question_only": {
        **_method("No retrieval", "baseline", "paper", "Answers from the question alone"),
        "retrieves": False,
    },
    "random_example": _method(
        "Random example", "control", "control", "Any item, chosen without looking at the question"
    ),
    "fixed_example": _method(
        "Same example every time", "control", "control", "One item for every question"
    ),
    "vector": _method("Vector search", "plain", "paper", "Nearest 100-word chunks by meaning"),
    "keyword": _method(
        "Word match (BM25)", "plain", "paper", "Items sharing the question's rare words"
    ),
    "hybrid": _method("Hybrid", "plain", "paper", "Vector search and word match, merged"),
    "hybrid_rerank": _method(
        "Hybrid + reranker", "modern", "modern", "Hybrid's top 50, reordered by a cross-encoder"
    ),
    "keyword_rag": _method(
        "Keyword-RAG",
        "modern",
        "new",
        "meaning + word match + match on predicted keywords, with the collection's channel weights",
        paid="with_model_tags",
    ),
    "iterative": _method(
        "Step by step",
        "modern",
        "modern",
        "The AI picks the next search, up to 3 hops",
        paid="always",
    ),
    "kw_all": _method(
        "Keywords, pooled",
        "keywords",
        "paper",
        "Top 9 keywords from all lists in one description",
        paid="with_summaries",
    ),
    "kw_individual": _method(
        "Keywords per list",
        "keywords",
        "paper",
        "Top 3 from each list in one description; the paper's best",
        paid="with_summaries",
    ),
    "kw_rrf": _method(
        "Keywords, list by list",
        "keywords",
        "paper",
        "One search per list, merged",
        paid="with_summaries",
    ),
    "kw_hybrid": _method(
        "Keywords + hybrid",
        "keywords",
        "new",
        "Keywords per list merged with hybrid",
        paid="with_summaries",
    ),
}


def _metric(label: str, says: str, kind: str, unit: str = "share") -> dict:
    return {"label": label, "says": says, "unit": unit, "kind": kind}


METRICS: dict[str, dict] = {
    "accuracy": _metric("Answered correctly", "Share of questions answered correctly", "answer"),
    "em": _metric("Exact answer", "Share of questions where the answer matched exactly", "answer"),
    "f1": _metric("Partly right", "Part marks for overlapping words", "answer"),
    "recall@1": _metric(
        "Found the table (top 1)",
        "Share of questions where the right table came first",
        "retrieval",
    ),
    "recall@3": _metric(
        "Found the table (top 3)",
        "Share of questions where the right table was in the top 3",
        "retrieval",
    ),
    "recall@2": _metric(
        "Found the evidence (top 2)", "Share of the needed passages found in the top 2", "retrieval"
    ),
    "recall@5": _metric(
        "Found the evidence (top 5)", "Share of the needed passages found in the top 5", "retrieval"
    ),
    "same_skill@1": _metric(
        "Example tests the same skill",
        "Share of questions where the example tested the same skill",
        "retrieval",
    ),
    "ndcg@10": _metric(
        "Relevant examples ranked first",
        "0 is none in the top 10, 100 every relevant example ranked ahead of the rest",
        "retrieval",
        unit="score",
    ),
    "solution_sim@1": _metric(
        "Reasoning match",
        "0 is a random example, 100 an identical solution",
        "retrieval",
        unit="score",
    ),
}

# The TAT-QA scorer gives numbers all-or-nothing, so on tables f1 is not "Partly right"
# for most answers: it is the word overlap, which only text answers can have in part.
# On worked collections recall@1 counts relevant examples, not a table.
WORKED_RECALL_LABEL = {"recall@1": "Relevant examples found (top 1)"}
WORKED_RECALL_SAYS = {
    "recall@1": "Share of a question's relevant examples found in the top 1; "
    "with 4 relevant, at most 1 in 4"
}
METRIC_LABELS_BY_TYPE: dict[QaType, dict[str, str]] = {
    "science": WORKED_RECALL_LABEL,
    "mcq": WORKED_RECALL_LABEL,
    "multihop": {},
    "table": {"em": "Exact answer", "f1": "Answer overlap"},
}

METRIC_SAYS_BY_TYPE: dict[QaType, dict[str, str]] = {
    "science": {
        "accuracy": "Share answered right: the choice, a number within 5% of the gold, "
        "or the exact short answer",
        **WORKED_RECALL_SAYS,
    },
    "mcq": WORKED_RECALL_SAYS,
    "multihop": {},
    "table": {
        "em": "Share answered exactly right; numbers are compared at 2 decimal places",
        "f1": "Share answered right, with part marks for text answers that overlap the gold "
        "one; numbers are all-or-nothing",
    },
}

QA_TYPES: dict[QaType, dict] = {
    "science": {
        "label": "Science problems",
        "item_noun": "worked example",
        "retrieves": "worked examples to copy the reasoning of, with or without choices",
    },
    "mcq": {
        "label": "Exam multiple choice",
        "item_noun": "example",
        "retrieves": "worked examples to copy the reasoning of",
    },
    "multihop": {
        "label": "Multi-hop",
        "item_noun": "passage",
        "retrieves": "passages holding the facts",
    },
    "table": {
        "label": "Tables",
        "item_noun": "table",
        "retrieves": "a table and its text to compute from",
    },
}


RECIPE_SAYS = {
    "worked_examples": "Whole items, tagged by AI on how they are solved, with abstracted examples",
    "reference": "About 100-word chunks, title prefix, tagged by similarity, with a classifier",
    "paper_2024": "The 2024 paper's way: 100-word chunks, your three keyword lists",
}
RECIPES = {r: {"label": RECIPE_LABELS[r], "says": RECIPE_SAYS[r]} for r in RECIPE_LABELS}


def catalog() -> dict:
    return {
        "methods": METHODS,
        "metrics": METRICS,
        "metric_labels_by_type": METRIC_LABELS_BY_TYPE,
        "metric_says_by_type": METRIC_SAYS_BY_TYPE,
        "qa_types": QA_TYPES,
        "recipes": RECIPES,
        "answer_pair": list(ANSWER_PAIR),
    }
