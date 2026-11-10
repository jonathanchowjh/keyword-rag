"""Plain-language names for methods, metrics and QA types (spec §7.1, §7.2, §8.4).
Served by GET /catalog so the API and every page agree."""

from ragqa_core.recipes import recipes_for
from ragqa_core.search_defaults import CHAT_METHODS
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
        "meaning + word match + match on predicted keywords, with the store's channel weights",
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
    "recall@10": _metric("Recall@10", "Share of the right items in the top 10", "retrieval"),
    "mrr@10": _metric(
        "MRR@10", "1 over the rank of the first right item in the top 10", "retrieval", unit="score"
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
    "documents": {
        "ndcg@10": "Right passages ranked first",
        "recall@10": "Right passages found (top 10)",
        "mrr@10": "How soon the first right passage comes",
        "accuracy": "Answers judged right",
    },
    "science": WORKED_RECALL_LABEL,
    "mcq": WORKED_RECALL_LABEL,
    "multihop": {},
    "table": {"em": "Exact answer", "f1": "Answer overlap"},
}

METRIC_SAYS_BY_TYPE: dict[QaType, dict[str, str]] = {
    "documents": {
        "ndcg@10": "Right passages near the top count most",
        "recall@10": "Share of a question's right passages in the top 10",
        "mrr@10": "1 when the first hit is right, 1/2 when the second is, 0 when none is in the"
        " top 10",
        "accuracy": "Share of judged answers the AI marked right against the question's"
        " reference answer; questions with no reference answer are left out",
    },
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

# Documents first: both the store form and the eval-set form lead with it.
QA_TYPES: dict[QaType, dict] = {
    "documents": {
        "label": "Documents",
        "item_noun": "passage",
        "retrieves": "passages that answer the question",
    },
    "science": {
        "label": "Science problems",
        "item_noun": "worked example",
        "retrieves": "worked examples whose reasoning carries over, with or without choices",
    },
    "mcq": {
        "label": "Exam multiple choice",
        "item_noun": "example",
        "retrieves": "worked examples whose reasoning carries over",
    },
    "multihop": {
        "label": "Multi-hop",
        "item_noun": "passage",
        "retrieves": "passages holding the facts",
    },
    "table": {
        "label": "Tables",
        "item_noun": "table",
        "retrieves": "the table and text that hold the figures for the answer",
    },
}


RECIPE_LABELS: dict[str, str] = {
    "worked_examples": "worked examples",
    "reference": "reference",
    "paper_2024": "paper 2024",
    "documents": "documents",
    "documents_tagged": "documents with keyword tags",
}
RECIPE_SAYS = {
    "worked_examples": "Whole items, tagged by AI on how they are solved, with abstracted examples",
    "reference": "About 100-word chunks, title prefix, tagged by similarity, with a classifier",
    "paper_2024": "The 2024 paper's way: 100-word chunks, your three keyword lists",
    "documents": "Your passages as uploaded, searched by meaning and by words; no AI",
    "documents_tagged": "Your passages as uploaded, tagged by similarity with an AI-made"
    " vocabulary",
}
RECIPES = {r: {"label": RECIPE_LABELS[r], "says": RECIPE_SAYS[r]} for r in RECIPE_LABELS}


# Every key a run's setup can hold (Evaluator.queue and _record_setup), as the run page shows it.
SETUP_LABELS: dict[str, str] = {
    "collection": "Store",
    "split": "Questions used",
    "methods": "Methods",
    "answer": "Answers written",
    "model": "Answer model",
    "k": "Passages per question",
    "rrf_k": "Fusion constant (RRF k)",
    "search": "Search settings",
    "eval_set": "Eval set",
    "rerank_pool": "Reranker pool",
    "rerank_model": "Reranker",
    "dense_model": "Embedding model",
    "query_instruction": "Query instruction",
    "judge_model": "Judge model",
    "keyword_version": "Keyword lists version",
    "vocabulary_version": "Vocabulary version",
    "seed": "Seed",
    "dataset_filter": "Dataset filter",
    "question_ids": "Questions asked",
    "disabled_sources": "Sources left out",
    "leaked_excluded": "Leaked questions left out",
    "evidence_named": "Questions naming evidence",
    "evidence_held": "Questions whose evidence the store holds",
    "evidence_off": "Questions whose evidence is switched off",
    "eval_set_version": "Eval set version",
    "leaked_ids": "Leaked questions",
    "evidence_size": "Evidence size",
    "judge": "Judged by the answer model",
    "answer_judge_model": "Judge model for answers",
    "chat_search": "Chat's search",
}

ROLE_LABELS = {
    "default": "Default model",
    "answer": "Answers",
    "summary": "Summaries",
    "keywords": "Keyword lists and tags",
    "query": "Query keywords",
    "rewrite": "Follow-up rewriting",
}


def catalog() -> dict:
    return {
        "methods": METHODS,
        "metrics": METRICS,
        "metric_labels_by_type": METRIC_LABELS_BY_TYPE,
        "metric_says_by_type": METRIC_SAYS_BY_TYPE,
        "qa_types": QA_TYPES,
        "recipes": RECIPES,
        "type_recipes": {t: list(recipes_for(t)) for t in QA_TYPES},
        "answer_pair": list(ANSWER_PAIR),
        "controls": list(CONTROLS),
        "chat_methods": list(CHAT_METHODS),
        "setup_labels": SETUP_LABELS,
        "roles": ROLE_LABELS,
    }
