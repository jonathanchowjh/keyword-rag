from ragqa_core.catalog import ANSWER_PAIR, METHODS, METRICS, QA_TYPES, catalog
from ragqa_core.retriever import METHOD_NAMES


def test_every_method_has_a_label():
    assert set(METHODS) == set(METHOD_NAMES)
    assert METHODS["kw_individual"]["label"] == "Keywords per list"
    assert METHODS["question_only"]["retrieves"] is False
    assert all(m["retrieves"] for k, m in METHODS.items() if k != "question_only")


def test_paid_marks():
    assert METHODS["iterative"]["paid"] == "always"
    assert METHODS["kw_rrf"]["paid"] == "with_summaries"
    assert METHODS["hybrid"]["paid"] == "never"
    assert METHODS["keyword_rag"]["paid"] == "with_model_tags"  # the AI picks its tags


def test_every_metric_the_evaluator_reports_has_a_label():
    from ragqa_core.evaluator import ANSWER_METRICS, RETRIEVAL_METRICS

    named = {m for ms in [*RETRIEVAL_METRICS.values(), *ANSWER_METRICS.values()] for m in ms}
    assert named <= set(METRICS)


def test_documents_leads_and_every_type_is_described():
    from typing import get_args

    from ragqa_core.catalog import METRIC_LABELS_BY_TYPE, METRIC_SAYS_BY_TYPE
    from ragqa_core.evaluator import ANSWER_METRICS, ANSWER_TOKENS, MISS_METRIC, RETRIEVAL_METRICS
    from ragqa_core.prompts import METHOD_HINT, QUERY_TASK
    from ragqa_core.types import QaType

    assert list(catalog()["qa_types"])[0] == "documents"
    assert list(catalog()["qa_types"])[1] == "science"  # the rest keep their order
    tables = (
        QA_TYPES,
        METRIC_LABELS_BY_TYPE,
        METRIC_SAYS_BY_TYPE,
        RETRIEVAL_METRICS,
        ANSWER_METRICS,
        MISS_METRIC,
        ANSWER_TOKENS,
        QUERY_TASK,
        METHOD_HINT,
        catalog()["type_recipes"],
    )
    for t in get_args(QaType):
        assert all(t in table for table in tables), t


def test_the_documents_type_in_plain_words():
    from ragqa_core.evaluator import (
        ANSWER_METRICS,
        ANSWER_TOKENS,
        MISS_METRIC,
        PAIR_ANSWER_METRIC,
        RETRIEVAL_METRICS,
    )
    from ragqa_core.prompts import CHAT_TOKENS, QUERY_TASK

    c = catalog()
    assert c["qa_types"]["documents"] == {
        "label": "Documents",
        "item_noun": "passage",
        "retrieves": "passages that answer the question",
    }
    assert c["metrics"]["recall@10"] == {
        "label": "Found in top 10",
        "says": "Share of the right items in the top 10",
        "unit": "share",
        "kind": "retrieval",
        "code": "Recall@10",
    }
    assert c["metrics"]["mrr@10"] == {
        "label": "First right item",
        "says": "1 over the rank of the first right item in the top 10",
        "unit": "score",
        "kind": "retrieval",
        "code": "MRR@10",
    }
    assert c["metrics"]["ndcg@10"]["label"] == "Ranking quality"
    assert c["metrics"]["ndcg@10"]["code"] == "nDCG@10"
    assert c["metrics"]["accuracy"]["code"] is None
    assert c["metric_labels_by_type"]["documents"] == {
        "ndcg@10": "Ranking quality",
        "recall@10": "Found in top 10",
        "mrr@10": "First right passage",
        "accuracy": "Answers judged right",
    }
    assert c["metric_labels_by_type"]["documents"]["accuracy"] == "Answers judged right"
    assert c["metric_says_by_type"]["documents"] == {
        "ndcg@10": "Right passages near the top count most",
        "recall@10": "Share of a question's right passages in the top 10",
        "mrr@10": "1 when the first hit is right, 1/2 when the second is, 0 when none is in"
        " the top 10",
        "accuracy": "Share of judged answers the AI marked right against the question's"
        " reference answer; questions with no reference answer are left out",
    }
    assert c["recipes"]["documents"] == {
        "label": "documents",
        "says": "Your passages as uploaded, searched by meaning and by words; no AI",
    }
    assert c["recipes"]["documents_tagged"] == {
        "label": "documents with keyword tags",
        "says": "Your passages as uploaded, tagged by similarity with an AI-made vocabulary",
    }
    assert c["type_recipes"]["documents"] == ["documents", "documents_tagged"]
    assert c["type_recipes"]["mcq"] == ["worked_examples", "reference", "paper_2024"]
    assert QUERY_TASK["documents"] == "Given a question, retrieve passages that answer it"
    assert RETRIEVAL_METRICS["documents"] == ("ndcg@10", "recall@10", "mrr@10")
    assert ANSWER_METRICS["documents"] == ("accuracy",)
    assert MISS_METRIC["documents"] == "recall@10"
    assert ANSWER_TOKENS["documents"] == CHAT_TOKENS
    assert PAIR_ANSWER_METRIC["documents"] == "accuracy"  # judged answers only


def test_qa_types_and_catalog_shape():
    assert QA_TYPES["mcq"]["label"] == "Exam multiple choice"
    assert QA_TYPES["table"]["item_noun"] == "table"
    assert QA_TYPES["multihop"]["retrieves"] == "passages holding the facts"
    c = catalog()
    assert c["answer_pair"] == list(ANSWER_PAIR)
    assert set(c) == {
        "methods",
        "metrics",
        "metric_labels_by_type",
        "metric_says_by_type",
        "qa_types",
        "recipes",
        "type_recipes",
        "answer_pair",
        "controls",
        "chat_methods",
        "setup_labels",
        "roles",
    }
    says = c["metric_says_by_type"]["table"]
    assert "part marks" not in says["em"] and "part marks" in says["f1"]  # exact is exact
    table = c["metric_labels_by_type"]["table"]
    assert table == {"em": "Exact answer", "f1": "Answer overlap"}  # distinct, as on the page


def test_catalog_lists_chat_methods_without_hybrid_rerank():
    chat_methods = catalog()["chat_methods"]
    assert "hybrid_rerank" not in chat_methods and "iterative" not in chat_methods
    assert all(m in METHODS for m in chat_methods)


def test_catalog_serves_setup_and_role_labels():
    got = catalog()
    assert got["setup_labels"]["dense_model"] == "Embedding model"
    assert got["setup_labels"]["chat_search"] == "Chat's search"
    assert got["roles"]["keywords"] == "Keyword lists and tags"
    assert got["roles"]["rewrite"] == "Follow-up rewriting"
