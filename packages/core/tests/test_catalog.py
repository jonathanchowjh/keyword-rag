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


def test_every_metric_the_evaluator_reports_has_a_label():
    from ragqa_core.evaluator import ANSWER_METRICS, RETRIEVAL_METRICS

    named = {m for ms in [*RETRIEVAL_METRICS.values(), *ANSWER_METRICS.values()] for m in ms}
    assert named <= set(METRICS)


def test_science_leads_and_every_type_is_described():
    from typing import get_args

    from ragqa_core.catalog import METRIC_LABELS_BY_TYPE, METRIC_SAYS_BY_TYPE
    from ragqa_core.types import QaType

    assert list(catalog()["qa_types"])[0] == "science"
    for t in get_args(QaType):
        assert t in QA_TYPES and t in METRIC_LABELS_BY_TYPE and t in METRIC_SAYS_BY_TYPE


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
        "answer_pair",
    }
    says = c["metric_says_by_type"]["table"]
    assert "partial credit" not in says["em"] and "partial credit" in says["f1"]  # exact is exact
    table = c["metric_labels_by_type"]["table"]
    assert table == {"em": "Exact answer", "f1": "Answer overlap"}  # distinct, as on the page
