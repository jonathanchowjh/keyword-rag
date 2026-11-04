"""scripts/experiments.py against a fake API: no network, nothing real is called."""

import importlib.util
import json
from pathlib import Path

import httpx
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "experiments.py"
spec = importlib.util.spec_from_file_location("experiments", SCRIPT)
assert spec and spec.loader
experiments = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiments)
arms = experiments.arms


def test_x1_is_one_run_of_four_methods():
    got = arms("x1", "c")
    assert len(got) == 1
    assert got[0]["methods"] == ["vector", "keyword", "hybrid", "keyword_rag"]
    assert got[0]["k"] == 10 and not got[0]["answer"] and got[0]["collection"] == "c"


def test_x2_arms_cover_four_query_tag_modes():
    got = arms("x2", "c")
    modes = ["similarity", "classifier", "oracle", "model"]
    assert [a["search"]["query_tags"] for a in got] == modes
    assert all(a["methods"] == ["keyword_rag"] and a["k"] == 10 for a in got[:3])
    assert got[3]["methods"] == ["hybrid", "keyword_rag"]  # paired with hybrid in one run


def test_x3_arms_single_channels_rrf_k_and_weights():
    got = arms("x3", "c")
    weights = [a["search"]["weights"] for a in got if "weights" in a.get("search", {})]
    singles = [w for w in weights if sorted(w.values()) == [0, 0, 1]]
    assert {next(c for c, v in w.items() if v) for w in singles} == {
        "meaning",
        "words",
        "keywords",
    }
    assert sorted(a["rrf_k"] for a in got if "weights" not in a.get("search", {})) == [
        10,
        30,
        60,
        100,
    ]
    assert {"meaning": 1, "words": 1, "keywords": 0.5} in weights
    assert {"meaning": 1, "words": 1, "keywords": 2} in weights
    assert all(a["methods"] == ["keyword_rag"] and not a["answer"] for a in got)


def test_x4_arms_answer_and_vary_example():
    got = arms("x4", "c")
    assert all(a["answer"] for a in got)
    assert [a["methods"][0] for a in got] == [
        "question_only",
        "keyword_rag",
        "keyword_rag",
        "random_example",
    ]
    assert [a.get("search", {}).get("example") for a in got[1:]] == [
        "full",
        "abstracted",
        "abstracted",
    ]


def test_a_recorded_search_still_names_the_example():
    # A run records its search settings in full, so weights and tags are always there;
    # the example must still tell the X4 arms apart.
    recorded = {"weights": {"meaning": 1, "words": 1, "keywords": 1}, "query_tags": "classifier"}
    names = {
        experiments.label(["keyword_rag"], 60, {**recorded, "example": e})
        for e in ("full", "abstracted")
    }
    assert names == {
        "keyword_rag w=1/1/1 tags=classifier ex=full",
        "keyword_rag w=1/1/1 tags=classifier ex=abstracted",
    }


def test_unknown_experiment_is_refused():
    with pytest.raises(ValueError):
        arms("x9", "c")


def fake_api(est_usd: float | None, posts: list[dict]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/eval":
            body = json.loads(request.content)
            posts.append(body)
            if not body["confirm"]:
                return httpx.Response(200, json={"estimate": {"est_usd": est_usd, "calls": 3}})
            return httpx.Response(202, json={"job_id": f"job{len(posts)}", "estimate": {}})
        if request.method == "GET" and request.url.path == "/eval/job1":
            return httpx.Response(
                200,
                json={
                    "setup": {
                        "methods": ["keyword_rag"],
                        "rrf_k": 60,
                        "search": {"query_tags": "oracle", "example": "full"},
                    },
                    "pairs": [
                        {
                            "a": "keyword_rag",
                            "b": "hybrid",
                            "metric": "ndcg@10",
                            "diff": 0.05,
                            "lo": 0.01,
                            "hi": 0.09,
                            "separated": True,
                        }
                    ],
                    "methods": {
                        "keyword_rag": {
                            "usd": 0.0,
                            "n": 4,
                            "ndcg@10": {"mean": 0.5, "lo": 0.4, "hi": 0.6, "n": 4},
                        }
                    },
                },
            )
        return httpx.Response(404)

    return httpx.Client(base_url="http://api", transport=httpx.MockTransport(handler))


def test_estimate_prints_each_arm_and_queues_nothing(monkeypatch, capsys):
    posts: list[dict] = []
    monkeypatch.setattr(experiments, "make_client", lambda api: fake_api(0.0, posts))
    assert experiments.main(["estimate", "x2", "c"]) == 0
    assert len(posts) == 4 and not any(p["confirm"] for p in posts)
    assert "oracle" in capsys.readouterr().out


def test_run_refuses_paid_without_flag(monkeypatch, capsys):
    posts: list[dict] = []
    monkeypatch.setattr(experiments, "make_client", lambda api: fake_api(0.4, posts))
    assert experiments.main(["run", "x4", "c", "--confirm"]) == 2
    assert not any(p["confirm"] for p in posts)  # estimated, nothing queued
    assert "--paid" in capsys.readouterr().err


def test_run_needs_confirm(monkeypatch):
    posts: list[dict] = []
    monkeypatch.setattr(experiments, "make_client", lambda api: fake_api(0.0, posts))
    assert experiments.main(["run", "x1", "c"]) == 2
    assert posts == []


def test_run_queues_free_arms_and_paid_with_flag(monkeypatch, capsys):
    posts: list[dict] = []
    monkeypatch.setattr(experiments, "make_client", lambda api: fake_api(0.0, posts))
    assert experiments.main(["run", "x1", "c", "--confirm"]) == 0
    assert sum(p["confirm"] for p in posts) == 1
    assert "queued" in capsys.readouterr().out
    posts.clear()
    monkeypatch.setattr(experiments, "make_client", lambda api: fake_api(0.4, posts))
    assert experiments.main(["run", "x4", "c", "--confirm", "--paid"]) == 0
    assert sum(p["confirm"] for p in posts) == 4


def test_report_prints_arm_metric_value_ci(monkeypatch, capsys):
    monkeypatch.setattr(experiments, "make_client", lambda api: fake_api(0.0, []))
    assert experiments.main(["report", "job1"]) == 0
    out = capsys.readouterr().out
    assert "keyword_rag" in out and "tags=oracle" in out
    assert "ndcg@10" in out and "0.500" in out and "0.400" in out and "0.600" in out


def test_unpriced_arm_is_shown_and_needs_paid(monkeypatch, capsys):
    posts: list[dict] = []
    monkeypatch.setattr(experiments, "make_client", lambda api: fake_api(None, posts))
    assert experiments.main(["estimate", "x1", "c"]) == 0
    assert "unpriced (no model)" in capsys.readouterr().out
    assert experiments.main(["run", "x1", "c", "--confirm"]) == 2
    assert not any(p["confirm"] for p in posts)
    assert experiments.main(["run", "x1", "c", "--confirm", "--paid"]) == 0


def test_report_names_each_method_of_a_one_run_experiment(monkeypatch, capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        cell = {"lo": 0.1, "hi": 0.9, "n": 4}
        return httpx.Response(
            200,
            json={
                "setup": {"methods": ["vector", "hybrid"], "rrf_k": 60, "search": {}},
                "methods": {
                    "vector": {"usd": 0.0, "n": 4, "ndcg@10": {"mean": 0.25, **cell}},
                    "hybrid": {"usd": 0.0, "n": 4, "ndcg@10": {"mean": 0.75, **cell}},
                },
            },
        )

    client = httpx.Client(base_url="http://api", transport=httpx.MockTransport(handler))
    monkeypatch.setattr(experiments, "make_client", lambda api: client)
    assert experiments.main(["report", "job1"]) == 0
    rows = [line.split() for line in capsys.readouterr().out.splitlines()[1:]]
    assert {(r[0], r[2]) for r in rows} == {("vector", "0.250"), ("hybrid", "0.750")}


def test_report_prints_paired_differences(monkeypatch, capsys):
    monkeypatch.setattr(experiments, "make_client", lambda api: fake_api(0.0, []))
    assert experiments.main(["report", "job1"]) == 0
    out = capsys.readouterr().out
    assert "keyword_rag - hybrid" in out and "0.050" in out and "0.010" in out and "(paired)" in out
