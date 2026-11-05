#!/usr/bin/env python3
"""Queue the method-v2 experiments X1-X4 through the API and compare their runs.

  experiments.py estimate X1 COLLECTION            price every arm; queues nothing
  experiments.py run X1 COLLECTION --confirm       queue the arms (--paid if any costs money)
  experiments.py report JOB_ID [JOB_ID ...]        one table from the finished runs

The guide, with the setup of each experiment, is scripts/experiments.md.
"""

import argparse
import sys

import httpx

CHANNELS = ("meaning", "words", "keywords")
RRF_KS = (10, 30, 60, 100)
KEYWORD_WEIGHTS = (0.5, 2)


def arm(collection: str, method: str, *, k: int = 10, answer: bool = False, **rest) -> dict:
    return {
        "collection": collection,
        "split": "dev",
        "methods": [method],
        "answer": answer,
        "k": k,
        **rest,
    }


def arms(experiment: str, collection: str) -> list[dict]:
    """The /eval request bodies (without confirm) of one experiment."""
    if experiment == "x1":
        # ONE run: paired differences exist only between methods of the same run.
        methods = ["vector", "keyword", "hybrid", "keyword_rag"]
        return [{**arm(collection, methods[0]), "methods": methods}]
    if experiment == "x2":
        out = [
            arm(collection, "keyword_rag", search={"query_tags": mode})
            for mode in ("similarity", "classifier", "oracle")
        ]
        # The AI's tags run beside hybrid, so the difference that decides it is paired.
        both = {**arm(collection, "hybrid", search={"query_tags": "model"})}
        return [*out, {**both, "methods": ["hybrid", "keyword_rag"]}]
    if experiment == "x3":
        out = [
            arm(
                collection,
                "keyword_rag",
                search={"weights": {c: 1 if c == only else 0 for c in CHANNELS}},
            )
            for only in CHANNELS
        ]
        out += [arm(collection, "keyword_rag", rrf_k=n) for n in RRF_KS]
        out += [
            arm(
                collection,
                "keyword_rag",
                search={"weights": {"meaning": 1, "words": 1, "keywords": w}},
            )
            for w in KEYWORD_WEIGHTS
        ]
        return out
    if experiment == "x4":
        return [
            arm(collection, "question_only", answer=True),
            arm(collection, "keyword_rag", answer=True, search={"example": "full"}),
            arm(collection, "keyword_rag", answer=True, search={"example": "abstracted"}),
            arm(collection, "random_example", answer=True, search={"example": "abstracted"}),
        ]
    raise ValueError(f"unknown experiment {experiment!r}; known: x1, x2, x3, x4")


def label(methods: list[str], rrf_k: int | None, search: dict) -> str:
    """A run's arm name: the method, and for keyword_rag the settings that vary between arms."""
    method = "+".join(methods) if len(methods) > 1 else methods[0]
    name = method
    if method == "keyword_rag":
        w = search.get("weights")
        if w:
            name += " w=" + "/".join(f"{w[c]:g}" for c in CHANNELS)
        if search.get("query_tags"):
            name += f" tags={search['query_tags']}"
    if method in ("keyword_rag", "random_example") and search.get("example"):
        name += f" ex={search['example']}"
    if rrf_k is not None and rrf_k != 60:
        name += f" rrf_k={rrf_k}"
    return name


def body_label(body: dict) -> str:
    return label(body["methods"], body.get("rrf_k"), body.get("search") or {})


def make_client(api: str) -> httpx.Client:
    return httpx.Client(base_url=api, timeout=60)


def estimates(client: httpx.Client, bodies: list[dict]) -> list[float | None]:
    """Each arm priced without confirm; prints one line per arm. None means unpriced."""
    costs: list[float | None] = []
    for body in bodies:
        r = client.post("/eval", json={**body, "confirm": False})
        r.raise_for_status()
        est = r.json()["estimate"]
        usd = est.get("est_usd")  # null when the job has no model to price it
        costs.append(None if usd is None else float(usd))
        price = "unpriced (no model)" if usd is None else f"${usd:.4f}"
        print(f"{body_label(body):55} {price}  {est.get('calls', 0)} calls")
    print(f"{'total':55} ${sum(c for c in costs if c is not None):.4f}")
    return costs


def run(client: httpx.Client, bodies: list[dict], paid: bool) -> int:
    costs = estimates(client, bodies)
    if any(c is None or c > 0 for c in costs) and not paid:
        print(
            "these arms cost money or are unpriced; read the estimate, then add --paid",
            file=sys.stderr,
        )
        return 2  # nothing is queued
    for body in bodies:
        r = client.post("/eval", json={**body, "confirm": True})
        r.raise_for_status()
        print(f"queued {body_label(body)}: {r.json()['job_id']}")
    return 0


def report(client: httpx.Client, job_ids: list[str]) -> int:
    print(f"{'arm':55} {'metric':20} {'value':>7}  CI")
    for job_id in job_ids:
        r = client.get(f"/eval/{job_id}")
        r.raise_for_status()
        table = r.json()
        setup = table["setup"]
        # A stopped run's means cover only what it got to: say so beside every row of it.
        unfinished = table.get("status") not in (None, "done")
        for method, entry in table["methods"].items():
            name = label([method], setup.get("rrf_k"), setup.get("search") or {})
            if unfinished:
                name += f" [{table['status']}]"
            for metric, cell in entry.items():
                if isinstance(cell, dict) and "mean" in cell:
                    ci = f"[{cell['lo']:.3f}, {cell['hi']:.3f}]"
                    print(f"{name:55} {metric:20} {cell['mean']:7.3f}  {ci}")
        # Paired differences exist only between methods of ONE run; one-method arms have none.
        for p in table.get("pairs", []):
            ci = f"[{p['lo']:.3f}, {p['hi']:.3f}]"
            print(f"{p['a']} - {p['b']:35} {p['metric']:20} {p['diff']:7.3f}  {ci} (paired)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--api", default="http://localhost:8080")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("estimate", "run"):
        p = sub.add_parser(name)
        p.add_argument("experiment", choices=["x1", "x2", "x3", "x4"], type=str.lower)
        p.add_argument("collection")
        p.add_argument("--model", help="the answer model for X4 (default: the one set up)")
        if name == "run":
            p.add_argument("--confirm", action="store_true", help="queue the arms")
            p.add_argument("--paid", action="store_true", help="allow arms that cost money")
    sub.add_parser("report").add_argument("job_ids", nargs="+")
    args = parser.parse_args(argv)

    client = make_client(args.api)
    if args.command == "report":
        return report(client, args.job_ids)
    bodies = arms(args.experiment, args.collection)
    if args.model:
        bodies = [{**b, "model": args.model} for b in bodies]
    if args.command == "estimate":
        estimates(client, bodies)
        return 0
    if not args.confirm:
        print("run queues work; add --confirm (estimate shows the price first)", file=sys.stderr)
        return 2
    return run(client, bodies, args.paid)


if __name__ == "__main__":
    sys.exit(main())
