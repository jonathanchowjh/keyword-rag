# Experiments X1-X4

`scripts/experiments.py` queues the arms of four experiments through the API and
prints one comparison from the finished runs. The experiments come from the
research report's plan (`docs/reports/RAG processing improvements since 2024.md`,
kept locally). The user runs them; nothing here is run for you.

```sh
python scripts/experiments.py [--api http://localhost:8080] estimate x1 bright_aops
python scripts/experiments.py run x1 bright_aops --confirm
python scripts/experiments.py report JOB_ID [JOB_ID ...]
```

- `estimate` posts every arm to `/eval` without `confirm` and prints each price. It
  queues nothing.
- `run` needs `--confirm`. It prices the arms first and refuses (exit 2, nothing
  queued) when any arm costs money unless you also pass `--paid`.
- `report` prints one table: arm, metric, value, 95% bootstrap CI. Arms are named
  by method and the settings that differ (`keyword_rag tags=oracle`, `w=1/1/2`,
  `rrf_k=30`, `ex=abstracted`). Take the job ids from the `run` output.
- X1 is one run of four methods, so `report` prints its paired differences (marked
  "paired"). X2, X3 and X4 vary the search settings per run, so each arm is a
  separate run and they compare unpaired: per-arm means with CIs. Overlapping CIs
  are not a verdict; the Run page shows the pairs inside a run.
- `--model` (X4) picks the answer model; the default is the one set up for the
  `answer` job.

Everything is stdlib plus `httpx`, and `run` only queues jobs: the service does the
work, and a run can be cancelled or resumed from the Runs page.

## Datasets and Qdrant

BRIGHT (X1-X3) has two tasks, `bright_aops` and `bright_theoremqa`. Both share one
pool of 188,002 documents, so loading either is large. **Run BRIGHT against server
Qdrant** (`RAGQA_QDRANT_URL`, set by `compose.yaml`; for a locally run API start
one with `scripts/dev-qdrant.sh up`). In the default embedded mode the excluded-id
filter on chunk searches is a linear scan, and every BRIGHT question has excluded
ids. For embedded mode, pass a load `limit` (for example `{"limit": 2000}`) and
read the results as a smoke test, not a measurement.

BRIGHT has no gold answers (all "N/A"), so an answering run on it scores 0. That is
why X4 uses a dataset with answers: **scibench** (small, worked solutions; the
default here) or scienceqa.

## Setup

Loads and copies show an estimate first. The build steps (inducing a vocabulary, AI
tagging, abstracted examples) call the model set up for `keywords`; without `confirm`
nothing is spent.

```sh
# X1-X3: the worked-examples recipe over BRIGHT
curl -X POST localhost:8080/collections -H 'content-type: application/json' \
  -d '{"name": "aops", "qa_type": "science", "recipe": "worked_examples"}'
curl -X POST localhost:8080/collections/aops/load -H 'content-type: application/json' \
  -d '{"dataset": "bright_aops", "confirm": true}'     # first without confirm: the estimate

# X4: a dataset with answers
curl -X POST localhost:8080/collections -H 'content-type: application/json' \
  -d '{"name": "sci", "qa_type": "science", "recipe": "worked_examples"}'
curl -X POST localhost:8080/collections/sci/load -H 'content-type: application/json' \
  -d '{"dataset": "scibench", "confirm": true}'
```

To run the same items under another recipe (`worked_examples`, `reference`,
`paper_2024`), copy instead of loading again:

```sh
curl -X POST localhost:8080/collections/sci/copy -H 'content-type: application/json' \
  -d '{"name": "sci-ref", "recipe": "reference", "confirm": true}'
```

## X1 Retrieval only

Dense, BM25, hybrid and the Keyword-RAG label channel, nDCG@10 on BRIGHT.
Queued as one run of `vector`, `keyword`, `hybrid` and `keyword_rag`; k 10; no
answering; free. One run, so `report` prints the paired difference of each method
against `hybrid`, the CI that decides this experiment.

```sh
python scripts/experiments.py run x1 aops --confirm
```

| | |
|---|---|
| Report's design | Retrieval only: dense, BM25, hybrid, plus label channel (labels from solutions, queries via classifier). Data: BRIGHT AoPS and TheoremQA-Q, nDCG@10. Paid cost: 0 |
| Proves it | Fused label channel beats hybrid with a bootstrap CI above zero |
| Kills it | No gain over hybrid: the facet idea adds nothing for same-skill retrieval |

Where the app differs: the label channel is `keyword_rag`'s `keywords` channel, fused
with meaning and BM25 by weighted RRF; labels are induced from the items, not taken
from solutions only. Repeat on `bright_theoremqa` in a second collection.

## X2 Query mapping

How the question's tags are predicted. Arms: `keyword_rag` with `query_tags`
`similarity` (cosine top-3, the paper's way), `classifier` (nearest-neighbour vote),
`oracle` (the tags of the question's gold items; evals only) and `model` (the query
job's AI picks 1 to 3 terms of the vocabulary, one call per question). The `model`
arm runs `hybrid` beside it, so their difference is paired. k 10; free but for the
`model` arm, which needs a priced query model and `--paid`.

```sh
python scripts/experiments.py run x2 aops --confirm --paid
```

| | |
|---|---|
| Report's design | Query mapping: cosine top-3 vs classifier vs local-LLM draft vs **oracle labels** from the gold positive's solution. Data: Same as X1. Paid cost: 0 |
| Proves it | Classifier closes most of the cosine-to-oracle gap |
| Kills it | Even oracle labels don't beat hybrid: keep facets only for routing and explanation |

Where the app differs: the LLM draft arm is `model`, on a priced provider rather
than a local model. To judge "beats hybrid", read the oracle row beside X1's
`hybrid` row, and the `model` arm's paired `hybrid - keyword_rag` row.

## X3 Fusion

Each channel alone, the RRF constant, and the weight of the keyword channel. Arms
(`keyword_rag`, k 10, free): one channel with weight 1 and the others 0 (`meaning`,
`words`, `keywords`); `rrf_k` 10, 30, 60 and 100 at equal weights; weights
`meaning 1, words 1, keywords w` for w 0.5 and 2.

```sh
python scripts/experiments.py run x3 aops --confirm
```

| | |
|---|---|
| Report's design | Fusion: each channel alone; RRF k ∈ {10, 30, 60, 100}; tuned convex weights. Data: BRIGHT plus held-out GPQA/ScienceQA same-skill@1. Paid cost: 0 |
| Proves it | Tuned convex beats RRF beyond noise |
| Kills it | RRF matches it: keep RRF for simplicity |

Where the app differs: the app fuses by weighted RRF, so the two weight arms stand in
for "tuned convex weights" (the weights scale each channel's RRF share; they are not
a convex score blend). The held-out same-skill set is a second run: create a
ScienceQA collection and run `python scripts/experiments.py run x3 scienceqa-w
--confirm` there.

## X4 Answer accuracy

Does an abstracted retrieved example beat zero-shot and a random example? Arms, all
with `answer: true`: `question_only`; `keyword_rag` with `example: full`;
`keyword_rag` with `example: abstracted`; `random_example` with
`example: abstracted`. Use a dataset with answers (scibench or scienceqa, see above).

```sh
python scripts/experiments.py estimate x4 sci
python scripts/experiments.py run x4 sci --confirm --paid
```

Every arm calls the `answer` model once per dev question, so X4 always costs money
(800 calls for SciBench's 200 dev questions). Read the estimate first and keep the
monthly cap set. The four arms share the dev questions, so the Run pages or the rows
of the four runs can be compared question by question.

| | |
|---|---|
| Design | Answer accuracy, four arms: zero-shot, full retrieved example, abstracted retrieved example, abstracted *random* example, on a priced answer model |
| Proves it | Abstracted retrieved beats both zero-shot and random |
| Kills it | Random ≈ retrieved: the benefit is format, not retrieval. No arm beats zero-shot: switch the example slot off by default for that model |

Where the app differs from the report's design: the retrieved arms run on Keyword-RAG,
not on a plain dense retriever, and the data is the collection you pass (scibench or
scienceqa), not GPQA Diamond. An example shows its abstracted form only when it has a
worked solution: on SciBench 76 of 273 items do, so the two Keyword-RAG arms differ only
on the questions whose top example is one of those. A numeric question also goes
through the program-and-run step when the collection's `program` setting is on; turn it
off for a clean example comparison with
`PUT /collections/{name}/search {"program": false}`.

## Setting the defaults from the results

The experiments decide each collection's defaults. Set them with
`PUT /collections/{name}/search`; `/eval` and `/query` bodies override them per call.

```sh
curl -X PUT localhost:8080/collections/aops/search -H 'content-type: application/json' \
  -d '{"query_tags": "classifier", "weights": {"meaning": 1, "words": 1, "keywords": 2}, "example": "abstracted"}'
```

Keys: `weights` (`meaning`, `words`, `keywords`; not all zero), `query_tags`
(`classifier`, `similarity`; `oracle` is for evals), `example` (`abstracted`, `full`,
`off`) and `program` (true or false). Differences under about 6 points on a few
hundred questions are inside the noise; trust the CI column, not the mean.
