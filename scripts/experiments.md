# Experiments X1-X5

`scripts/experiments.py` queues the arms of five experiments through the API and
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
  by method and their search settings (`keyword_rag w=1/1/2 tags=oracle
  ex=abstracted`, `rrf_k=30`). A run that has not finished is marked with its status. Take the job ids from the `run` output.
- X1 is one run of four methods, so `report` prints its paired differences (marked
  "paired"). X2, X3 and X4 vary the search settings per run, so each arm is a
  separate run and they compare unpaired: per-arm means with CIs. The exception is
  X2's `model` arm, which runs `hybrid` beside it and so prints a paired row; X4's
  arms share their questions, so their rows can be paired by question id. Overlapping CIs
  are not a verdict; the Run page shows the pairs inside a run.
- `--model` (X4) picks the answer model; the default is the one set up for the
  `answer` job.

Everything is stdlib plus `httpx`, and `run` only queues jobs: the service does the
work, and a run can be cancelled or resumed from its Run page.

## Datasets and Qdrant

BRIGHT (X1-X3) has two tasks, `bright_aops` and `bright_theoremqa`. Both share one
pool of 188,002 documents, so loading either is large. **Run BRIGHT against server
Qdrant** (`RAGQA_QDRANT_URL`, set by `compose.yaml`; for a locally run API start
one with `scripts/dev-qdrant.sh up`). In the default embedded mode the excluded-id
filter on chunk searches is a linear scan, and every BRIGHT question has excluded
ids. For embedded mode, pass a load `limit` (for example `{"limit": 2000}`) and
read the results as a smoke test, not a measurement.

BRIGHT has no gold answers (all "N/A"), so an answering run on it measures no accuracy:
its answers are unmeasured, not wrong. That is
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

A load writes its dataset's questions into the store's default eval set (the one
named by its `eval_set`, else the one named like the store), creating it when there is
none. Every arm of an experiment asks that default set unless the arm's body gives an
`eval_set`, and each run records the set's version, so editing the questions later
does not change a finished run. Reloading a dataset upserts its questions by id, which
overwrites hand edits to them.

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

To answer the arms without a provider, choose a model answered by hand for the
`answer` job (README, "Answered by hand"; a price of 0). Each run then stops once
with `stopped_replies`. Export its prompts with `python -m ragqa_core.replies export
--job JOB_ID`, write the replies, import them, and resume the run. The same works
for the `model` arm of X2, on the `query` job.

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

## Results

X1–X4 ran on stores embedded with Qwen/Qwen3-Embedding-0.6B-Q, the default until
2026-11-08. Run 2026-11 on the running service. The CIs are the 95% bootstrap CIs of
`report`; where a job id was not kept with the number, only the mean is given.

### X1 Retrieval only (aops, BRIGHT AoPS, nDCG@10, dev)

| Method | nDCG@10 |
|---|---|
| `vector` | .352 |
| `keyword` (BM25) | .261 |
| `hybrid` | .376 |
| `keyword_rag` (classifier tags) | .314 |

Hybrid is ahead, so the label channel with classifier tags adds nothing: this is
X1's kill outcome.

### X2 Query mapping (aops, `keyword_rag` query tags, nDCG@10)

| Query tags | nDCG@10 | Paired against `hybrid` |
|---|---|---|
| `similarity` | .343 | |
| `classifier` | .314 | |
| `oracle` | .545 | |
| `model`, Opus | .406 | +.030 [-.003, +.062], not separated |
| `model`, Haiku | .430 | +.054 [+.022, +.085], separated |

The oracle is far above the rest, so tag prediction is the bottleneck: a model
that picks the tags (Haiku) clears hybrid, the classifier does not.

### X3 Fusion (aops, with model query tags)

Every arm is within noise of the others. The best is `rrf_k` 10 at .422. The
defaults are kept (`rrf_k` 60, equal channel weights).

### X4 Answer accuracy (sci, SciBench, 200 dev questions, Haiku answers)

| Arm | Accuracy |
|---|---|
| `question_only` (zero-shot) | .825 |
| `keyword_rag`, `example: full` | .820 |
| `keyword_rag`, `example: abstracted` | .820 |
| `random_example`, abstracted | .850 |

Nothing separates from zero-shot: the kill outcome, so the example slot has not
been shown to help on this model and data. Of the 35 zero-shot misses, retrieved
examples fixed 8 and a random example fixed 8. Retrieved examples broke 9
zero-shot hits and a random one broke 3, which points to anchoring on the
example.

## X5 Rerank against hybrid at chat's pool size

Does reranking hybrid's top 20 help, with the reranker chat uses? One run of
`hybrid_rerank` and `hybrid`, k 10, with `rerank_pool` 20, no answering and free.
The run sends no `rerank_model`, so it takes the Settings reranker when it is
queued and records it in `setup.rerank_model`; the report's rows name it. `/eval`
takes a pool of 1 to 50, so chat candidates of 51 to 60 cannot be reproduced.

```sh
curl -s localhost:8080/search-defaults        # in_force.rerank_model is the one the run will use
python scripts/experiments.py estimate x5 aops     # expect $0
python scripts/experiments.py run x5 aops --confirm
python scripts/experiments.py run x5 mq --confirm  # MuSiQue, the home page's default store
python scripts/experiments.py report AOPS_JOB MQ_JOB
```

Read the paired row `hybrid_rerank - hybrid`: nDCG@10 on aops and recall@2 on
MuSiQue.

Decision: rerank is **clearly worse on both** when the CI's upper bound is below 0
on aops and on MuSiQue. Then rerank's recommended value becomes off, with the
reason shown beside it in Settings. Otherwise it stays on.

Result: pending.

## Setting the defaults from the results

The experiments decide each collection's defaults. Set them with
`PUT /collections/{name}/search`; `/eval` and `/query` bodies override them per call.

```sh
curl -X PUT localhost:8080/collections/aops/search -H 'content-type: application/json' \
  -d '{"query_tags": "classifier", "weights": {"meaning": 1, "words": 1, "keywords": 2}, "example": "abstracted"}'
```

Keys: `weights` (`meaning`, `words`, `keywords`; not all zero), `query_tags`
(`classifier`, `similarity`, `model`, which costs a query-job call per question;
`oracle` is for evals), `example` (`abstracted`, `full`,
`off`) and `program` (true or false). Differences under about 6 points on a few
hundred questions are inside the noise; trust the CI column, not the mean.

## Reranker timing

`scripts/time_rerank.py` times the default reranker inside the service (Settings' choice, else `RAGQA_RERANK_MODEL`, else the built-in default). It is free:
a local model, no AI. The first run downloads the model into the `ragqa` volume.

```sh
docker compose -p rag-qa exec -T api python - < scripts/time_rerank.py
```

It reranks 20, 40 and 60 English passages of 1,500 characters against one question, in
ONNX batches of 8 and 16. Each row is the median of 3 calls, after one untimed call. The
batch with the lower 60-pair time becomes `models.RERANK_BATCH`. Under that batch, the
largest pool that reranks in 2 s or less becomes chat's pool cap.

Measured 2026-11-08 with `BAAI/bge-reranker-v2-m3-int8` (the earlier default; these are its figures) on an Apple M4 Pro Mac (12
cores, 24 GB), in Docker Desktop with 11.4 GB of memory and no CPU limit on `api`. No
job was running.

| Batch | 20 pairs | 40 pairs | 60 pairs |
|---|---|---|---|
| 8 | 15.0 s | 25.0 s | 42.3 s |
| 16 | 11.1 s | 23.7 s | 37.8 s |

Result: `batch 16, cap none`. `RERANK_BATCH` is 16. Even 20 passages took 11.1 s, over
five times the 2 s limit, so bge was too slow to rerank in a chat on this machine: a chat
would wait about 11 s for every 20 passages. Re-run the script after changing the
reranker or the machine.

### Smaller rerankers

The same script, with `RAGQA_RERANK_MODEL=<name>`
(`docker compose -p rag-qa exec -T -e RAGQA_RERANK_MODEL=<name> api python - < scripts/time_rerank.py`),
on 1,500-character passages; the times are at the batch the script picked, and the
cap is the largest pool at or under 2 s. fastembed lists each model's size but no
maximum length; the lengths below come from its descriptions.

| Model | Size (GB) | Max length (listed) | Batch | 20 | 40 | 60 | Cap (2 s or less) |
|---|---|---|---|---|---|---|---|
| `jinaai/jina-reranker-v1-tiny-en` | 0.13 | 8K context | 16 | 1.50 s | 2.89 s | 3.41 s | 20 |
| `jinaai/jina-reranker-v1-turbo-en` | 0.15 | 8K context | 16 | 1.76 s | 3.47 s | 5.29 s | 20 |
| `Xenova/ms-marco-MiniLM-L-6-v2` | 0.08 | not listed | 16 | 1.92 s | 3.65 s | 4.96 s | 20 |
| `Xenova/ms-marco-MiniLM-L-12-v2` | 0.12 | not listed | 16 | 3.90 s | 8.00 s | 8.74 s | none |
| `BAAI/bge-reranker-base` | 1.04 | not listed | 8 | 9.01 s | 17.72 s | 29.39 s | none |
| `jinaai/jina-reranker-v2-base-multilingual` | 1.11 | 1K context | 16 | 17.90 s | 27.21 s | 41.74 s | none |
| `BAAI/bge-reranker-v2-m3-int8`, the earlier default | 0.57 | not listed | 16 | 11.08 s | 23.74 s | 37.82 s | none |
| `BAAI/bge-reranker-v2-m3-int8`, paired rerun | 0.57 | not listed | 16 | 18.34 s | 28.33 s | 37.15 s | none |

Passages cut to 600 characters made bge about 3.5 to 4.5 times faster (4.1 s for 20
at batch 16, against 18.3 s at 1,500 in the paired rerun), still over 2 s.

**The machine was busy** (host load average 16 to 27 on 12 cores, other projects'
containers using CPU; no rag-qa job running), so the times are rough: identical bge
runs differed by up to about 1.6 times (11.1 s and 18.3 s for 20 passages). The three
models that fit 20 passages are close to the 2 s line (1.5 to 1.9 s), and a quieter or
busier machine could move them across it.

**Decision.** The reranker is a Setting (Settings → Search defaults → Reranker model,
and Rerank up to), and the default is `jinaai/jina-reranker-v1-tiny-en`, the fastest
with a cap of at least 20 (1.50 s for 20 passages). Whether it ranks as well on these
stores as bge is what X5 checks. To go back to bge, pick
`BAAI/bge-reranker-v2-m3-int8` in Settings, or set it in `RAGQA_RERANK_MODEL`.

## Embedding timing

`scripts/time_embed.py` times each embedding model the Settings page offers
(`BAAI/bge-small-en-v1.5`, `BAAI/bge-base-en-v1.5`, `Qwen/Qwen3-Embedding-0.6B-Q`), inside the
service. It is free: local models, no AI. The first run downloads bge-base and Qwen into the
`ragqa` volume. Qwen takes minutes, and the machine runs hot meanwhile.

```sh
docker compose -p rag-qa exec -T api python - < scripts/time_embed.py
```

Name models as arguments to time others. For each model it times the load, then one pass over
1,000 English passages of 1,500 characters in batches of `EMBED_BATCH` = 8 (after one untimed
call of 8), and the median of 20 single-question calls. Each `per_1000_s`, rounded to whole
seconds, becomes `search_defaults.EMBED_SECONDS[name]`, and the machine becomes `EMBED_MACHINE`.

bge-small and bge-base read at most 512 tokens (about 2,000 characters of English) and Qwen
32k. Longer passages are cut short for the BGE models, though not these 1,500-character ones.

| Model | Dimensions | load_s | per_1000_s | query_ms |
|---|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | | | |
| `BAAI/bge-base-en-v1.5` | 768 | | | |
| `Qwen/Qwen3-Embedding-0.6B-Q` | 1024 | | | |

Result: pending — run after the build, with the service started by Jonathan. Record the machine
(Mac model, cores, memory, Docker memory) and whether a job was running alongside the table.
