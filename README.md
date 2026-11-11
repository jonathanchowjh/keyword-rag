# keyword-rag

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![Runs in Docker](https://img.shields.io/badge/runs%20in-Docker-2496ED.svg)
[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](paper/keyword-vectorization-rag-science-qa.pdf)

**A local bench and API for searching your own documents and QA benchmarks.**

Put your documents in a store and compare search methods on them. Mark which
passages are right, and the bench scores each method against your marks. Then
chat across several stores at once, on the page or from your own app. It handles
five QA types: Documents (your own files), science problems, exam multiple
choice, multi-hop and tables. Search runs on your machine. An AI is optional, and
every paid step except chat and the Lab shows its price before it spends.

- **Status:** version 0.2.0, in active development. Runs locally in Docker; no
  published image.
- **License:** Apache-2.0.

## Contents

[Quick start](#quick-start) · [Words used here](#words-used-here) ·
[Your first store](#your-first-store) · [An eval set by labelling](#an-eval-set-by-labelling) ·
[Run and read it](#run-and-read-it) · [Chat across stores](#chat-across-stores) ·
[Chat from your own app](#chat-from-your-own-app) ·
[AI connections and cost safety](#ai-connections-and-cost-safety) ·
[Configuration](#configuration) · [Benchmarks and the paper](#benchmarks-and-the-paper) ·
[More](#more)

## Quick start

You need Docker, and nothing else.

```sh
git clone https://github.com/jonathanchowjh/rag-qa.git
cd rag-qa
cp .env.example .env    # optional: every line in it may stay blank
docker compose up --build
```

**What the first start costs.** No money: nothing calls an AI until you connect
one. The build downloads Python, the dependencies and the Qdrant image. At start
the service fetches its reranker (`jinaai/jina-reranker-v1-tiny-en`, about
0.13 GB) in the background. Your first import fetches the embedding model that
Settings names for new stores (`BAAI/bge-small-en-v1.5`, about 0.07 GB). All of
it stays in the `ragqa` and `qdrant` volumes, so later starts download nothing.

Docker on macOS cannot use the Apple GPU, so embedding runs on the CPU: expect
minutes, not seconds, for thousands of passages (MuSiQue has 11,656).

Then open http://127.0.0.1:8080/. The side nav reads Stores, Chat, Eval sets,
Lab, Settings, Worker (the job queue) and API docs (every endpoint). Its
**Theme** button cycles System, Light and Dark.

**The login is optional.** The port is published on 127.0.0.1 only, and with no
login anyone who can reach it can use it. To add one, set `RAGQA_USERNAME` and
`RAGQA_PASSWORD` in `.env` and restart. Set both or neither: one alone makes
every request a 503. See [SECURITY.md](SECURITY.md).

**Without an AI model the service still runs.** Search, imports into a Documents
store, labelling, runs that only search, and chat as passages all work and cost
nothing. Anything that would spend answers 503 until you connect a model.

## Words used here

- **Store**: what is searched. It holds items (your passages, or a benchmark's
  examples), has one QA type, and keeps the embedding model it was built with.
- **Eval set**: what is asked. A named, versioned list of questions of one QA
  type, each naming its right passages, and maybe a reference answer. One eval
  set can be asked of several stores.
- **Run**: one eval set asked of one store by several methods, each scored with
  a 95% range.
- **Method**: a way to search, such as vector search, word match (BM25) or
  hybrid. There are 13; see [Retrieval methods](REFERENCE.md#retrieval-methods).
- **Recipe**: how a store is built. A Documents store keeps your passages as you
  upload them; see [Recipes](REFERENCE.md#recipes).
- **Connection**: an AI provider, with its address and key, set up on Settings.
- **Chat**: a saved conversation that searches every store it lists, reranks
  their passages together and answers from the best of them.

The API paths and JSON fields say `collections` for stores (`POST /collections`,
`GET /collections/{name}`). The pages and these docs say store.

## Your first store

The Stores page leads with Start here: Connect AI (optional), Pick or make a
store, Chat, and Test with an eval set. Each step is marked done from what
exists, and the strip hides once the last three are done.

1. On Stores, press **+ New store**. The form starts on a Documents store, the
   kind for your own files. Give it a name and press **Create**. The store's
   page opens.
2. Under Add your own passages, choose one or more `.md` or `.txt` files, in
   UTF-8. The preview lists each file and how many passages it adds.
3. Press **Import**. Its price sits beside it, and nothing is spent before the
   click. For a Documents store it is $0: passages are embedded on your machine
   and no AI is called. The line follows the job (`Adding… 40%`) and ends with
   how many passages were added.

Each file is split into passages of up to 1,500 characters, at headings first,
each starting with a breadcrumb such as `notes.md › Setup › Install`. Add an
edited file again, and only the passages that changed are embedded again.

Then the store reads Ready, and **Chat with this store** is enabled. Its search
is vector, word match and hybrid, all free. **Add keyword tags** adds one more
method, `keyword_rag`: the one paid search method on a Documents store, priced first.
Other file types, the import API and passage ids are in
[Importing](REFERENCE.md#importing).

## An eval set by labelling

An eval set tells the bench which passages are right for each question. For a
Documents store you build one by labelling, with no AI.

1. On the store page, under Eval set, follow **Create one**. The form is named
   after the store and linked to it. Press **Create**; the eval set's page opens.
2. Under Label a question, type a question someone would ask of your documents,
   and press **Find passages**. It searches the store with every free method,
   drops repeats and sorts the rest with the reranker, best first.
3. Mark each passage **Right** or **Wrong**. Press a mark again to clear it. If
   the right passage is missing, search the store in the box below the list.
4. Optionally, give a reference answer, so a run can judge the AI's answers.
   Type one, or press **Get the AI's answer**: it shows its price, and **Start**
   pays for one answer from the passages marked Right. Then press **Right** to
   keep it or **Wrong** to drop it.
5. Press **Save question**, or **Save and write the next**.

Finding, marking and saving are free. The table under the panel lists each
question with its right and marked counts, and **Label** reopens one. Every save
makes a new version, and a run always reads the version it started with. More
in [Labelling](REFERENCE.md#labelling).

## Run and read it

**Which search method.** Start with the default, Hybrid (`hybrid`): vector
search and word match, merged by reciprocal rank fusion, with no AI. Chat also
reranks by default with a local cross-encoder, which a run calls Hybrid +
reranker (`hybrid_rerank`). Vector search (`vector`) and Word match (`keyword`)
are hybrid's two halves; Keyword-RAG (`keyword_rag`) adds keyword tags. The
other defaults are 20 candidates per store, 8 passages of evidence, rerank over
up to 20, follow-ups rewritten and RRF k 60. See
[Retrieval methods](REFERENCE.md#retrieval-methods).

**Start a run.**

1. On the store page, press **+ New run**. On a Documents store it ticks the
   search your chats use, plus Hybrid, Vector search and Word match.
2. Pick the eval set (the store's default is chosen) and the questions: the Dev
   slice (up to 200, picked by a hash of each id, so it stays put) or All.
3. To score answers too, tick **Have the AI answer the questions** and **Judge
   answers**: the run's answer model marks each answer right or wrong against
   its reference answer. Both cost money; without them the run is free.
4. Press **Estimate**. The price and the number of AI calls appear beside
   **Start**; a run that only searches shows $0. Press **Start**, and the run's
   page opens.

**Read it.** The store page leads with "How well does chat find the right
passages?": nDCG@10 for the search your chats use, with its 95% range and the
number of questions measured (n). nDCG@10 is 1 when the right passages top the
list, and 0 when none is in the top 10. The run page has each method's score,
with Recall@10 and MRR@10.

- **Another method did better:** the store page says so, with the gap. It was
  picked after the fact, so the gap is likely overstated. To switch, pick it for
  the store in a chat's Settings for this chat and press **Set as default**.
- **Unjudged:** a method whose top 10 hold many passages nobody marked (more than
  3 on average). An unmarked passage counts as wrong, so the score can't be
  trusted. The run page flags it `score unreliable`; for the chat's search the
  store page reads `No score yet`. **Label** those questions again and mark the
  new passages.
- **Unmeasured:** a question whose right passages the store no longer holds. It
  is left out, not counted wrong, and the run page counts them.

Everything a run returns is in [Reading a run](REFERENCE.md#reading-a-run).

## Chat across stores

Open Chat, press **+ New chat** and pick its stores, or press **Chat with this
store** on a store page.

- **Each message** searches every store on the chat and pools their best
  passages. The pool is reranked against the question, and the top 8 go to the
  answer. A follow-up is first rewritten into a standalone question when an AI
  is connected: one paid call per follow-up.
- **Evidence** is under every answer, which cites passages as `[n]`. The
  Retrieval step lists each passage with its store, file and part (`part 2 of
  5` for a passage of a split file), its rank in its store and its rerank score.
- **Settings for this chat**, under the latest message, holds Evidence size,
  Rerank, Rewrite follow-ups, and each store's Method and Candidates.
  **Re-run** searches again with your edits; **Answer from this attempt**
  answers it. A blank field follows the store, then Settings.
- **With no answer model**, Send still works: the reply is the passages alone,
  marked `Passages only: no answer model`, and costs nothing.
- **Cost:** no estimate before a message. Each reply shows its cost: the
  rewrite, any keyword-tag calls and the answer. Reranking is free.

The whole pipeline is in [Chat in detail](REFERENCE.md#chat-in-detail).

## Chat from your own app

Your app can chat over the API with the same pipeline. Set up a chat on the page
first: its stores, evidence size, rerank and so on. Then open **Use from your
app** on the chat page. It holds two curl lines, each with a **Copy** button:

```sh
# start a conversation set up like chat <id>
curl -s -u user:password http://127.0.0.1:8080/chats \
  -H 'content-type: application/json' -d '{"from": "<id>"}'

# send each next message; the URL is quoted because zsh reads ? as a glob
curl -s -u user:password 'http://127.0.0.1:8080/chats/<id>/messages?brief=true' \
  -H 'content-type: application/json' -d '{"content":"How do I reset my password?"}'
```

Use the new chat's `id` in the second call. The server keeps the history, so
your app sends only the new message. Every API conversation is a saved chat:
open it on the chat page to see why it answered.

- **The reply** has `n`, `answer`, `citations`, `passages`, `passages_only`,
  `error` and `usd`. Each passage has `n`, `store`, `id`, `text`, `file`,
  `source`, `part`, `parts` and `relevance` (0 to 1, or null when merged by rank).
- **Your own model:** send `"answer": false` to get the reranked passages with
  no answer written and nothing paid for one.
- **Check `error`:** a failed answer is still HTTP 200, and the message is saved.
  Retrying would save it twice.
- **The login:** `-u user:password` is the `RAGQA_USERNAME` and `RAGQA_PASSWORD`
  in the server's `.env`; leave it out when none is set. Put HTTPS in front
  before exposing it beyond localhost. There are no API keys, CORS or streaming
  yet. Every body is in [Chat request bodies](REFERENCE.md#chat-request-bodies).

## AI connections and cost safety

Connect an AI on Settings; nothing is needed in `.env`. **Give the key its own
spend limit at the provider:** the service sees only its own calls, so the
provider's limit is the real ceiling, since other keys on the same account draw
on the same budget. For Anthropic, Claude Max 5x, Max 20x and
Team plans include monthly API credit ($100 a month on Max 5x):

1. On claude.ai, open **Settings → Billing → API credits** and link a Console
   organisation.
2. In the Console, create a workspace for keyword-rag and give it its own spend
   limit.
3. Add no payment method and leave auto-reload off. When the credit runs out,
   requests stop; nothing is charged to your plan.
4. Create an API key in that workspace and paste it into `ANTHROPIC_API_KEY` in
   `.env` (then restart), or add it on the Settings page.

The claude.ai login itself must not be used: Anthropic's Consumer Terms allow
scripted access only through an API key. OpenAI, Gemini, xAI and OpenRouter
have presets; LM Studio, vLLM and other OpenAI-compatible servers use **Custom**
with their base URL.

- **Prices.** A price is per million input and output tokens. A model you fetch
  or add starts at $0, and a $0 model's calls cost the budget nothing, so price
  a paid model before you use it. A model set to No price (in Settings, or
  null prices through the API) is refused until you price it.
- **The monthly cap**, on Settings (default $20). Each call is priced and
  written to a ledger. A call that would cross the cap is refused before it is
  sent.
- **A price before every spend.** Imports, loads, runs, keyword tags and the
  AI's answer while labelling show their price, and spend only on **Start** or
  **Import**. If the price changed since you saw it, the step is refused, so you
  can check the new one.
- **Two things spend at once,** with no estimate: chat (on the page and over the
  API) and the Lab. Each shows its cost after the call; the cap still holds.
- **Every response is cached,** so a repeat with the same prompts is free.
- **[Answered by hand](REFERENCE.md#answered-by-hand)** is a connection that
  sends nothing: a person writes the replies.

Rough prices on Haiku 5.5: a one-shot multiple-choice answer is about $0.0005,
and a 200-question × 10-method run with answers about $1. Sonnet costs about 20
times as much. Jobs, model refs and batches are in
[AI connections in detail](REFERENCE.md#ai-connections-in-detail).

## Configuration

Environment variables, read at start. `compose.yaml` passes `.env` through. All
are optional.

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Becomes the Anthropic connection: stored, read-only on Settings, applied at every start |
| `RAGQA_MODEL` | The `default` job's model: `<connection>/<model id>`, or a bare model id one enabled model has, such as `claude-haiku-5-5`. Read-only on Settings while set |
| `RAGQA_MONTHLY_CAP_USD` | The monthly cap. Read-only on Settings while set; unset, the stored cap stands (default `20`) |
| `RAGQA_USERNAME`, `RAGQA_PASSWORD` | HTTP basic auth over every route. Set both or neither |
| `RAGQA_DENSE_MODEL` | The embedding model new stores take. Default `BAAI/bge-small-en-v1.5`; a store keeps its own |
| `RAGQA_JUDGE_MODEL` | The model that scores solution similarity on benchmarks. Default `BAAI/bge-small-en-v1.5` |
| `RAGQA_RERANK_MODEL` | The default reranker. Default `jinaai/jina-reranker-v1-tiny-en`; Settings can pick another |
| `RAGQA_QDRANT_URL` | Set by compose. Unset means embedded Qdrant in the API process, fine for a small store |
| `HF_TOKEN` | Hugging Face token: read, to load GPQA (gated); write, only for the owner's build publishing |
| `RAGQA_BUILD_HUB` | The Hub repo of [published builds](REFERENCE.md#published-builds) a load applies. Default `jonathanmiro97/keyword-rag-builds`; empty turns it off |
| `RAGQA_HOME` | State directory. `/home/app/.ragqa` in the image |
| `PORT` | Listen port. `8080` |

## Benchmarks and the paper

keyword-rag grew out of a 2024 paper on keyword vectorization for science QA. It
loads seven built-in datasets to measure the paper's methods beside modern ones
and two controls. The paper, the datasets and the experiments are in
[RESEARCH.md](RESEARCH.md); loading one is in
[Loading a built-in dataset](REFERENCE.md#loading-a-built-in-dataset).

## More

- [REFERENCE.md](REFERENCE.md): importing, recipes, methods, search defaults,
  chat in detail, every page and endpoint, and reading a run.
- [CONTRIBUTING.md](CONTRIBUTING.md): the code's layout, the commands and hooks,
  and what to do when the gate fails.
- [SECURITY.md](SECURITY.md): what is stored, and what runs code.
- [CHANGELOG.md](CHANGELOG.md): what changed in each version, with the notes for
  upgrading from 0.1.0.
- [packages/aiconnect/README.md](packages/aiconnect/README.md): the AI layer, a
  package another project can use on its own.

## License

Apache-2.0. See [LICENSE](LICENSE). The paper is © its author and is included
for reference.
