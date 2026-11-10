# keyword-rag research

keyword-rag grew out of a paper: does describing every document with keywords
find better evidence than searching its text? This file holds the paper, the
built-in datasets that measure its methods, and the experiments. Using the
service on your own documents is in the [README](README.md), and every option
and endpoint is in [REFERENCE.md](REFERENCE.md).

## Contents

- [The paper](#the-paper)
- [Datasets](#datasets)
- [Experiments](#experiments)
- [References and resources](#references-and-resources)

## The paper

> **Keyword Vectorization for Retrieval Augmented Generation on Science
> Question Answering**
> Jonathan Chow (Hartono Tjakrawinata), Singapore Management University, 2024.
> [Read the paper (PDF)](paper/keyword-vectorization-rag-science-qa.pdf)

The paper asks how to pick the one solved example that best helps a language
model answer a long, table-heavy science question. Plain vector search over
whole questions does badly: a 361-word question with tables and units embeds
as noise. Instead, each stored question is described by keywords chosen from
three hand-built lists (science topics, experiment types and common terms)
plus a short AI summary, and the new question is described the same way. The
example whose keywords match best is shown to the model as a one-shot
chain-of-thought demonstration.

It was tested on ACT Science: 446 questions from 12 past papers as the
example store, 76 questions from 2 further papers as the test, gpt-3.5-turbo
answering, `text-embedding-3-large` embeddings, Faiss, BM25 and reciprocal
rank fusion (k = 1). One-shot results:

| Retrieval | Method here | Correct |
|---|---|---|
| No retrieval (question only) | `question_only` | 53 / 76 (69.7%) |
| Vector search (parent-document retriever) | `vector` | 48 / 76 (63.1%) |
| Keyword search (BM25) | `keyword` | 56 / 76 (73.6%) |
| Hybrid: BM25 + vector, fused with RRF | `hybrid` | 58 / 76 (76.3%) |
| Keyword vectors: all keywords in one string | `kw_all` | 51 / 76 (67.1%) |
| **Keyword vectors: top 3 from each list** | **`kw_individual`** | **60 / 76 (78.9%)** |
| Keyword vectors: one store per list, fused with RRF | `kw_rrf` | 58 / 76 (76.3%) |

Two-shot prompts (about 8,000 characters) did worse across the board. With 76
questions, the gap between the best two rows is two questions, so the finding
is a direction, not a settled size. This service exists to measure it
properly: every row above is a method here, run on larger question sets with
paired confidence intervals, and beside controls the paper did not have (see
[Retrieval methods](REFERENCE.md#retrieval-methods)).

The two controls, `random_example` and `fixed_example`, answer a question the
paper could not: in 2024 plain vector search (63.1%) scored below no retrieval
(69.7%), which is either a badly chosen example or any example at all getting in
the way. A method that does not separate from `random_example` has not shown that
choosing the example helped.

To cite it:

```bibtex
@misc{chow2024keywordvectorization,
  title  = {Keyword Vectorization for Retrieval Augmented Generation on
            Science Question Answering},
  author = {Chow, Jonathan},
  year   = {2024},
  note   = {Singapore Management University. Author also known as Hartono Tjakrawinata},
  howpublished = {\url{https://github.com/jonathanchowjh/rag-qa/blob/main/paper/keyword-vectorization-rag-science-qa.pdf}}
}
```

## Datasets

All are downloaded at runtime into the `ragqa` volume and never committed. To
load one, see [Loading a built-in dataset](REFERENCE.md#loading-a-built-in-dataset):
on the Stores page, **Try a built-in benchmark** loads one in a click, or the API
loads it into a store of its QA type.

| QA type | Dataset (id) | Licence | Used as | Free retrieval metric | Paid answer metric |
|---|---|---|---|---|---|
| Science problems | SciBench (`scibench`), physics and chemistry textbook problems | MIT | 273 examples (76 worked solutions) and 219 questions: solved problems are all examples, the 416 unsolved split by a hash of their id. Small, a stand-in until your own corpus is imported | same-skill@1 | accuracy |
| Exam multiple choice | ScienceQA (`scienceqa`) | CC BY-SA 4.0 | 693 solved train examples; 291 test questions | solution similarity@1, same-skill@1 | accuracy |
| Exam multiple choice, hard | GPQA (`gpqa`, gated) | CC BY 4.0 | GPQA Extended minus Diamond as examples; Diamond's 198 as questions | solution similarity@1, same-subdomain@1 | accuracy |
| Science problems, reasoning-intensive retrieval | BRIGHT AoPS (`bright_aops`) and BRIGHT TheoremQA (`bright_theoremqa`), from xlangai/BRIGHT | CC BY 4.0 | Each task's documents as examples (the pool of both is 188,002 documents; use server Qdrant, or a load `limit`); its questions, with the documents to exclude removed from every search | nDCG@10, recall@1 | none: BRIGHT has no gold answers |
| Multi-hop | MuSiQue (`musique`), HippoRAG-2 protocol | CC BY 4.0 | 11,656 passages; 1,000 dev questions | recall@2, recall@5 | EM, F1 |
| Tables (table + text numeric) | TAT-QA (`tatqa`) | CC BY 4.0 | 2,757 contexts; 1,668 dev questions | recall@1, recall@3 | EM, F1 |

ScienceQA is filtered to text-only natural science, grades 7–12, keeping one
item per near-duplicate template. Its human-written solutions serve as the
paper's chain-of-thought examples. TAT-QA answers use simplified scoring:
numbers are compared at 2 decimal places after stripping `$`, `,`, `%` and
scale words, and text is scored like SQuAD.

ScienceQA turned out too easy to separate methods on: plain vector search
finds a same-skill example for 84% of test questions (a random one, 11%),
and its templated solutions put every content method near 0.91 solution
similarity. GPQA is the harder stand-in for the paper's ACT data: expert
questions with a written explanation each. Diamond is held out of the
example store by record, so no question retrieves itself; choices are
shuffled with a seed per record. "Same skill" there means same subdomain.

GPQA is gated. Accept its terms on
[its Hugging Face page](https://huggingface.co/datasets/Idavidrein/gpqa) and
set `HF_TOKEN` to a read token from that account. The terms ask that examples
are not shown online, so keep screenshots of it private.

MuSiQue recall can be set beside published HippoRAG-2 numbers, but only as
**gain over our own dense baseline**: their baseline is NV-Embed-v2, ours a
small local model. EM and F1 come from a different reader and are not
comparable to the papers.

BRIGHT's two tasks load as Science problems stores. Their questions name their
gold documents, so a worked-examples store that has them also reports nDCG@10
(binary relevance) and recall@1, measured after the question's excluded ids are
removed. Answering runs on BRIGHT score 0; use SciBench or ScienceQA to compare
answers. Copy a store under another recipe to compare recipes on the same items.

## Experiments

`scripts/experiments.py` queues the method-v2 experiments (X1 retrieval, X2 query
tags, X3 fusion, X4 answer accuracy, and X5, rerank against hybrid at chat's pool
size) through the API: `estimate` prices every arm,
`run --confirm` queues them (`--paid` for the arms that cost money: X2's `model`
arm, one call per question, and every X4 arm, one answer call per question), and
`report` prints one table with confidence intervals. Setup,
commands, the X1 to X4 results and what each result proves or kills are in
[scripts/experiments.md](scripts/experiments.md).

## References and resources

### The paper this is built on

- Chow, J. (2024). [*Keyword Vectorization for Retrieval Augmented Generation
  on Science Question Answering*](paper/keyword-vectorization-rag-science-qa.pdf).
  Singapore Management University.

### What the paper builds on

- Lewis et al. (2020). [Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401).
- Brown et al. (2020). [Language Models are Few-Shot
  Learners](https://arxiv.org/abs/2005.14165).
- Wei et al. (2022). [Chain-of-Thought Prompting Elicits Reasoning in Large
  Language Models](https://arxiv.org/abs/2201.11903).
- Li et al. (2023). [Guiding Large Language Models via Directional Stimulus
  Prompting](https://arxiv.org/abs/2302.11520).
- Cormack, Clarke & Büttcher (2009). [Reciprocal Rank Fusion outperforms
  Condorcet and individual Rank Learning
  Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf).
- Robertson, Zaragoza & Taylor (2004). [Simple BM25 Extension to Multiple
  Weighted
  Fields](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=10b2752c9940e9c761a161a0fef7ae08cae66301).
- Çano & Bojar (2019). [Keyphrase Generation: A Multi-Aspect
  Survey](https://arxiv.org/abs/1910.05059).
- Gao et al. (2022). [Retrieval-Augmented Multilingual Keyphrase Generation
  with Retriever-Generator Iterative
  Training](https://arxiv.org/abs/2205.10471).
- OpenAI (2023). [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774).
- Tools: [Faiss](https://faiss.ai), LangChain's parent-document and ensemble
  (hybrid) retrievers,
  [reichenbch/RAG-examples](https://github.com/reichenbch/RAG-examples),
  [brexhq/prompt-engineering](https://github.com/brexhq/prompt-engineering).
- Data: ACT Science past papers ([ACT Test Prep](https://www.actexam.net/))
  with worked explanations from
  [PrepSharp](https://www.prepsharp.com/act-answer-keys/).

### Later work this project draws on

- Didolkar et al. (2024). [Metacognitive Capabilities of LLMs: An Exploration
  in Mathematical Problem Solving](https://arxiv.org/abs/2405.12205): pick
  examples by the skill the solution uses, not the question's topic.
- [Doc2Query++](https://arxiv.org/html/2510.09557v2) (2025): a separate
  keyword index fused with text search beats either alone; the idea behind
  `kw_hybrid`.
- Li et al. (2024). [mFAR: multi-field adaptive
  retrieval](https://arxiv.org/html/2410.20056v2): index each field and the
  whole document.
- Kang et al. (2024). [TaxoIndex](https://arxiv.org/abs/2410.19218): a
  classifier maps a question onto the corpus's concepts.
- Wan et al. (2024). [TnT-LLM](https://arxiv.org/abs/2403.12173): an AI
  proposes a taxonomy once, a light classifier labels the rest.
- Weller et al. (2024). [When do Generative Query and Document Expansions
  Fail?](https://aclanthology.org/2024.findings-eacl.134): expansion helps weak
  retrievers and hurts strong ones.
- Liu et al. (2023). [Lost in the Middle](https://arxiv.org/abs/2307.03172):
  where evidence sits in the prompt matters.
- Chen et al. (2025). [SA-ICL](https://arxiv.org/abs/2510.13905) and Wang et
  al. (2025). [I2S](https://arxiv.org/abs/2509.23196): show an example's
  method, not the full worked example.
- [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5): the
  default embedding model.
- [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B): an
  option, and the model of stores built before this change.

### Benchmarks

[ScienceQA](https://arxiv.org/abs/2209.09513) ·
[SciBench](https://arxiv.org/abs/2307.10635) ·
[GPQA](https://arxiv.org/abs/2311.12022) ·
[MuSiQue](https://arxiv.org/abs/2108.00573) ·
[TAT-QA](https://arxiv.org/abs/2105.07624) ·
[HippoRAG 2](https://arxiv.org/abs/2502.14802) (the MuSiQue protocol) ·
[BRIGHT](https://arxiv.org/abs/2407.12883) (the AoPS and TheoremQA tasks)

## License

The paper is © its author and is included for reference. The code is Apache-2.0;
see [LICENSE](LICENSE).
