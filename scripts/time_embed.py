"""Times each offered embedding model on 1,000 passages and on single queries.

Run it inside the service, which has the model cache and no uv or scripts/:

    docker compose exec -T api python - < scripts/time_embed.py

Models are argv, else EMBEDDERS. For each it prints the load time (a first run downloads
the model into the cache), the time to embed 1,000 passages of 1,500 characters in
batches of EMBED_BATCH after one untimed call, and the median of 20 single-query calls.
"""

import statistics
import sys
import time

from fastembed import TextEmbedding

from ragqa_core.models import EMBED_BATCH
from ragqa_core.search_defaults import EMBEDDERS
from ragqa_core.settings import load_settings

PASSAGES = 1000
CHARS = 1500
QUERIES = 20
QUESTION = "How does the rate of an enzyme-catalysed reaction change with temperature?"
SENTENCES = [
    "The river rose slowly through the night and by morning it had covered the lower fields.",
    "Most enzymes work fastest within a narrow band of temperature and acidity.",
    "A ledger kept by the harbour master lists every ship that docked that winter.",
    "Heating a gas in a sealed container raises its pressure in proportion to its temperature.",
    "The committee met twice before agreeing on the wording of the final report.",
    "Bread rises because yeast turns sugar into carbon dioxide and a little alcohol.",
    "Early maps of the coast left whole bays blank where no surveyor had been.",
    "When a catalyst is added, the reaction reaches the same balance in less time.",
    "She kept the old letters in a tin box under the stairs for forty years.",
    "A bridge must carry its own weight as well as the traffic that crosses it.",
    "Prices in the market fell sharply once the new harvest came in from the south.",
    "Cooling a solution too quickly can trap impurities inside the growing crystals.",
]


def passages(n: int, chars: int = 1500) -> list[str]:
    """n distinct English prose passages, each exactly `chars` long."""
    out = []
    for i in range(n):
        start = (i * 5) % len(SENTENCES)
        order = SENTENCES[start:] + SENTENCES[:start]
        text = f"Passage {i + 1}. "
        while len(text) < chars:
            text += " ".join(order) + " "
        out.append(text[:chars])
    return out


def load_model(name: str, settings) -> TextEmbedding:
    return TextEmbedding(model_name=name, cache_dir=str(settings.home / "models"))


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    names = list(argv) if argv else list(EMBEDDERS)
    texts = passages(PASSAGES, CHARS)
    print("model  dim  load_s  per_1000_s  query_ms", flush=True)
    for name in names:
        began = time.perf_counter()
        model = load_model(name, settings)
        load_s = time.perf_counter() - began
        list(model.embed(passages(8, CHARS), batch_size=EMBED_BATCH))  # untimed: warms ONNX
        began = time.perf_counter()
        vectors = list(model.embed(texts, batch_size=EMBED_BATCH))  # a generator: consume it
        per_1000 = time.perf_counter() - began
        runs = []
        for _ in range(QUERIES):
            began = time.perf_counter()
            list(model.query_embed([QUESTION]))
            runs.append(time.perf_counter() - began)
        query_ms = statistics.median(runs) * 1000
        print(
            f"{name}  {len(vectors[0])}  {load_s:.1f}  {per_1000:.1f}  {query_ms:.0f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
