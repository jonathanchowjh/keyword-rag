"""Times the configured reranker on 20, 40 and 60 passages, in batches of 8 and 16.

Run it inside the service, which has the model cache and no uv or scripts/:

    docker compose exec -T api python - < scripts/time_rerank.py

It times the configured reranker: the service's own (RAGQA_RERANK_MODEL, else the
default), not one picked under Settings. It prints the median of 3 calls per (batch,
pairs), then the batch to use and the largest pool that reranks in 2 s or less ("cap
none" when even 20 passages take longer). That is a guide only: chat's cap is the
Settings "Rerank up to", which the user chooses.
"""

import statistics
import sys
import time

from fastembed.rerank.cross_encoder import TextCrossEncoder

from ragqa_core.settings import load_settings

PAIRS = (20, 40, 60)
BATCHES = (8, 16)
RUNS = 3
QUERY = "How does the rate of an enzyme-catalysed reaction change with temperature?"
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


def load_encoder(settings) -> TextCrossEncoder:
    return TextCrossEncoder(
        model_name=settings.rerank_model, cache_dir=str(settings.home / "models")
    )


def choose_cap(seconds: dict[int, dict[int, float]], limit: float = 2.0) -> tuple[int, int | None]:
    """The batch with the lower 60-pair time, and under it the largest pool within limit."""
    for batch, by_pairs in seconds.items():
        missing = [pairs for pairs in PAIRS if pairs not in by_pairs]
        if missing:
            raise ValueError(f"batch {batch} has no timing for pool size {missing[0]}")
    batch = min(seconds, key=lambda b: seconds[b][60])
    within = [pairs for pairs in PAIRS if seconds[batch][pairs] <= limit]
    return batch, max(within) if within else None


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    print(f"model {settings.rerank_model}", flush=True)
    encoder = load_encoder(settings)
    list(encoder.rerank(QUERY, passages(4), batch_size=BATCHES[0]))  # untimed: warms ONNX
    seconds: dict[int, dict[int, float]] = {}
    print("batch  pairs  seconds")
    for batch in BATCHES:
        seconds[batch] = {}
        for pairs in PAIRS:
            texts, runs = passages(pairs), []
            for _ in range(RUNS):
                began = time.perf_counter()
                list(encoder.rerank(QUERY, texts, batch_size=batch))  # a generator: consume it
                runs.append(time.perf_counter() - began)
            seconds[batch][pairs] = statistics.median(runs)
            print(f"{batch:5d}  {pairs:5d}  {seconds[batch][pairs]:7.3f}", flush=True)
    batch, cap = choose_cap(seconds)
    if cap is None:
        print(f"batch {batch}, cap none (rerank off by default)")
    else:
        print(f"batch {batch}, cap {cap}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
