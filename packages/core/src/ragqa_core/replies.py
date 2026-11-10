"""Answer by hand the prompts of a connection of kind `manual`. A job that asks such a
model what its cache does not hold stops (`stopped_replies`) and its prompts wait. Write
them out, reply to each, read the replies back, then resume the run or load again: the
job finds them cached. Run it where the database is, inside the api container:

    docker compose exec api python -m ragqa_core.replies export OUT.jsonl [--job ID]
    docker compose exec api python -m ragqa_core.replies import REPLIES.jsonl

OUT.jsonl has one prompt a line: {"key", "model", "purpose", "job_id", "max_tokens",
"system", "prompt"}. REPLIES.jsonl has {"key", "reply"} a line, for any of them. A reply is
recorded as the model's, ledgered at its price on estimated tokens. The service never
imports this module.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from aiconnect.ai import AI, UnknownPrice
from ragqa_core.settings import load_settings


def read_replies(path: Path) -> dict[str, str]:
    replies: dict[str, str] = {}
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        if line.strip():
            row = json.loads(line)
            key, reply = (row.get("key"), row.get("reply")) if isinstance(row, dict) else (0, 0)
            if not isinstance(key, str) or key in replies or not isinstance(reply, str):
                raise ValueError(f"{path.name} line {n}: a missing or repeated key, or no reply")
            replies[key] = reply
    return replies


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Answer waiting prompts by hand.")
    commands = parser.add_subparsers(dest="command", required=True)
    out = commands.add_parser("export", help="write the waiting prompts out")
    out.add_argument("out", type=Path)
    out.add_argument("--job", help="only this job's prompts")
    back = commands.add_parser("import", help="record replies to waiting prompts")
    back.add_argument("replies", type=Path)
    args = parser.parse_args(argv)
    ai = AI(load_settings().home / "ragqa.db")
    if args.command == "export":
        waiting = ai.waiting(args.job)
        fields = ("key", "model", "purpose", "job_id", "max_tokens", "system", "prompt")
        args.out.write_text("".join(json.dumps({f: w[f] for f in fields}) + "\n" for w in waiting))
        print(
            json.dumps(
                {"waiting": len(waiting), "by_purpose": Counter(w["purpose"] for w in waiting)}
            )
        )
        return 0
    try:
        recorded = ai.answer(read_replies(args.replies))
    except (OSError, ValueError, UnknownPrice) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    print(json.dumps({"recorded": recorded, "still_waiting": len(ai.waiting())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
