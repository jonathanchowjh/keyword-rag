"""What a job is: its kinds, the statuses it ends in and what each ending says."""

from aiconnect.ai import AwaitingReplies, BudgetExceeded, Cancelled, NoConnection

# The state a store shows while a job of each kind is queued or running.
BUSY_STATE = {
    "load": "loading",
    "rekey": "rekeying",
    "retag": "retagging",
    "copy": "copying",
    "eval": "evaluating",
    "import": "importing",
    "backfill": "labelling",
    "reembed": "rebuilding",
    "tag_build": "tagging",
}

# The statuses of a job that has not ended.
BUSY = ("queued", "running")

# The statuses a job can be queued again from.
RESUMABLE = (
    "interrupted",
    "cancelled",
    "stopped_budget",
    "stopped_no_key",
    "stopped_replies",
    "failed",
)
PARTLY_DONE = ("failed", "cancelled", "interrupted")
# The lead-in of a job's label: the page shows "<word> <collection>".
WORD = {
    "load": "Loading",
    "import": "Importing into",
    "copy": "Copying into",
    "rekey": "Re-keying",
    "retag": "Re-tagging",
    "backfill": "Labelling",
    "reembed": "Re-embedding",
    "tag_build": "Adding keyword tags",
    "eval": "Evaluating",
}
NOUN = {
    "load": "load",
    "import": "import",
    "copy": "copy",
    "rekey": "re-key",
    "retag": "re-tag",
    "backfill": "labelling job",
    "reembed": "rebuild",
    "tag_build": "tag build",
    "eval": "run",
}
# The kinds whose stopped jobs run again; any other is started again by hand.
RESUMING_KINDS = ("eval", "copy", "reembed", "tag_build")
# What each stopped status says about a job in a problem line.
PROBLEM = {
    "failed": "failed",
    "interrupted": "was interrupted by a restart",
    "stopped_budget": "stopped at the monthly cap",
    "stopped_no_key": "stopped: no AI model is set up for this job (see the Settings page)",
    "stopped_replies": "stopped: its prompts wait for replies by hand",
}


def label(job: dict) -> str:
    """What the pages call a job: "Evaluating wl", "Importing into docs"."""
    return f"{WORD.get(job['kind'], job['kind'])} {job['collection']}"


def resumable(job: dict) -> bool:
    return job["kind"] in RESUMING_KINDS and job["status"] in RESUMABLE


def not_resumable(job: dict) -> str | None:
    """Why a job cannot be resumed (its kind never does, or its status rules it out),
    None when it can be."""
    noun = NOUN.get(job["kind"], "job")
    if job["kind"] not in RESUMING_KINDS:
        return f"this {noun} cannot be resumed; start it again"
    if job["status"] in RESUMABLE:
        return None
    return f"a {job['status']} {noun} cannot be resumed"


def ended(e: Exception) -> dict:
    """The status and error a job that raised e ends with: stopped (resumable) when it ran
    out of budget, a model or replies; cancelled; otherwise failed."""
    if isinstance(e, Cancelled):
        return {"status": "cancelled"}
    if isinstance(e, BudgetExceeded):
        return {"status": "stopped_budget", "error": str(e)}
    if isinstance(e, NoConnection):
        return {"status": "stopped_no_key", "error": no_ai(e.role)}
    if isinstance(e, AwaitingReplies):
        return {"status": "stopped_replies", "error": awaiting(e)}
    return {"status": "failed", "error": repr(e)}


def no_ai(role: str) -> str:
    return f"no AI model is set up for the '{role}' job; choose one on the Settings page"


def awaiting(e: AwaitingReplies) -> str:
    return (
        f"{e}: export them with `docker compose exec api python -m ragqa_core.replies"
        " export /home/app/.ragqa/prompts.jsonl`, import the replies, then resume the run"
        " or start the job again; replies already imported stay cached"
    )
