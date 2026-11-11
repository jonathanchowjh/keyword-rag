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
NOUN = {"eval": "run", "copy": "copy", "tag_build": "tag build", "reembed": "rebuild"}


def not_resumable(job: dict) -> str | None:
    """Why a job's status rules out resuming it, None when it can be resumed."""
    if job["status"] in RESUMABLE:
        return None
    return f"a {job['status']} {NOUN.get(job['kind'], 'job')} cannot be resumed"


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
        f"{e}: export them with `python -m ragqa_core.replies export`, import the replies,"
        " then resume the run or start the job again; replies already imported stay cached"
    )
