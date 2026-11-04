"""aiconnect: provider-neutral AI connections, models, roles and budget. No HTTP server."""

from aiconnect.ai import (
    AI,
    KEEP,
    AwaitingReplies,
    BudgetExceeded,
    Cancelled,
    Conflict,
    NoConnection,
    ProviderError,
    UnknownModel,
    UnknownPrice,
)
from aiconnect.types import Completion

__all__ = [
    "AI",
    "KEEP",
    "AwaitingReplies",
    "BudgetExceeded",
    "Cancelled",
    "Conflict",
    "NoConnection",
    "ProviderError",
    "UnknownModel",
    "UnknownPrice",
    "Completion",
]
