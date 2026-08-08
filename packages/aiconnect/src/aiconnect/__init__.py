"""aiconnect: provider-neutral AI connections, models, roles and budget. No HTTP server."""

from aiconnect.ai import (
    AI,
    KEEP,
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
    "BudgetExceeded",
    "Cancelled",
    "Conflict",
    "NoConnection",
    "ProviderError",
    "UnknownModel",
    "UnknownPrice",
    "Completion",
]
