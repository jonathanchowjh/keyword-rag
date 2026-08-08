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
]
