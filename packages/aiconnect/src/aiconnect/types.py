"""Plain data returned by AI calls. No logic lives here."""

from dataclasses import dataclass


@dataclass
class Completion:
    text: str
    usd: float
    cached: bool
    input_tokens: int
    output_tokens: int
