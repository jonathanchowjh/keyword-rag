import pytest

from ragqa_core.types import slug


@pytest.mark.parametrize(
    "raw,want",
    [
        ("Team Handbook", "team-handbook"),
        ("Café notes", "cafe-notes"),
        ("***", "store"),
        ("  a__b  ", "a__b"),
        ("x" * 80, "x" * 64),
    ],
)
def test_slug(raw, want):
    assert slug(raw) == want
