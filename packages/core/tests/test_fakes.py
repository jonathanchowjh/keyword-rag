import pytest

from ragqa_core.fakes import until


def test_until_returns_the_first_truthy_value():
    calls = []

    def check():
        calls.append(1)
        return "ready" if len(calls) == 3 else None

    assert until(check, step=0.001) == "ready"
    assert len(calls) == 3


def test_until_says_what_did_not_happen():
    with pytest.raises(AssertionError) as e:
        until(lambda: None, 0.05, what="job")
    assert str(e.value) == "job did not happen in 0.05s"
