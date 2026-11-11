import pytest

from ragqa_core import jobs


def test_every_kind_has_a_label_and_a_noun():
    for kind in jobs.BUSY_STATE:
        assert kind in jobs.WORD and kind in jobs.NOUN
        assert jobs.label({"kind": kind, "collection": "c"}) == f"{jobs.WORD[kind]} c"


def test_an_unknown_kind_is_labelled_by_its_name():
    assert jobs.label({"kind": "mystery", "collection": "c"}) == "mystery c"


@pytest.mark.parametrize("kind", list(jobs.BUSY_STATE))
@pytest.mark.parametrize("status", ["queued", "running", "done", *jobs.RESUMABLE])
def test_resumable_is_resumes_and_resumable_status(kind, status):
    job = {"kind": kind, "status": status}
    assert jobs.resumable(job) is (kind in jobs.RESUMING_KINDS and status in jobs.RESUMABLE)


def test_a_kind_that_never_resumes_says_start_it_again():
    for kind, noun in (("load", "load"), ("rekey", "re-key"), ("backfill", "labelling job")):
        said = jobs.not_resumable({"kind": kind, "status": "interrupted"})
        assert said == f"this {noun} cannot be resumed; start it again"
    assert jobs.not_resumable({"kind": "mystery", "status": "failed"}) == (
        "this job cannot be resumed; start it again"
    )


def test_a_resumable_kind_is_refused_by_status_only():
    assert jobs.not_resumable({"kind": "eval", "status": "done"}) == "a done run cannot be resumed"
    assert jobs.not_resumable({"kind": "eval", "status": "failed"}) is None
