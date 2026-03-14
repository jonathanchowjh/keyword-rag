from ragqa_core.prompts import (
    answer_prompt,
    hop_prompt,
    parse_answer,
    parse_json,
    propose_prompt,
    summary_prompt,
)
from ragqa_core.types import Hit

HIT_WITH_ANSWER_B = Hit(
    "e1", "Which is a mammal?\nChoices: A. shark; B. whale", 1.0, {}, [], "B", "Whales nurse.", {}
)


def test_parse_mcq_letter():
    assert parse_answer("mcq", "blah\nAnswer: (c) because") == "C"


def test_parse_takes_last_answer_line():
    assert parse_answer("mcq", "Answer: A\nwait\nAnswer: D") == "D"


def test_parse_free_text():
    assert parse_answer("multihop", "x\nAnswer: June 1982") == "June 1982"


def test_parse_missing():
    assert parse_answer("mcq", "no idea") == ""


def test_mcq_prompt_uses_paper_format():
    _, user = answer_prompt("mcq", "Q?", ["x", "y"], [HIT_WITH_ANSWER_B])
    assert "Example question:" in user and "Example answer: B" in user
    assert "Whales nurse." in user and "A. x" in user and "B. y" in user
    assert user.rstrip().endswith("`Answer: <letter>`.")


def test_mcq_prompt_without_hits_has_no_example():
    _, user = answer_prompt("mcq", "Q?", ["x", "y"], [])
    assert "Example" not in user


def test_passage_prompt_numbers_passages():
    hits = [HIT_WITH_ANSWER_B, Hit("e2", "Second passage", 0.5, {}, [], None, None, {})]
    _, user = answer_prompt("multihop", "Who?", None, hits)
    assert "[1] Which is a mammal?" in user and "[2] Second passage" in user
    assert "`Answer: <short answer>`" in user


def test_other_prompts_carry_their_input():
    assert "enzyme" in summary_prompt("enzyme rates")
    assert "ratio" in propose_prompt(["sample"], "table")
    assert "experiment" in propose_prompt(["sample"], "mcq")
    assert "next_query" in hop_prompt("Q?", ["p1"])


def test_parse_json_tolerates_prose():
    assert parse_json('Sure: {"done": true, "next_query": ""} ok') == {
        "done": True,
        "next_query": "",
    }
    assert parse_json("not json") == {}


def test_parse_mcq_letter_inside_a_sentence():
    assert parse_answer("mcq", "Answer: The answer is B") == "B"
    assert parse_answer("mcq", "Answer: option (d), because") == "D"


def test_parse_mcq_without_a_clear_letter_is_empty():
    assert parse_answer("mcq", "Answer: None of these") == ""


def test_parse_mcq_bare_letter_with_punctuation():
    assert parse_answer("mcq", "Answer: B.") == "B"
    assert parse_answer("mcq", "Answer: c) sulfate") == "C"


def test_parse_markdown_wrapped_answers():
    assert parse_answer("mcq", "so...\nAnswer: **B**") == "B"
    assert parse_answer("mcq", "so...\n**Answer: B**") == "B"
    assert parse_answer("mcq", "so...\n`Answer: C`") == "C"
    assert parse_answer("multihop", "x\n**Answer:** June 1982") == "June 1982"


def test_science_prompt_with_and_without_choices():
    hit = Hit("i1", "Example problem text", 1.0, {}, [], "4.2 J", "Worked solution", {})
    _, free = answer_prompt("science", "How much work?", None, [hit])
    assert "Example solution:\nWorked solution" in free and "Choices" not in free
    assert free.rstrip().endswith("`Answer: <number in the unit asked, or a short phrase>`.")
    _, mcq = answer_prompt("science", "Which?", ["x", "y"], [hit])
    assert "A. x\nB. y" in mcq and mcq.rstrip().endswith("`Answer: <letter>`.")


def test_science_parse_reads_letters_only_with_choices():
    assert parse_answer("science", "work...\nAnswer: (b) sulfate", has_choices=True) == "B"
    assert parse_answer("science", "work...\nAnswer: 4.2 J", has_choices=False) == "4.2 J"
    assert parse_answer("science", "no answer line", has_choices=False) == ""
    assert parse_answer("science", "work...\nAnswer: sulfate", has_choices=True) == ""
