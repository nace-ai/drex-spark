"""The checks made before any request, which fail the query."""

import json

import numpy as np
import pytest

from drex_spark._contract import DrexError, answer_value, parse_questions, typed_question, wire_questions


def refused(call, *args, match: str) -> None:
    with pytest.raises(DrexError, match=match):
        call(*args)


@pytest.mark.parametrize(
    "labels",
    [
        ["billing", "tech"],
        '["billing", "tech"]',
        np.array(["billing", "tech"], dtype=object),
        [{"label": "billing", "description": "Payments"}, {"label": "tech"}],
        {"billing": "Payments", "tech": None},
    ],
)
def test_choice_takes_the_three_shapes_as_values_or_json(labels):
    question = typed_question("drex_choice", "choice", "What is this about?", labels)
    assert [label.label for label in question.labels] == ["billing", "tech"]
    assert set(question.to_wire()["criteria"]) == {"billing", "tech"}


def test_a_choice_option_with_a_null_description_is_sent_as_null():
    question = typed_question("drex_choice", "choice", "Topic?", {"billing": "Payments", "other": None})
    assert question.to_wire() == {
        "type": "choice",
        "instructions": "Topic?",
        "criteria": {"billing": "Payments", "other": None},
    }


def test_noul_labels_alone_are_no_criteria_and_described_ones_are_sent():
    assert "criteria" not in typed_question("drex_noul", "noul", "Refund?", ["true", "false"]).to_wire()
    wire = typed_question("drex_noul", "noul", None, {"true": "Many users are down"}).to_wire()
    assert wire == {"type": "noul", "instructions": "", "criteria": {"true": "Many users are down"}}


@pytest.mark.parametrize(
    ("kind", "instructions", "criteria", "match"),
    [
        ("noul", None, None, "needs instructions, or a description of true or false"),
        ("noul", "", ["true", "false"], "needs instructions"),
        ("noul", "Refund?", ["yes", "no"], "labels must be exactly true and false"),
        ("noul", "Refund?", ["true"], "labels must be exactly true and false"),
        ("choice", "Topic?", ["only"], "takes 2 to 255 labels, got 1"),
        ("choice", "Topic?", [f"l{i}" for i in range(256)], "takes 2 to 255 labels, got 256"),
        ("choice", "Topic?", ["a", "a"], "must be unique; 'a' is a duplicate"),
        ("choice", "Topic?", ["a", ""], "labels must not be empty"),
        ("choice", "Topic?", ["a", None], "drex_choice: labels must not contain a NULL label"),
        ("choice", "Topic?", None, "choice needs labels"),
        ("choice", "Topic?", "not json", "drex_choice: labels is not valid JSON"),
        ("choice", "Topic?", [{"label": "a", "colour": "red"}, "b"], "must be a list of labels"),
        ("score", "How upset?", {"calm": None, "angry": None}, "drex_score: levels must be a list of labels"),
        ("score", "How upset?", [{"label": "calm", "description": "x"}, "angry"], "score levels take no description"),
    ],
)
def test_typed_helpers_refuse(kind, instructions, criteria, match):
    refused(typed_question, f"drex_{kind}", kind, instructions, criteria, match=match)


def question(**fields):
    return {"type": "noul", "instructions": "Refund?", **fields}


@pytest.mark.parametrize(
    ("questions", "match"),
    [
        ({}, "takes 1 to 512 questions, got 0"),
        ({f"q{i}": question() for i in range(513)}, "takes 1 to 512 questions, got 513"),
        ({" ": question()}, "question ids must not be empty"),
        ({"Refund": question(), "refund": question()}, "unique ignoring case; 'refund' is a duplicate"),
        ({"__proto__": question()}, "'__proto__' is not allowed"),
        ({"a": question(colour="red")}, "unknown key 'colour'"),
        ({"a": {"type": "multiple_choice", "criteria": ["x", "y"]}}, "multiple_choice questions are not supported"),
        ({"a": {"type": "box"}}, "box questions are not supported"),
        ({"a": {"type": "ranking"}}, "must be noul, choice or score, not 'ranking'"),
        ({"a": {"type": "choice", "instructions": "Topic?"}}, "questions.a needs criteria"),
        ({"a": {"type": "noul", "instructions": 3}}, "instructions must be a string"),
        ([question()], "must be a JSON object of question id to question"),
        ("{", "questions is not valid JSON"),
    ],
)
def test_questions_refuse(questions, match):
    refused(parse_questions, "drex_decide", questions, match=match)


def test_questions_are_sent_with_string_instructions_and_drex_criteria():
    parsed = parse_questions(
        "drex_decide",
        json.dumps(
            {
                "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]},
                "topic": {"type": "choice", "criteria": [{"label": "billing", "description": "Payments"}, "tech"]},
                "refund": {"type": "noul", "instructions": None, "criteria": {"true": "Asks for money back"}},
            }
        ),
    )
    assert wire_questions(parsed) == {
        "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]},
        "topic": {"type": "choice", "instructions": "", "criteria": {"billing": "Payments", "tech": None}},
        "refund": {"type": "noul", "instructions": "", "criteria": {"true": "Asks for money back"}},
    }


def test_answers_are_read_from_the_response_and_never_invented():
    choice = typed_question("drex_choice", "choice", "Topic?", ["tech", "billing"])
    answers = {"choice": {"choice": "billing", "confidence": 0.9, "probabilities": {"tech": 0.1, "billing": 0.9}}}
    assert answer_value("drex_choice", choice, answers) == {
        "choice": "billing",
        "probabilities": [{"value": "billing", "probability": 0.9}, {"value": "tech", "probability": 0.1}],
        "confidence": 0.9,
    }
    score = typed_question("drex_score", "score", "Urgency?", ["low", "high"])
    answers = {"score": {"score": 0.7, "confidence": 0.6, "probabilities": {"0": 0.3, "1": 0.7}}}
    assert answer_value("drex_score", score, answers)["probabilities"] == [
        {"index": 0, "value": "low", "probability": 0.3},
        {"index": 1, "value": "high", "probability": 0.7},
    ]
    with pytest.raises(DrexError, match="has no number confidence"):
        answer_value("drex_score", score, {"score": {"score": 0.7, "probabilities": {"0": 0.3, "1": 0.7}}})
    with pytest.raises(DrexError, match="no answer to question 'score'"):
        answer_value("drex_score", score, {})
