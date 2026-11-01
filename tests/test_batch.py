"""One batch of each function against the Drex mock, in this process: the answers, the requests sent, and the error
rules. tests/test_spark.py runs the same code inside Spark's Python workers."""

import itertools
import json
import logging
import time

import pandas as pd
import pytest
from conftest import BLUE, RED

from drex_spark import _client
from drex_spark._batch import decide_batch, typed_batch
from drex_spark._contract import DrexError

REFUND = "Is the customer asking for a refund?"
TOPIC = "What is this support ticket about?"
TOPICS = {
    "billing": "Payments, invoices, refunds, or charges",
    "technical": "Bugs, errors, or outages",
    "account": "Login, profile, or permissions",
    "other": None,
}
UPSET = ["Calm", "Mildly annoyed", "Frustrated", "Angry"]
COLOURS = {"type": "choice", "criteria": ["red", "blue"]}


def typed(target, kind, inputs, instructions, criteria=None, model=None, concurrency=8):
    return typed_batch(
        target,
        f"drex_{kind}",
        kind,
        pd.Series(inputs),
        itertools.repeat(instructions),
        itertools.repeat(criteria),
        itertools.repeat(model),
        concurrency,
    )


def decide(target, states, questions, model=None, images=None, concurrency=8):
    function = "drex_decide" if images is None else "drex_decide_image"
    image_series = None if images is None else pd.Series(images, dtype="object")
    questions = json.dumps(questions) if not isinstance(questions, str) else questions
    return decide_batch(
        target,
        function,
        pd.Series(states, dtype="object"),
        image_series,
        itertools.repeat(questions),
        itertools.repeat(model),
        concurrency,
    )


@pytest.mark.parametrize("model", ["drex-v2.0", "drex-max"])
def test_noul_choice_and_score(mock, ns, model):
    target = mock.target(ns)
    ticket = "I was charged twice. Please refund the duplicate payment."
    assert typed(target, "noul", [ticket], REFUND, model=model).tolist() == [0.9868]

    choice = typed(target, "choice", [ticket], TOPIC, TOPICS, model=model)
    assert choice.to_dict("records") == [
        {
            "choice": "billing",
            "probabilities": [
                {"value": "account", "probability": 0.0006},
                {"value": "billing", "probability": 0.9895},
                {"value": "other", "probability": 0.0049},
                {"value": "technical", "probability": 0.005},
            ],
            "confidence": 0.986,
        }
    ]
    score = typed(target, "score", [ticket], "How upset is the customer?", UPSET, model=model)
    assert score.loc[0, "score"] == 1.5306
    assert [row["index"] for row in score.loc[0, "probabilities"]] == [0, 1, 2, 3]
    assert score.loc[0, "probabilities"][1] == {"index": 1, "value": "Mildly annoyed", "probability": 0.3516}

    sent = mock.log(ns)
    assert [record["served_model"] for record in sent] == [model] * 3
    assert {record["x_nace_client"] for record in sent} == {"drex-spark/test"}
    assert all(record["instructions_present"] for record in sent)
    assert sent[1]["state"] == json.dumps(ticket)


def test_a_mixed_batch_sends_one_request_per_non_null_row_in_order(mock, ns):
    target = mock.target(ns)
    rows = ["a", None, "b", "c", None]
    result = typed(target, "noul", rows, "Is this a?")
    assert result.isna().tolist() == [False, True, False, False, True]
    states = sorted(record["state"] for record in mock.log(ns))
    assert states == [json.dumps(text) for text in ["a", "b", "c"]]


def test_json_state_from_a_struct_or_map_value(mock, ns):
    target = mock.target(ns)
    typed(target, "noul", [{"message": "charged twice", "amount": 12}], REFUND)
    typed(target, "noul", [[("message", "charged twice")]], REFUND)
    sent = mock.log(ns)
    assert [record["state_type"] for record in sent] == ["object", "object"]
    assert json.loads(sent[0]["state"]) == {"message": "charged twice", "amount": 12}
    assert json.loads(sent[1]["state"]) == {"message": "charged twice"}


def test_drex_decide_returns_the_body_unchanged(mock, ns):
    questions = {"refund": {"type": "noul", "instructions": REFUND}, "topic": {"type": "choice", "criteria": TOPICS}}
    [body] = decide(mock.target(ns), ['{"message": "I was charged twice"}'], questions).tolist()
    parsed = json.loads(body)
    assert set(parsed) == {"model", "answers", "usage", "evaluation_time_ms", "request_id"}
    assert parsed["model"] == "drex-v2.0"
    assert parsed["answers"]["refund"]["noul"] == 0.9868
    assert json.loads(mock.log(ns)[0]["body"])["state"] == {"message": "I was charged twice"}


def test_drex_decide_refuses_proto_and_invalid_state(mock, ns):
    with pytest.raises(DrexError, match="'__proto__' is not allowed"):
        decide(mock.target(ns), ["{}"], {"__proto__": {"type": "noul", "instructions": REFUND}})
    with pytest.raises(DrexError, match="drex_decide: state is not valid JSON"):
        decide(mock.target(ns), ["{not json"], {"refund": {"type": "noul", "instructions": REFUND}})
    assert mock.log(ns) == []


def test_drex_max_with_one_image_then_two_then_reversed(mock, ns):
    target = mock.target(ns)
    which = {"colour": {**COLOURS, "instructions": "Which colour is Image 1?"}}
    second = {"colour": {**COLOURS, "instructions": "Which colour is Image 2?"}}
    answers = [
        json.loads(body)["answers"]["colour"]["choice"]
        for body in [
            decide(target, ["{}"], which, images=[RED])[0],
            decide(target, ["{}"], which, images=[[RED, BLUE]])[0],
            decide(target, ["{}"], which, images=[[BLUE, RED]])[0],
            decide(target, ["{}"], second, images=[[BLUE, RED]])[0],
        ]
    ]
    assert answers == ["red", "red", "blue", "red"]
    sent = mock.log(ns)
    # drex_decide_image defaults to drex-max
    assert {record["served_model"] for record in sent} == {"drex-max"}
    assert [record["image_colours"] for record in sent] == [["red"], ["red", "blue"], ["blue", "red"], ["blue", "red"]]


def test_a_typed_helper_with_images_needs_an_image_model(mock, ns):
    target = mock.target(ns)
    with pytest.raises(DrexError, match="drex_choice: drex-v2.0 does not read images; use an image model"):
        typed(target, "choice", [RED], "Which colour?", ["red", "blue"])
    with pytest.raises(DrexError, match="drex-v1.5 does not read images"):
        typed(target, "choice", [RED], "Which colour?", ["red", "blue"], model="drex-v1.5")
    assert mock.log(ns) == []
    result = typed(target, "choice", [RED], "Which colour is Image 1?", ["red", "blue"], model="drex-max")
    assert result.loc[0, "choice"] == "red"
    assert mock.log(ns)[0]["state"] == '""'


def test_a_null_image_or_an_empty_list_sends_nothing(mock, ns):
    which = {"colour": {**COLOURS, "instructions": "Which colour is Image 1?"}}
    result = decide(mock.target(ns), ["{}", "{}", "{}"], which, images=[None, [], [RED, None]])
    assert result.tolist() == [None, None, None]
    assert mock.log(ns) == []


def test_noul_criteria_without_instructions(mock, ns):
    criteria = {"true": "Many users cannot use the product", "false": "A single user or a question"}
    result = typed(mock.target(ns), "noul", ["Nobody can log in"], None, criteria)
    assert result.notna().all()
    [sent] = json.loads(mock.log(ns)[0]["questions"])
    assert sent == {"id": "noul", "type": "noul", "instructions": "", "criteria": criteria}


def test_401_fails_the_query(mock, ns):
    with pytest.raises(DrexError, match=r"drex_noul: could not list Drex's models|Invalid API key"):
        typed(mock.target(ns, "401"), "noul", ["x"], REFUND)


def test_402_fails_the_query_with_the_billing_page(mock, ns):
    target = mock.target(ns, "402")
    with pytest.raises(DrexError, match="insufficient available funds.*Billing: .*/ns/.*/402/dashboard/billing"):
        typed(target, "noul", ["x"], REFUND)


def test_a_422_about_the_questions_names_each_issue(mock, ns):
    with pytest.raises(DrexError) as raised:
        typed(mock.target(ns, "422"), "noul", ["x"], REFUND)
    message = str(raised.value)
    assert message.startswith("drex_noul: Drex refused the request: ")
    assert "questions.a.instructions: must be a string; questions.a.instructions: noul needs instructions" in message


def test_an_unknown_model_fails_the_query_and_names_the_listed_ones(mock, ns):
    with pytest.raises(DrexError, match="unknown model 'drex-v9'; GET /v1/models lists .*drex-v2.0.*drex-max"):
        typed(mock.target(ns), "noul", ["x"], REFUND, model="drex-v9")
    assert mock.log(ns) == []


@pytest.mark.parametrize(("scenario", "wait_ms"), [("429-once", 1000), ("529-once", 2000)])
def test_429_and_529_are_retried_after_retry_after_ms_with_the_same_body(mock, ns, scenario, wait_ms):
    result = typed(mock.target(ns, scenario), "noul", ["x"], REFUND)
    assert result.tolist() == [0.9868]
    first, second = mock.log(ns)
    assert (first["status"], second["status"]) == (int(scenario[:3]), 200)
    assert second["waited_ms"] >= wait_ms
    assert first["body_sha256"] == second["body_sha256"]
    assert first["idempotency_key"] == second["idempotency_key"] and first["idempotency_key"]


def test_a_row_drex_stays_at_capacity_for_is_null_once_the_budget_passes(mock, ns, caplog, monkeypatch):
    monkeypatch.setattr(_client, "REQUEST_BUDGET_SECONDS", 5.0)
    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="drex_spark"):
        result = typed(mock.target(ns, "529-always"), "noul", ["x"], REFUND)
    assert result.isna().all()
    assert len(mock.log(ns)) in (2, 3)
    assert 4.0 <= time.monotonic() - started <= 5.0
    assert "drex_noul: a row returned NULL: no answer after" in caplog.text
    assert "the last HTTP 529: Drex is at capacity" in caplog.text


def test_rows_over_the_accounts_in_flight_limit_wait_their_turn(mock, ns):
    """The Databricks run's failure: more requests offered than the account's limit. Every row is answered, after as
    many refusals as it takes, and no row's own requests number more than the batch's concurrency at once."""
    result = typed(mock.target(ns, "limit-2"), "noul", [f"row {i}" for i in range(40)], REFUND, concurrency=8)
    assert result.notna().all()
    log = mock.log(ns)
    answered = [record for record in log if record["status"] == 200]
    assert len(answered) == 40
    assert len({record["body_sha256"] for record in answered}) == 40
    assert any(record["status"] == 429 for record in log)
    assert max(record["attempt"] for record in log) > 5
    assert max(record["in_flight"] for record in log) <= 8


def test_an_oversized_row_is_null_and_the_query_continues(mock, ns, caplog):
    big = "x" * (4 * 1024 * 1024)
    with caplog.at_level(logging.WARNING, logger="drex_spark"):
        result = typed(mock.target(ns), "noul", ["small", big, "small too"], REFUND)
    assert result.isna().tolist() == [False, True, False]
    assert len(mock.log(ns)) == 2
    assert "over Drex's limit of 4194304" in caplog.text
    assert "xxxx" not in caplog.text


def test_a_row_drex_refuses_as_too_long_is_null_and_the_warning_names_state_only(mock, ns, caplog):
    with caplog.at_level(logging.WARNING, logger="drex_spark"):
        result = typed(mock.target(ns, "too-long"), "noul", ["short", "secret words " * 1000], REFUND)
    assert result.isna().tolist() == [False, True]
    assert "HTTP 422: " in caplog.text
    assert "secret words" not in caplog.text


def test_an_image_that_is_not_png_jpeg_or_webp_is_null(mock, ns, caplog):
    which = {"colour": {**COLOURS, "instructions": "Which colour is Image 1?"}}
    with caplog.at_level(logging.WARNING, logger="drex_spark"):
        result = decide(mock.target(ns), ["{}", "{}"], which, images=[b"GIF89a....", RED])
    assert result.isna().tolist() == [True, False]
    assert "image 1 is not a PNG, JPEG or WebP image" in caplog.text
    assert len(mock.log(ns)) == 1


def test_the_concurrency_cap_holds_within_a_batch(mock, ns):
    target = mock.target(ns, "slow-150")
    result = typed(target, "noul", [f"row {i}" for i in range(12)], REFUND, concurrency=3)
    assert result.notna().all()
    assert max(record["in_flight"] for record in mock.log(ns)) == 3


def test_concurrency_below_one_fails(mock, ns):
    with pytest.raises(DrexError, match="concurrency must be at least 1"):
        typed(mock.target(ns), "noul", ["x"], REFUND, concurrency=0)


def test_constants_are_checked_on_every_row_that_is_sent(mock, ns):
    """A Spark SQL or Databricks call passes the constants as columns: a bad one fails the query on any row it sends."""
    target = mock.target(ns)
    with pytest.raises(DrexError, match="takes 2 to 255 labels"):
        typed_batch(
            target, "drex_choice", "choice", pd.Series(["a", "b"]), ["Topic?"] * 2, [TOPICS, '["only"]'], [None] * 2, 8
        )
    with pytest.raises(DrexError, match="unknown model 'nope'"):
        typed_batch(
            target, "drex_noul", "noul", pd.Series(["a", "b"]), [REFUND, REFUND], [None, None], [None, "nope"], 8
        )


def test_a_null_row_checks_nothing_since_it_sends_nothing(mock, ns):
    """Arguments made from the row, such as concat('Is ', body, ' a refund?'), are NULL with it."""
    target = mock.target(ns)
    answers = typed_batch(
        target, "drex_noul", "noul", pd.Series(["a", None]), [REFUND, None], [None, '["only"]'], [None, "nope"], 8
    )
    assert answers.isna().tolist() == [False, True]
    bodies = decide_batch(
        target,
        "drex_decide",
        pd.Series(['{"a": 1}', None]),
        None,
        ['{"q": {"type": "noul", "instructions": "a?"}}', "{"],
        [None, "nope"],
        8,
    )
    assert [body is None for body in bodies] == [False, True]
    assert len(mock.log(ns)) == 2
