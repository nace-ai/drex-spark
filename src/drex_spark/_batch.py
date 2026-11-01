"""One pandas batch of a Drex function: each row's input as a request, the constants checked once per distinct value,
and Drex's answers as the function's SQL values. The PySpark functions and the Unity Catalog handlers both call it.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal
from typing import Any

import pandas as pd

from drex_spark._client import DrexRequest, Target, check_model, decide_all
from drex_spark._contract import (
    DEFAULT_IMAGE_MODEL,
    DEFAULT_MODEL,
    Question,
    answer_value,
    fail,
    parse_questions,
    typed_question,
    wire_questions,
)

#: The fields of the choice and score results, in order
CHOICE_FIELDS = ("choice", "probabilities", "confidence")
SCORE_FIELDS = ("score", "probabilities", "confidence")


def _missing(value: Any) -> bool:
    """SQL NULL as pandas hands it over: None, NaN or NaT. An array, a mapping or bytes is a value, even when empty."""
    if value is None:
        return True
    if isinstance(value, str | bytes | bytearray | memoryview | Mapping | list | tuple):
        return False
    if getattr(value, "ndim", 0) != 0:
        # A NumPy array
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _jsonable(value: Any) -> Any:
    """A STRUCT, MAP, ARRAY or VARIANT value as JSON, the way Spark's to_json writes it."""
    if _missing(value):
        return None
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "toJson"):
        # A VARIANT (pyspark.sql.types.VariantVal)
        return json.loads(value.toJson())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, bytes | bytearray | memoryview):
        return base64.b64encode(bytes(value)).decode()
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, dt.datetime | dt.date | dt.time):
        return value.isoformat()
    if getattr(value, "ndim", None) == 0:
        # A NumPy scalar
        return _jsonable(value.item())
    if hasattr(value, "tolist"):
        # A NumPy array, which is how Arrow hands an ARRAY to pandas
        value = value.tolist()
    if isinstance(value, list | tuple):
        # Arrow hands a MAP to pandas as a list of (key, value) pairs
        if value and all(isinstance(item, tuple) and len(item) == 2 for item in value):
            return {str(key): _jsonable(item) for key, item in value}
        return [_jsonable(item) for item in value]
    return str(value)


def _rows(values: pd.Series | pd.DataFrame) -> list[Any]:
    """A batch's argument as one value per row. A STRUCT argument arrives as a DataFrame, one column per field."""
    if isinstance(values, pd.DataFrame):
        return [None if row is None else dict(row) for row in values.to_dict("records")]
    return list(values)


def _images(value: Any) -> list[bytes] | None:
    """A BINARY or ARRAY<BINARY> value as its images, or `None` when it is neither."""
    if isinstance(value, bytes | bytearray | memoryview):
        return [bytes(value)]
    if hasattr(value, "tolist") and not isinstance(value, str):
        value = value.tolist()
    if isinstance(value, list) and all(isinstance(item, bytes | bytearray | memoryview | None) for item in value):
        # An empty list is images too: none, so the row sends nothing
        return [bytes(item) if item is not None else None for item in value]  # type: ignore[misc]
    return None


def _input(value: Any) -> tuple[Any, list[bytes]] | None:
    """A typed helper's input as (state, images), or `None` when the row sends nothing.

    STRING is text, BINARY or ARRAY<BINARY> is images (with an empty text state, as in drex-duckdb), and a STRUCT, MAP
    or VARIANT is JSON state. A NULL input, a NULL image or an empty list of images sends nothing.
    """
    if _missing(value):
        return None
    if isinstance(value, str):
        return value, []
    images = _images(value)
    if images is not None:
        return None if not images or any(image is None for image in images) else ("", images)
    return _jsonable(value), []


def _state(function: str, value: Any) -> Any:
    """drex_decide's state: JSON text, or a STRUCT, MAP or VARIANT value. Invalid JSON fails the query."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError as error:
            raise fail(function, f"state is not valid JSON: {error}") from None
    return _jsonable(value)


def _text(function: str, name: str, value: Any) -> str | None:
    if _missing(value) or (isinstance(value, str) and not value):
        return None
    if not isinstance(value, str):
        raise fail(function, f"{name} must be a string")
    return value


def _key(value: Any) -> str:
    """A constant argument's identity, so each distinct value is checked once per batch."""
    return json.dumps(_jsonable(value), sort_keys=True)


def _send(
    target: Target,
    function: str,
    concurrency: int,
    models: set[str],
    rows: list[tuple[int, str, DrexRequest]],
    size: int,
) -> list[str | None]:
    """Checks every model the batch names, then sends each (row, model, request), one call per model, and returns each
    row's body, or `None`."""
    bodies: list[str | None] = [None] * size
    by_model: dict[str, list[tuple[int, DrexRequest]]] = {model: [] for model in sorted(models)}
    for row, model, request in rows:
        by_model[model].append((row, request))
    for model, entries in by_model.items():
        check_model(target, function, model, images=any(request.images for _, request in entries))
        if entries:
            answers = decide_all(target, function, model, [request for _, request in entries], concurrency)
            for (row, _), body in zip(entries, answers, strict=True):
                bodies[row] = body
    return bodies


def typed_batch(
    target: Target,
    function: str,
    kind: str,
    inputs: pd.Series | pd.DataFrame,
    instructions: Iterable[Any],
    criteria: Iterable[Any],
    models: Iterable[Any],
    concurrency: int,
) -> pd.Series | pd.DataFrame:
    """`drex_noul`, `drex_choice` or `drex_score` over one batch: a Series of probabilities, or a DataFrame of rows."""
    questions: dict[str, Question] = {}
    models_seen: set[str] = set()
    plan: list[tuple[int, str, DrexRequest]] = []
    asked: list[Question | None] = []
    for row, (value, row_instructions, row_criteria, row_model) in enumerate(
        zip(_rows(inputs), instructions, criteria, models, strict=False)
    ):
        request = _input(value)
        if request is None:
            # Nothing is sent, so nothing of the row is checked: its other arguments may be NULL with it
            asked.append(None)
            continue
        text = _text(function, "instructions", row_instructions)
        criteria_value = (
            None if _missing(row_criteria) or (isinstance(row_criteria, str) and not row_criteria) else row_criteria
        )
        key = f"{text!r}\0{_key(criteria_value)}"
        question = questions.get(key)
        if question is None:
            question = questions[key] = typed_question(function, kind, text, criteria_value)
        model = _text(function, "model", row_model) or DEFAULT_MODEL
        models_seen.add(model)
        asked.append(question)
        state, images = request
        plan.append((row, model, DrexRequest(state, wire_questions([question]), images)))

    bodies = _send(target, function, concurrency, models_seen, plan, len(asked))
    values = [
        None if body is None or question is None else answer_value(function, question, json.loads(body).get("answers"))
        for question, body in zip(asked, bodies, strict=True)
    ]
    if kind == "noul":
        return pd.Series(values, dtype="float64")
    fields = CHOICE_FIELDS if kind == "choice" else SCORE_FIELDS
    # A struct result is a DataFrame of its fields, so a row with no answer has every field NULL
    return pd.DataFrame(
        {name: [None if value is None else value[name] for value in values] for name in fields}, columns=list(fields)
    )


def decide_batch(
    target: Target,
    function: str,
    states: pd.Series | pd.DataFrame,
    images: pd.Series | None,
    questions: Iterable[Any],
    models: Iterable[Any],
    concurrency: int,
) -> pd.Series:
    """`drex_decide` or, with `images`, `drex_decide_image` over one batch: Drex's response bodies as written."""
    default_model = DEFAULT_MODEL if images is None else DEFAULT_IMAGE_MODEL
    parsed: dict[str, dict[str, Any]] = {}
    models_seen: set[str] = set()
    plan: list[tuple[int, str, DrexRequest]] = []
    image_values: Sequence[Any] = [None] * len(states) if images is None else list(images)
    rows = _rows(states)
    for row, (state, row_images, row_questions, row_model) in enumerate(
        zip(rows, image_values, questions, models, strict=False)
    ):
        if _missing(state) or _missing(row_questions):
            continue
        sent: list[bytes] = []
        if images is not None:
            found = None if _missing(row_images) else _images(row_images)
            if found is None and not _missing(row_images):
                raise fail(function, "images must be BINARY or ARRAY<BINARY>")
            if not found or any(image is None for image in found):
                # Nothing is sent, so nothing of the row is checked, as for a NULL state
                continue
            sent = found
        model = _text(function, "model", row_model) or default_model
        models_seen.add(model)
        key = row_questions if isinstance(row_questions, str) else _key(row_questions)
        wire = parsed.get(key)
        if wire is None:
            wire = parsed[key] = wire_questions(parse_questions(function, row_questions))
        plan.append((row, model, DrexRequest(_state(function, state), wire, sent)))

    return pd.Series(_send(target, function, concurrency, models_seen, plan, len(rows)), dtype="object")
