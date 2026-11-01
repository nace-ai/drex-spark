"""Drex's functions as PySpark columns, and `register` for Spark SQL.

Each function returns a `Column` backed by a pandas UDF. Its constant arguments (instructions, criteria, questions,
model) are checked here, on the driver, before any row is read; the model is checked against `GET /v1/models` on the
executors, which hold the key (`spark.executorEnv.NACE_API_KEY`).
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterator
from typing import Any, Tuple  # noqa: UP035 - PySpark reads typing.Tuple hints

import pandas as pd
from pyspark.sql import Column, SparkSession
from pyspark.sql.functions import pandas_udf  # type: ignore[attr-defined]
from pyspark.sql.types import (
    ArrayType,
    DataType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from drex_spark._batch import decide_batch, typed_batch
from drex_spark._client import Target
from drex_spark._contract import DEFAULT_IMAGE_MODEL, DEFAULT_MODEL, fail, parse_questions, typed_question
from drex_spark._version import __version__

__all__ = ["drex_choice", "drex_decide", "drex_decide_image", "drex_noul", "drex_score", "register"]

logger = logging.getLogger("drex_spark")

CONCURRENCY_CONF = "spark.drex.concurrency"
DEFAULT_CONCURRENCY = 8

TARGET = Target(client_info=f"drex-spark/{__version__}", title="Drex for Apache Spark")

CHOICE_TYPE = StructType(
    [
        StructField("choice", StringType()),
        StructField(
            "probabilities",
            ArrayType(StructType([StructField("value", StringType()), StructField("probability", DoubleType())])),
        ),
        StructField("confidence", DoubleType()),
    ]
)
SCORE_TYPE = StructType(
    [
        StructField("score", DoubleType()),
        StructField(
            "probabilities",
            ArrayType(
                StructType(
                    [
                        StructField("index", IntegerType()),
                        StructField("value", StringType()),
                        StructField("probability", DoubleType()),
                    ]
                )
            ),
        ),
        StructField("confidence", DoubleType()),
    ]
)
RESULT_TYPES: dict[str, DataType] = {"noul": DoubleType(), "choice": CHOICE_TYPE, "score": SCORE_TYPE}

ColumnOrName = Column | str


def worker_concurrency(budget: int | None = None, spark: SparkSession | None = None) -> int:
    """Each Python worker's share of a job-wide in-flight budget: `max(1, budget // defaultParallelism)`.

    The budget is `budget`, else `spark.drex.concurrency`, else 8. Where the parallelism can't be read (Spark Connect),
    each worker gets the whole budget, and Drex's 429s with their retries pace the rest.
    """
    spark = spark or SparkSession.getActiveSession()
    if budget is None:
        try:
            configured = spark.conf.get(CONCURRENCY_CONF, None) if spark is not None else None
        except Exception:  # noqa: BLE001 - Databricks serverless refuses to read a configuration it doesn't know
            configured = None
        try:
            budget = int(configured) if configured else DEFAULT_CONCURRENCY
        except ValueError:
            raise fail("drex_spark", f"{CONCURRENCY_CONF} must be a whole number, not {configured!r}") from None
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise fail("drex_spark", "concurrency must be at least 1")
    try:
        parallelism = spark.sparkContext.defaultParallelism if spark is not None else None
    except Exception:  # noqa: BLE001 - Spark Connect has no SparkContext
        parallelism = None
    if not parallelism:
        logger.warning("drex_spark: can't read the parallelism, so each worker may send %d requests at once", budget)
        return budget
    return max(1, budget // parallelism)


def _udf(function: str, run: Any, return_type: DataType) -> Any:
    """`run` as a pandas UDF named after the function, so an unaliased column reads `drex_noul(body)`.

    Every call is a billed request, so it is nondeterministic: Spark must never fold, reorder or duplicate it.
    """
    run.__name__ = run.__qualname__ = function
    return pandas_udf(run, return_type).asNondeterministic()


def _model(function: str, model: str | None, default: str) -> str:
    if model is not None and not isinstance(model, str):
        raise fail(function, "model must be a string")
    return model or default


def _typed(
    function: str, kind: str, input: ColumnOrName, instructions: Any, criteria: Any, model: Any, concurrency: Any
) -> Column:
    # Checked now, so a refused question fails before any row is read
    typed_question(function, kind, instructions or None, criteria)
    model = _model(function, model, DEFAULT_MODEL)
    limit = worker_concurrency(concurrency)

    def run(batches: Iterator[pd.Series]) -> Iterator[pd.Series]:
        for batch in batches:
            yield typed_batch(
                TARGET,
                function,
                kind,
                batch,
                itertools.repeat(instructions),
                itertools.repeat(criteria),
                itertools.repeat(model),
                limit,
            )

    return _udf(function, run, RESULT_TYPES[kind])(input)


def drex_noul(
    input: ColumnOrName,
    instructions: str | None,
    criteria: Any = None,
    model: str = DEFAULT_MODEL,
    concurrency: int | None = None,
) -> Column:
    """The probability that the answer is yes, a DOUBLE in [0, 1].

    `input` is STRING (text), STRUCT, MAP or VARIANT (JSON state), or BINARY or ARRAY<BINARY> (images, for an image
    model). `criteria` describes `true` and `false`, as a list of labels, a list of `{"label", "description"}`, an
    object of label to description, or the same as JSON text. A NULL input returns NULL and sends nothing.
    """
    return _typed("drex_noul", "noul", input, instructions, criteria, model, concurrency)


def drex_choice(
    input: ColumnOrName,
    instructions: str | None,
    labels: Any,
    model: str = DEFAULT_MODEL,
    concurrency: int | None = None,
) -> Column:
    """One of `labels`: `STRUCT<choice, probabilities ARRAY<STRUCT<value, probability>>, confidence>`."""
    return _typed("drex_choice", "choice", input, instructions, labels, model, concurrency)


def drex_score(
    input: ColumnOrName,
    instructions: str | None,
    levels: Any,
    model: str = DEFAULT_MODEL,
    concurrency: int | None = None,
) -> Column:
    """A score on `levels`, in order.

    The result is `STRUCT<score, probabilities ARRAY<STRUCT<index, value, probability>>, confidence>`.
    """
    return _typed("drex_score", "score", input, instructions, levels, model, concurrency)


def drex_decide(
    state: ColumnOrName, questions: Any, model: str = DEFAULT_MODEL, concurrency: int | None = None
) -> Column:
    """Drex's response body for `state` (JSON text, or a STRUCT, MAP or VARIANT) and `questions`, as a JSON STRING.

    `questions` is an object of id to `{type, instructions, criteria}`, or the same as JSON text.
    """
    return _decide("drex_decide", state, None, questions, _model("drex_decide", model, DEFAULT_MODEL), concurrency)


def drex_decide_image(
    state: ColumnOrName,
    images: ColumnOrName,
    questions: Any,
    model: str = DEFAULT_IMAGE_MODEL,
    concurrency: int | None = None,
) -> Column:
    """`drex_decide` with the images of `images`, a BINARY or ARRAY<BINARY> column. The model defaults to drex-max."""
    function = "drex_decide_image"
    return _decide(function, state, images, questions, _model(function, model, DEFAULT_IMAGE_MODEL), concurrency)


def _decide(
    function: str, state: ColumnOrName, images: ColumnOrName | None, questions: Any, model: str, concurrency: Any
) -> Column:
    parse_questions(function, questions)
    limit = worker_concurrency(concurrency)

    def answer(states: pd.Series, pictures: pd.Series | None) -> pd.Series:
        return decide_batch(
            TARGET, function, states, pictures, itertools.repeat(questions), itertools.repeat(model), limit
        )

    if images is None:

        def run(batches: Iterator[pd.Series]) -> Iterator[pd.Series]:
            for states in batches:
                yield answer(states, None)

        return _udf(function, run, StringType())(state)

    def run_images(batches: Iterator[Tuple[pd.Series, pd.Series]]) -> Iterator[pd.Series]:  # noqa: UP006 - PySpark reads typing.Tuple
        for states, pictures in batches:
            yield answer(states, pictures)

    return _udf(function, run_images, StringType())(state, images)


def _arguments(function: str, batch: Any, names: tuple[str, ...]) -> tuple[pd.Series, ...]:
    """A registered function's batch as its arguments. Spark SQL passes a pandas UDF whatever it was called with."""
    if not isinstance(batch, tuple) or len(batch) != len(names):
        count = len(batch) if isinstance(batch, tuple) else 1
        raise fail(
            function,
            f"takes {len(names)} arguments ({', '.join(names)}), got {count}; pass NULL for the ones you leave out",
        )
    return batch


def register(spark: SparkSession, concurrency: int | None = None) -> None:
    """Registers the five functions for Spark SQL, as temporary functions of `spark`.

    Registered pandas UDFs take no defaults, so every argument is required: a NULL model is the default model, and a
    NULL criteria a noul without criteria. Constants are checked once per distinct value in each batch.
    """
    limit = worker_concurrency(concurrency, spark)

    def typed(function: str, kind: str) -> Any:
        names = ("input", "instructions", "labels" if kind == "choice" else "levels" if kind == "score" else "criteria")

        def run(batches: Iterator[Tuple[pd.Series, pd.Series, pd.Series, pd.Series]]) -> Iterator[pd.Series]:  # noqa: UP006
            for batch in batches:
                inputs, instructions, criteria, models = _arguments(function, batch, (*names, "model"))
                yield typed_batch(TARGET, function, kind, inputs, instructions, criteria, models, limit)

        return _udf(function, run, RESULT_TYPES[kind])

    def decide(batches: Iterator[Tuple[pd.Series, pd.Series, pd.Series]]) -> Iterator[pd.Series]:  # noqa: UP006
        for batch in batches:
            states, questions, models = _arguments("drex_decide", batch, ("state", "questions", "model"))
            yield decide_batch(TARGET, "drex_decide", states, None, questions, models, limit)

    def decide_image(batches: Iterator[Tuple[pd.Series, pd.Series, pd.Series, pd.Series]]) -> Iterator[pd.Series]:  # noqa: UP006
        for batch in batches:
            names = ("state", "images", "questions", "model")
            states, images, questions, models = _arguments("drex_decide_image", batch, names)
            yield decide_batch(TARGET, "drex_decide_image", states, images, questions, models, limit)

    for kind in ("noul", "choice", "score"):
        spark.udf.register(f"drex_{kind}", typed(f"drex_{kind}", kind))
    spark.udf.register("drex_decide", _udf("drex_decide", decide, StringType()))
    spark.udf.register("drex_decide_image", _udf("drex_decide_image", decide_image, StringType()))
