"""Opt-in tests against the live Drex API: billed requests, synthetic content only.

    DREX_LIVE=1 NACE_API_KEY=... uv run pytest tests/live -m live

Run on their own (not with the mock suite, whose SparkSession points at the mock). `LiveTests.yml` runs them with
the `NACE_LIVE_API_KEY` secret, on workflow_dispatch only. 10 requests.
"""

import itertools
import json
import os
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mock"))
import pngs

from drex_spark import __version__, databricks
from drex_spark._batch import decide_batch, typed_batch
from drex_spark._client import Target

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("DREX_LIVE") != "1" or not os.environ.get("NACE_API_KEY"),
        reason="set DREX_LIVE=1 and NACE_API_KEY to call the live API",
    ),
]

RED = pngs.solid_png(pngs.COLOURS["red"])
BLUE = pngs.solid_png(pngs.COLOURS["blue"])
TARGET = Target(client_info=f"drex-spark/{__version__}", title="Drex for Apache Spark")


def colour_question(number: int) -> str:
    return json.dumps(
        {"colour": {"type": "choice", "instructions": f"Which colour is Image {number}?", "criteria": ["red", "blue"]}}
    )


def colours(images: list[list[bytes]], question: str, model: str | None = None) -> list[str]:
    bodies = decide_batch(
        TARGET,
        "drex_decide_image",
        pd.Series(["{}"] * len(images)),
        pd.Series(images, dtype="object"),
        itertools.repeat(question),
        itertools.repeat(model),
        8,
    )
    return [json.loads(body)["answers"]["colour"]["choice"] for body in bodies]


def test_drex_v2_answers_text():
    refund = typed_batch(
        TARGET,
        "drex_noul",
        "noul",
        pd.Series(["I was charged twice for one order. Please refund the duplicate.", "What are your opening hours?"]),
        itertools.repeat("Is the customer asking for a refund?"),
        itertools.repeat(None),
        itertools.repeat(None),
        8,
    )
    assert refund[0] > 0.5 > refund[1]
    team = typed_batch(
        TARGET,
        "drex_choice",
        "choice",
        pd.Series(["My invoice shows the wrong amount."]),
        itertools.repeat("Which team should handle this ticket?"),
        itertools.repeat('["billing", "technical", "account"]'),
        itertools.repeat("drex-v2.0"),
        8,
    )
    assert team.loc[0, "choice"] == "billing"
    assert sum(row["probability"] for row in team.loc[0, "probabilities"]) == pytest.approx(1, abs=0.01)


def test_drex_max_reads_one_image_then_two_then_the_same_two_reversed():
    assert colours([[RED]], colour_question(1)) == ["red"]
    assert colours([[RED, BLUE], [BLUE, RED]], colour_question(1)) == ["red", "blue"]
    assert colours([[RED, BLUE]], colour_question(2), model="drex-max") == ["blue"]


def test_the_unity_catalog_handler_sends_drex_databricks(monkeypatch):
    module = types.ModuleType("databricks.secrets")
    module.get = lambda *, catalog, schema, key: os.environ["NACE_API_KEY"]
    monkeypatch.setitem(sys.modules, "databricks", types.ModuleType("databricks"))
    monkeypatch.setitem(sys.modules, "databricks.secrets", module)
    databricks._read_secret.cache_clear()
    handler = databricks.handler("noul", secret="main.drex.nace_api_key", base_url=os.environ.get("NACE_BASE_URL"))
    batch = (
        pd.Series(["Please cancel my subscription and refund this month."]),
        pd.Series(["Is a refund requested?"]),
        pd.Series([None]),
        pd.Series([None]),
    )
    [result] = list(handler(iter([batch])))
    assert result[0] > 0.5


def test_pyspark_functions_in_a_local_session():
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    from drex_spark.functions import drex_decide_image, drex_noul

    builder = SparkSession.builder.master("local[2]").config("spark.ui.enabled", "false")
    for name in ("NACE_API_KEY", "NACE_BASE_URL"):
        if os.environ.get(name):
            builder = builder.config(f"spark.executorEnv.{name}", os.environ[name])
    spark = builder.getOrCreate()
    try:
        frame = spark.createDataFrame([("I need a refund for a double charge.", RED)], "body STRING, image BINARY")
        [row] = frame.select(
            drex_noul("body", "Is the customer asking for a refund?").alias("refund"),
            F.get_json_object(
                drex_decide_image(F.lit("{}"), "image", colour_question(1)), "$.answers.colour.choice"
            ).alias("colour"),
        ).collect()
        assert row.refund > 0.5
        assert row.colour == "red"
    finally:
        spark.stop()
