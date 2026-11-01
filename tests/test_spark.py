"""The PySpark functions and Spark SQL in a local SparkSession, whose Python workers call the mock."""

import json
from unittest import mock

import pytest
from conftest import BLUE, RED, SPARK_PARALLELISM
from pyspark.errors import PythonException
from pyspark.sql import functions as F
from pyspark.sql.types import BinaryType, StringType, StructField, StructType

import drex_spark
from drex_spark import DrexError
from drex_spark.functions import (
    drex_choice,
    drex_decide,
    drex_decide_image,
    drex_noul,
    drex_score,
    worker_concurrency,
)

pytestmark = pytest.mark.spark

REFUND = "Is the customer asking for a refund?"
TICKETS = [
    (1, "I was charged twice. Please refund the duplicate payment."),
    (2, None),
    (3, "How do I reset my password?"),
]


@pytest.fixture
def tickets(spark):
    return spark.createDataFrame(TICKETS, "id INT, body STRING")


def test_the_typed_helpers_return_spark_values(tickets, spark_log):
    rows = sorted(
        tickets.select(
            "id",
            drex_noul("body", REFUND).alias("refund"),
            drex_choice(F.col("body"), "Which team?", ["billing", "technical"]).alias("team"),
            drex_score("body", "How urgent?", ["low", "medium", "high"], model="drex-max").alias("urgency"),
        ).collect(),
        key=lambda row: row.id,
    )
    assert [row.refund for row in rows] == [0.9868, None, 0.9868]
    assert rows[0].team.choice == "billing"
    assert [p.value for p in rows[0].team.probabilities] == ["billing", "technical"]
    assert rows[1].team.choice is None
    assert [(p["index"], p["value"]) for p in rows[2].urgency.probabilities] == [(0, "low"), (1, "medium"), (2, "high")]
    sent = spark_log()
    assert len(sent) == 6
    assert {record["x_nace_client"] for record in sent} == {f"drex-spark/{drex_spark.__version__}"}


def test_an_unaliased_column_is_named_after_the_function(tickets, spark_log):
    assert tickets.select(drex_noul("body", REFUND)).columns == ["drex_noul(body)"]


def test_a_struct_or_map_input_is_json_state(spark, spark_log):
    frame = spark.createDataFrame([("charged twice", 12)], "message STRING, amount INT")
    frame.select(
        drex_noul(F.struct("message", "amount"), REFUND).alias("a"),
        drex_noul(F.create_map(F.lit("message"), F.col("message")), REFUND).alias("b"),
    ).collect()
    states = sorted(record["state"] for record in spark_log())
    assert states == ['{"message":"charged twice","amount":12}', '{"message":"charged twice"}']


def test_drex_decide_and_drex_decide_image(spark, spark_log):
    schema = StructType([StructField("state", StringType()), StructField("image", BinaryType())])
    frame = spark.createDataFrame([('{"order": 1}', RED), ('{"order": 2}', BLUE)], schema)
    questions = {"colour": {"type": "choice", "instructions": "Which colour is Image 1?", "criteria": ["red", "blue"]}}
    rows = frame.select(
        F.get_json_object(drex_decide_image("state", "image", questions), "$.answers.colour.choice").alias("colour"),
        F.get_json_object(drex_decide("state", {"refund": {"type": "noul", "instructions": REFUND}}), "$.model").alias(
            "model"
        ),
    ).collect()
    assert sorted(row.colour for row in rows) == ["blue", "red"]
    assert {row.model for row in rows} == {"drex-v2.0"}
    assert sorted(record["served_model"] for record in spark_log()) == [
        "drex-max",
        "drex-max",
        "drex-v2.0",
        "drex-v2.0",
    ]


def test_order_by_after_a_drex_column_sends_every_row_twice(tickets, spark_log):
    """Spark samples a sort's input to plan its ranges, which runs the plan, and so the requests, once more. The docs
    tell users to save answers (a table, or a cached and counted DataFrame) before they sort or reuse them."""
    tickets.select("id", drex_noul("body", REFUND).alias("refund")).orderBy("refund").collect()
    assert len(spark_log()) == 2 * 2

    answers = tickets.select("id", drex_noul("body", REFUND).alias("refund")).cache()
    answers.count()
    spark_log_after_save = len(spark_log())
    answers.orderBy("refund").collect()
    assert len(spark_log()) == spark_log_after_save == 2 * 2 + 2
    answers.unpersist()


def test_spark_sql_after_register(spark, tickets, spark_log):
    drex_spark.register(spark)
    tickets.createOrReplaceTempView("tickets")
    rows = spark.sql(
        """
        SELECT id,
               drex_noul(body, 'Is the customer asking for a refund?', NULL, NULL) AS refund,
               drex_choice(body, 'Which team?', '["billing", "technical"]', 'drex-max').choice AS team,
               drex_score(body, 'How urgent?', array('low', 'high'), NULL).score AS urgency,
               get_json_object(drex_decide('{"a": 1}', '{"q": {"type": "noul", "instructions": "Is a 1?"}}', NULL),
                               '$.answers.q.noul') AS q
        FROM tickets
        """
    ).collect()
    rows = sorted(rows, key=lambda row: row.id)
    assert [row.refund for row in rows] == [0.9868, None, 0.9868]
    assert rows[0].team == "billing"
    assert rows[2].urgency is not None
    assert {row.q for row in rows} == {"0.75"}


def test_a_registered_function_called_without_every_argument_names_them(spark, tickets, spark_log):
    drex_spark.register(spark)
    tickets.createOrReplaceTempView("tickets")
    message = r"drex_choice: takes 4 arguments \(input, instructions, labels, model\), got 3; pass NULL"
    with pytest.raises(PythonException, match=message):
        spark.sql("""SELECT drex_choice(body, 'Which team?', '["billing", "technical"]') FROM tickets""").collect()
    assert spark_log() == []


def test_refused_constants_fail_on_the_driver_before_any_row(tickets, spark_log):
    with pytest.raises(DrexError, match="drex_choice: choice takes 2 to 255 labels, got 1"):
        tickets.select(drex_choice("body", "Which team?", ["billing"]))
    with pytest.raises(DrexError, match="'__proto__' is not allowed"):
        tickets.select(drex_decide("body", {"__proto__": {"type": "noul", "instructions": REFUND}}))
    assert spark_log() == []


def test_an_unknown_model_fails_the_job(tickets, spark_log):
    with pytest.raises(PythonException, match="drex_noul: unknown model 'drex-v9'"):
        tickets.select(drex_noul("body", REFUND, model="drex-v9")).collect()
    assert spark_log() == []


def test_images_with_the_default_model_fail_the_job(spark, spark_log):
    frame = spark.createDataFrame([(RED,)], "image BINARY")
    with pytest.raises(PythonException, match="drex-v2.0 does not read images"):
        frame.select(drex_choice("image", "Which colour?", ["red", "blue"])).collect()


def test_the_job_budget_is_split_across_workers(spark, spark_log):
    assert worker_concurrency(8, spark) == 8 // SPARK_PARALLELISM
    frame = spark.range(64).repartition(SPARK_PARALLELISM).select(F.concat(F.lit("row "), F.col("id")).alias("text"))
    assert frame.select(drex_noul("text", REFUND, concurrency=8).alias("p")).where("p IS NOT NULL").count() == 64
    peak = max(record["in_flight"] for record in spark_log())
    # Four workers, each capped at two: more than one worker's share at once, never more than the budget
    assert 2 < peak <= 8


def test_spark_drex_concurrency_is_the_default_budget(spark):
    spark.conf.set("spark.drex.concurrency", "16")
    try:
        assert worker_concurrency(spark=spark) == 16 // SPARK_PARALLELISM
    finally:
        spark.conf.unset("spark.drex.concurrency")
    assert worker_concurrency(spark=spark) == max(1, 8 // SPARK_PARALLELISM)
    with mock.patch.object(type(spark.conf), "get", side_effect=RuntimeError("CONFIG_NOT_AVAILABLE")):
        # Databricks serverless refuses to read a configuration it doesn't know
        assert worker_concurrency(spark=spark) == max(1, 8 // SPARK_PARALLELISM)
    spark.conf.set("spark.drex.concurrency", "lots")
    try:
        with pytest.raises(DrexError, match="spark.drex.concurrency must be a whole number, not 'lots'"):
            worker_concurrency(spark=spark)
    finally:
        spark.conf.unset("spark.drex.concurrency")


def test_a_variant_input_is_json_state(spark, spark_log):
    if not hasattr(F, "parse_json"):
        pytest.skip("VARIANT needs Spark 4")
    frame = spark.createDataFrame([('{"message": "charged twice"}',)], "raw STRING")
    frame.select(drex_noul(F.parse_json("raw"), REFUND)).collect()
    assert json.loads(spark_log()[0]["state"]) == {"message": "charged twice"}
