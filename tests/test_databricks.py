"""The Unity Catalog functions: the statements install_sql writes, and the batch handlers run as Databricks runs them
(an iterator of tuples of pandas Series), with `databricks.secrets` faked."""

import json
import sys
import types

import pandas as pd
import pytest
from conftest import KEY, RED

from drex_spark import __version__, databricks


@pytest.fixture
def secrets(monkeypatch):
    reads = []

    def get(*, catalog, schema, key):
        reads.append((catalog, schema, key))
        return KEY

    package = types.ModuleType("databricks")
    module = types.ModuleType("databricks.secrets")
    module.get = get
    package.secrets = module
    monkeypatch.setitem(sys.modules, "databricks", package)
    monkeypatch.setitem(sys.modules, "databricks.secrets", module)
    databricks._read_secret.cache_clear()
    yield reads
    databricks._read_secret.cache_clear()


def run(function, batches, mock, ns, concurrency=8):
    handler = databricks.handler(
        function, secret="main.drex.nace_api_key", base_url=mock.base_url(ns), concurrency=concurrency
    )
    return list(handler(iter(batches)))


def test_the_handlers_read_the_secret_and_send_drex_databricks(mock, ns, secrets):
    [noul] = run(
        "noul",
        [
            (
                pd.Series(["I was charged twice", None]),
                pd.Series(["Is the customer asking for a refund?"] * 2),
                pd.Series([None, None]),
                pd.Series([None, None]),
            )
        ],
        mock,
        ns,
    )
    assert noul.isna().tolist() == [False, True]
    assert secrets == [("main", "drex", "nace_api_key")]
    [sent] = mock.log(ns)
    assert sent["x_nace_client"] == f"drex-databricks/{__version__}"
    assert sent["served_model"] == "drex-v2.0"


def test_choice_and_score_rows_and_decide_image(mock, ns, secrets):
    [choice] = run(
        "choice",
        [
            (
                pd.Series(["My card was charged"]),
                pd.Series(["Topic?"]),
                pd.Series(['["billing", "tech"]']),
                pd.Series(["drex-max"]),
            )
        ],
        mock,
        ns,
    )
    assert list(choice.columns) == ["choice", "probabilities", "confidence"]
    assert choice.loc[0, "choice"] == "billing"
    [score] = run(
        "score",
        [
            (
                pd.Series(["Down for everyone"]),
                pd.Series(["Urgency?"]),
                pd.Series(['["low", "high"]']),
                pd.Series([None]),
            )
        ],
        mock,
        ns,
    )
    assert [row["value"] for row in score.loc[0, "probabilities"]] == ["low", "high"]
    questions = json.dumps(
        {"colour": {"type": "choice", "instructions": "Which colour is Image 1?", "criteria": ["red", "blue"]}}
    )
    [bodies] = run(
        "decide_image",
        [(pd.Series(["{}"]), pd.Series([[RED]]), pd.Series([questions]), pd.Series([None]))],
        mock,
        ns,
    )
    assert json.loads(bodies[0])["answers"]["colour"]["choice"] == "red"
    assert mock.log(ns)[-1]["served_model"] == "drex-max"


def test_install_sql_writes_a_batch_udf_and_a_sql_function_with_defaults_for_each():
    sql = databricks.install_sql("main", "drex", "main.drex.nace_api_key", concurrency=4)
    for function in databricks.FUNCTIONS:
        assert f"CREATE OR REPLACE FUNCTION main.drex.drex_{function}_batch(" in sql
        assert f"HANDLER 'drex_{function}_batch'" in sql
        assert f"RETURN main.drex.drex_{function}_batch(" in sql
    assert sql.count("PARAMETER STYLE PANDAS") == 5
    assert sql.count("SECRETS (main.drex.nace_api_key)") == 5
    assert (
        sql.count(f"""ENVIRONMENT (dependencies = '["drex-spark=={__version__}"]', environment_version = '6')""") == 5
    )
    assert "criteria STRING DEFAULT NULL, model STRING DEFAULT 'drex-v2.0')" in sql
    assert "questions STRING, model STRING DEFAULT 'drex-max')\nRETURNS STRING\nLANGUAGE SQL" in sql
    assert "concurrency=4" in sql
    assert sql.count("NOT DETERMINISTIC") == 5


def test_install_sql_bakes_in_the_openrouter_identity():
    sql = databricks.install_sql(
        "main", "drex", "main.drex.nace_api_key", openrouter_referer="https://app.example", openrouter_title="Example"
    )
    assert sql.count("openrouter_referer='https://app.example', openrouter_title='Example'") == 5
    assert "openrouter_referer=None, openrouter_title=None" in databricks.install_sql("main", "drex", "main.drex.k")


def test_install_sql_refuses_an_openrouter_identity_a_header_cant_carry():
    with pytest.raises(ValueError, match="openrouter_title must be printable ASCII"):
        databricks.install_sql("main", "drex", "main.drex.k", openrouter_title="Acme\u2122")


def test_the_cli_takes_the_openrouter_identity(capsys):
    databricks.main(
        ["--catalog", "main", "--schema", "drex", "--secret", "main.drex.k", "--openrouter-title", "Example"]
    )
    assert "openrouter_title='Example'" in capsys.readouterr().out


def test_install_sql_bodies_are_valid_python():
    sql = databricks.install_sql("main", "drex", "main.drex.nace_api_key", base_url="https://console.nace.ai")
    bodies = [part.split("$$")[0] for part in sql.split("AS $$\n")[1:]]
    assert len(bodies) == 5
    for body in bodies:
        compile(body, "<udf>", "exec")


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        (("main", "drex", "nace_api_key"), "three-part name"),
        (("main", "drex; DROP", "main.drex.k"), "plain identifiers"),
        (("main", "drex", "main.drex.k'"), "three-part name"),
    ],
)
def test_install_sql_refuses_names_that_are_not_identifiers(arguments, match):
    with pytest.raises(ValueError, match=match):
        databricks.install_sql(*arguments)


def test_the_cli_prints_the_statements(capsys):
    databricks.main(["--catalog", "main", "--schema", "drex", "--secret", "main.drex.nace_api_key"])
    assert "CREATE OR REPLACE FUNCTION main.drex.drex_noul(" in capsys.readouterr().out
