"""Fixtures: the Drex mock and a local SparkSession.

tests/mock is copied unchanged from nace-ai/drex-duckdb at 8b48eac (test/mock), except openapi.json, regenerated with
its openapi_subset.py from nace-ai/drex at c0f68741 (the schemas are unchanged).
"""

from __future__ import annotations

import itertools
import os
import shutil
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "mock"))
import drex_mock
import pngs

from drex_spark._client import Target

#: A well-formed test key the mock accepts, never a real one
KEY = "nace_sk_mockKeyForDrexSparkTests_" + "3" * 18
assert drex_mock.KEY_PATTERN.match(f"Bearer {KEY}")

RED = pngs.solid_png(pngs.COLOURS["red"])
BLUE = pngs.solid_png(pngs.COLOURS["blue"])

_names = itertools.count()


@dataclass
class Mock:
    url: str
    store: Any

    def namespace(self) -> str:
        return f"t{next(_names)}"

    def base_url(self, ns: str, scenario: str = "ok") -> str:
        return f"{self.url}/ns/{ns}/{scenario}"

    def target(self, ns: str, scenario: str = "ok", client_info: str = "drex-spark/test") -> Target:
        return Target(
            client_info=client_info, title="Drex for Apache Spark", api_key=KEY, base_url=self.base_url(ns, scenario)
        )

    def log(self, ns: str, endpoint: str | None = "/v1/systemone") -> list[dict[str, Any]]:
        with self.store.lock:
            records = list(self.store.logs.get(ns, []))
        return [record for record in records if endpoint is None or record["endpoint"] == endpoint]

    def reset(self, ns: str) -> None:
        self.store.reset(ns)


@pytest.fixture(scope="session")
def mock() -> Iterator[Mock]:
    server = drex_mock.make_server()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    yield Mock(f"http://{host}:{port}", server.RequestHandlerClass.store)
    server.shutdown()


@pytest.fixture
def ns(mock: Mock) -> str:
    return mock.namespace()


#: The namespace and scenario every Spark test's workers call, since the workers read NACE_BASE_URL when they start
SPARK_NS = "spark"
SPARK_SCENARIO = "slow-100"
SPARK_PARALLELISM = 4


def _java_home() -> str | None:
    if os.environ.get("JAVA_HOME"):
        return os.environ["JAVA_HOME"]
    for candidate in ("/opt/homebrew/opt/openjdk@17", "/usr/local/opt/openjdk@17"):
        if Path(candidate, "bin", "java").exists():
            return candidate
    return None


@pytest.fixture(scope="session")
def spark(mock: Mock) -> Iterator[Any]:
    java_home = _java_home()
    if java_home is None and shutil.which("java") is None:
        pytest.skip("needs Java 17 for a local SparkSession")
    if java_home:
        os.environ["JAVA_HOME"] = java_home
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master(f"local[{SPARK_PARALLELISM}]")
        .appName("drex-spark-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", str(SPARK_PARALLELISM))
        .config("spark.executorEnv.NACE_API_KEY", KEY)
        .config("spark.executorEnv.NACE_BASE_URL", mock.base_url(SPARK_NS, SPARK_SCENARIO))
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "16")
        # Spark's task retries would resend a failed task's requests; the tests count requests
        .config("spark.task.maxFailures", "1")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture
def spark_log(mock: Mock, spark: Any) -> Iterator[Any]:
    """The Spark namespace's requests, from this test only."""
    mock.reset(SPARK_NS)
    yield lambda endpoint="/v1/systemone": mock.log(SPARK_NS, endpoint)
