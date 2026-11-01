"""Drex for Apache Spark: ask typed questions about rows, from PySpark, Spark SQL and Databricks SQL.

The PySpark functions are in `drex_spark.functions`, and `register(spark)` makes them Spark SQL functions. The Unity
Catalog functions for Databricks SQL warehouses are created by `drex_spark.databricks`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from drex_spark._contract import DrexError
from drex_spark._version import __version__

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

__all__ = ["DrexError", "__version__", "register"]


def register(spark: SparkSession, concurrency: int | None = None) -> None:
    """Registers drex_noul, drex_choice, drex_score, drex_decide and drex_decide_image for Spark SQL.

    PySpark is imported only here, so the Unity Catalog functions, which don't need it, never import it.
    """
    from drex_spark.functions import register as register_functions

    register_functions(spark, concurrency)
