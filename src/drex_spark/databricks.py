"""Drex's functions as Unity Catalog SQL functions, for Databricks SQL warehouses.

Each function is a batch Python UDF (`PARAMETER STYLE PANDAS`) named `drex_<fn>_batch`, which takes every argument,
and a `LANGUAGE SQL` function under the canonical name that calls it with defaults, since Unity Catalog allows
`DEFAULT` only in `LANGUAGE SQL` functions and has no overloading. The API key is a Unity Catalog secret, declared in
the batch UDF's `SECRETS` clause and read with `databricks.secrets.get`, which needs `environment_version` 6.

    python -m drex_spark.databricks --catalog main --schema drex --secret main.drex.nace_api_key > install.sql

prints the statements that create them.
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import sys
from collections.abc import Callable, Iterator

import pandas as pd

from drex_spark._batch import decide_batch, typed_batch
from drex_spark._client import Target, is_header_text
from drex_spark._contract import DEFAULT_IMAGE_MODEL, DEFAULT_MODEL
from drex_spark._version import __version__

__all__ = ["handler", "install_sql"]

#: Requests in flight per batch. A warehouse runs many batches at once (about 20 on a Small serverless warehouse), so
#: a few each already reach a paid account's limit of 16; more only add refusals
DEFAULT_CONCURRENCY = 2
ENVIRONMENT_VERSION = "6"
FUNCTIONS = ("noul", "choice", "score", "decide", "decide_image")

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

PROBABILITY_ROW = {
    "choice": "STRUCT<value: STRING, probability: DOUBLE>",
    "score": "STRUCT<index: INT, value: STRING, probability: DOUBLE>",
}
RETURNS = {
    "noul": "DOUBLE",
    "choice": f"STRUCT<choice: STRING, probabilities: ARRAY<{PROBABILITY_ROW['choice']}>, confidence: DOUBLE>",
    "score": f"STRUCT<score: DOUBLE, probabilities: ARRAY<{PROBABILITY_ROW['score']}>, confidence: DOUBLE>",
    "decide": "STRING",
    "decide_image": "STRING",
}
#: Each function's parameters as (name, type, the SQL function's DEFAULT or None)
PARAMETERS: dict[str, list[tuple[str, str, str | None]]] = {
    "noul": [("input", "STRING", None), ("instructions", "STRING", None), ("criteria", "STRING", "NULL")],
    "choice": [("input", "STRING", None), ("instructions", "STRING", None), ("labels", "STRING", None)],
    "score": [("input", "STRING", None), ("instructions", "STRING", None), ("levels", "STRING", None)],
    "decide": [("state", "STRING", None), ("questions", "STRING", None)],
    "decide_image": [("state", "STRING", None), ("images", "ARRAY<BINARY>", None), ("questions", "STRING", None)],
}
COMMENTS = {
    "noul": "Drex: the probability that the answer to instructions about input is yes",
    "choice": "Drex: which of labels (a JSON list) answers instructions about input",
    "score": "Drex: a score on levels (a JSON list, in order) for instructions about input",
    "decide": "Drex: the response body, as JSON, for questions (JSON) about state (JSON)",
    "decide_image": "Drex: the response body, as JSON, for questions (JSON) about state (JSON) and images",
}


def _secret_parts(secret: str) -> tuple[str, str, str]:
    parts = secret.split(".")
    if len(parts) != 3 or not all(_IDENTIFIER.fullmatch(part) for part in parts):
        raise ValueError(
            f"the secret must be a Unity Catalog secret's three-part name, catalog.schema.secret: {secret!r}"
        )
    return parts[0], parts[1], parts[2]


@functools.cache
def _read_secret(secret: str) -> str:
    from databricks.secrets import get  # Only inside a Unity Catalog UDF

    catalog, schema, key = _secret_parts(secret)
    return get(catalog=catalog, schema=schema, key=key)


def handler(
    function: str,
    *,
    secret: str,
    base_url: str | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    openrouter_referer: str | None = None,
    openrouter_title: str | None = None,
) -> Callable[[Iterator[tuple[pd.Series, ...]]], Iterator[pd.Series | pd.DataFrame]]:
    """The batch handler of `drex_<function>_batch`, which `install_sql` writes into each UDF's body.

    The key is read from the secret on the first batch. `concurrency` caps the requests in flight per batch; the UDF
    can't see the warehouse's parallelism, so the requests over the account's limit are refused and wait their turn
    (see `_client._with_retries`).
    """
    if function not in FUNCTIONS:
        raise ValueError(f"function must be one of {', '.join(FUNCTIONS)}, not {function!r}")
    _secret_parts(secret)
    name = f"drex_{function}"

    def run(batches: Iterator[tuple[pd.Series, ...]]) -> Iterator[pd.Series | pd.DataFrame]:
        target = Target(
            client_info=f"drex-databricks/{__version__}",
            title="Drex for Databricks",
            api_key=_read_secret(secret),
            base_url=base_url,
            openrouter_referer=openrouter_referer,
            openrouter_title=openrouter_title,
        )
        for batch in batches:
            if function in ("noul", "choice", "score"):
                inputs, instructions, criteria, models = batch
                yield typed_batch(target, name, function, inputs, instructions, criteria, models, concurrency)
            elif function == "decide":
                states, questions, models = batch
                yield decide_batch(target, name, states, None, questions, models, concurrency)
            else:
                states, images, questions, models = batch
                yield decide_batch(target, name, states, images, questions, models, concurrency)

    return run


def _sql_string(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def install_sql(
    catalog: str,
    schema: str,
    secret: str,
    *,
    dependency: str | None = None,
    base_url: str | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    openrouter_referer: str | None = None,
    openrouter_title: str | None = None,
) -> str:
    """The statements that create the five functions, and their batch UDFs, in `catalog.schema`.

    `dependency` is what each UDF's environment installs: `drex-spark==<this version>` from PyPI by default, or a wheel
    in a volume (`/Volumes/...whl`). `base_url` is baked into the handlers (console.nace.ai when omitted), and so are
    `openrouter_referer` and `openrouter_title`, the app identity OpenRouter attributes requests to when `base_url` is
    OpenRouter's (https://nace.ai and Drex for Databricks when omitted).
    """
    for part in (catalog, schema):
        if not _IDENTIFIER.fullmatch(part):
            raise ValueError(f"catalog and schema must be plain identifiers, not {part!r}")
    _secret_parts(secret)
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    for name, value in (("openrouter_referer", openrouter_referer), ("openrouter_title", openrouter_title)):
        if value is not None and not is_header_text(value):
            raise ValueError(f"{name} must be printable ASCII, since it is sent as a header: {value!r}")
    environment = json.dumps([dependency or f"drex-spark=={__version__}"])
    statements = [f"-- Drex for Databricks {__version__}: run as the owner the functions will run as."]
    for function in FUNCTIONS:
        name = f"{catalog}.{schema}.drex_{function}"
        parameters = [
            *PARAMETERS[function],
            ("model", "STRING", _sql_string(DEFAULT_IMAGE_MODEL if function == "decide_image" else DEFAULT_MODEL)),
        ]
        arguments = ", ".join(f"{parameter} {kind}" for parameter, kind, _ in parameters)
        configured = (
            f"{function!r}, secret={secret!r}, base_url={base_url!r}, concurrency={concurrency}, "
            f"openrouter_referer={openrouter_referer!r}, openrouter_title={openrouter_title!r}"
        )
        statements.append(
            f"""CREATE OR REPLACE FUNCTION {name}_batch({arguments})
RETURNS {RETURNS[function]}
LANGUAGE PYTHON
PARAMETER STYLE PANDAS
HANDLER 'drex_{function}_batch'
SECRETS ({secret})
ENVIRONMENT (dependencies = {_sql_string(environment)}, environment_version = {_sql_string(ENVIRONMENT_VERSION)})
COMMENT {_sql_string(COMMENTS[function] + "; every argument is required, a NULL model is the default")}
AS $$
from drex_spark.databricks import handler

_run = handler({configured})


def drex_{function}_batch(batches):
    yield from _run(batches)
$$;"""
        )
        with_defaults = ", ".join(
            f"{parameter} {kind}" + (f" DEFAULT {default}" if default is not None else "")
            for parameter, kind, default in parameters
        )
        call = ", ".join(parameter for parameter, _, _ in parameters)
        statements.append(
            f"""CREATE OR REPLACE FUNCTION {name}({with_defaults})
RETURNS {RETURNS[function]}
LANGUAGE SQL
NOT DETERMINISTIC
COMMENT {_sql_string(COMMENTS[function])}
RETURN {name}_batch({call});"""
        )
    return "\n\n".join(statements) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m drex_spark.databricks", description=install_sql.__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument(
        "--secret", required=True, help="the Unity Catalog secret holding the key, catalog.schema.secret"
    )
    parser.add_argument("--dependency", help=f"what each UDF installs (default drex-spark=={__version__})")
    parser.add_argument("--base-url", help="the Drex API to call (default https://console.nace.ai)")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="requests in flight per batch")
    parser.add_argument(
        "--openrouter-referer", help="the HTTP-Referer app identity on OpenRouter (default https://nace.ai)"
    )
    parser.add_argument("--openrouter-title", help="the X-OpenRouter-Title app name on OpenRouter")
    args = parser.parse_args(argv)
    try:
        sql = install_sql(
            args.catalog,
            args.schema,
            args.secret,
            dependency=args.dependency,
            base_url=args.base_url,
            concurrency=args.concurrency,
            openrouter_referer=args.openrouter_referer,
            openrouter_title=args.openrouter_title,
        )
    except ValueError as error:
        parser.error(str(error))
    sys.stdout.write(sql)


if __name__ == "__main__":
    main()
