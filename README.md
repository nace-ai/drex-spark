<p align="center">
  <a href="https://nace.ai"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/nace-logo-white.svg"><img src="docs/assets/nace-logo-black.svg" alt="Nace" height="40"></picture></a>
</p>

<h1 align="center">Drex for Apache Spark™</h1>

<p align="center"><b>Ask typed questions about rows from PySpark, Apache Spark SQL and Databricks SQL.</b></p>

<p align="center">
  <a href="https://github.com/nace-ai/drex-spark/actions/workflows/Tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/nace-ai/drex-spark/Tests.yml?branch=main&amp;label=tests&amp;logo=github" alt="Tests"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license"></a>
</p>

Ask typed questions about your rows from [Apache Spark](https://spark.apache.org/) through PySpark and SQL,
or from Databricks SQL. `drex-spark` sends each row to
[Drex](https://console.nace.ai/docs) and returns calibrated probabilities as Spark values.

The PySpark interface uses pandas UDFs. A separate generator installs Unity Catalog SQL functions for
Databricks SQL warehouses. CI tests Python 3.10 with Spark 3.5, and Python 3.12 and 3.13 with Spark 4.

**Also from Drex:** [Drex Browser Use](https://github.com/nace-ai/drex-browser-harness-pub), an AI browser agent and MCP server. One sentence fills a DoorDash cart in 16 moves.

[Getting started](#getting-started) · [Databricks SQL](#databricks-sql) · [API reference](docs/API.md) ·
[Requests and cost](#requests-and-cost) · [Development](#development)

## What it does

| Function | Result | Use case |
| --- | --- | --- |
| `drex_noul` | `DOUBLE` probability of yes | Flag refund requests |
| `drex_choice` | Struct: label, probabilities, confidence | Route tickets to a team |
| `drex_score` | Struct: score, probabilities, confidence | Rank urgency on an ordered scale |
| `drex_decide` | JSON string of the full response | Ask several questions per row |
| `drex_decide_image` | JSON string about image bytes | Classify photos or receipts |

## Apache Spark integration

<a href="https://spark.apache.org/"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/spark-logo-dark.svg"><img src="docs/assets/spark-logo-light.png" alt="Apache Spark project logo" height="40"></picture></a>

The PySpark interface runs on [Apache Spark](https://spark.apache.org/) and exposes pandas UDFs as columns
and SQL functions. The [Databricks SQL](#databricks-sql) interface uses separately generated Unity Catalog functions.

## Getting started

Requires Python 3.10+ and Spark 3.5+. For the local setup below, use Python 3.12 and Java 17, as in the tests.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install drex-spark 'pyspark==4.2.1' 'pandas<3'
```

On a managed cluster, install `drex-spark` on the driver and executors and use the platform's PySpark: on Databricks,
`%pip install drex-spark` in a notebook, or a PyPI library on the cluster or job. See
[Databricks SQL](#databricks-sql) for SQL warehouses.

### The key

Get an API key from the [Nace console](https://console.nace.ai). The functions read `NACE_API_KEY` (and an optional
`NACE_BASE_URL`) on the executors, never from a function argument. Set them from your platform's secret store:

- **Databricks clusters:** in the cluster's Spark config, `spark.executorEnv.NACE_API_KEY {{secrets/<scope>/<key>}}`.
  Serverless compute takes no Spark config, so its executors can't get the key: on serverless notebooks and jobs, call
  the [Unity Catalog functions](#databricks-sql) with `spark.sql` or `%sql` instead.
- **Spark elsewhere:** `--conf spark.executorEnv.NACE_API_KEY=...` from your secret manager, or the environment of your
  executors. In `local` mode the driver's environment is used.

For [OpenRouter](https://openrouter.ai), set `NACE_BASE_URL=https://openrouter.ai/api` and an OpenRouter key as
`OPENROUTER_API_KEY` (an `sk-or-` key in `NACE_API_KEY` also works); a `nace_` key is refused before any request. The
requests carry `HTTP-Referer: https://nace.ai` and `X-OpenRouter-Title: Drex for Apache Spark` unless you set
`NACE_OPENROUTER_REFERER` and `NACE_OPENROUTER_TITLE` the same way, to report your own app. On Databricks, pass
`--openrouter-referer` and `--openrouter-title` to `python -m drex_spark.databricks`, which bakes them into the functions.

### Query

For local mode, set `NACE_API_KEY` in the driver environment before starting Python, using your secret manager.
Then run the following in one Python session:

```python
from pyspark.sql import SparkSession
from drex_spark.functions import drex_noul, drex_score

spark = SparkSession.builder.master("local[2]").appName("drex-example").getOrCreate()
tickets = spark.createDataFrame([
    (1, "I was charged twice. Please refund the duplicate payment."),
    (2, "How do I change my email address?"),
], "id long, body string")
tickets.createOrReplaceTempView("tickets")

answers = tickets.select(
    "id",
    drex_noul("body", "Is the customer asking for a refund?").alias("refund"),
    drex_score("body", "How urgent is this ticket?", ["low", "medium", "high"]).alias("urgency"),
)
answers.write.saveAsTable("ticket_answers")
```

`refund` is the probability of yes; `urgency` is a struct with a score, probabilities and confidence.
Answers come from the API and can vary. Saving them in a table lets you query them without sending the rows again; see
[requests and cost](#requests-and-cost).

## Examples

### Apache Spark SQL

Continue in the Python session above, which defines the `tickets` temporary view.

```python
import drex_spark

drex_spark.register(spark)
spark.sql("""
    SELECT id, drex_choice(body, 'Which team?', '["billing", "technical"]', NULL).choice AS team
    FROM tickets
""").show(truncate=False)
```

Registered functions take every argument (Spark's pandas UDFs have no defaults): a NULL `model` is the default model.

## Databricks SQL

On a SQL warehouse, the functions are Unity Catalog functions that run `drex-spark` inside a batch Python UDF. Create
them once, as the principal they should run as:

1. **The key** as a Unity Catalog secret, for example `main.drex.nace_api_key` (Catalog Explorer › Create › Secret). The
   functions run with their owner's permissions: the owner needs `READ SECRET` on it; callers need only `EXECUTE`.
2. **Network access.** Python UDFs on a serverless SQL warehouse can't reach the internet until a workspace admin turns
   on the Public Preview *Enable networking for isolated workloads in Serverless SQL Warehouses* (Previews page), then
   restarts the warehouse: one that was running keeps refusing. If the
   workspace has a serverless network policy that restricts egress, allow `console.nace.ai` and `pypi.org`. Pro
   warehouses and classic compute use your workspace's own network, which must reach both.
3. **The functions:**

   ```sh
   python -m pip install drex-spark
   python -m drex_spark.databricks --catalog main --schema drex --secret main.drex.nace_api_key > drex.sql
   ```

   Use an existing catalog and schema that your function owner can access. Run `drex.sql` in the SQL editor. Each
   function installs the same version of `drex-spark` from PyPI into its own environment (`environment_version` 6); to
   install a wheel from a Unity Catalog volume instead, add `--dependency /Volumes/<catalog>/<schema>/<volume>/<wheel>`.

```sql
SELECT id,
       main.drex.drex_noul(body, 'Is the customer asking for a refund?') AS refund,
       main.drex.drex_choice(body, 'Which team?', '["billing", "technical"]', model => 'drex-max') AS team
FROM VALUES (1, 'I was charged twice. Please refund one.') AS tickets(id, body);
```

Unity Catalog has no overloading, so `input` is text (`STRING`): ask about JSON state with `drex_decide`, and about
images with `drex_decide_image`. Requires Databricks SQL (pro or serverless) or classic compute on Databricks Runtime 19
in standard access mode; functions with a `SECRETS` clause can't be called directly on dedicated access mode compute.

## Configuration and behavior

Text and JSON calls default to `drex-v2.0`; images default to `drex-max`. PySpark helpers accept `model=`;
registered SQL functions require all arguments and use NULL for the default model. Read the
[API reference](docs/API.md) for structured inputs, image bytes, validation and retry rules.
A missing choice or score is a struct with NULL fields: check `team.choice IS NULL`.

## Requests and cost

Inputs, questions and any image bytes leave the Spark workers for the configured API. Use only data you may
send there. A custom endpoint receives the configured key as well as the data.

Each non-NULL row sends one request, and Drex bills only a `200`. A row with no answer is NULL, never a guess; treat
NULL as "unanswered", not as a negative answer. Spark can send a row more than once, and repeated successful requests can be billed again:

- **A retried task** (an executor lost, a failed stage) sends its rows again.
- **A sort over a Drex column** samples its input first: `orderBy` after a Drex column sends every row twice.
- **Reusing a DataFrame** (two actions, a self-join) runs it again.

Save answers first: write them to a table, or `cache()` the DataFrame and run `count()` before you sort, filter on, or
reuse it. Filter and limit the input before calling Drex.

### Concurrency

`spark.drex.concurrency` (default 8) is a budget divided among Python workers using
`max(1, budget // spark.default.parallelism)`. This is not a strict cluster-wide cap: more workers than the budget
can exceed it. When Spark Connect cannot expose parallelism, each worker gets the whole budget.
`concurrency=` overrides the budget for one call. The Unity Catalog
functions cap each batch at 2 (`--concurrency` when you create them), since a UDF can't see the warehouse's
parallelism and a warehouse runs many batches at once (about 20 on a Small serverless warehouse). Above your account's
limit, Drex answers `429` with `retry-after-ms`: the request waits its turn, and a row returns NULL only if Drex answers
nothing for two minutes.

## Development

```sh
git clone https://github.com/nace-ai/drex-spark.git && cd drex-spark
uv sync
uv run pytest                 # needs Java 17 for the Spark tests (JAVA_HOME, or Homebrew's openjdk@17)
uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src
```

The tests call `tests/mock/drex_mock.py`, a mock of Drex's API copied from nace-ai/drex-duckdb (and checked against
Drex's `openapi.json` by `python -m unittest discover -s tests/mock`). `tests/live` calls the real API and runs only
with `DREX_LIVE=1` (`LiveTests.yml`, on dispatch).

## Support and contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for local checks and pull requests, and the
[API reference](docs/API.md) for the complete contract. Report bugs through
[GitHub Issues](https://github.com/nace-ai/drex-spark/issues), with your runtime version and a minimal example
using synthetic data. Remove keys, signed URLs and private data from reports.

## License and trademarks

[MIT](LICENSE). Nace artwork belongs to Nace AI. The software license does not grant trademark rights.
See [asset sources and branding guidance](docs/assets/README.md).

This integration is maintained by Nace AI. References to the platform describe compatibility,
not sponsorship or endorsement by its owner.

[Apache Spark](https://spark.apache.org/), Spark, and the Spark logo are trademarks of the
[Apache Software Foundation](https://www.apache.org/).
