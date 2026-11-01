# API

The five functions of Drex's shared SQL surface are available in [Apache Spark™](https://spark.apache.org/)
as PySpark columns (`drex_spark.functions`) and SQL functions (`drex_spark.register(spark)`), and in
Databricks as Unity Catalog functions (`python -m drex_spark.databricks`). Set up the key as in
the [README](../README.md#the-key).

| Function | Result |
| --- | --- |
| `drex_noul(input, instructions, criteria=None, model='drex-v2.0')` | `DOUBLE`, the probability that the answer is yes |
| `drex_choice(input, instructions, labels, model='drex-v2.0')` | `STRUCT<choice STRING, probabilities ARRAY<STRUCT<value STRING, probability DOUBLE>>, confidence DOUBLE>` |
| `drex_score(input, instructions, levels, model='drex-v2.0')` | `STRUCT<score DOUBLE, probabilities ARRAY<STRUCT<index INT, value STRING, probability DOUBLE>>, confidence DOUBLE>` |
| `drex_decide(state, questions, model='drex-v2.0')` | `STRING`: Drex's response body, JSON, unchanged |
| `drex_decide_image(state, images, questions, model='drex-max')` | `STRING`: Drex's response body, JSON, unchanged |

A choice's `probabilities` are sorted by label, a score's by index, as in drex-duckdb. Read a `drex_decide` result with
`from_json` or `get_json_object`, or `parse_json` on Databricks.

## Arguments

- **`input`** (PySpark and Spark SQL): `STRING` is sent as text; `STRUCT`, `MAP` or `VARIANT` as JSON state; `BINARY` or
  `ARRAY<BINARY>` as images, which need an image model (`model='drex-max'`): with any other model the query fails. In
  Unity Catalog `input` is `STRING`: use `drex_decide` for JSON state and `drex_decide_image` for images.
- **`instructions`**: a string; NULL or `''` means none. A `noul` needs instructions or a described `true` or `false`.
- **`criteria`, `labels`, `levels`**: a list of labels (`["billing", "tech"]`), a list of `{"label", "description"}`, or
  an object of label to description (a description may be null), as a Python value or JSON text. Spark SQL also takes
  an `ARRAY<STRING>`; Unity Catalog takes JSON text. `noul` labels are exactly `true` and `false`, and labels with no
  description are the same as no criteria. `choice` and `score` take 2–255 unique, non-empty labels; `score` levels
  take no descriptions (describe the scale in `instructions`) and their order is the scale.
- **`state`**: JSON text, or (PySpark, Spark SQL) a `STRUCT`, `MAP` or `VARIANT`. State that isn't valid JSON fails
  the query.
- **`questions`**: an object of id to `{type, instructions, criteria}`, as a Python value or JSON text. 1–512 questions;
  ids are non-empty after trimming and unique ignoring case; `__proto__` is refused. `type` is `noul`, `choice` or
  `score`; `multiple_choice`, `box` and unknown keys are refused.
- **`images`**: a `BINARY` or `ARRAY<BINARY>` column (Unity Catalog: `ARRAY<BINARY>`) of 1–8 PNG, JPEG or WebP
  images, read for example with `spark.read.format("binaryFile")`. They are sent inline; Drex never fetches a URL.
- **`model`**: checked against `GET /v1/models` on the executors, cached per worker. An unknown model fails the query.
- **`concurrency`** (PySpark and `register` only): the job's in-flight budget for this call, split over the workers.

A NULL `input` or `state`, a NULL image or an empty list of images, and a NULL `questions` return NULL and send
nothing. PySpark checks `instructions`, `criteria` and `questions` when you build the column, before any row is read;
Spark SQL and Unity Catalog pass them as columns, so each distinct value is checked once per batch, and a value that
fails fails the query in any row that would be sent. A row that sends nothing checks nothing, so arguments built from
the row, such as `concat('Is ', body, ' a refund?')`, may be NULL with it. A registered function called with fewer
arguments fails the query and names them.

## Errors

| Outcome | When |
| --- | --- |
| **The query fails** | 401; 402 (with the billing page); a 422 about anything but the row (its issues as `path: message; …`); an unknown model; images with a model that doesn't read them; a TLS failure; any check above |
| **NULL and a warning** | a 422 whose message starts `Request too long:` or whose issues are all under `state` or `images` (the warning names `state`, never the row's keys); a body over 4 MiB; an image that isn't PNG, JPEG or WebP; more than 8 images; retries exhausted; any other status |
| **Retried** | 429 and 529 (Drex is at the account's or its own limit) while Drex answers the worker's other requests; no response, 408, 500, 502, 503 and 504 up to 5 attempts |

A failed query fails its Spark task; Spark's task retries then send that task's rows again. Warnings go to the executor
log (logger `drex_spark`). A struct result (`drex_choice`, `drex_score`) with no answer is a struct whose fields are
all NULL, since a pandas UDF can't return a NULL struct: test `team.choice IS NULL`.

A request waits for `retry-after-ms` (or `retry-after`, at most 30 seconds) before each retry, else 1, 2, 4 and 8
seconds less up to 25 % jitter; a 429 or 529's wait gains up to 25 %, so the rows it refused don't return at once. A 429
or 529 means Drex refused the request before doing any work, so it doesn't count as an attempt: the request is retried
until 120 seconds pass in which Drex answers none of the worker's requests. Any other retryable failure ends the request
after 5 attempts, or 120 seconds after its first attempt (or the worker's latest answer, if later). A request keeps its
place in the concurrency cap while it waits, so a batch never has more requests started than its concurrency, and the
rows over an account's limit queue instead of returning NULL.

Every attempt sends the same body and `Idempotency-Key`. Drex bills only a `200`, so a retry is billed twice only when
a `200` was lost on its way back. drex-duckdb counts 429 and 529 as attempts (5 within 120 seconds); a Spark job or a
Databricks query offers many workers' requests at once, which a DuckDB process doesn't.

## OpenRouter

Set `NACE_BASE_URL=https://openrouter.ai/api` and an OpenRouter key as `NACE_API_KEY`. Models resolve under
`nace-ai/`, images travel as `state` content parts, and requests carry `HTTP-Referer: https://nace.ai` and
`X-OpenRouter-Title: Drex for Apache Spark` (`Drex for Databricks` from Unity Catalog).

The key is `OPENROUTER_API_KEY` when set, else `NACE_API_KEY`, which may hold an `sk-or-` key; a `nace_` key fails the
query before any request. Unity Catalog functions read their one secret, which must then hold the OpenRouter key.

The two headers go only to OpenRouter. To report your own app, set `NACE_OPENROUTER_REFERER` and
`NACE_OPENROUTER_TITLE` on the executors, pass `openrouter_referer` and `openrouter_title` to `Target`, or, for Unity
Catalog, to `install_sql` and `handler` (`--openrouter-referer` and `--openrouter-title` on the command line). Both
are HTTP headers, so they must be printable ASCII (`™` or `ü` fail the query, or the generator); a blank one is the
default.

## Attribution

Requests carry `X-Nace-Client: drex-spark/<version>` from PySpark and Spark SQL, and `drex-databricks/<version>` from the
Unity Catalog functions.

---

[Apache Spark](https://spark.apache.org/) and Spark are trademarks of the
[Apache Software Foundation](https://www.apache.org/).
