"""Sending requests to Drex: one `AsyncNaceClient` per Python worker, concurrency within a batch, and the shared
surface's retry and error rules (drex-duckdb's src/drex_client.cpp), except that a 429 or 529 is not an attempt: many
Spark workers offer requests at once, so the rows over the account's limit wait their turn (`_with_retries`).

A Spark or Unity Catalog handler is synchronous, so every worker process runs one event loop on a thread of its own
and submits each batch to it; the client and its connections live on that loop and are reused from batch to batch.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import math
import os
import random
import ssl
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any, TypeVar
from urllib.parse import urlsplit

import httpx
from nace_sdk import AsyncNaceClient, NaceAPIConnectionError, NaceAPIError, NaceAPIResponseValidationError, RetryPolicy

from drex_spark._contract import MAX_IMAGES, DrexError, fail

logger = logging.getLogger("drex_spark")

T = TypeVar("T")

DEFAULT_BASE_URL = "https://console.nace.ai"
REQUEST_TIMEOUT_SECONDS = 60.0
#: Attempts per request that fail other than for capacity, all with the same body and Idempotency-Key
MAX_ATTEMPTS = 5
MAX_RETRY_WAIT_SECONDS = 30.0
#: From a request's first attempt, or this worker's latest answer from Drex if later; a retry that would start after
#: it is not made
REQUEST_BUDGET_SECONDS = 120.0
#: Drex refuses a larger body with a 413
MAX_BODY_BYTES = 4 * 1024 * 1024
RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 529})
#: Drex refused the request before doing any work, because the account or Drex is at its limit: retried for as long as
#: Drex answers this worker's other requests, and never counted as an attempt
CAPACITY_STATUSES = frozenset({429, 529})

OPENROUTER_REFERER = "https://nace.ai"
OPENROUTER_BILLING_URL = "https://openrouter.ai/settings/credits"
#: The versions that read images. Drex's GET /v1/models has no image field, so the list is kept here; a listed model
#: reads images when it is one of these or its alias_for names one. OpenRouter lists each model's modalities
IMAGE_MODELS = frozenset({"drex-max", "drex-max-v1.1"})


@dataclass(frozen=True)
class Target:
    """Where a call's requests go, and how they say who sent them."""

    client_info: str
    #: The title OpenRouter shows for these requests, unless `openrouter_title` or NACE_OPENROUTER_TITLE replaces it
    title: str
    #: `None` reads NACE_API_KEY, and NACE_BASE_URL (else console.nace.ai), from the environment when first used; on
    #: OpenRouter it reads OPENROUTER_API_KEY first
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    #: The app identity OpenRouter attributes requests to, in place of https://nace.ai and `title`; `None` reads
    #: NACE_OPENROUTER_REFERER and NACE_OPENROUTER_TITLE from the environment
    openrouter_referer: str | None = None
    openrouter_title: str | None = None


@dataclass
class DrexRequest:
    """One row's request. `state` is any JSON value; `images` are raw bytes, sent inline."""

    state: Any
    questions: dict[str, Any]
    images: Sequence[bytes] = ()


def is_open_router(base_url: str) -> bool:
    host = (urlsplit(base_url).hostname or "").lower()
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


def is_header_text(value: str) -> bool:
    """Whether `value` can be an HTTP header's value: printable ASCII, the only text headers carry."""
    return value.isascii() and value.isprintable()


def image_mime_type(image: bytes) -> str | None:
    if image[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(image) >= 12 and image[:4] == b"RIFF" and image[8:12] == b"WEBP":
        return "image/webp"
    return None


def _dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _routed_model(model: str, open_router: bool) -> str:
    return f"nace-ai/{model}" if open_router and "/" not in model else model


def _body(model: str, request: DrexRequest, open_router: bool) -> dict[str, Any]:
    """The body as sent. On OpenRouter the images are data URLs in `state` content parts after the state as text."""
    if not open_router:
        body: dict[str, Any] = {"model": model, "state": request.state, "questions": request.questions}
        if request.images:
            body["images"] = [base64.b64encode(image).decode() for image in request.images]
        return body
    state = request.state
    if request.images:
        parts: list[dict[str, Any]] = [{"type": "text", "text": state if isinstance(state, str) else _dumps(state)}]
        for image in request.images:
            url = f"data:{image_mime_type(image)};base64,{base64.b64encode(image).decode()}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        state = parts
    return {"model": _routed_model(model, True), "state": state, "questions": request.questions}


def _idempotency_key(model: str, body: str) -> str:
    """The same for every attempt of a body. Drex's /v1/systemone ignores it and bills only a 200, so a retry is billed
    twice only when a 200 was lost on its way back."""
    return hashlib.sha256(f"{model}\0{body}".encode()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class _Failure:
    """An attempt without an answer: an HTTP status, or 0 when no response arrived."""

    status: int
    body: Any = None
    headers: httpx.Headers = field(default_factory=httpx.Headers)
    transport_error: str = ""
    tls: bool = False

    @property
    def error(self) -> dict[str, Any]:
        error = self.body.get("error") if isinstance(self.body, dict) else None
        if isinstance(error, str):
            return {"message": error}
        return error if isinstance(error, dict) else {}

    @property
    def message(self) -> str:
        message = self.error.get("message")
        return message if isinstance(message, str) else ""

    @property
    def issues(self) -> list[tuple[str, str]]:
        issues = self.error.get("issues")
        if not isinstance(issues, list):
            return []
        return [
            (str(issue.get("path") or ""), str(issue.get("message") or ""))
            for issue in issues
            if isinstance(issue, dict)
        ]

    @property
    def request_note(self) -> str:
        request_id = self.body.get("request_id") if isinstance(self.body, dict) else None
        return f" (request {request_id})" if isinstance(request_id, str) and request_id else ""


def _is_tls(error: BaseException) -> bool:
    """Whether a connection failed in the TLS handshake or on the certificate, which no retry fixes."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ssl.SSLError):
            return True
        text = str(current).lower()
        if "ssl" in text or "certificate" in text:
            return True
        current = current.__cause__ or current.__context__
    return False


def _failure(function: str, error: Exception) -> _Failure:
    if isinstance(error, NaceAPIResponseValidationError):
        raise fail(function, f"Drex answered with an unexpected body: {error}") from None
    if isinstance(error, NaceAPIError):
        return _Failure(error.status, error.body, error.headers)
    if isinstance(error, NaceAPIConnectionError | httpx.RequestError):
        return _Failure(0, transport_error=str(error) or type(error).__name__, tls=_is_tls(error))
    raise error


def _issue_text(path: str, message: str) -> str:
    return f"{path}: {message}" if path else message


def _is_under(path: str, name: str) -> bool:
    return path == name or path.startswith(f"{name}.")


def _refuses_row(failure: _Failure) -> bool:
    """Whether a 422 is about the row rather than the query's constants (questions, model): a request over the token
    limit, or issues only under `state` and `images`."""
    if failure.message.startswith("Request too long:"):
        return True
    issues = failure.issues
    return bool(issues) and all(_is_under(path, "state") or _is_under(path, "images") for path, _ in issues)


def _describe(failure: _Failure) -> str:
    """The response's own message, else its status. A row refusal's issues name `state`, never the row's own keys."""
    if failure.status == 0:
        return failure.transport_error or "no response"
    if failure.status == 422:
        reason = "; ".join(
            _issue_text("state" if _is_under(path, "state") else path, message) for path, message in failure.issues
        )
    else:
        reason = failure.message
    return f"HTTP {failure.status}" + (f": {reason}" if reason else "") + failure.request_note


def _raise_if_fatal(function: str, failure: _Failure, base_url: str) -> None:
    """Raises for the answers that fail the query: 401, 402, a 422 that is not about the row alone, and TLS."""
    if failure.tls:
        raise fail(function, f"could not reach Drex over TLS: {failure.transport_error}")
    if failure.status == 401:
        raise fail(function, (failure.message or "the API key was refused (HTTP 401)") + failure.request_note)
    if failure.status == 402:
        billing = OPENROUTER_BILLING_URL if is_open_router(base_url) else f"{base_url}/dashboard/billing"
        message = failure.message or "the account has no funds left (HTTP 402)."
        raise fail(function, f"{message} Billing: {billing}{failure.request_note}")
    if failure.status == 422 and not _refuses_row(failure):
        issues = "; ".join(_issue_text(path, message) for path, message in failure.issues) or failure.message
        raise fail(function, f"Drex refused the request: {issues or 'HTTP 422'}{failure.request_note}")


def _parse_wait(text: str | None, scale: float) -> float | None:
    if text is None:
        return None
    try:
        value = float(text.strip())
    except ValueError:
        if scale != 1.0:
            return None
        try:
            value = parsedate_to_datetime(text).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return None
        return min(max(0.0, value), MAX_RETRY_WAIT_SECONDS)
    if not math.isfinite(value) or value < 0:
        return None
    return min(value * scale, MAX_RETRY_WAIT_SECONDS)


def _retry_wait(failure: _Failure, retry: int) -> float:
    """The wait before retry number `retry` (1-based): what the response asked for, capped at 30 s, else 1, 2, 4 and 8 s
    less up to 25 % jitter. A capacity refusal's wait gains up to 25 %, so the rows it refused don't all return at
    once."""
    asked = _parse_wait(failure.headers.get("retry-after-ms"), 0.001)
    if asked is None:
        asked = _parse_wait(failure.headers.get("retry-after"), 1.0)
    if asked is not None:
        if failure.status in CAPACITY_STATUSES:
            return min(asked + random.uniform(0, asked / 4), MAX_RETRY_WAIT_SECONDS)
        return asked
    backoff = float(1 << min(retry - 1, 4))
    return min(backoff - random.uniform(0, backoff / 4), MAX_RETRY_WAIT_SECONDS)


class _Lost(Exception):
    """A request with no answer: its row is NULL, with a warning that holds no row content."""


class _Answers:
    """When this worker last had an answer from Drex. A request Drex refuses for capacity keeps waiting while others are
    answered: the query is then being paced to the account's limit, not failing."""

    def __init__(self) -> None:
        self.latest = float("-inf")

    def answered(self) -> None:
        self.latest = time.monotonic()


async def _with_retries(
    function: str,
    base_url: str,
    attempt: Callable[[], Awaitable[T]],
    slot: asyncio.Semaphore | None = None,
    answers: _Answers | None = None,
) -> T:
    """`attempt` until it answers, by the retry rules. The request holds its `slot` through every wait, so a batch never
    has more requests started than its concurrency, and a request's budget starts only when it gets a slot."""
    async with slot if slot is not None else contextlib.nullcontext():
        started = time.monotonic()
        tries = failures = 0
        while True:
            tries += 1
            try:
                result = await attempt()
            except Exception as error:  # noqa: BLE001 - _failure re-raises anything that isn't a failed request
                failure = _failure(function, error)
            else:
                if answers is not None:
                    answers.answered()
                return result
            _raise_if_fatal(function, failure, base_url)
            if failure.status != 0 and failure.status not in RETRYABLE_STATUSES:
                raise _Lost(_describe(failure))
            if failure.status not in CAPACITY_STATUSES:
                failures += 1
            wait = _retry_wait(failure, tries)
            deadline = max(started, answers.latest if answers is not None else started) + REQUEST_BUDGET_SECONDS
            if failures == MAX_ATTEMPTS or time.monotonic() + wait > deadline:
                raise _Lost(f"no answer after {tries} attempts, the last {_describe(failure)}")
            await asyncio.sleep(wait)


# ---------------------------------------------------------------------------------------------------------------------
# The worker's session
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ListedModel:
    name: str
    reads_images: bool


class _Session:
    """A client for one key, base URL and client name, with the models its base URL lists."""

    def __init__(self, target: Target) -> None:
        base_url = (target.base_url or os.environ.get("NACE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        api_key = target.api_key
        if api_key is None:
            api_key = os.environ.get("NACE_API_KEY", "")
            if is_open_router(base_url):
                api_key = os.environ.get("OPENROUTER_API_KEY", "").strip() or api_key
        if not api_key.strip():
            raise DrexError(
                "Drex: no API key; set NACE_API_KEY on the executors (spark.executorEnv.NACE_API_KEY). Databricks "
                "serverless compute can't set it: call the Unity Catalog functions there (drex_spark.databricks)"
            )
        if is_open_router(base_url) and api_key.strip().startswith("nace_"):
            raise DrexError(
                "Drex: a Nace API key (nace_...) is never sent to OpenRouter; set OPENROUTER_API_KEY on the executors "
                "(spark.executorEnv.OPENROUTER_API_KEY), or store an OpenRouter key in the Unity Catalog secret"
            )
        parts = urlsplit(base_url)
        local = parts.hostname in ("localhost", "127.0.0.1", "::1")
        if parts.scheme != "https" and not (parts.scheme == "http" and local):
            raise DrexError(
                f"Drex: NACE_BASE_URL must be an https:// URL (http:// only for localhost), not {base_url!r}"
            )
        self.base_url = base_url
        self.open_router = is_open_router(base_url)
        headers = {"X-Nace-Client": target.client_info}
        if self.open_router:
            referer = (target.openrouter_referer or os.environ.get("NACE_OPENROUTER_REFERER") or "").strip()
            title = (target.openrouter_title or os.environ.get("NACE_OPENROUTER_TITLE") or "").strip()
            for name, value in (("NACE_OPENROUTER_REFERER", referer), ("NACE_OPENROUTER_TITLE", title)):
                if not is_header_text(value):
                    raise DrexError(f"Drex: {name} must be printable ASCII, since it is sent as a header: {value!r}")
            headers |= {"HTTP-Referer": referer or OPENROUTER_REFERER, "X-OpenRouter-Title": title or target.title}
        self._headers = headers
        self._api_key = api_key.strip()
        self._http = httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT_SECONDS, limits=httpx.Limits(max_keepalive_connections=64)
        )
        self.client = AsyncNaceClient(
            api_key=self._api_key,
            base_url=base_url,
            client_info=target.client_info,
            timeout=REQUEST_TIMEOUT_SECONDS,
            # Retries are ours, by the shared surface's rules
            retry=RetryPolicy(max_retries=0),
            http_client=self._http,
            headers=headers,
        )
        self._models: list[ListedModel] | None = None
        self._answers = _Answers()

    async def _fetch_models(self, function: str) -> list[ListedModel]:
        async def attempt() -> Any:
            if not self.open_router:
                listed = await self.client.models.list()
                return [ListedModel(m.name, (m.alias_for or m.name) in IMAGE_MODELS) for m in listed.models]
            # OpenRouter's list, of which only Nace's models are Drex; it lists only text models unless asked
            response = await self._http.get(
                f"{self.base_url}/v1/models",
                params={"output_modalities": "decisions"},
                headers={**self._headers, "Authorization": f"Bearer {self._api_key}", "Accept": "application/json"},
            )
            if not response.is_success:
                raise NaceAPIError(response.status_code, _json_or_text(response), response.headers)
            data = _json_or_text(response)
            if not isinstance(data, dict) or not isinstance(data.get("data"), list):
                raise DrexError(f"{function}: GET /v1/models answered with an unexpected body")
            models = []
            for entry in data["data"]:
                model_id = entry.get("id") if isinstance(entry, dict) else None
                if not isinstance(model_id, str):
                    raise DrexError(f"{function}: GET /v1/models answered with an unexpected body")
                if model_id.startswith("nace-ai/"):
                    modalities = (entry.get("architecture") or {}).get("input_modalities") or []
                    models.append(ListedModel(model_id, "image" in modalities))
            return models

        try:
            return await _with_retries(function, self.base_url, attempt)
        except _Lost as lost:
            raise fail(function, f"could not list Drex's models: {lost}") from None

    async def find_model(self, function: str, model: str) -> ListedModel:
        """The listed model a request for `model` is sent as. Drex answers any unregistered drex-* name with drex-v2.0,
        so only this check refuses one. The list is fetched again for a model it lacks, so a new model is found."""
        name = _routed_model(model, self.open_router)
        for refresh in (False, True):
            if self._models is None or refresh:
                self._models = await self._fetch_models(function)
            for listed in self._models:
                if listed.name == name:
                    return listed
        names = ", ".join(listed.name for listed in self._models or [])
        raise fail(function, f"unknown model '{model}'; GET /v1/models lists {names}")

    async def decide(self, function: str, model: str, request: DrexRequest, slot: asyncio.Semaphore) -> str:
        """Drex's response body as written, or `_Lost` for a row with no answer."""
        for index, image in enumerate(request.images, 1):
            if image_mime_type(image) is None:
                raise _Lost(f"image {index} is not a PNG, JPEG or WebP image")
        if len(request.images) > MAX_IMAGES:
            raise _Lost(f"{len(request.images)} images, over Drex's limit of {MAX_IMAGES}")
        body = _body(model, request, self.open_router)
        text = _dumps(body)
        size = len(text.encode())
        if size > MAX_BODY_BYTES:
            raise _Lost(f"the request body is {size} bytes, over Drex's limit of {MAX_BODY_BYTES}")
        headers = {"Idempotency-Key": _idempotency_key(model, text)}
        images = body.pop("images", None)

        async def attempt() -> str:
            response = await self.client.system_one(
                body["state"],
                body["questions"],
                model=body["model"],
                images=images,
                extra_headers=headers,
            )
            # As Drex wrote it: drex_decide returns it unchanged
            return response.raw_http_response.text

        return await _with_retries(function, self.base_url, attempt, slot, self._answers)


def _json_or_text(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


class _Worker:
    """This process's event loop thread and sessions. A forked process starts its own: a loop doesn't survive fork."""

    def __init__(self) -> None:
        self.pid = os.getpid()
        self.loop = asyncio.new_event_loop()
        self.sessions: dict[Target, _Session] = {}
        threading.Thread(target=self.loop.run_forever, name="drex-spark", daemon=True).start()

    def run(self, coroutine: Awaitable[T]) -> T:
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result()  # type: ignore[arg-type]

    async def session(self, target: Target) -> _Session:
        session = self.sessions.get(target)
        if session is None:
            session = self.sessions[target] = _Session(target)
        return session


_worker: _Worker | None = None
_worker_lock = threading.Lock()


def _current() -> _Worker:
    global _worker
    with _worker_lock:
        if _worker is None or _worker.pid != os.getpid():
            _worker = _Worker()
        return _worker


def check_model(target: Target, function: str, model: str, *, images: bool) -> None:
    """Fails the query for a model the base URL doesn't list, or images for a model that doesn't read them."""

    async def check() -> None:
        listed = await (await worker.session(target)).find_model(function, model)
        if images and not listed.reads_images:
            raise fail(function, f"{model} does not read images; use an image model such as drex-max")

    worker = _current()
    worker.run(check())


def decide_all(
    target: Target, function: str, model: str, requests: Sequence[DrexRequest], concurrency: int
) -> list[str | None]:
    """Each request's response body, or `None` (logged as a warning) for a row with no answer, in order.

    Raises `DrexError` for anything that fails the query; the requests still in flight are then cancelled.
    """
    if concurrency < 1:
        raise fail(function, "concurrency must be at least 1")

    async def run() -> list[str | None]:
        session = await worker.session(target)
        slot = asyncio.Semaphore(concurrency)

        async def one(request: DrexRequest) -> str | None:
            try:
                return await session.decide(function, model, request, slot)
            except _Lost as lost:
                logger.warning("%s: a row returned NULL: %s", function, lost)
                return None

        tasks = [asyncio.ensure_future(one(request)) for request in requests]
        try:
            return list(await asyncio.gather(*tasks))
        finally:
            for task in tasks:
                task.cancel()

    worker = _current()
    return worker.run(run())
