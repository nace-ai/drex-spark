"""The retry rules of `_with_retries`, on scripted answers (no HTTP): a capacity refusal waits while Drex answers other
requests, any other retryable failure has 5 attempts, and a request keeps its slot while it waits."""

import asyncio

import httpx
import pytest
from nace_sdk import NaceAPIError

from drex_spark import _client
from drex_spark._client import _Answers, _Lost, _with_retries


def refusal(status: int, retry_after_ms: int = 10) -> NaceAPIError:
    body = {"error": {"type": "overloaded", "message": "busy"}, "request_id": "req_1"}
    return NaceAPIError(status, body, httpx.Headers({"retry-after-ms": str(retry_after_ms)}))


def scripted(*outcomes: object):
    """An attempt that raises or returns each outcome in turn, counting its calls."""
    remaining = list(outcomes)

    async def attempt() -> object:
        attempt.calls += 1
        outcome = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    attempt.calls = 0
    return attempt


def run(attempt, slot=None, answers=None):
    return asyncio.run(_with_retries("drex_noul", "https://console.nace.ai", attempt, slot, answers))


@pytest.mark.parametrize("status", [429, 529])
def test_capacity_refusals_are_not_counted_as_attempts(status):
    attempt = scripted(*[refusal(status)] * 12, "answer")
    assert run(attempt) == "answer"
    assert attempt.calls == 13


@pytest.mark.parametrize("status", [408, 500, 503])
def test_other_retryable_failures_end_after_five_attempts(status):
    attempt = scripted(refusal(status))
    with pytest.raises(_Lost, match=f"no answer after 5 attempts, the last HTTP {status}: busy"):
        run(attempt)
    assert attempt.calls == 5


def test_capacity_refusals_end_once_drex_answers_nothing_for_the_budget(monkeypatch):
    monkeypatch.setattr(_client, "REQUEST_BUDGET_SECONDS", 0.3)
    attempt = scripted(refusal(529, retry_after_ms=50))
    with pytest.raises(_Lost, match=r"no answer after \d+ attempts, the last HTTP 529: busy"):
        run(attempt, answers=_Answers())
    assert 3 <= attempt.calls <= 7


def test_capacity_refusals_wait_past_the_budget_while_other_requests_are_answered(monkeypatch):
    monkeypatch.setattr(_client, "REQUEST_BUDGET_SECONDS", 0.3)
    answers = _Answers()

    async def main():
        async def others():
            # Another row of the worker, answered every 0.1 s for a second
            for _ in range(10):
                await asyncio.sleep(0.1)
                answers.answered()

        background = asyncio.ensure_future(others())
        refused = scripted(*[refusal(529, retry_after_ms=50)] * 10, "answer")
        result = await _with_retries("drex_noul", "https://console.nace.ai", refused, None, answers)
        await background
        return result, refused.calls

    result, calls = asyncio.run(main())
    assert (result, calls) == ("answer", 11)


def test_a_request_keeps_its_slot_while_it_waits():
    async def main():
        slot = asyncio.Semaphore(1)
        order = []

        def recording(name, *outcomes):
            attempt = scripted(*outcomes)

            async def recorded():
                order.append(name)
                return await attempt()

            return recorded

        first = recording("first", refusal(529, retry_after_ms=50), refusal(529, retry_after_ms=50), "first")
        second = recording("second", "second")
        results = await asyncio.gather(
            _with_retries("drex_noul", "https://console.nace.ai", first, slot),
            _with_retries("drex_noul", "https://console.nace.ai", second, slot),
        )
        return results, order

    results, order = asyncio.run(main())
    assert results == ["first", "second"]
    assert order == ["first", "first", "first", "second"]


def test_a_capacity_wait_is_what_drex_asked_for_plus_up_to_a_quarter():
    failure = _client._failure("drex_noul", refusal(429, retry_after_ms=1000))
    waits = [_client._retry_wait(failure, 1) for _ in range(200)]
    assert all(1.0 <= wait <= 1.25 for wait in waits)
    assert max(waits) - min(waits) > 0.1
    other = _client._failure("drex_noul", refusal(503, retry_after_ms=1000))
    assert _client._retry_wait(other, 1) == 1.0
