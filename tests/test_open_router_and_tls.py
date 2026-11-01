"""OpenRouter's body, headers and model list, and a TLS failure, with httpx mocked in this process (respx)."""

import base64
import itertools
import json

import httpx
import pandas as pd
import pytest
import respx
from conftest import RED

from drex_spark import _client
from drex_spark._batch import decide_batch, typed_batch
from drex_spark._client import Target
from drex_spark._contract import DrexError

OPEN_ROUTER = "https://openrouter.ai/api"
MODELS = {
    "data": [
        {"id": "liquid/d1.1", "architecture": {"input_modalities": ["text", "image"]}},
        {"id": "nace-ai/drex-v2.0", "architecture": {"input_modalities": ["text"]}},
        {"id": "nace-ai/drex-max", "architecture": {"input_modalities": ["text", "image"]}},
    ]
}


def answer(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    answers = {qid: {"type": "noul", "noul": 0.25} for qid in body["questions"]}
    usage = {"input_tokens": 10, "output_tokens": 2}
    return httpx.Response(200, json={"model": body["model"], "answers": answers, "usage": usage, "request_id": "r1"})


@respx.mock
def test_open_router_namespaces_the_model_and_puts_images_in_state():
    models = respx.get(f"{OPEN_ROUTER}/v1/models").mock(return_value=httpx.Response(200, json=MODELS))
    decisions = respx.post(f"{OPEN_ROUTER}/v1/systemone").mock(side_effect=answer)
    target = Target(
        client_info="drex-spark/test", title="Drex for Apache Spark", api_key="sk-or-v1-test", base_url=OPEN_ROUTER
    )

    questions = json.dumps({"red": {"type": "noul", "criteria": {"true": "The image is red"}}})
    [body] = decide_batch(
        target, "drex_decide_image", pd.Series(['{"n": 1}']), pd.Series([RED]), [questions], [None], 8
    )
    assert json.loads(body)["model"] == "nace-ai/drex-max"

    assert models.calls.last.request.url.params["output_modalities"] == "decisions"
    request = decisions.calls.last.request
    sent = json.loads(request.content)
    assert "images" not in sent
    assert sent["model"] == "nace-ai/drex-max"
    assert sent["state"][0] == {"type": "text", "text": '{"n":1}'}
    assert sent["state"][1]["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(RED).decode()
    # OpenRouter refuses a question without instructions, which Drex allows when it has criteria
    assert sent["questions"]["red"]["instructions"] == ""
    assert request.headers["HTTP-Referer"] == "https://nace.ai"
    assert request.headers["X-OpenRouter-Title"] == "Drex for Apache Spark"
    assert request.headers["X-Nace-Client"] == "drex-spark/test"
    assert request.headers["Authorization"] == "Bearer sk-or-v1-test"


@pytest.fixture(autouse=True)
def fresh_sessions():
    """A worker keeps one session per target, and reads the environment when it makes one."""
    _client._current().sessions.clear()


def open_router_routes():
    models = respx.get(f"{OPEN_ROUTER}/v1/models").mock(return_value=httpx.Response(200, json=MODELS))
    decisions = respx.post(f"{OPEN_ROUTER}/v1/systemone").mock(side_effect=answer)
    return models, decisions


def noul(target: Target) -> None:
    typed_batch(target, "drex_noul", "noul", pd.Series(["x"]), ["Refund?"], [None], [None], 8)


@respx.mock
def test_open_router_attribution_is_set_by_the_caller():
    models, decisions = open_router_routes()
    target = Target(
        client_info="drex-spark/test",
        title="Drex for Apache Spark",
        api_key="sk-or-v1-test",
        base_url=OPEN_ROUTER,
        openrouter_referer="https://app.example",
        openrouter_title="Example",
    )
    noul(target)
    for route in (models, decisions):
        headers = route.calls.last.request.headers
        assert headers.get_list("http-referer") == ["https://app.example"]
        assert headers.get_list("x-openrouter-title") == ["Example"]


@respx.mock
def test_open_router_attribution_comes_from_the_environment_when_the_target_has_none(monkeypatch):
    monkeypatch.setenv("NACE_OPENROUTER_REFERER", "https://env.example")
    monkeypatch.setenv("NACE_OPENROUTER_TITLE", "From the environment")
    _, decisions = open_router_routes()
    noul(
        Target(
            client_info="drex-spark/test", title="Drex for Apache Spark", api_key="sk-or-v1-test", base_url=OPEN_ROUTER
        )
    )
    headers = decisions.calls.last.request.headers
    assert headers.get_list("http-referer") == ["https://env.example"]
    assert headers.get_list("x-openrouter-title") == ["From the environment"]


@respx.mock
def test_open_router_attribution_must_be_ascii_and_blank_is_the_default(monkeypatch):
    models, decisions = open_router_routes()
    target = Target(
        client_info="drex-spark/test", title="Drex for Apache Spark", api_key="sk-or-v1-test", base_url=OPEN_ROUTER
    )
    monkeypatch.setenv("NACE_OPENROUTER_TITLE", "Acme\u2122 Analytics")
    with pytest.raises(DrexError, match="NACE_OPENROUTER_TITLE must be printable ASCII"):
        noul(target)
    assert models.call_count == decisions.call_count == 0
    _client._current().sessions.clear()
    monkeypatch.setenv("NACE_OPENROUTER_TITLE", "  ")
    noul(target)
    assert decisions.calls.last.request.headers["X-OpenRouter-Title"] == "Drex for Apache Spark"


@respx.mock
def test_a_nace_host_gets_no_attribution_headers():
    route = respx.post("https://drex.example/v1/systemone").mock(side_effect=answer)
    respx.get("https://drex.example/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "models": [{"name": "drex-v2.0", "description": "", "release_date": "2026-01-01", "alias_for": None}]
            },
        )
    )
    target = Target(
        client_info="drex-spark/test",
        title="Drex for Apache Spark",
        api_key="nace_sk_x",
        base_url="https://drex.example",
        openrouter_referer="https://app.example",
        openrouter_title="Example",
    )
    noul(target)
    headers = route.calls.last.request.headers
    assert "http-referer" not in headers and "x-openrouter-title" not in headers


@respx.mock
def test_a_nace_key_is_never_sent_to_open_router(monkeypatch):
    models, decisions = open_router_routes()
    for target in (
        Target(client_info="drex-spark/test", title="Drex for Apache Spark", api_key="nace_sk_x", base_url=OPEN_ROUTER),
        Target(client_info="drex-spark/test", title="Drex for Apache Spark", base_url=OPEN_ROUTER),
    ):
        monkeypatch.setenv("NACE_API_KEY", "nace_sk_x")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(DrexError, match="OPENROUTER_API_KEY"):
            noul(target)
    assert models.call_count == decisions.call_count == 0


@respx.mock
def test_open_router_prefers_openrouter_api_key_from_the_environment(monkeypatch):
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-from-openrouter-var")
    _, decisions = open_router_routes()
    noul(Target(client_info="drex-spark/test", title="Drex for Apache Spark", base_url=OPEN_ROUTER))
    assert decisions.calls.last.request.headers["Authorization"] == "Bearer sk-or-v1-from-openrouter-var"


@respx.mock
def test_an_open_router_key_in_nace_api_key_still_works(monkeypatch):
    monkeypatch.setenv("NACE_API_KEY", "sk-or-v1-test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _, decisions = open_router_routes()
    noul(Target(client_info="drex-spark/test", title="Drex for Apache Spark", base_url=OPEN_ROUTER))
    assert decisions.calls.last.request.headers["Authorization"] == "Bearer sk-or-v1-test"


@respx.mock
def test_open_router_402_names_its_credits_page():
    respx.get(f"{OPEN_ROUTER}/v1/models").mock(return_value=httpx.Response(200, json=MODELS))
    respx.post(f"{OPEN_ROUTER}/v1/systemone").mock(
        return_value=httpx.Response(402, json={"error": {"message": "Insufficient credits", "code": 402}})
    )
    target = Target(
        client_info="drex-spark/test", title="Drex for Apache Spark", api_key="sk-or-v1-402", base_url=OPEN_ROUTER
    )
    with pytest.raises(DrexError, match="Insufficient credits Billing: https://openrouter.ai/settings/credits"):
        typed_batch(target, "drex_noul", "noul", pd.Series(["x"]), ["Refund?"], [None], [None], 8)


@respx.mock
def test_a_tls_failure_fails_the_query_at_once():
    route = respx.get("https://drex.invalid/v1/models").mock(
        side_effect=httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
    )
    target = Target(
        client_info="drex-spark/test",
        title="Drex for Apache Spark",
        api_key="nace_sk_x",
        base_url="https://drex.invalid",
    )
    with pytest.raises(DrexError, match="could not reach Drex over TLS"):
        typed_batch(target, "drex_noul", "noul", pd.Series(["x"]), itertools.repeat("Refund?"), [None], [None], 8)
    assert route.call_count == 1


def test_plain_http_is_refused_except_on_localhost():
    target = Target(
        client_info="drex-spark/test",
        title="Drex for Apache Spark",
        api_key="nace_sk_x",
        base_url="http://drex.example",
    )
    with pytest.raises(DrexError, match=r"must be an https:// URL \(http:// only for localhost\)"):
        typed_batch(target, "drex_noul", "noul", pd.Series(["x"]), ["Refund?"], [None], [None], 8)
