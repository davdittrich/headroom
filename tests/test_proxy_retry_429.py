"""Upstream 429 rate-limit retry + Retry-After honoring (fixes #1221).

Both the non-streaming (``server.py:_retry_request``) and streaming
(``streaming.py:_stream_response``) forwarders must retry an upstream 429 with
backoff instead of passing it straight back to the client, since a parallel
agent fan-out that exceeds the per-minute limit otherwise aborts every run.
"""

from __future__ import annotations

import asyncio

import httpx

from headroom.proxy.server import ProxyConfig, create_app


class _RateLimitTransport(httpx.AsyncBaseTransport):
    """Returns ``fail_status`` for the first ``fail_times`` calls, then 200.

    Records ``calls`` so a test can assert whether a retry happened.
    """

    def __init__(
        self,
        *,
        fail_status: int = 429,
        fail_times: int = 1,
        retry_after: str | None = None,
        sse: bool = False,
    ) -> None:
        self.fail_status = fail_status
        self.fail_times = fail_times
        self.retry_after = retry_after
        self.sse = sse
        self.calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        async for _ in request.stream:  # drain the request body
            pass
        if self.calls <= self.fail_times:
            headers = {"retry-after": self.retry_after} if self.retry_after is not None else {}
            return httpx.Response(
                self.fail_status,
                headers=headers,
                json={"type": "error", "error": {"type": "rate_limit_error"}},
            )
        if self.sse:
            body = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )


def _proxy_with(
    transport: _RateLimitTransport,
    *,
    max_attempts: int = 3,
    retry_after_budget_ms: int | None = None,
):
    kwargs: dict[str, object] = {
        "optimize": False,
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "cost_tracking_enabled": False,
        "log_requests": False,
        "ccr_inject_tool": False,
        "ccr_handle_responses": False,
        "ccr_context_tracking": False,
        "image_optimize": False,
        "retry_enabled": True,
        "retry_max_attempts": max_attempts,
        "retry_base_delay_ms": 1,
        "retry_max_delay_ms": 5000,
    }
    if retry_after_budget_ms is not None:
        kwargs["retry_after_budget_ms"] = retry_after_budget_ms
    config = ProxyConfig(**kwargs)
    proxy = create_app(config).state.proxy
    proxy.http_client = httpx.AsyncClient(transport=transport)
    return proxy


# --- non-streaming: _retry_request ---------------------------------------


def test_retry_request_retries_429_then_succeeds() -> None:
    transport = _RateLimitTransport(fail_status=429, fail_times=1, retry_after="0")
    proxy = _proxy_with(transport)
    resp = asyncio.run(proxy._retry_request("POST", "https://up/v1/messages", {}, {"messages": []}))
    assert resp.status_code == 200
    assert transport.calls == 2  # one 429 + one success — the retry happened


def test_retry_request_returns_429_verbatim_on_exhaustion() -> None:
    # Always 429: must return the 429 to the client, NOT raise / convert to 5xx.
    transport = _RateLimitTransport(fail_status=429, fail_times=99, retry_after="0")
    proxy = _proxy_with(transport, max_attempts=3)
    resp = asyncio.run(proxy._retry_request("POST", "https://up/v1/messages", {}, {"messages": []}))
    assert resp.status_code == 429
    assert transport.calls == 3  # exhausted all attempts


def test_retry_request_honors_retry_after(monkeypatch) -> None:
    slept: list[float] = []

    async def _fake_wait(self, seconds: float) -> bool:  # type: ignore[no-untyped-def]
        slept.append(seconds)
        return False

    monkeypatch.setattr(
        "headroom.proxy.server.HeadroomProxy._wait_for_retry_delay_or_shutdown", _fake_wait
    )
    transport = _RateLimitTransport(fail_status=429, fail_times=1, retry_after="2")
    proxy = _proxy_with(transport)
    asyncio.run(proxy._retry_request("POST", "https://up/v1/messages", {}, {"messages": []}))
    # Retry-After: 2s honored (not the ~1-10ms jittered exponential backoff).
    assert slept and abs(slept[0] - 2.0) < 0.01


def test_retry_request_does_not_retry_other_4xx() -> None:
    transport = _RateLimitTransport(fail_status=400, fail_times=99)
    proxy = _proxy_with(transport)
    resp = asyncio.run(proxy._retry_request("POST", "https://up/v1/messages", {}, {"messages": []}))
    assert resp.status_code == 400
    assert transport.calls == 1  # 4xx (non-429) still short-circuits — no retry


def test_retry_request_still_retries_5xx() -> None:
    transport = _RateLimitTransport(fail_status=503, fail_times=1)
    proxy = _proxy_with(transport)
    resp = asyncio.run(proxy._retry_request("POST", "https://up/v1/messages", {}, {"messages": []}))
    assert resp.status_code == 200
    assert transport.calls == 2  # 5xx retry path unchanged


# --- streaming: _stream_response -----------------------------------------


def test_stream_response_retries_429() -> None:
    transport = _RateLimitTransport(fail_status=429, fail_times=1, retry_after="0", sse=True)
    proxy = _proxy_with(transport)
    asyncio.run(
        proxy._stream_response(
            "https://up/v1/messages",
            {},
            {"messages": []},
            "anthropic",
            "claude-3",
            "r1",
            0,
            0,
            0,
            [],
            {},
            0.0,
        )
    )
    assert transport.calls == 2  # streaming 429 retried, not forwarded raw


# --- 529 overloaded: same transient-retry path as 429 --------------------
#
# 529 is Anthropic's ``overloaded_error``. Like 429 it means "try again
# shortly", so both forwarders must retry it honoring Retry-After. Before this
# fix the streaming path forwarded a 529 to the client raw (zero retries), and
# _retry_request retried it only via the generic 5xx path — raising on
# exhaustion instead of returning the 529 verbatim, and ignoring Retry-After.


def test_retry_request_retries_529_then_succeeds() -> None:
    transport = _RateLimitTransport(fail_status=529, fail_times=1, retry_after="0")
    proxy = _proxy_with(transport)
    resp = asyncio.run(proxy._retry_request("POST", "https://up/v1/messages", {}, {"messages": []}))
    assert resp.status_code == 200
    assert transport.calls == 2  # one 529 + one success — the retry happened


def test_retry_request_returns_529_verbatim_on_exhaustion() -> None:
    # Always 529: must return the 529 to the client, NOT raise / convert to 5xx.
    transport = _RateLimitTransport(fail_status=529, fail_times=99, retry_after="0")
    proxy = _proxy_with(transport, max_attempts=3)
    resp = asyncio.run(proxy._retry_request("POST", "https://up/v1/messages", {}, {"messages": []}))
    assert resp.status_code == 529
    assert transport.calls == 3  # exhausted all attempts, returned verbatim


def test_retry_request_honors_retry_after_on_529(monkeypatch) -> None:
    slept: list[float] = []

    async def _fake_wait(self, seconds: float) -> bool:  # type: ignore[no-untyped-def]
        slept.append(seconds)
        return False

    monkeypatch.setattr(
        "headroom.proxy.server.HeadroomProxy._wait_for_retry_delay_or_shutdown", _fake_wait
    )
    transport = _RateLimitTransport(fail_status=529, fail_times=1, retry_after="2")
    proxy = _proxy_with(transport)
    asyncio.run(proxy._retry_request("POST", "https://up/v1/messages", {}, {"messages": []}))
    # Retry-After: 2s honored for 529 just like 429.
    assert slept and abs(slept[0] - 2.0) < 0.01


def test_stream_response_retries_529() -> None:
    # The gap this PR closes: an interactive (streaming) session hitting a 529
    # used to get "Overloaded" surfaced immediately, with no retry.
    transport = _RateLimitTransport(fail_status=529, fail_times=1, retry_after="0", sse=True)
    proxy = _proxy_with(transport)
    asyncio.run(
        proxy._stream_response(
            "https://up/v1/messages",
            {},
            {"messages": []},
            "anthropic",
            "claude-3",
            "r1",
            0,
            0,
            0,
            [],
            {},
            0.0,
        )
    )
    assert transport.calls == 2  # streaming 529 retried, not forwarded raw


# --- WU1 (headroom-8z2.1): Retry-After is a floor, not clamped down --------
#
# Root cause: retry_after_ms used to clamp the upstream-demanded wait DOWN to
# retry_max_delay_ms, guaranteeing a repeat 429 when a caller asked for more
# than the cap (measured: 276/292 live retries waited exactly the 30000ms
# clamp). The fix gives the "how long are we willing to hold this request"
# budget its own config field (retry_after_budget_ms) and, when the demand
# exceeds it, returns the 429 verbatim on the first attempt instead of
# sleeping short and retrying into a wait we already know is insufficient.


def test_retry_after_honored_beyond_old_backoff_cap(monkeypatch) -> None:
    slept: list[float] = []

    async def _fake_wait(self, seconds: float) -> bool:  # type: ignore[no-untyped-def]
        slept.append(seconds)
        return False

    monkeypatch.setattr(
        "headroom.proxy.server.HeadroomProxy._wait_for_retry_delay_or_shutdown", _fake_wait
    )
    # retry_max_delay_ms=5000 (from _proxy_with) used to clamp this to 5s;
    # retry_after_budget_ms=90000 is the new, separate "willing to wait" cap.
    transport = _RateLimitTransport(fail_status=429, fail_times=1, retry_after="60")
    proxy = _proxy_with(transport, retry_after_budget_ms=90000)
    asyncio.run(proxy._retry_request("POST", "https://up/v1/messages", {}, {"messages": []}))
    assert slept and abs(slept[0] - 60.0) < 0.01


def test_stream_response_retry_after_honored_beyond_old_backoff_cap(monkeypatch) -> None:
    slept: list[float] = []

    async def _fake_sleep(seconds: float) -> None:  # type: ignore[no-untyped-def]
        slept.append(seconds)

    monkeypatch.setattr("headroom.proxy.handlers.streaming.asyncio.sleep", _fake_sleep)
    # Streaming-path counterpart of test_retry_after_honored_beyond_old_backoff_cap:
    # retry_max_delay_ms=5000 (from _proxy_with) used to clamp this to 5s;
    # retry_after_budget_ms=90000 is the new, separate "willing to wait" cap.
    transport = _RateLimitTransport(fail_status=429, fail_times=1, retry_after="60", sse=True)
    proxy = _proxy_with(transport, retry_after_budget_ms=90000)
    asyncio.run(
        proxy._stream_response(
            "https://up/v1/messages",
            {},
            {"messages": []},
            "anthropic",
            "claude-3",
            "r1",
            0,
            0,
            0,
            [],
            {},
            0.0,
        )
    )
    assert slept and abs(slept[0] - 60.0) < 0.01


def test_retry_after_beyond_budget_returns_429_verbatim_single_call(monkeypatch) -> None:
    async def _fail_if_called(self, seconds: float) -> bool:  # type: ignore[no-untyped-def]
        raise AssertionError("must not sleep when Retry-After exceeds the retry budget")

    monkeypatch.setattr(
        "headroom.proxy.server.HeadroomProxy._wait_for_retry_delay_or_shutdown", _fail_if_called
    )
    # Subscription-style 429s commonly carry Retry-After well beyond any
    # sane in-loop budget; default retry_after_budget_ms is 30000ms.
    transport = _RateLimitTransport(fail_status=429, fail_times=99, retry_after="3600")
    proxy = _proxy_with(transport)
    resp = asyncio.run(proxy._retry_request("POST", "https://up/v1/messages", {}, {"messages": []}))
    assert resp.status_code == 429
    assert resp.headers.get("retry-after") == "3600"
    assert transport.calls == 1  # no retry attempted — budget exceeded on the first look


def test_retry_after_zero_does_not_produce_a_zero_wait(monkeypatch) -> None:
    # Retry-After: 0 parses to 0.0, which is a valid ("not None") delay but
    # not a *usable* one -- sleeping 0ms would immediately re-fire into the
    # same rate limit. It must fall back to jittered backoff instead, same
    # as an absent header.
    slept: list[float] = []

    async def _fake_wait(self, seconds: float) -> bool:  # type: ignore[no-untyped-def]
        slept.append(seconds)
        return False

    monkeypatch.setattr(
        "headroom.proxy.server.HeadroomProxy._wait_for_retry_delay_or_shutdown", _fake_wait
    )
    transport = _RateLimitTransport(fail_status=429, fail_times=1, retry_after="0")
    proxy = _proxy_with(transport)
    asyncio.run(proxy._retry_request("POST", "https://up/v1/messages", {}, {"messages": []}))
    assert slept and slept[0] > 0.0


def test_retry_after_past_http_date_does_not_produce_a_zero_wait(monkeypatch) -> None:
    slept: list[float] = []

    async def _fake_wait(self, seconds: float) -> bool:  # type: ignore[no-untyped-def]
        slept.append(seconds)
        return False

    monkeypatch.setattr(
        "headroom.proxy.server.HeadroomProxy._wait_for_retry_delay_or_shutdown", _fake_wait
    )
    # Any date in the past parses to a negative delta, floored to 0.0 by
    # retry_after_ms -- same "not usable" case as Retry-After: 0.
    transport = _RateLimitTransport(
        fail_status=429, fail_times=1, retry_after="Mon, 01 Jan 2001 00:00:00 GMT"
    )
    proxy = _proxy_with(transport)
    asyncio.run(proxy._retry_request("POST", "https://up/v1/messages", {}, {"messages": []}))
    assert slept and slept[0] > 0.0


def test_stream_response_retry_after_zero_does_not_produce_a_zero_wait(monkeypatch) -> None:
    slept: list[float] = []

    async def _fake_sleep(seconds: float) -> None:  # type: ignore[no-untyped-def]
        slept.append(seconds)

    monkeypatch.setattr("headroom.proxy.handlers.streaming.asyncio.sleep", _fake_sleep)
    transport = _RateLimitTransport(fail_status=429, fail_times=1, retry_after="0", sse=True)
    proxy = _proxy_with(transport)
    asyncio.run(
        proxy._stream_response(
            "https://up/v1/messages",
            {},
            {"messages": []},
            "anthropic",
            "claude-3",
            "r1",
            0,
            0,
            0,
            [],
            {},
            0.0,
        )
    )
    assert slept and slept[0] > 0.0


def test_stream_response_retry_after_past_http_date_does_not_produce_a_zero_wait(
    monkeypatch,
) -> None:
    slept: list[float] = []

    async def _fake_sleep(seconds: float) -> None:  # type: ignore[no-untyped-def]
        slept.append(seconds)

    monkeypatch.setattr("headroom.proxy.handlers.streaming.asyncio.sleep", _fake_sleep)
    # Same "not usable" case as Retry-After: 0 -- streaming-path counterpart
    # of test_retry_after_past_http_date_does_not_produce_a_zero_wait.
    transport = _RateLimitTransport(
        fail_status=429, fail_times=1, retry_after="Mon, 01 Jan 2001 00:00:00 GMT", sse=True
    )
    proxy = _proxy_with(transport)
    asyncio.run(
        proxy._stream_response(
            "https://up/v1/messages",
            {},
            {"messages": []},
            "anthropic",
            "claude-3",
            "r1",
            0,
            0,
            0,
            [],
            {},
            0.0,
        )
    )
    assert slept and slept[0] > 0.0


def test_retry_after_absent_still_uses_jitter_backoff() -> None:
    transport = _RateLimitTransport(fail_status=429, fail_times=1, retry_after=None)
    proxy = _proxy_with(transport)
    resp = asyncio.run(proxy._retry_request("POST", "https://up/v1/messages", {}, {"messages": []}))
    assert resp.status_code == 200
    assert transport.calls == 2  # jittered backoff fallback unchanged, retry still happens


def test_retry_after_529_not_bound_by_429_budget() -> None:
    # 529 keeps its pre-fix behavior (clamped to retry_max_delay_ms, not the
    # 429 budget) because Anthropic's overloaded_error carries no trustworthy
    # Retry-After. budget=1000ms would short-circuit a 429 demanding 3s, but
    # 529 must still retry since 3000ms <= retry_max_delay_ms=5000.
    transport = _RateLimitTransport(fail_status=529, fail_times=1, retry_after="3")
    proxy = _proxy_with(transport, retry_after_budget_ms=1000)
    resp = asyncio.run(proxy._retry_request("POST", "https://up/v1/messages", {}, {"messages": []}))
    assert resp.status_code == 200
    assert transport.calls == 2  # 529 retried despite exceeding the 429 budget


def test_stream_response_429_beyond_budget_forwards_without_retry() -> None:
    transport = _RateLimitTransport(fail_status=429, fail_times=99, retry_after="3600", sse=True)
    proxy = _proxy_with(transport)
    asyncio.run(
        proxy._stream_response(
            "https://up/v1/messages",
            {},
            {"messages": []},
            "anthropic",
            "claude-3",
            "r1",
            0,
            0,
            0,
            [],
            {},
            0.0,
        )
    )
    assert transport.calls == 1  # budget exceeded — forwarded on first attempt, no retry
