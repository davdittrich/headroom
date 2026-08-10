"""WU0 (headroom-8z2.4): deterministic offline harness for the parallel-agent
429 storm (epic headroom-8z2).

Drives N concurrent simulated clients through the real proxy retry path
(``HeadroomProxy._retry_request`` and the streaming sibling in
``handlers/streaming.py``) against a scripted, non-replenishing-capacity mock
upstream, and reports the metrics every later work unit's before/after claim
depends on. This module does NOT change proxy behavior — it only proves the
harness reproduces the CURRENT, BROKEN failure shape:

  * a rate-limited request makes ``retry_max_attempts`` (3) upstream calls,
  * failing retries wait the ``retry_max_delay_ms`` cap (the upstream
    ``Retry-After`` in the mock always exceeds the cap, so every retry
    clamps to it — the same clamp bug that produced the 30000ms figure in
    the live log, at test-friendly millisecond scale),
  * the great majority of over-capacity requests still surface as a
    client-visible ``rate_limit_error`` (low rescue rate).

Deterministic-time: no wall-clock sleeps of real duration. ``retry_base_delay_ms``
/ ``retry_max_delay_ms`` are configured in single-digit milliseconds, so a
full N=20 run completes in well under a second, and no randomness is on the
delay path (``retry_after_ms`` is always non-None here, so the ``or
jitter_delay_ms(...)`` fallback — the only source of ``random.random()`` in
this loop — never executes).
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any

import httpx
import pytest

from headroom.proxy.server import ProxyConfig, create_app

# Mirrors RETRYABLE_OVERLOAD_STATUSES semantics (headroom/proxy/helpers.py:848)
# without importing private internals not needed here.
_RATE_LIMIT_STATUS = 429

# Test-scale retry timing: same shape as production defaults
# (retry_max_attempts=3, models.py:310) but three orders of magnitude
# smaller delays so the suite stays fast and offline.
_RETRY_MAX_ATTEMPTS = 3
_RETRY_BASE_DELAY_MS = 2
_RETRY_MAX_DELAY_MS = 10
# Mock upstream always asks for more than the cap, so every retry clamps to
# it deterministically -- this is the harness analog of the live-log figure
# ("276 of 292 retries logged retrying in 30000ms").
_MOCK_RETRY_AFTER_S = 5.0


class _CapacityLimitedTransport(httpx.AsyncBaseTransport):
    """Scripted mock upstream: a fixed, non-replenishing capacity budget.

    The first ``capacity`` calls (global, first-come-first-served across all
    concurrent clients, serialized by ``self._lock``) succeed with 200;
    every call after that returns 429 with a ``Retry-After`` header. No
    randomness -- deterministic given ``capacity`` and call order, and
    asyncio's cooperative scheduling makes call order reproducible run to
    run for a fixed set of coroutines. Calls are bucketed per client via the
    ``x-client-id`` header the harness attaches to each simulated client's
    requests, so the test can assert per-client call counts without
    threading extra plumbing through the proxy.
    """

    def __init__(self, *, capacity: int, sse: bool = False) -> None:
        self.capacity = capacity
        self.sse = sse
        self.calls = 0
        self.calls_by_client: dict[str, int] = defaultdict(int)
        self._granted = 0
        self._lock = asyncio.Lock()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        client_id = request.headers.get("x-client-id", "?")
        async with self._lock:
            self.calls += 1
            self.calls_by_client[client_id] += 1
            allow = self._granted < self.capacity
            if allow:
                self._granted += 1
        async for _ in request.stream:  # drain the request body
            pass
        if not allow:
            return httpx.Response(
                _RATE_LIMIT_STATUS,
                headers={"retry-after": str(_MOCK_RETRY_AFTER_S)},
                json={
                    "type": "error",
                    "error": {
                        "type": "rate_limit_error",
                        "message": (
                            "This request would exceed your account's rate "
                            "limit. Please try again later."
                        ),
                    },
                },
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


def _proxy_with(transport: _CapacityLimitedTransport, *, governor_enabled: bool = False):
    """Build a HeadroomProxy wired to ``transport``.

    ``governor_enabled`` is the A/B arm switch the epic guard requires
    (headroom-8z2 guard: "gate is keyed... one install point"; headroom-8z2.4
    guard: "expose the arm as an explicit parameter"). WU0 builds no
    governor -- there is nothing in ``ProxyConfig`` yet to flip -- so the
    parameter is currently inert and both arms produce a byte-identical
    ``ProxyConfig``. WU2/WU3 wire this into the real kill switch; until then
    it exists so callers of this harness never change call signature.
    """
    del governor_enabled  # no-op until WU2/WU3 land a governor to toggle
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        retry_enabled=True,
        retry_max_attempts=_RETRY_MAX_ATTEMPTS,
        retry_base_delay_ms=_RETRY_BASE_DELAY_MS,
        retry_max_delay_ms=_RETRY_MAX_DELAY_MS,
    )
    proxy = create_app(config).state.proxy
    proxy.http_client = httpx.AsyncClient(transport=transport)
    return proxy


def _capped_retry_recorder(monkeypatch: pytest.MonkeyPatch, module_path: str) -> list[float]:
    """Wrap ``retry_after_ms`` in ``module_path`` to record every computed delay.

    Both ``server.py`` and ``handlers/streaming.py`` import ``retry_after_ms``
    by name (``from headroom.proxy.helpers import retry_after_ms``), so each
    module holds its own reference and can be patched independently without
    touching the shared helper.
    """
    from headroom.proxy import helpers

    real = helpers.retry_after_ms
    delays: list[float] = []

    def _wrapped(response: httpx.Response, max_ms: int) -> float | None:
        result = real(response, max_ms)
        if result is not None:
            delays.append(result)
        return result

    monkeypatch.setattr(f"{module_path}.retry_after_ms", _wrapped)
    return delays


async def _run_buffered_storm(
    n_clients: int, capacity: int, monkeypatch: pytest.MonkeyPatch, *, governor_enabled: bool = False
) -> dict[str, Any]:
    transport = _CapacityLimitedTransport(capacity=capacity)
    proxy = _proxy_with(transport, governor_enabled=governor_enabled)
    delays_ms = _capped_retry_recorder(monkeypatch, "headroom.proxy.server")

    start = time.perf_counter()
    responses = await asyncio.gather(
        *(
            proxy._retry_request(
                "POST", "https://up/v1/messages", {"x-client-id": str(i)}, {"messages": []}
            )
            for i in range(n_clients)
        )
    )
    wallclock = time.perf_counter() - start

    return _summarize(responses, transport, delays_ms, wallclock, n_clients, capacity)


async def _run_streaming_storm(
    n_clients: int, capacity: int, monkeypatch: pytest.MonkeyPatch, *, governor_enabled: bool = False
) -> dict[str, Any]:
    transport = _CapacityLimitedTransport(capacity=capacity, sse=True)
    proxy = _proxy_with(transport, governor_enabled=governor_enabled)
    delays_ms = _capped_retry_recorder(monkeypatch, "headroom.proxy.handlers.streaming")

    async def _one(i: int) -> httpx.Response | Any:
        return await proxy._stream_response(
            "https://up/v1/messages",
            {"x-client-id": str(i)},
            {"messages": []},
            "anthropic",
            "claude-3",
            f"r{i}",
            0,
            0,
            0,
            [],
            {},
            0.0,
        )

    start = time.perf_counter()
    responses = await asyncio.gather(*(_one(i) for i in range(n_clients)))
    wallclock = time.perf_counter() - start

    return _summarize(responses, transport, delays_ms, wallclock, n_clients, capacity)


def _summarize(
    responses: list[Any],
    transport: _CapacityLimitedTransport,
    delays_ms: list[float],
    wallclock: float,
    n_clients: int,
    capacity: int,
) -> dict[str, Any]:
    rate_limited_call_counts = [
        transport.calls_by_client[str(i)]
        for i, resp in enumerate(responses)
        if resp.status_code == _RATE_LIMIT_STATUS
    ]
    client_visible_errors = sum(1 for resp in responses if resp.status_code == _RATE_LIMIT_STATUS)
    upstream_429_count = transport.calls - capacity
    retries_at_cap = sum(1 for d in delays_ms if d == _RETRY_MAX_DELAY_MS)

    return {
        "n_clients": n_clients,
        "capacity": capacity,
        "total_upstream_calls": transport.calls,
        "upstream_429_count": upstream_429_count,
        "client_visible_rate_limit_error_count": client_visible_errors,
        "rate_limited_call_counts": rate_limited_call_counts,
        "retries_at_max_delay_cap": retries_at_cap,
        "wallclock_seconds": wallclock,
    }


# --------------------------------------------------------------------------- #
# Buffered path (_retry_request)                                              #
# --------------------------------------------------------------------------- #

_N_CLIENTS = 20
_CAPACITY = 2


def test_buffered_path_reproduces_429_storm(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = asyncio.run(_run_buffered_storm(_N_CLIENTS, _CAPACITY, monkeypatch))

    n_rate_limited = _N_CLIENTS - _CAPACITY
    assert len(metrics["rate_limited_call_counts"]) == n_rate_limited
    # 3 upstream calls per rate-limited request -- current, unfixed shape.
    assert all(c == _RETRY_MAX_ATTEMPTS for c in metrics["rate_limited_call_counts"])
    # Every failing attempt returns 429; every retry it triggers clamps to
    # the max-delay cap (2 retries per exhausted request: after attempt 0
    # and attempt 1, none after the final attempt 2).
    assert metrics["upstream_429_count"] == n_rate_limited * _RETRY_MAX_ATTEMPTS
    assert metrics["retries_at_max_delay_cap"] == n_rate_limited * (_RETRY_MAX_ATTEMPTS - 1)
    # Low rescue rate: great majority of rate-limited requests still end as
    # a client-visible error.
    assert metrics["client_visible_rate_limit_error_count"] == n_rate_limited
    assert metrics["wallclock_seconds"] < 2.0


def test_buffered_path_governor_arms_are_config_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """WU0 ships no governor; both arms must be byte-identical configs until
    WU2/WU3 land something to flip. Guards the harness's own contract, not
    proxy behavior."""
    off = asyncio.run(_run_buffered_storm(_N_CLIENTS, _CAPACITY, monkeypatch, governor_enabled=False))
    monkeypatch.undo()
    on = asyncio.run(_run_buffered_storm(_N_CLIENTS, _CAPACITY, monkeypatch, governor_enabled=True))
    off.pop("wallclock_seconds")
    on.pop("wallclock_seconds")
    assert off == on


def test_buffered_path_deterministic_across_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    results = []
    for _ in range(3):
        m = asyncio.run(_run_buffered_storm(_N_CLIENTS, _CAPACITY, monkeypatch))
        m.pop("wallclock_seconds")
        results.append(m)
        monkeypatch.undo()
    assert results[0] == results[1] == results[2]


# --------------------------------------------------------------------------- #
# Streaming path (_stream_response)                                           #
# --------------------------------------------------------------------------- #


def test_streaming_path_reproduces_429_storm(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = asyncio.run(_run_streaming_storm(_N_CLIENTS, _CAPACITY, monkeypatch))

    n_rate_limited = _N_CLIENTS - _CAPACITY
    assert len(metrics["rate_limited_call_counts"]) == n_rate_limited
    assert all(c == _RETRY_MAX_ATTEMPTS for c in metrics["rate_limited_call_counts"])
    assert metrics["upstream_429_count"] == n_rate_limited * _RETRY_MAX_ATTEMPTS
    assert metrics["retries_at_max_delay_cap"] == n_rate_limited * (_RETRY_MAX_ATTEMPTS - 1)
    assert metrics["client_visible_rate_limit_error_count"] == n_rate_limited
    assert metrics["wallclock_seconds"] < 2.0


def test_streaming_path_deterministic_across_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    results = []
    for _ in range(3):
        m = asyncio.run(_run_streaming_storm(_N_CLIENTS, _CAPACITY, monkeypatch))
        m.pop("wallclock_seconds")
        results.append(m)
        monkeypatch.undo()
    assert results[0] == results[1] == results[2]
