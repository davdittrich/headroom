"""WU0 (headroom-8z2.4): deterministic offline harness for the parallel-agent
429 storm (epic headroom-8z2).

Drives N concurrent simulated clients through the real proxy retry path
(``HeadroomProxy._retry_request`` and the streaming sibling in
``handlers/streaming.py``) against a scripted, non-replenishing-capacity mock
upstream, and reports the metrics every later work unit's before/after claim
depends on.

Updated by WU1 (headroom-8z2.1) to encode the FIXED behavior: ``retry_after_ms``
no longer clamps the upstream-demanded wait down to ``retry_max_delay_ms``.
The mock's ``Retry-After`` (5000ms at test scale) now exceeds
``retry_after_budget_ms`` (10ms at test scale, mirroring the production
default's derivation from the pre-fix ``retry_max_delay_ms`` value), so a
rate-limited request is returned verbatim on its FIRST upstream call instead
of retrying twice more into a wait the proxy already knows is insufficient.
Pre-fix baseline (3 calls/429, clamped retries, ~5% rescue) is preserved in
git history and in the WU1 `bd comment` A/B evidence on headroom-8z2.1 — this
module encodes only the current, correct behavior.

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
# Test-scale retry_after_budget_ms, mirroring production's default derivation
# (retry_after_budget_ms defaults to the same numeric value retry_max_delay_ms
# used to clamp to pre-fix — see headroom/proxy/models.py, headroom-8z2.1).
_RETRY_AFTER_BUDGET_MS = _RETRY_MAX_DELAY_MS
# Mock upstream always asks for more than the budget, so a rate-limited
# request is now returned verbatim on its first upstream call instead of
# retrying -- this is the harness analog of the live-log figure ("276 of 292
# retries logged retrying in 30000ms", pre-fix; post-fix those retries never
# happen at all).
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
        retry_after_budget_ms=_RETRY_AFTER_BUDGET_MS,
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

    def _wrapped(response: httpx.Response) -> float | None:
        result = real(response)
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


def test_buffered_path_fix_returns_429_on_first_call(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = asyncio.run(_run_buffered_storm(_N_CLIENTS, _CAPACITY, monkeypatch))

    n_rate_limited = _N_CLIENTS - _CAPACITY
    assert len(metrics["rate_limited_call_counts"]) == n_rate_limited
    # 1 upstream call per rate-limited request post-fix -- the mock's
    # Retry-After (5000ms) exceeds retry_after_budget_ms (10ms), so
    # _retry_request returns the 429 verbatim instead of retrying twice
    # more into a wait it already knows is insufficient. Pre-fix this was 3.
    assert all(c == 1 for c in metrics["rate_limited_call_counts"])
    assert metrics["upstream_429_count"] == n_rate_limited * 1
    # No retries happen at all -- nothing is clamped anymore. Pre-fix this
    # was n_rate_limited * (_RETRY_MAX_ATTEMPTS - 1).
    assert metrics["retries_at_max_delay_cap"] == 0
    # Client-visible outcome is unchanged: every rate-limited request still
    # surfaces its 429 to the client (the fix removes wasted upstream calls,
    # not the 429 itself -- rescuing beyond-budget waits is WU2/WU3's job).
    assert metrics["client_visible_rate_limit_error_count"] == n_rate_limited


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


def test_streaming_path_fix_returns_429_on_first_call(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = asyncio.run(_run_streaming_storm(_N_CLIENTS, _CAPACITY, monkeypatch))

    n_rate_limited = _N_CLIENTS - _CAPACITY
    assert len(metrics["rate_limited_call_counts"]) == n_rate_limited
    # Streaming overload branch obeys the same budget rule (headroom-8z2.1).
    assert all(c == 1 for c in metrics["rate_limited_call_counts"])
    assert metrics["upstream_429_count"] == n_rate_limited * 1
    assert metrics["retries_at_max_delay_cap"] == 0
    assert metrics["client_visible_rate_limit_error_count"] == n_rate_limited


def test_streaming_path_deterministic_across_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    results = []
    for _ in range(3):
        m = asyncio.run(_run_streaming_storm(_N_CLIENTS, _CAPACITY, monkeypatch))
        m.pop("wallclock_seconds")
        results.append(m)
        monkeypatch.undo()
    assert results[0] == results[1] == results[2]
