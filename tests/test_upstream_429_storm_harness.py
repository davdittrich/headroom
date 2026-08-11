"""WU0 (headroom-8z2.4): deterministic offline harness for the parallel-agent
429 storm (epic headroom-8z2).

Drives N concurrent simulated clients through the real proxy retry path
(``HeadroomProxy._retry_request`` and the streaming sibling in
``handlers/streaming.py``) against a scripted, non-replenishing-capacity mock
upstream, and reports the metrics every later work unit's before/after claim
depends on.

Updated by WU1 (headroom-8z2.1) to encode the FIXED behavior: ``retry_after_ms``
no longer clamps the upstream-demanded wait down to ``retry_max_delay_ms``.
Two ``retry_after_budget_ms`` arms are covered, both against the same mock
``Retry-After`` (5000ms at test scale):

* fail-fast arm (``_RETRY_AFTER_BUDGET_MS``, 10ms): the demand exceeds the
  budget, so a rate-limited request is returned verbatim on its FIRST
  upstream call instead of retrying into a wait the proxy already knows is
  insufficient.
* retry-exercised arm (``_RETRY_AFTER_BUDGET_MS_GENEROUS``, just above
  5000ms): the demand is within budget, so the retry/backoff loop still runs
  its full ``retry_max_attempts`` and the true (uncapped) 5000ms delay is
  what gets recorded — proving the clamp stays removed even on the path
  that still retries, not just on the path that now fails fast.

Pre-fix baseline (3 calls/429, clamped-to-10ms retries) is preserved in git
history and in the WU1 `bd comment` A/B evidence on headroom-8z2.1 — this
module encodes only the current, correct behavior.

Deterministic-time: no wall-clock sleeps of real duration. ``retry_base_delay_ms``
/ ``retry_max_delay_ms`` are configured in single-digit milliseconds, and the
retry-exercised arm's real 5000ms delay is never actually slept — both storm
runners stub the retry wait (``_wait_for_retry_delay_or_shutdown`` for the
buffered path, ``asyncio.sleep`` for the streaming path) to return
instantly, so a full N=20 run completes in well under a second regardless of
arm. No randomness is on the delay path: whichever branch a status/arm
combination takes, it either returns the parsed ``Retry-After`` value
directly or falls back to ``jitter_delay_ms`` (only reachable for an absent
or non-positive ``Retry-After``, which this mock never sends, so the harness
itself never exercises ``random.random()``).

Exception, added by WU2 (headroom-8z2.2): the governor A/B section at the end
of this module does sleep for real (sub-second), because the thing under test
IS a wait. Stubbing the wait in either arm would decide the comparison, so both
arms run unstubbed and differ only in the kill switch.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any

import httpx
import pytest

from headroom.proxy.server import ProxyConfig, create_app
from headroom.proxy.upstream_rate_gate import RateGateTransport, UpstreamRateGate

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
# Retry-exercised arm: a budget just above the mock's Retry-After, so the
# demand is within budget and the retry/backoff loop actually runs instead
# of failing fast. Real retry waits under this arm are stubbed (see
# ``_stub_retry_wait``) so the suite stays fast despite the realistic delay.
_RETRY_AFTER_BUDGET_MS_GENEROUS = int(_MOCK_RETRY_AFTER_S * 1000) + 1


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


def _proxy_with(
    transport: _CapacityLimitedTransport,
    *,
    governor_enabled: bool = False,
    retry_after_budget_ms: int = _RETRY_AFTER_BUDGET_MS,
):
    """Build a HeadroomProxy wired to ``transport``.

    ``governor_enabled`` is the A/B arm switch the epic guard requires
    (headroom-8z2 guard: "gate is keyed... one install point"; headroom-8z2.4
    guard: "expose the arm as an explicit parameter"). WU2 (headroom-8z2.2)
    wired it to the real kill switch ``ProxyConfig.upstream_rate_gate_enabled``:
    the enabled arm installs the same :class:`RateGateTransport` wrapper that
    ``HeadroomProxy.startup`` installs in production, around the mock upstream.
    Everything else in both arms is identical.
    """
    config = ProxyConfig(
        upstream_rate_gate_enabled=governor_enabled,
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
        retry_after_budget_ms=retry_after_budget_ms,
    )
    proxy = create_app(config).state.proxy
    outbound: httpx.AsyncBaseTransport = transport
    if governor_enabled:
        proxy.upstream_rate_gate = UpstreamRateGate(config, proxy._get_shutdown_event)
        outbound = RateGateTransport(proxy.upstream_rate_gate, transport)
    proxy.http_client = httpx.AsyncClient(transport=outbound)
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


def _stub_buffered_retry_wait(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Make the buffered path's retry wait return instantly, recording the
    ``seconds`` the caller actually applied.

    Under the retry-exercised arm the recorded delay is a realistic 5000ms
    (the mock's true ``Retry-After``), which the harness must never actually
    sleep for -- but the applied value itself IS the thing under test: it is
    what ``server.py`` computed as ``delay_ms`` after any clamp/budget logic,
    not the raw ``retry_after_ms`` return (see ``_capped_retry_recorder``,
    which only observes the pre-clamp value). Inert under the fail-fast arm,
    which returns before ever calling this.
    """
    applied_seconds: list[float] = []

    async def _instant_wait(self: Any, seconds: float) -> bool:
        applied_seconds.append(seconds)
        return False

    monkeypatch.setattr(
        "headroom.proxy.server.HeadroomProxy._wait_for_retry_delay_or_shutdown", _instant_wait
    )
    return applied_seconds


def _stub_streaming_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Make the streaming path's retry ``asyncio.sleep`` return instantly,
    recording the ``seconds`` the caller actually applied.

    Same rationale as ``_stub_buffered_retry_wait``, for the streaming
    overload branch's bare ``await asyncio.sleep(...)`` call.
    """
    applied_seconds: list[float] = []

    async def _instant_sleep(seconds: float) -> None:
        applied_seconds.append(seconds)

    monkeypatch.setattr("headroom.proxy.handlers.streaming.asyncio.sleep", _instant_sleep)
    return applied_seconds


async def _run_buffered_storm(
    n_clients: int,
    capacity: int,
    monkeypatch: pytest.MonkeyPatch,
    *,
    governor_enabled: bool = False,
    retry_after_budget_ms: int = _RETRY_AFTER_BUDGET_MS,
) -> dict[str, Any]:
    transport = _CapacityLimitedTransport(capacity=capacity)
    proxy = _proxy_with(
        transport, governor_enabled=governor_enabled, retry_after_budget_ms=retry_after_budget_ms
    )
    delays_ms = _capped_retry_recorder(monkeypatch, "headroom.proxy.server")
    applied_wait_seconds = _stub_buffered_retry_wait(monkeypatch)

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

    return _summarize(
        responses, transport, delays_ms, applied_wait_seconds, wallclock, n_clients, capacity
    )


async def _run_streaming_storm(
    n_clients: int,
    capacity: int,
    monkeypatch: pytest.MonkeyPatch,
    *,
    governor_enabled: bool = False,
    retry_after_budget_ms: int = _RETRY_AFTER_BUDGET_MS,
) -> dict[str, Any]:
    transport = _CapacityLimitedTransport(capacity=capacity, sse=True)
    proxy = _proxy_with(
        transport, governor_enabled=governor_enabled, retry_after_budget_ms=retry_after_budget_ms
    )
    delays_ms = _capped_retry_recorder(monkeypatch, "headroom.proxy.handlers.streaming")
    applied_wait_seconds = _stub_streaming_retry_sleep(monkeypatch)

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

    return _summarize(
        responses, transport, delays_ms, applied_wait_seconds, wallclock, n_clients, capacity
    )


def _summarize(
    responses: list[Any],
    transport: _CapacityLimitedTransport,
    delays_ms: list[float],
    applied_wait_seconds: list[float],
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
        # The delay the caller (server.py/streaming.py) actually applied to
        # its wait, in seconds -- as opposed to delays_ms/retries_at_max_delay_cap,
        # which only observe retry_after_ms's raw (pre-clamp) return. This is
        # the value a caller-side clamp regression would corrupt.
        "applied_wait_seconds": sorted(applied_wait_seconds),
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
    # No wait is ever applied -- the request returns before the retry loop
    # computes or sleeps on a delay at all.
    assert metrics["applied_wait_seconds"] == []
    # Client-visible outcome is unchanged: every rate-limited request still
    # surfaces its 429 to the client (the fix removes wasted upstream calls,
    # not the 429 itself -- rescuing beyond-budget waits is WU2/WU3's job).
    assert metrics["client_visible_rate_limit_error_count"] == n_rate_limited


def test_buffered_path_retries_when_budget_exceeds_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry-exercised arm: budget (_RETRY_AFTER_BUDGET_MS_GENEROUS) exceeds the
    mock's Retry-After (5000ms), so the retry/backoff loop still runs its full
    retry_max_attempts instead of failing fast. This does not weaken the
    fail-fast arm's assertions above -- it is a second, independent arm."""
    metrics = asyncio.run(
        _run_buffered_storm(
            _N_CLIENTS,
            _CAPACITY,
            monkeypatch,
            retry_after_budget_ms=_RETRY_AFTER_BUDGET_MS_GENEROUS,
        )
    )

    n_rate_limited = _N_CLIENTS - _CAPACITY
    assert len(metrics["rate_limited_call_counts"]) == n_rate_limited
    # Within budget, so every rate-limited client retries to exhaustion.
    assert all(c == _RETRY_MAX_ATTEMPTS for c in metrics["rate_limited_call_counts"])
    assert metrics["upstream_429_count"] == n_rate_limited * _RETRY_MAX_ATTEMPTS
    # The recorded delay is the true, uncapped Retry-After (5000ms), not the
    # old clamp value (_RETRY_MAX_DELAY_MS=10ms) -- proves the clamp stays
    # removed even on the path that is actually exercised, not just the one
    # that now fails fast.
    assert metrics["retries_at_max_delay_cap"] == 0
    # This is the assertion that actually catches a caller-side clamp
    # regression: the delay APPLIED to the wait (not retry_after_ms's raw
    # pre-clamp return) is the true 5.0s on both of the two retries every
    # rate-limited client makes (attempt 0 and 1; attempt 2 exhausts without
    # retrying). If a future change reintroduced `min(retry_after, cap)` in
    # server.py's delay_ms computation, this would go red.
    assert metrics["applied_wait_seconds"] == [5.0] * (n_rate_limited * (_RETRY_MAX_ATTEMPTS - 1))
    assert metrics["client_visible_rate_limit_error_count"] == n_rate_limited


def test_buffered_path_governor_arms_are_config_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """WU0 ships no governor; both arms must be byte-identical configs until
    WU2/WU3 land something to flip. Guards the harness's own contract, not
    proxy behavior."""
    off = asyncio.run(
        _run_buffered_storm(_N_CLIENTS, _CAPACITY, monkeypatch, governor_enabled=False)
    )
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
    assert metrics["applied_wait_seconds"] == []
    assert metrics["client_visible_rate_limit_error_count"] == n_rate_limited


def test_streaming_path_retries_when_budget_exceeds_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming-path counterpart of test_buffered_path_retries_when_budget_exceeds_retry_after."""
    metrics = asyncio.run(
        _run_streaming_storm(
            _N_CLIENTS,
            _CAPACITY,
            monkeypatch,
            retry_after_budget_ms=_RETRY_AFTER_BUDGET_MS_GENEROUS,
        )
    )

    n_rate_limited = _N_CLIENTS - _CAPACITY
    assert len(metrics["rate_limited_call_counts"]) == n_rate_limited
    assert all(c == _RETRY_MAX_ATTEMPTS for c in metrics["rate_limited_call_counts"])
    assert metrics["upstream_429_count"] == n_rate_limited * _RETRY_MAX_ATTEMPTS
    assert metrics["retries_at_max_delay_cap"] == 0
    assert metrics["applied_wait_seconds"] == [5.0] * (n_rate_limited * (_RETRY_MAX_ATTEMPTS - 1))
    assert metrics["client_visible_rate_limit_error_count"] == n_rate_limited


# ---------------------------------------------------------------------------
# WU2 (headroom-8z2.2) A/B: the per-upstream-host rate gate on vs off.
#
# The non-replenishing mock above cannot show a governor win -- capacity never
# comes back, so no amount of waiting rescues a request. This arm uses a
# windowed mock (capacity refills every window, like a real per-minute quota)
# and staggered client arrivals, which is the measured failure shape: agents
# keep arriving while the host is already limited and each burns its own 429
# rediscovering that. The two arms differ ONLY in ``governor_enabled``.
# ---------------------------------------------------------------------------
_AB_WINDOW_S = 0.25
_AB_CAPACITY = 8
_AB_CLIENTS = 24
_AB_ARRIVAL_STAGGER_S = 0.01


class _WindowedCapacityTransport(httpx.AsyncBaseTransport):
    """Mock upstream granting ``capacity`` calls per ``window_s`` window."""

    def __init__(self, *, capacity: int, window_s: float) -> None:
        self.capacity = capacity
        self.window_s = window_s
        self.calls = 0
        self.rate_limited = 0
        self._window_start: float | None = None
        self._granted = 0
        self._lock = asyncio.Lock()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        async with self._lock:
            now = time.monotonic()
            if self._window_start is None or now - self._window_start >= self.window_s:
                self._window_start = now
                self._granted = 0
            self.calls += 1
            allow = self._granted < self.capacity
            if allow:
                self._granted += 1
            else:
                self.rate_limited += 1
        async for _ in request.stream:  # drain the request body
            pass
        if not allow:
            return httpx.Response(
                _RATE_LIMIT_STATUS,
                headers={"retry-after": str(self.window_s)},
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
        return httpx.Response(200, json={"id": "msg_1", "type": "message", "role": "assistant"})


async def _run_governor_ab_storm(*, governor_enabled: bool) -> dict[str, Any]:
    transport = _WindowedCapacityTransport(capacity=_AB_CAPACITY, window_s=_AB_WINDOW_S)
    proxy = _proxy_with(
        transport,
        governor_enabled=governor_enabled,
        # Generous budget so the retry loop actually runs in both arms; real
        # (unstubbed) waits, so neither arm gets a stub-induced advantage.
        retry_after_budget_ms=int(_AB_WINDOW_S * 1000) + 1,
    )

    async def _one(i: int) -> httpx.Response:
        await asyncio.sleep(i * _AB_ARRIVAL_STAGGER_S)
        return await proxy._retry_request(
            "POST", "https://up/v1/messages", {"x-client-id": str(i)}, {"messages": []}
        )

    start = time.perf_counter()
    responses = await asyncio.gather(*(_one(i) for i in range(_AB_CLIENTS)))
    return {
        "upstream_429_count": transport.rate_limited,
        "upstream_call_count": transport.calls,
        "client_visible_rate_limit_error_count": sum(
            1 for r in responses if r.status_code == _RATE_LIMIT_STATUS
        ),
        "wallclock_seconds": time.perf_counter() - start,
    }


def test_governor_arm_reduces_upstream_429s() -> None:
    """A/B: the only difference between the arms is the gate kill switch."""
    off = asyncio.run(_run_governor_ab_storm(governor_enabled=False))
    on = asyncio.run(_run_governor_ab_storm(governor_enabled=True))

    assert on["upstream_429_count"] < off["upstream_429_count"], (
        f"gate must cut wasted upstream 429s: off={off}, on={on}"
    )
    assert on["upstream_call_count"] < off["upstream_call_count"]
    assert (
        on["client_visible_rate_limit_error_count"]
        <= (off["client_visible_rate_limit_error_count"])
    )


def test_streaming_path_deterministic_across_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    results = []
    for _ in range(3):
        m = asyncio.run(_run_streaming_storm(_N_CLIENTS, _CAPACITY, monkeypatch))
        m.pop("wallclock_seconds")
        results.append(m)
        monkeypatch.undo()
    assert results[0] == results[1] == results[2]
