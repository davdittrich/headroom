"""WU2 (headroom-8z2.2) A/B harness: per-upstream-host rate gate on vs off
(epic headroom-8z2).

N=1 coverage of ``retry_after_ms`` / the retry-budget behavior itself lives
in ``tests/test_proxy_retry_429.py``; per WU1 the fix is a per-request branch
with no shared state, so N=20 concurrency buys no discrimination that N=1
does not already give. This module now covers only what the gate's shared
state actually needs concurrency to demonstrate: a windowed mock upstream
plus staggered client arrivals, arms differing solely in the kill switch.

Deterministic-time note: the A/B section below does sleep for real
(sub-second), because the thing under test IS a wait — stubbing it in either
arm would decide the comparison, so both arms run unstubbed.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

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


def _proxy_with(
    transport: httpx.AsyncBaseTransport,
    *,
    governor_enabled: bool = False,
    retry_after_budget_ms: int = _RETRY_AFTER_BUDGET_MS,
):
    """Build a HeadroomProxy wired to ``transport``.

    ``governor_enabled`` is the A/B arm switch the epic guard requires
    (headroom-8z2 guard: "gate is keyed... one install point"; headroom-8z2.4
    guard: "expose the arm as an explicit parameter"). It does NOT go through
    ``ProxyConfig.upstream_rate_gate_enabled`` at runtime: ``create_app``
    never calls ``HeadroomProxy.startup`` (only the ASGI lifespan does), and
    that flag is read nowhere else, so setting it on the config is inert here
    -- it is set anyway so the config object matches what the enabled arm
    describes. The arm variable that actually matters is this function
    hand-building the same :class:`RateGateTransport` wrapper
    ``HeadroomProxy.startup`` installs in production, around the mock
    upstream. Everything else in both arms is identical.
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
        # No gate-only bound: the gate's willingness to wait IS
        # ``retry_after_budget_ms``, so the arm cannot be tuned independently of
        # the retry loop it shares a policy with.
        proxy.upstream_rate_gate = UpstreamRateGate(config, proxy._get_shutdown_event)
        outbound = RateGateTransport(proxy.upstream_rate_gate, transport)
    proxy.http_client = httpx.AsyncClient(transport=outbound)
    return proxy


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


async def _run_governor_ab_storm(
    *,
    governor_enabled: bool,
    # Generous budget so the retry loop actually runs in both arms. The
    # default is `2*window + 1ms`, so the mock's Retry-After (= one window)
    # sits exactly AT the gate's threshold (2x remaining <= budget) rather
    # than comfortably inside it -- only elapsed time between observing the
    # 429 and evaluating the threshold keeps it parking; real (unstubbed)
    # waits, so neither arm gets a stub-induced advantage. Lower it below
    # `2*window` and the gate fails fast instead: that is the production band
    # (Retry-After >= 30s against a 30s budget) and the AIMD limiter's own
    # regime -- see ``test_limiter_arm_reduces_upstream_429s_above_the_budget``.
    retry_after_budget_ms: int = int(_AB_WINDOW_S * 2000) + 1,
) -> dict[str, Any]:
    transport = _WindowedCapacityTransport(capacity=_AB_CAPACITY, window_s=_AB_WINDOW_S)
    proxy = _proxy_with(
        transport,
        governor_enabled=governor_enabled,
        retry_after_budget_ms=retry_after_budget_ms,
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


# The above-bound A/B arm that used to live here is gone with the behavior it
# measured: past ``retry_after_budget_ms`` the gate now fails fast instead of
# parking (module docstring of ``upstream_rate_gate``), so there is nothing to
# throttle in that band. Its measured win was 13 -> 7 wasted upstream 429s and
# zero for a tight arrival burst, against a request paying the full budget
# before receiving the same 429 it would have gotten in ~200ms.
# ``tests/test_upstream_rate_gate.py::test_deadline_beyond_the_budget_dispatches_immediately``
# pins the replacement contract deterministically.


# ---------------------------------------------------------------------------
# WU3 (headroom-8z2.3) A/B: the AIMD concurrency limiter, above the budget.
#
# Measured negative result that motivates this second harness: against the
# WINDOWED mock above the limiter changes nothing (24/48 off, 2/26 on, both
# before and after WU3), because that mock is a per-window RATE limiter whose
# offered load exceeds capacity -- nothing but waiting past the window helps,
# and above the budget waiting is exactly what the gate refuses to do. Below
# the budget the gate's park already collects the whole win.
#
# The shape the limiter actually governs is a BURST/concurrency cap: the
# upstream rejects the (cap+1)-th SIMULTANEOUS request. N parallel agents trip
# that on arrival, no amount of Retry-After honouring prevents it, and above
# the budget the gate declines to park at all -- which is where the user's
# real traffic sits (276 of 292 retries demanded >= 30s against a 30s budget).
# Arms differ ONLY in ``governor_enabled``.
# ---------------------------------------------------------------------------
_AB_CONCURRENCY_CAP = 4
_AB_UPSTREAM_LATENCY_S = 0.05
_AB_WAVES = 3
_AB_WAVE_CLIENTS = 12
_AB_WAVE_GAP_S = 0.3
# Below 2 * the mock's Retry-After, so the gate fails fast and never parks:
# the production band, and the only band where the limiter is the sole defence.
_AB_ABOVE_BUDGET_MS = 100


class _ConcurrentCapacityTransport(httpx.AsyncBaseTransport):
    """Mock upstream that 429s any request beyond ``cap`` simultaneously in flight.

    ``quota_headers`` reproduces API-key auth, which reports the remaining
    request quota on every response; subscription/OAuth auth reports nothing,
    which is the ``False`` case.
    """

    def __init__(self, *, cap: int, latency_s: float, quota_headers: bool) -> None:
        self.cap = cap
        self.latency_s = latency_s
        self.quota_headers = quota_headers
        self.calls = 0
        self.rate_limited = 0
        self.inflight = 0
        self.max_inflight = 0

    def _headers(self) -> dict[str, str]:
        if not self.quota_headers:
            return {}
        reset = datetime.now(timezone.utc) + timedelta(seconds=_AB_WINDOW_S)
        return {
            "anthropic-ratelimit-requests-limit": str(self.cap),
            "anthropic-ratelimit-requests-remaining": str(max(0, self.cap - self.inflight)),
            "anthropic-ratelimit-requests-reset": reset.isoformat().replace("+00:00", "Z"),
        }

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        async for _ in request.stream:  # drain the request body
            pass
        self.calls += 1
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            if self.inflight > self.cap:
                self.rate_limited += 1
                return httpx.Response(
                    _RATE_LIMIT_STATUS,
                    headers={"retry-after": str(_AB_WINDOW_S), **self._headers()},
                    json={"type": "error", "error": {"type": "rate_limit_error"}},
                )
            await asyncio.sleep(self.latency_s)
            return httpx.Response(200, headers=self._headers(), json={"id": "msg_1"})
        finally:
            self.inflight -= 1


async def _run_limiter_ab_storm(
    *, governor_enabled: bool, quota_headers: bool = False
) -> dict[str, Any]:
    transport = _ConcurrentCapacityTransport(
        cap=_AB_CONCURRENCY_CAP, latency_s=_AB_UPSTREAM_LATENCY_S, quota_headers=quota_headers
    )
    proxy = _proxy_with(
        transport,
        governor_enabled=governor_enabled,
        retry_after_budget_ms=_AB_ABOVE_BUDGET_MS,
    )

    async def _one(index: int) -> httpx.Response:
        await asyncio.sleep((index // _AB_WAVE_CLIENTS) * _AB_WAVE_GAP_S)
        return await proxy._retry_request(
            "POST", "https://up/v1/messages", {"x-client-id": str(index)}, {"messages": []}
        )

    start = time.perf_counter()
    responses = await asyncio.gather(*(_one(i) for i in range(_AB_WAVES * _AB_WAVE_CLIENTS)))
    return {
        "upstream_429_count": transport.rate_limited,
        "upstream_call_count": transport.calls,
        "max_upstream_concurrency": transport.max_inflight,
        "client_visible_rate_limit_error_count": sum(
            1 for r in responses if r.status_code == _RATE_LIMIT_STATUS
        ),
        "wallclock_seconds": time.perf_counter() - start,
    }


def test_limiter_arm_reduces_upstream_429s_above_the_budget() -> None:
    """Pure AIMD, no quota headers: wave 1 pays, waves 2-3 are bounded."""
    off = asyncio.run(_run_limiter_ab_storm(governor_enabled=False))
    on = asyncio.run(_run_limiter_ab_storm(governor_enabled=True))

    assert on["upstream_429_count"] < off["upstream_429_count"], (
        f"the limiter must cut 429s the gate cannot park on: off={off}, on={on}"
    )
    assert (
        on["client_visible_rate_limit_error_count"]
        <= (off["client_visible_rate_limit_error_count"])
    ), f"off={off}, on={on}"


# Header seeding is deliberately NOT an A/B arm here: it cannot be measured in
# this harness. The real ``anthropic-ratelimit-requests-remaining`` is a
# per-WINDOW request quota, but this mock's limit is a concurrency cap, so the
# only "remaining" it can report is an instantaneous in-flight reading -- a
# different quantity, and one that sits at 0-1 for the whole burst, i.e.
# precisely the tight limit whose synchronized fail-open costs more than it
# saves (measured 9 -> 14 wasted 429s at this budget, while above the mock's
# Retry-After the same seeding wins, 3 -> 1). Rather than pick the budget that
# flatters it, the seeding contract is pinned deterministically by
# ``tests/test_upstream_rate_gate.py::test_ratelimit_headers_seed_the_limit_without_probing``
# and its header-absent twin.
