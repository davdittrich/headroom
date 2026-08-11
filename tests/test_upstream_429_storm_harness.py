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


async def _run_governor_ab_storm(*, governor_enabled: bool) -> dict[str, Any]:
    transport = _WindowedCapacityTransport(capacity=_AB_CAPACITY, window_s=_AB_WINDOW_S)
    proxy = _proxy_with(
        transport,
        governor_enabled=governor_enabled,
        # Generous budget so the retry loop actually runs in both arms. The
        # budget is set to `2*window + 1ms`, so the mock's Retry-After (= one
        # window) sits exactly AT the gate's threshold (2x remaining <=
        # budget) rather than comfortably inside it -- only elapsed time
        # between observing the 429 and evaluating the threshold keeps it
        # parking; real (unstubbed) waits, so neither arm gets a
        # stub-induced advantage.
        retry_after_budget_ms=int(_AB_WINDOW_S * 2000) + 1,
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
