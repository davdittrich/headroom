"""WU2 (headroom-8z2.2): per-upstream-host rate gate as an httpx transport wrapper.

Covers the guards from the ticket, one test per lettered requirement (3a-3p):
gating, per-host isolation, dispersed release, idle no-op, the fail-fast bound,
529 exclusion, shutdown interruption, the kill switch, streaming transparency,
non-``_retry_request`` call sites, headerless/unparseable ``Retry-After``,
post-wake re-check, cross-client shared state, the bound on quota-exhaustion
waits, and preservation of everything the client was built with -- including
the environment proxy map, which ``transport=`` silently discards.

All waits are test-scale (tens to hundreds of milliseconds) so the suite stays
fast while exercising the real ``asyncio`` wait path -- the gate's wait is not
stubbed anywhere in this module.
"""

from __future__ import annotations

import asyncio
import random
import ssl
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest

from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import HeadroomProxy
from headroom.proxy.upstream_rate_gate import (
    RateGateTransport,
    UpstreamRateGate,
    install_gate,
    release_delay_seconds,
)

pytestmark = pytest.mark.asyncio


def _config(**overrides: Any) -> ProxyConfig:
    base: dict[str, Any] = {
        "optimize": False,
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "cost_tracking_enabled": False,
        "log_requests": False,
        "ccr_inject_tool": False,
        "ccr_handle_responses": False,
        "ccr_context_tracking": False,
        "image_optimize": False,
        "subscription_tracking_enabled": False,
        "retry_base_delay_ms": 40,
        "retry_max_delay_ms": 80,
    }
    base.update(overrides)
    return ProxyConfig(**base)


class _Recorder(httpx.AsyncBaseTransport):
    """Mock upstream that records dispatch times and replays scripted responses.

    ``script`` maps host -> list of ``(status, headers)``; the last entry is
    reused once exhausted. Returns without ever awaiting, so a dispatch and the
    gate's observation of its response happen in the same event-loop tick.
    """

    def __init__(self, script: dict[str, list[tuple[int, dict[str, str]]]] | None = None) -> None:
        self.script = script or {}
        self.dispatches: list[tuple[str, float]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        self.dispatches.append((host, time.monotonic()))
        entries = self.script.get(host) or [(200, {})]
        index = min(sum(1 for h, _ in self.dispatches if h == host) - 1, len(entries) - 1)
        status, headers = entries[index]
        return httpx.Response(status, headers=headers, request=request, json={"ok": True})

    def times(self, host: str) -> list[float]:
        return [t for h, t in self.dispatches if h == host]


def _gate(config: ProxyConfig | None = None, **kwargs: Any) -> UpstreamRateGate:
    return UpstreamRateGate(config or _config(), asyncio.Event, **kwargs)


def _wired(
    recorder: _Recorder, gate: UpstreamRateGate | None = None
) -> tuple[UpstreamRateGate, RateGateTransport]:
    gate = gate or _gate()
    return gate, RateGateTransport(gate, recorder)


def _request(host: str) -> httpx.Request:
    return httpx.Request("POST", f"https://{host}/v1/messages")


# --------------------------------------------------------------------------- 3a
async def test_gate_holds_second_request_to_same_host() -> None:
    recorder = _Recorder({"a.upstream": [(429, {"retry-after": "0.3"}), (200, {})]})
    _, transport = _wired(recorder)

    first = await transport.handle_async_request(_request("a.upstream"))
    assert first.status_code == 429
    opened_at = time.monotonic()

    await transport.handle_async_request(_request("a.upstream"))
    dispatched_at = recorder.times("a.upstream")[1]
    assert dispatched_at - opened_at >= 0.3, "second dispatch must wait for the gate"


# --------------------------------------------------------------------------- 3b
async def test_gate_on_one_host_does_not_delay_another_host() -> None:
    recorder = _Recorder({"a.upstream": [(429, {"retry-after": "5"})]})
    gate, transport = _wired(recorder)

    await transport.handle_async_request(_request("a.upstream"))
    assert gate.deadline("a.upstream") is not None, (
        "host A must be gated for this to prove anything"
    )
    started = time.monotonic()
    await transport.handle_async_request(_request("b.upstream"))
    assert time.monotonic() - started < 0.1, "a 429 on host A must not gate host B"


# --------------------------------------------------------------------------- 3c
async def test_release_delay_disperses_waiters_in_every_parking_regime() -> None:
    """Dispersal is asserted on the computed delay, not on observed wakeups.

    The previous form of this test compared order statistics of 8 unsynchronized
    ``asyncio`` wakeups and failed at ``spread=0.0491`` against a ``>= 0.05``
    threshold on a loaded combined run while passing in isolation.
    :func:`release_delay_seconds` is pure in ``(remaining, roll)``, so a
    seeded RNG makes the same property deterministic.

    Both parking regimes are covered. Only "remaining far below the budget" was
    exercised before, which is why the clamped form of this computation --
    ``min(remaining * (1 + random()), park_until - now)``, where every waiter
    whose demanded wait reached the ceiling got the identical clamped timer --
    shipped with dead jitter unnoticed.
    """
    budget = 30.0
    rng = random.Random(20260810)
    # Far below the budget, and at the largest remaining wait the gate parks on
    # at all (2 * remaining == budget) -- the regime the clamp used to kill.
    for remaining in (0.2, 7.5, 15.0):
        delays = [release_delay_seconds(remaining, rng.random()) for _ in range(8)]
        assert min(delays) >= remaining, "no waiter may dispatch before the deadline"
        assert max(delays) <= budget, "no waiter may be held past the budget"
        spread = max(delays) - min(delays)
        assert spread > 0.3 * remaining, (
            f"waiters share one timer at remaining={remaining} (spread={spread:.4f}s)"
        )


async def test_waiters_do_not_all_dispatch_in_one_tick() -> None:
    """End-to-end companion to the pure-function assertion above.

    Deliberately only an upper bound (everyone dispatches inside the jitter
    window) plus an exact count -- lower-bounding an observed spread is what
    made the old version of this flaky.
    """
    recorder = _Recorder({"a.upstream": [(429, {"retry-after": "0.2"}), (200, {})]})
    _, transport = _wired(recorder)
    await transport.handle_async_request(_request("a.upstream"))
    opened_at = time.monotonic()

    await asyncio.gather(
        *(transport.handle_async_request(_request("a.upstream")) for _ in range(8))
    )
    released = recorder.times("a.upstream")[1:]
    assert len(released) == 8
    assert min(released) - opened_at >= 0.2, "no waiter may dispatch before the deadline"


# --------------------------------------------------------------------------- 3d
async def test_idle_path_performs_no_await(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    _, transport = _wired(recorder)

    async def _boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("idle path must not await the gate")

    monkeypatch.setattr("headroom.proxy.upstream_rate_gate.asyncio.wait_for", _boom)
    started = time.monotonic()
    response = await transport.handle_async_request(_request("a.upstream"))
    assert response.status_code == 200
    assert time.monotonic() - started < 0.01


# --------------------------------------------------------------------------- 3e
async def test_deadline_beyond_the_budget_dispatches_immediately() -> None:
    """Above the willingness-to-wait budget the gate does not park at all.

    Same rule ``helpers.overload_retry_delay_ms`` applies to the retry loop: a
    demanded wait past ``retry_after_budget_ms`` is not worth holding the
    request for, so the 429 goes back to the client and its own backoff handles
    it. Parking anyway burned the full budget and then returned the same 429.
    """
    assert _gate()._max_wait_seconds == _config().retry_after_budget_ms / 1000.0, (
        "the gate budget must BE retry_after_budget_ms, not a second constant that can drift"
    )
    recorder = _Recorder()
    gate, transport = _wired(recorder, _gate(max_wait_seconds=0.2))
    gate._until["a.upstream"] = time.monotonic() + 10.0
    assert gate.deadline("a.upstream") is not None, (
        "the gate must be live for this to prove anything"
    )

    started = time.monotonic()
    await transport.handle_async_request(_request("a.upstream"))
    assert time.monotonic() - started < 0.15, "an over-budget deadline must not park the request"
    assert len(recorder.dispatches) == 1
    assert gate.deadline("a.upstream") is not None, "declining to park must not clear the deadline"

    # The budget bounds the WHOLE hold, jitter included, so the threshold sits
    # at half of it: a wait whose 100-200% release band still fits does park.
    gate._until["a.upstream"] = time.monotonic() + 0.08  # 2 * 0.08 <= 0.2
    started = time.monotonic()
    await transport.handle_async_request(_request("a.upstream"))
    assert time.monotonic() - started >= 0.08, "a wait that fits the budget must still park"


# --------------------------------------------------------------------------- 3f
async def test_529_does_not_open_the_gate() -> None:
    recorder = _Recorder({"a.upstream": [(529, {"retry-after": "5"}), (429, {"retry-after": "5"})]})
    gate, transport = _wired(recorder)

    await transport.handle_async_request(_request("a.upstream"))
    assert gate.deadline("a.upstream") is None
    started = time.monotonic()
    await transport.handle_async_request(_request("a.upstream"))
    assert time.monotonic() - started < 0.1
    # ...and the same header on a 429 does open it, so the 529 exclusion is a
    # real discrimination and not a dead code path.
    assert gate.deadline("a.upstream") is not None


# --------------------------------------------------------------------------- 3g
async def test_shutdown_abandons_the_gate_wait() -> None:
    """Shutdown must abandon the wait AND the request.

    ``HeadroomProxy.shutdown`` sets this same event and then ``aclose()``s the
    client, so dispatching after the wake races the close and can surface
    ``RuntimeError: client has been closed``. Return the same 503 shape
    ``_shutdown_retry_response`` uses instead.
    """
    recorder = _Recorder()
    event = asyncio.Event()
    gate = UpstreamRateGate(_config(), lambda: event)
    transport = RateGateTransport(gate, recorder)
    gate._until["a.upstream"] = time.monotonic() + 10.0  # inside the budget, so it parks

    started = time.monotonic()
    task = asyncio.create_task(transport.handle_async_request(_request("a.upstream")))
    await asyncio.sleep(0.05)
    event.set()
    response = await asyncio.wait_for(task, timeout=2.0)
    assert time.monotonic() - started < 1.0, "shutdown must abandon the wait promptly"
    assert response.status_code == 503
    assert recorder.dispatches == [], "shutdown must not dispatch into a closing client"


# --------------------------------------------------------------------------- 3h
async def test_kill_switch_prevents_installation() -> None:
    config = _config(upstream_rate_gate_enabled=False, http2=False)
    proxy = HeadroomProxy(config)
    await proxy.startup()
    try:
        assert proxy.upstream_rate_gate is None
        assert not isinstance(proxy.http_client._transport, RateGateTransport)
        assert isinstance(proxy.http_client._transport, httpx.AsyncHTTPTransport)
    finally:
        await proxy.shutdown()


async def test_install_gate_is_a_no_op_without_a_gate() -> None:
    client = httpx.AsyncClient()
    try:
        original = client._transport
        assert install_gate(client, None) is client
        assert client._transport is original
    finally:
        await client.aclose()


# --------------------------------------------------------------------------- 3i
async def test_streaming_response_is_passed_through_unaltered() -> None:
    chunks = [b"event: a\n\n", b"event: b\n\n"]
    consumed: list[bytes] = []

    class _Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for chunk in chunks:
                consumed.append(chunk)
                yield chunk

    stream = _Stream()
    inner_response = httpx.Response(
        200, headers={"content-type": "text/event-stream", "x-mark": "1"}, stream=stream
    )

    class _Streaming(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return inner_response

    transport = RateGateTransport(_gate(), _Streaming())
    response = await transport.handle_async_request(_request("a.upstream"))

    assert response is inner_response
    assert response.status_code == 200
    assert response.headers["x-mark"] == "1"
    assert consumed == [], "the wrapper must not buffer the body"
    assert [c async for c in response.aiter_raw()] == chunks


async def test_transport_does_not_swallow_exceptions() -> None:
    class _Boom(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("nope")

    transport = RateGateTransport(_gate(), _Boom())
    with pytest.raises(httpx.ConnectError):
        await transport.handle_async_request(_request("a.upstream"))


# --------------------------------------------------------------------------- 3j
async def test_direct_client_post_bypassing_retry_request_is_gated() -> None:
    """``handlers/anthropic.py:3241``-style ``http_client.post`` must be gated."""
    recorder = _Recorder({"a.upstream": [(429, {"retry-after": "0.3"}), (200, {})]})
    _, transport = _wired(recorder)
    client = httpx.AsyncClient(transport=transport)
    try:
        first = await client.post("https://a.upstream/v1/messages", json={})
        assert first.status_code == 429
        opened_at = time.monotonic()
        await client.post("https://a.upstream/v1/messages", json={})
        assert recorder.times("a.upstream")[1] - opened_at >= 0.3
    finally:
        await client.aclose()


# --------------------------------------------------------------------------- 3k
async def test_429_without_retry_after_opens_the_gate() -> None:
    recorder = _Recorder({"a.upstream": [(429, {}), (200, {})]})
    gate, transport = _wired(recorder)

    await transport.handle_async_request(_request("a.upstream"))
    opened_at = time.monotonic()
    deadline = gate.deadline("a.upstream")
    assert deadline is not None, "every 429 must open the gate"
    # jitter_delay_ms(retry_base_delay_ms=40, retry_max_delay_ms=80, 0) -> 20-60ms.
    assert 0.02 <= deadline - opened_at <= 0.06

    await transport.handle_async_request(_request("a.upstream"))
    assert recorder.times("a.upstream")[1] >= deadline


# --------------------------------------------------------------------------- 3l
async def test_429_with_unparseable_retry_after_uses_the_same_fallback() -> None:
    recorder = _Recorder({"a.upstream": [(429, {"retry-after": "soon-ish"}), (200, {})]})
    gate, transport = _wired(recorder)

    await transport.handle_async_request(_request("a.upstream"))
    opened_at = time.monotonic()
    deadline = gate.deadline("a.upstream")
    assert deadline is not None
    assert 0.02 <= deadline - opened_at <= 0.06

    await transport.handle_async_request(_request("a.upstream"))
    assert recorder.times("a.upstream")[1] >= deadline


# --------------------------------------------------------------------------- 3m
async def test_waiters_recheck_the_deadline_after_waking() -> None:
    recorder = _Recorder(
        {
            "a.upstream": [
                (429, {"retry-after": "0.1"}),  # opens the first gate
                (429, {"retry-after": "0.5"}),  # first released waiter re-trips it
                (200, {}),
            ]
        }
    )
    gate, transport = _wired(recorder)
    await transport.handle_async_request(_request("a.upstream"))

    await asyncio.gather(
        *(transport.handle_async_request(_request("a.upstream")) for _ in range(6))
    )
    times = recorder.times("a.upstream")
    new_deadline = times[1] + 0.5
    for dispatched_at in times[2:]:
        assert dispatched_at >= new_deadline - 0.01, "waiters must re-read the advanced deadline"


# --------------------------------------------------------------------------- 3n
async def test_gate_state_is_shared_across_both_clients() -> None:
    config = _config(http2=True, upstream_rate_gate_enabled=True)
    proxy = HeadroomProxy(config)
    await proxy.startup()
    try:
        assert proxy.http_client_h1 is not proxy.http_client, "test needs two distinct clients"
        primary = proxy.http_client._transport
        h1 = proxy.http_client_h1._transport
        assert isinstance(primary, RateGateTransport) and isinstance(h1, RateGateTransport)
        assert primary._gate is h1._gate

        recorder = _Recorder({"a.upstream": [(429, {"retry-after": "0.3"}), (200, {})]})
        primary._inner = recorder
        h1._inner = recorder

        await primary.handle_async_request(_request("a.upstream"))
        opened_at = time.monotonic()
        await h1.handle_async_request(_request("a.upstream"))
        assert recorder.times("a.upstream")[1] - opened_at >= 0.3
    finally:
        await proxy.shutdown()


# --------------------------------------------------------------------------- 3o
async def test_quota_exhaustion_retry_after_is_recorded_but_not_waited_on() -> None:
    """An hour-long Retry-After is learned, then failed fast.

    ``observe()`` must record the deadline even though ``wait()`` will decline
    to park on it: the gate keeps learning, so it throttles again the moment a
    fresh 429 lands inside the budget.
    """
    recorder = _Recorder({"a.upstream": [(429, {"retry-after": "3600"}), (200, {})]})
    gate, transport = _wired(recorder, _gate(max_wait_seconds=0.2))

    await transport.handle_async_request(_request("a.upstream"))
    deadline = gate.deadline("a.upstream")
    assert deadline is not None and deadline - time.monotonic() > 3000
    started = time.monotonic()
    await transport.handle_async_request(_request("a.upstream"))
    assert time.monotonic() - started < 0.15, "an hour-long Retry-After must not park at all"


async def test_gate_deadline_does_not_ratchet_upward() -> None:
    """The freshest 429 wins, in both directions.

    With ``max()`` merge semantics an early ``Retry-After: 60`` would pin the
    host for a minute even after the upstream started answering ``Retry-After:
    5``, and every fresh long header would push the deadline further out.
    """
    recorder = _Recorder(
        {"a.upstream": [(429, {"retry-after": "60"}), (429, {"retry-after": "0.1"})]}
    )
    gate, transport = _wired(recorder, _gate(max_wait_seconds=0.05))

    await transport.handle_async_request(_request("a.upstream"))
    await transport.handle_async_request(_request("a.upstream"))
    deadline = gate.deadline("a.upstream")
    assert deadline is not None
    assert deadline - time.monotonic() <= 0.1, "a shorter, fresher Retry-After must win"


# --------------------------------------------------------------------------- 3p
async def test_client_kwargs_survive_transport_installation() -> None:
    config = _config(
        http2=True,
        upstream_rate_gate_enabled=True,
        max_connections=123,
        max_keepalive_connections=45,
        keepalive_expiry=67.0,
    )
    proxy = HeadroomProxy(config)
    await proxy.startup()
    try:
        primary = proxy.http_client._transport
        assert isinstance(primary, RateGateTransport)
        pool = primary._inner._pool
        assert pool._http2 is True, "HTTP/2 must survive the transport= install"
        assert pool._max_connections == 123
        assert pool._max_keepalive_connections == 45
        assert pool._keepalive_expiry == 67.0
        assert pool._ssl_context.verify_mode == ssl.CERT_REQUIRED
        # The forced-h1 twin keeps HTTP/2 off but the same pool settings.
        h1_pool = proxy.http_client_h1._transport._inner._pool
        assert h1_pool._http2 is False
        assert h1_pool._max_connections == 123
        # Timeouts are client-level (applied per I/O inside httpcore), so they
        # are unaffected by transport= and still come from _client_kwargs.
        assert proxy.http_client.timeout.read == float(config.request_timeout_seconds)
    finally:
        await proxy.shutdown()


async def test_configured_http_proxy_is_mounted_and_gated() -> None:
    """A configured ``proxy=`` keeps its ``all://`` mount, and the mount is gated.

    ``Client._transport_for_url`` consults mounts BEFORE ``transport=``, so a
    gate installed only on ``_transport`` would be routed around entirely.
    """
    config = _config(http_proxy="http://corp-proxy:3128", upstream_rate_gate_enabled=True)
    proxy = HeadroomProxy(config)
    await proxy.startup()
    try:
        client = proxy.http_client
        assert client._mounts, "the configured proxy mount must survive"
        routed = client._transport_for_url(httpx.URL("https://api.anthropic.com/v1/messages"))
        assert isinstance(routed, RateGateTransport)
        assert type(routed._inner._pool).__name__ == "AsyncHTTPProxy"
    finally:
        await proxy.shutdown()


async def test_environment_proxy_map_survives_and_is_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """httpx builds its env proxy map only when ``transport is None``.

    ``_client.py``: ``allow_env_proxies = trust_env and transport is None``.
    Passing any ``transport=`` therefore drops HTTP_PROXY/HTTPS_PROXY/ALL_PROXY
    and every NO_PROXY exemption -- in a deployment whose egress only works
    through the env proxy that is total breakage, and the gate is on by default.
    """
    monkeypatch.setenv("HTTPS_PROXY", "http://corp-egress:3128")
    monkeypatch.setenv("NO_PROXY", "internal.test")
    config = _config(upstream_rate_gate_enabled=True)
    proxy = HeadroomProxy(config)
    await proxy.startup()
    try:
        client = proxy.http_client
        assert client._mounts, "environment proxies must survive gate installation"
        routed = client._transport_for_url(httpx.URL("https://api.anthropic.com/v1/messages"))
        assert isinstance(routed, RateGateTransport), "the env proxy mount must be gated"
        assert type(routed._inner._pool).__name__ == "AsyncHTTPProxy"
        # NO_PROXY exemptions still bypass the proxy, and are still gated.
        exempt = client._transport_for_url(httpx.URL("https://internal.test/v1"))
        assert isinstance(exempt, RateGateTransport)
        assert type(exempt._inner._pool).__name__ == "AsyncConnectionPool"
    finally:
        await proxy.shutdown()


# ===========================================================================
# WU3 (headroom-8z2.3): AIMD concurrency limiter + header-driven seeding.
#
# One test per lettered requirement (2a-2i) of the ticket. The limiter is
# inspected through ``concurrency_limit`` (the effective limit) and driven
# through ``admit``/``release``/``observe`` -- the same three calls
# ``RateGateTransport`` makes -- so the assertions are on the governor's
# contract, never on elapsed wall clock.
# ===========================================================================

_HOST = "a.upstream"


def _response(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, headers=headers or {}, json={"ok": True})


def _ratelimit_headers(remaining: int, *, reset_in_seconds: float = 60.0) -> dict[str, str]:
    """Real API-key-auth header shape (tests/test_proxy_streaming_ratelimit_headers.py)."""
    reset = datetime.now(timezone.utc) + timedelta(seconds=reset_in_seconds)
    return {
        "anthropic-ratelimit-requests-limit": "60",
        "anthropic-ratelimit-requests-remaining": str(remaining),
        "anthropic-ratelimit-requests-reset": reset.isoformat().replace("+00:00", "Z"),
    }


# --------------------------------------------------------------------------- 2a
async def test_sustained_success_raises_the_concurrency_limit_additively() -> None:
    """+1 per ceil(limit) successes: one extra slot per full round at the current limit."""
    gate = _gate()
    assert await gate.admit(_HOST) is False
    gate.observe(_HOST, _response(429, {"retry-after": "0.001"}))
    gate.release(_HOST, _response(429))
    assert gate.concurrency_limit(_HOST) == 1.0

    # ceil(1) == 1 success buys one slot.
    assert await gate.admit(_HOST) is False
    gate.release(_HOST, _response(200))
    assert gate.concurrency_limit(_HOST) == 2.0

    # ceil(2) == 2 successes buy the next one: additive, and slower as it grows.
    assert await gate.admit(_HOST) is False
    gate.release(_HOST, _response(200))
    assert gate.concurrency_limit(_HOST) == 2.0
    assert await gate.admit(_HOST) is False
    gate.release(_HOST, _response(200))
    assert gate.concurrency_limit(_HOST) == 3.0


# --------------------------------------------------------------------------- 2b
async def test_429_halves_the_concurrency_limit() -> None:
    """The first 429 halves the in-flight count that earned it; later ones halve the limit."""
    gate = _gate()
    for _ in range(8):
        assert await gate.admit(_HOST) is False
    gate.observe(_HOST, _response(429, {"retry-after": "0.001"}))
    assert gate.concurrency_limit(_HOST) == 4.0

    gate.observe(_HOST, _response(429, {"retry-after": "0.001"}))
    assert gate.concurrency_limit(_HOST) == 2.0


# --------------------------------------------------------------------------- 2c
async def test_one_burst_of_429s_halves_the_limit_exactly_once() -> None:
    """Chiu & Jain assume ONE decrease per feedback round; N decreases is collapse.

    Eight requests admitted together all 429: that is one congestion episode,
    not eight, and un-collapsing 8 -> 1 would cost 1+2+...+7 = 28 successes
    serialised at limit 1.
    """
    gate = _gate()
    generations = []
    for _ in range(8):
        assert await gate.admit(_HOST) is False
        generations.append(gate.generation(_HOST))

    rate_limited = _response(429, {"retry-after": "0.001"})
    for generation in generations:
        gate.observe(_HOST, rate_limited, generation)
        gate.release(_HOST, rate_limited)
    assert gate.concurrency_limit(_HOST) == 4.0

    # A 429 from a request admitted AFTER the decrease is a new episode.
    assert await gate.admit(_HOST) is False
    gate.observe(_HOST, _response(429, {"retry-after": "0.001"}), gate.generation(_HOST))
    assert gate.concurrency_limit(_HOST) == 2.0


async def test_cancelling_a_parked_admit_leaves_no_pending_tasks() -> None:
    """Client disconnects and httpx timeouts cancel admit on the hot path."""
    gate = _gate()
    gate.observe(_HOST, _response(200, _ratelimit_headers(1)))
    assert await gate.admit(_HOST) is False  # fills the single slot

    task = asyncio.create_task(gate.admit(_HOST))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    leaked = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    assert leaked == [], f"cancelled admit leaked waiter tasks: {leaked}"


async def test_concurrency_limit_never_drops_below_one() -> None:
    gate = _gate()
    assert await gate.admit(_HOST) is False
    for _ in range(5):
        gate.observe(_HOST, _response(429, {"retry-after": "0.001"}))
    assert gate.concurrency_limit(_HOST) == 1.0


# --------------------------------------------------------------------------- 2d
async def test_529_does_not_change_the_concurrency_limit() -> None:
    """Upstream overload is not account quota: it neither lowers nor raises the limit."""
    gate = _gate()
    for _ in range(4):
        assert await gate.admit(_HOST) is False
    gate.observe(_HOST, _response(429, {"retry-after": "0.001"}))
    assert gate.concurrency_limit(_HOST) == 2.0

    for _ in range(4):
        overloaded = _response(529, {"retry-after": "5"})
        gate.observe(_HOST, overloaded)
        gate.release(_HOST, overloaded)
    assert gate.concurrency_limit(_HOST) == 2.0


# --------------------------------------------------------------------------- 2e
async def test_ratelimit_headers_seed_the_limit_without_probing() -> None:
    """API-key auth: ``requests-remaining`` bounds concurrency BEFORE any 429."""
    gate = _gate()
    gate.observe(_HOST, _response(200, _ratelimit_headers(3)))
    assert gate.concurrency_limit(_HOST) == 3.0, "the limit must be seeded, not probed"
    assert gate.deadline(_HOST) is None, "seeding must not open the 429 gate"

    # A stale header set (reset already past) is ignored rather than pinning us.
    gate.observe(_HOST, _response(200, _ratelimit_headers(1, reset_in_seconds=-10.0)))
    assert gate.concurrency_limit(_HOST) == 3.0

    # Headers and AIMD compose as a minimum: a generous quota cannot erase what
    # a 429 taught, and a scarce quota still binds.
    gate.observe(_HOST, _response(200, _ratelimit_headers(50)))
    for _ in range(8):
        assert await gate.admit(_HOST) is False
    gate.observe(_HOST, _response(429, dict(_ratelimit_headers(50), **{"retry-after": "0.001"})))
    assert gate.concurrency_limit(_HOST) == 4.0, "a generous quota must not erase AIMD"
    gate.observe(_HOST, _response(200, _ratelimit_headers(2)))
    assert gate.concurrency_limit(_HOST) == 2.0, "a scarce quota must still bind"


# --------------------------------------------------------------------------- 2f
async def test_absent_ratelimit_headers_fall_back_to_pure_aimd() -> None:
    """Subscription/OAuth auth sends no such headers: nothing raises, nothing breaks."""
    gate = _gate()
    for _ in range(3):
        ok = _response(200)
        assert await gate.admit(_HOST) is False
        gate.observe(_HOST, ok)
        gate.release(_HOST, ok)
    assert gate.concurrency_limit(_HOST) is None, "no headers, no 429: no limit at all"

    for _ in range(2):
        assert await gate.admit(_HOST) is False
    gate.observe(_HOST, _response(429, {"retry-after": "0.001"}))
    assert gate.concurrency_limit(_HOST) == 1.0, "pure AIMD still learns from the 429"


# --------------------------------------------------------------------------- 2g
async def test_single_in_flight_request_is_never_delayed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A solo agent pays nothing, even at the floor limit of 1."""
    gate = _gate()
    gate.observe(_HOST, _response(200, _ratelimit_headers(1)))
    assert gate.concurrency_limit(_HOST) == 1.0

    async def _boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a single in-flight request must never wait on the limiter")

    monkeypatch.setattr("headroom.proxy.upstream_rate_gate.asyncio.wait", _boom)
    monkeypatch.setattr("headroom.proxy.upstream_rate_gate.asyncio.wait_for", _boom)
    for _ in range(3):
        assert await gate.admit(_HOST) is False
        gate.release(_HOST, _response(200))


# --------------------------------------------------------------------------- 2h
async def test_limiter_wait_is_bounded_by_the_budget() -> None:
    """At the budget the limiter fails open, exactly as the gate's park does.

    ``asyncio.wait_for`` here is the assertion: an unbounded limiter hangs, and
    the bound itself is pinned by the constructor budget, not by a wall-clock
    threshold.
    """
    gate = _gate(max_wait_seconds=0.05)
    gate.observe(_HOST, _response(200, _ratelimit_headers(1)))
    assert await gate.admit(_HOST) is False  # fills the single slot
    assert await asyncio.wait_for(gate.admit(_HOST), timeout=2.0) is False


async def test_limiter_wait_is_shutdown_interruptible() -> None:
    event = asyncio.Event()
    gate = UpstreamRateGate(_config(), lambda: event)  # 30s budget: only shutdown can free it
    gate.observe(_HOST, _response(200, _ratelimit_headers(1)))
    assert await gate.admit(_HOST) is False

    task = asyncio.create_task(gate.admit(_HOST))
    await asyncio.sleep(0.05)
    event.set()
    assert await asyncio.wait_for(task, timeout=2.0) is True


# 2i (kill switch) needs no test of its own: with the gate off no
# RateGateTransport is installed, so the limiter is unreachable by
# construction, and ``test_kill_switch_prevents_installation`` above already
# asserts exactly that. A limiter-specific copy passed before the limiter
# existed, which makes it evidence of nothing.


# --------------------------------------------------------------------------- behaviour
async def test_limiter_bounds_concurrent_in_flight_requests() -> None:
    """The point of the whole unit: a seeded limit caps real parallel dispatch.

    No 429 is involved, so the gate deadline cannot be the cause of the bound --
    the limiter is the only variable.
    """
    offered = 12

    class _ConcurrencyRecorder(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.inflight = 0
            self.max_inflight = 0

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
            try:
                await asyncio.sleep(0.01)
                return httpx.Response(200, headers=_ratelimit_headers(3), request=request)
            finally:
                self.inflight -= 1

    recorder = _ConcurrencyRecorder()
    gate = _gate()
    transport = RateGateTransport(gate, recorder)
    gate.observe(_HOST, _response(200, _ratelimit_headers(3)))

    await asyncio.gather(*(transport.handle_async_request(_request(_HOST)) for _ in range(offered)))
    # Exactly the seeded limit: every response re-seeds 3 and no 429 ever moves
    # AIMD off None, so the effective limit is 3 for the whole run.
    assert recorder.max_inflight == 3, "the limiter must bound parallel dispatch to the limit"
