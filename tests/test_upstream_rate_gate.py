"""WU2 (headroom-8z2.2): per-upstream-host rate gate as an httpx transport wrapper.

Covers the guards from the ticket, one test per lettered requirement (3a-3p):
gating, per-host isolation, dispersed release, idle no-op, the fail-fast bound,
529 exclusion, shutdown interruption, the kill switch, streaming transparency,
non-``_retry_request`` call sites, headerless/unparseable ``Retry-After``,
post-wake re-check, cross-client shared state, quota-exhaustion fail-fast, and
preservation of the ``AsyncClient`` kwargs that ``transport=`` would otherwise
silence.

All waits are test-scale (tens to hundreds of milliseconds) so the suite stays
fast while exercising the real ``asyncio`` wait path -- the gate's wait is not
stubbed anywhere in this module.
"""

from __future__ import annotations

import asyncio
import ssl
import time
from typing import Any

import httpx
import pytest

from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import HeadroomProxy, _provider_httpx_client_options
from headroom.proxy.upstream_rate_gate import (
    GATE_MAX_WAIT_SECONDS,
    RateGateTransport,
    UpstreamRateGate,
    client_kwargs_with_gate,
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
async def test_release_is_dispersed_across_waiters() -> None:
    recorder = _Recorder({"a.upstream": [(429, {"retry-after": "0.2"}), (200, {})]})
    _, transport = _wired(recorder)
    await transport.handle_async_request(_request("a.upstream"))

    await asyncio.gather(
        *(transport.handle_async_request(_request("a.upstream")) for _ in range(8))
    )
    released = recorder.times("a.upstream")[1:]
    spread = max(released) - min(released)
    assert spread >= 0.05, f"waiters dispatched in one tick (spread={spread:.4f}s)"


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
async def test_deadline_beyond_bound_dispatches_instead_of_sleeping() -> None:
    recorder = _Recorder()
    gate, transport = _wired(recorder)
    gate._until["a.upstream"] = time.monotonic() + GATE_MAX_WAIT_SECONDS + 60
    assert gate.deadline("a.upstream") is not None, (
        "the gate must be live for this to prove anything"
    )

    started = time.monotonic()
    await transport.handle_async_request(_request("a.upstream"))
    assert time.monotonic() - started < 0.1, "a deadline beyond the bound must not park"
    assert len(recorder.dispatches) == 1


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
    recorder = _Recorder()
    event = asyncio.Event()
    gate = UpstreamRateGate(_config(), lambda: event)
    transport = RateGateTransport(gate, recorder)
    gate._until["a.upstream"] = time.monotonic() + 20.0

    started = time.monotonic()
    task = asyncio.create_task(transport.handle_async_request(_request("a.upstream")))
    await asyncio.sleep(0.05)
    event.set()
    await asyncio.wait_for(task, timeout=2.0)
    dispatched_at = recorder.dispatches[0][1]
    assert dispatched_at - started >= 0.05, "the request must actually have been parked"
    assert dispatched_at - started < 1.0, "shutdown must abandon the wait promptly"


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


async def test_kill_switch_off_yields_no_transport_kwarg() -> None:
    kwargs = {"timeout": httpx.Timeout(1.0), "verify": True}
    assert client_kwargs_with_gate(None, http2=True, client_kwargs=kwargs) == kwargs


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
async def test_quota_exhaustion_retry_after_does_not_park() -> None:
    recorder = _Recorder({"a.upstream": [(429, {"retry-after": "3600"}), (200, {})]})
    gate, transport = _wired(recorder)

    await transport.handle_async_request(_request("a.upstream"))
    deadline = gate.deadline("a.upstream")
    assert deadline is not None and deadline - time.monotonic() > 3000
    started = time.monotonic()
    await transport.handle_async_request(_request("a.upstream"))
    assert time.monotonic() - started < 0.1, "an hour-long Retry-After must fail fast"


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


async def test_verify_object_is_handed_to_the_inner_transport() -> None:
    context = ssl.create_default_context()
    gate = _gate()
    kwargs = client_kwargs_with_gate(
        gate,
        http2=False,
        client_kwargs={
            "timeout": httpx.Timeout(1.0),
            "limits": httpx.Limits(max_connections=3),
            "verify": context,
        },
    )
    assert kwargs["transport"]._inner._pool._ssl_context is context


async def test_unknown_client_kwarg_disables_the_gate_rather_than_dropping_it() -> None:
    """A future ``_client_kwargs`` key must never be silently lost.

    Dropping one would silently downgrade TLS or the connection pool; the gate
    fails open (not installed) instead, which is exactly the kill-switch path.
    """
    kwargs = client_kwargs_with_gate(
        _gate(),
        http2=False,
        client_kwargs={
            "timeout": httpx.Timeout(1.0),
            "limits": httpx.Limits(max_connections=3),
            "verify": True,
            "cert": ("/nope.pem",),
        },
    )
    assert "transport" not in kwargs
    assert kwargs["cert"] == ("/nope.pem",)


async def test_provider_client_kwargs_keys_are_all_mirrored() -> None:
    """Guards the mirroring list against drift in ``_provider_httpx_client_options``."""
    _, client_kwargs = _provider_httpx_client_options(_config(http_proxy="http://p:1"), verify=True)
    assert set(client_kwargs) == {"timeout", "limits", "verify", "proxy"}
    kwargs = client_kwargs_with_gate(_gate(), http2=False, client_kwargs=client_kwargs)
    assert "transport" in kwargs


async def test_configured_http_proxy_does_not_mount_around_the_gate() -> None:
    """A client-level ``proxy=`` registers an ``all://`` mount that
    ``Client._transport_for_url`` prefers over ``transport=``, which would route
    every request around the gate. The proxy must live on the inner transport.
    """
    config = _config(http_proxy="http://corp-proxy:3128", upstream_rate_gate_enabled=True)
    proxy = HeadroomProxy(config)
    await proxy.startup()
    try:
        client = proxy.http_client
        assert client._mounts == {}
        routed = client._transport_for_url(httpx.URL("https://api.anthropic.com/v1/messages"))
        assert routed is client._transport
        assert isinstance(routed, RateGateTransport)
        assert type(routed._inner._pool).__name__ == "AsyncHTTPProxy"
    finally:
        await proxy.shutdown()
