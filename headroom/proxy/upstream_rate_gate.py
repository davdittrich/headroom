"""Per-upstream-host rate gate, installed as an ``httpx`` transport wrapper.

All Headroom-wrapped agents on a host share one proxy process, but each
outbound request used to discover an upstream rate limit on its own: N parallel
agents each burned a 429 learning the same fact, then resynchronized on the same
backoff boundary and re-tripped it together. This module holds ONE per-upstream-
host deadline for the whole process. A 429 opens that host's gate; every other
outbound request to the same host parks until it expires, then dispatches with
per-waiter jitter.

It is a transport wrapper rather than call-site wiring because several live
upstream calls never pass through ``HeadroomProxy._retry_request``
(``handlers/anthropic.py`` CCR continuation, ``handlers/batch.py``,
``handlers/bedrock.py``, ``handlers/openai.py``). All of them do go through
``self.http_client`` / ``self.http_client_h1``, so one install point at those two
constructions covers ``.post``, ``.send`` and ``.stream`` uniformly.

Policy, and why:

* **Bounded wait, fail open at the bound.** The gate parks a request for at most
  :data:`GATE_MAX_WAIT_SECONDS` (30s). Two ceilings were considered and the
  smaller taken. (a) Anthropic's prompt-cache TTL is 5 minutes and ~97.6% of this
  workload's input arrives as cache reads, so a wait that approaches the TTL
  converts a rate-limit problem into a much larger uncached-input problem;
  with ``retry_max_attempts=3`` (``models.py``) a request can meet the gate up to
  three times, so 3 x 30s = 90s keeps a >3x margin under the 300s TTL. (b) There
  is NO client-visible inbound request timeout to compete with: ``uvicorn.run``
  (``server.py``, in ``run_server``) sets ``timeout_graceful_shutdown`` only, and
  ``ProxyConfig.request_timeout_seconds`` / ``connect_timeout_seconds`` are
  OUTBOUND httpx timeouts (``server.py:_provider_httpx_client_options``), not an
  inbound cap. 30s also matches the already-approved
  ``ProxyConfig.retry_after_budget_ms`` default, i.e. the gate never holds a
  request longer than the retry loop itself already would. A remaining gate
  longer than the bound (quota exhaustion: ``Retry-After`` of minutes or hours)
  is NOT slept on -- the request dispatches, the upstream 429 flows back, and the
  client's own backoff handles it. Never convert a 429 into a client-visible hang.
* **Release jitter.** With one shared deadline, every parked waiter's timer would
  otherwise fire in the same event-loop tick and recreate the burst. Each waiter
  sleeps ``remaining * (1.0 + random())``, i.e. it dispatches uniformly in
  ``[deadline, deadline + remaining]``. This is the proportional 50-150% band of
  ``helpers.jitter_delay_ms`` shifted to 100-200% so that no waiter can dispatch
  *before* the deadline; the dispersal window equals the gate length itself, so
  there is no new tuned constant.
* **The wait is a loop.** After each wake the deadline is re-read: if the first
  released waiter tripped a fresh 429, the rest keep waiting on the new deadline.
* **Every 429 opens the gate**, including one with an absent or unparseable
  ``Retry-After`` (5.5% of the measured baseline). Those fall back to
  ``jitter_delay_ms(retry_base_delay_ms, retry_max_delay_ms, 0)`` -- the same
  backoff the retry loop already uses for that case. A non-positive
  ``Retry-After`` counts as unusable and takes the same fallback.
* **529 does not open the gate**: upstream overload, not account quota
  (``helpers.py``, ``RETRYABLE_OVERLOAD_STATUSES``).

Limitations, deliberate:

* State is process-local and intentionally lost on restart; a fresh process
  re-learns from its next 429. There is no persistence and no cross-process
  coordination.
* Requests already in flight when the gate opens are NOT cancelled. The gate
  governs dispatch only.
* ``headroom/backends/litellm.py`` and ``headroom/backends/anyllm.py`` drive
  their own HTTP stacks and never touch ``self.http_client``, so traffic through
  a configured backend is not gated. The default direct-Anthropic path (where
  ``anthropic_backend`` is ``None``) is, which is the traffic this governs.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Callable
from typing import Any

import httpx

from .helpers import jitter_delay_ms, retry_after_ms
from .models import ProxyConfig

logger = logging.getLogger(__name__)

RATE_LIMIT_STATUS = 429

#: Longest a single request may be parked on a gate. See the module docstring
#: for the derivation (min of the prompt-cache-TTL margin and the client-visible
#: inbound timeout, of which none is enforced).
GATE_MAX_WAIT_SECONDS = 30.0

#: ``_client_kwargs`` keys that must be mirrored onto the inner transport,
#: because ``transport=`` makes the ``AsyncClient``-level ones silent no-ops.
_MIRRORED_CLIENT_KWARGS = frozenset({"verify", "limits", "proxy"})

#: ``_client_kwargs`` keys that stay effective at the client level: httpx passes
#: timeouts down as ``request.extensions["timeout"]`` and httpcore applies them
#: per I/O operation, so replacing the transport does not weaken them.
_CLIENT_LEVEL_KWARGS = frozenset({"timeout"})


class UpstreamRateGate:
    """One per-upstream-host deadline map, shared by every wrapped client."""

    def __init__(
        self,
        config: ProxyConfig,
        shutdown_event: Callable[[], asyncio.Event],
        *,
        max_wait_seconds: float = GATE_MAX_WAIT_SECONDS,
    ) -> None:
        self._config = config
        self._shutdown_event = shutdown_event
        self._max_wait_seconds = max_wait_seconds
        self._until: dict[str, float] = {}

    def deadline(self, host: str) -> float | None:
        """Live gate deadline for ``host``, or ``None`` when it is not gated."""
        until = self._until.get(host)
        if until is None:
            return None
        if until <= time.monotonic():
            self._until.pop(host, None)
            return None
        return until

    def observe(self, host: str, response: httpx.Response) -> None:
        """Open ``host``'s gate when the response is a 429."""
        if response.status_code != RATE_LIMIT_STATUS:
            return
        demanded_ms = retry_after_ms(response)
        if demanded_ms is None or demanded_ms <= 0:
            demanded_ms = jitter_delay_ms(
                self._config.retry_base_delay_ms, self._config.retry_max_delay_ms, 0
            )
        until = time.monotonic() + demanded_ms / 1000.0
        previous = self._until.get(host)
        if previous is None or until > previous:
            self._until[host] = until

    async def wait(self, host: str) -> None:
        """Park until ``host``'s gate expires, the bound is hit, or shutdown."""
        park_until = time.monotonic() + self._max_wait_seconds
        shutdown = self._shutdown_event()
        while True:
            deadline = self.deadline(host)
            if deadline is None:
                return
            now = time.monotonic()
            remaining = deadline - now
            if remaining > self._max_wait_seconds or now >= park_until:
                # Quota exhaustion or a gate that keeps advancing: dispatch and
                # let the upstream 429 reach the client instead of hanging.
                return
            delay = min(remaining * (1.0 + random.random()), park_until - now)
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=delay)
            except (TimeoutError, asyncio.TimeoutError):
                continue  # woke on the timer: re-read the deadline
            return  # shutdown: abandon the wait


class RateGateTransport(httpx.AsyncBaseTransport):
    """Transparent wrapper: gate before dispatch, observe the response status.

    Does not buffer bodies, alter status/headers/bytes, or swallow exceptions;
    the inner response object is returned as-is so streaming stays streaming.
    """

    def __init__(self, gate: UpstreamRateGate, inner: httpx.AsyncBaseTransport) -> None:
        self._gate = gate
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if self._gate.deadline(host) is not None:
            await self._gate.wait(host)
        response = await self._inner.handle_async_request(request)
        self._gate.observe(host, response)
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


def client_kwargs_with_gate(
    gate: UpstreamRateGate | None,
    *,
    http2: bool,
    client_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """``AsyncClient`` kwargs with ``gate`` installed, or ``client_kwargs`` as-is.

    ``transport=`` makes the client's own ``http2``/``limits``/``verify``/
    ``proxy`` silent no-ops, so the inner :class:`httpx.AsyncHTTPTransport` is
    built with those exact values and they are dropped from the client kwargs.
    Dropping ``proxy`` is load-bearing, not tidiness: a client-level ``proxy``
    registers an ``all://`` mount, and ``Client._transport_for_url`` consults
    mounts BEFORE ``transport=``, so leaving it would route every request around
    the gate. An unrecognized ``_client_kwargs`` key would mean a setting
    silently lost (a TLS or connection-pool downgrade), so the gate refuses to
    install rather than mirror an incomplete set.
    """
    if gate is None:
        return dict(client_kwargs)
    unknown = set(client_kwargs) - _MIRRORED_CLIENT_KWARGS - _CLIENT_LEVEL_KWARGS
    if unknown:
        logger.error(
            "Upstream rate gate not installed: unmirrored httpx client options %s "
            "would be silently dropped by transport=. Update _MIRRORED_CLIENT_KWARGS.",
            sorted(unknown),
        )
        return dict(client_kwargs)
    inner = httpx.AsyncHTTPTransport(
        http2=http2,
        verify=client_kwargs["verify"],
        limits=client_kwargs["limits"],
        proxy=client_kwargs.get("proxy"),
    )
    gated = {k: v for k, v in client_kwargs.items() if k not in _MIRRORED_CLIENT_KWARGS}
    gated["transport"] = RateGateTransport(gate, inner)
    return gated
