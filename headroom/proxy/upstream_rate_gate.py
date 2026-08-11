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

* **Bounded wait, capped not skipped.** The gate parks a request for at most
  :data:`GATE_MAX_WAIT_SECONDS` (30s), and a longer demanded wait is CAPPED at
  the bound, never skipped: skipping would make the gate a permanent no-op for
  exactly the ``Retry-After`` values subscription 429s carry (60s+), letting the
  herd re-trip the limit and push the deadline out again with each fresh 429.
  Two ceilings bound the number, and the binding one is (a):

  (a) *Prompt-cache TTL.* Anthropic's default TTL is 5 minutes and ~97.6% of
  this workload's input arrives as cache reads, so time spent holding a request
  is time the cache entry is aging out; blowing the TTL converts a rate-limit
  problem into a much larger uncached-input one. Gate parks are ADDITIVE with
  the retry loop's own sleeps: ``retry_max_attempts=3`` (``models.py``) means a
  request can meet the gate 3 times with 2 ``Retry-After`` sleeps between them,
  each of those capped by ``retry_after_budget_ms`` (30s default), so the worst
  case is ``3*B + 2*30s``. At ``B = 30s`` that is 150s, i.e. a 2x margin under
  the 300s TTL. Solving ``3*B + 60 <= 150`` gives B <= 30s: the bound is the
  largest value that keeps that 2x margin, not a round number.

  (b) *Inbound concurrency, the client-visible cap.* There is no per-request
  inbound timeout (``uvicorn.run`` in ``server.py`` sets only
  ``timeout_graceful_shutdown``; ``ProxyConfig.request_timeout_seconds`` and
  ``connect_timeout_seconds`` are OUTBOUND httpx timeouts), but
  ``limit_concurrency`` (default 1000, ``HEADROOM_LIMIT_CONCURRENCY``) IS one:
  uvicorn answers 503 once the slots are full, and a parked request holds its
  slot the whole time. Parked slots are ``arrival_rate * B``; at the measured
  baseline (~146 rate-limited requests over 29 minutes, ~0.08/s) that is ~3 of
  1000 slots at B = 30s, and B = 30s stays under the cap for any arrival rate
  below ~33 req/s. Not binding here, but it is why the bound cannot simply be
  raised to the TTL.

  30s also coincides with the already-approved ``ProxyConfig.retry_after_budget_ms``
  default, i.e. the gate never holds a request longer than the retry loop
  already would.
* **Release jitter.** With one shared deadline, every parked waiter's timer would
  otherwise fire in the same event-loop tick and recreate the burst. Each waiter
  sleeps ``remaining * (1.0 + random())``, i.e. it dispatches uniformly in
  ``[deadline, deadline + remaining]``. This is the proportional 50-150% band of
  ``helpers.jitter_delay_ms`` shifted to 100-200% so that no waiter can dispatch
  *before* the deadline; the dispersal window equals the gate length itself, so
  there is no new tuned constant.
* **The wait is a loop.** After each wake the deadline is re-read: if the first
  released waiter tripped a fresh 429, the rest keep waiting on the new deadline.
  The freshest 429 wins in both directions -- a later, shorter ``Retry-After``
  lowers the deadline instead of being ignored by a ``max()`` merge.
* **Shutdown aborts the request, not just the wait.** ``HeadroomProxy.shutdown``
  sets the shutdown event and then closes the client, so a waiter that woke on
  it and dispatched anyway would race the close. It answers 503 in the same
  shape as ``HeadroomProxy._shutdown_retry_response``.
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
  coordination. The deadline map is swept of expired hosts on every 429, so a
  host that is 429'd once and never contacted again does not leak an entry.
* Requests already in flight when the gate opens are NOT cancelled. The gate
  governs dispatch only.
* ``headroom/backends/litellm.py`` and ``headroom/backends/anyllm.py`` drive
  their own HTTP stacks and never touch ``self.http_client``, so traffic through
  a configured backend is not gated. The default direct-Anthropic path (where
  ``anthropic_backend`` is ``None``) is, which is the traffic this governs.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable

import httpx

from .helpers import jitter_delay_ms, retry_after_ms
from .models import ProxyConfig

RATE_LIMIT_STATUS = 429

#: Longest a single request may be parked on a gate. See the module docstring
#: for the derivation (prompt-cache-TTL margin under additive retry sleeps, and
#: the inbound concurrency cap).
GATE_MAX_WAIT_SECONDS = 30.0


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
        now = time.monotonic()
        # The freshest 429 wins in BOTH directions. A max() merge would let one
        # early ``Retry-After: 60`` pin the host for a minute after the upstream
        # started answering ``Retry-After: 5``, and would let every fresh long
        # header ratchet the deadline further out.
        self._until = {h: t for h, t in self._until.items() if t > now}  # cheap sweep
        self._until[host] = now + demanded_ms / 1000.0

    async def wait(self, host: str) -> bool:
        """Park until ``host``'s gate expires or the bound is hit.

        Returns True if shutdown interrupted the wait, matching
        ``HeadroomProxy._wait_for_retry_delay_or_shutdown``.
        """
        park_until = time.monotonic() + self._max_wait_seconds
        shutdown = self._shutdown_event()
        while True:
            deadline = self.deadline(host)
            if deadline is None:
                return False
            now = time.monotonic()
            if now >= park_until:
                # Bound reached: dispatch and let the upstream 429 reach the
                # client rather than hold the request (and its inbound
                # concurrency slot) any longer.
                return False
            # min(remaining, bound), NOT "skip the wait when remaining > bound":
            # skipping would make the gate a permanent no-op for exactly the
            # Retry-After values subscription 429s carry.
            delay = min((deadline - now) * (1.0 + random.random()), park_until - now)
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=delay)
            except (TimeoutError, asyncio.TimeoutError):
                continue  # woke on the timer: re-read the deadline
            return True  # shutdown


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
        if self._gate.deadline(host) is not None and await self._gate.wait(host):
            # Shutdown woke us. HeadroomProxy.shutdown sets that event and then
            # closes the client, so dispatching now races the close; answer with
            # the same shape _shutdown_retry_response uses.
            return httpx.Response(
                503,
                request=request,
                headers={"content-type": "application/json", "retry-after": "0"},
                json={
                    "error": {
                        "type": "shutdown",
                        "message": "Proxy is shutting down; upstream rate gate wait cancelled.",
                    }
                },
            )
        response = await self._inner.handle_async_request(request)
        self._gate.observe(host, response)
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


def install_gate(client: httpx.AsyncClient, gate: UpstreamRateGate | None) -> httpx.AsyncClient:
    """Wrap every transport an already-built ``AsyncClient`` routes through.

    Deliberately wraps AFTER construction instead of passing ``transport=``.
    ``transport=`` is not a neutral override: httpx sets
    ``allow_env_proxies = trust_env and transport is None``, so supplying one
    empties the environment proxy map (HTTP_PROXY/HTTPS_PROXY/ALL_PROXY and
    every NO_PROXY exemption), and ``Client._transport_for_url`` consults
    ``_mounts`` before ``_transport``, so any mount would route around the gate.
    Re-deriving those from ``_client_kwargs`` is unwinnable -- it only mirrors
    what is passed explicitly, never httpx's implicit behavior.

    ponytail: reaches into ``_transport`` and ``_mounts``, which are private.
    That is the ceiling: httpx has no public "wrap what you just built" hook.
    Upgrade path is to drop this function the day it grows one. A ``None`` mount
    value means "use the default transport" and is left alone, since that
    default is already wrapped.
    """
    if gate is None:
        return client
    client._transport = RateGateTransport(gate, client._transport)
    client._mounts = {
        pattern: (None if transport is None else RateGateTransport(gate, transport))
        for pattern, transport in client._mounts.items()
    }
    return client
