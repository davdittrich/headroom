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

* **Fail fast above the budget.** The gate parks a request only while the whole
  hold -- the remaining deadline plus its release jitter, i.e. up to 2x the
  remaining wait -- fits inside ONE willingness-to-wait number,
  ``ProxyConfig.retry_after_budget_ms`` (30s default): the same number
  ``helpers.overload_retry_delay_ms`` uses to decide that a demanded wait is
  not worth holding a request for, applied to the total this component actually
  holds for. Past it the gate dispatches immediately and
  lets the upstream 429 flow back to the client, whose own backoff handles it,
  exactly as the retry loop already does. There is deliberately no second
  gate-only ceiling: a separate constant is what let the two halves of one
  policy disagree (the gate parked 30s on a wait the retry loop had already
  given up on), so the budget is read from the config and the
  ``max_wait_seconds`` constructor argument exists only for tests.

  Consequence, accepted deliberately: **above the budget the gate no longer
  throttles at all.** The measured above-bound benefit was small (13 -> 7
  wasted upstream 429s in the A/B harness) and zero for a tight arrival burst,
  while the cost was concrete: a request meeting an over-budget gate burned the
  full 30s and then got its 429 anyway, where the ungated path answered in
  ~200ms. The gate still RECORDS every 429 deadline -- :meth:`observe` is
  independent of :meth:`wait` -- so it keeps learning while it declines to
  park, and throttles again as soon as a fresh ``Retry-After`` lands inside the
  budget.

  Why 30s is also the right park ceiling, i.e. why sharing the retry loop's
  budget is not merely convenient -- two ceilings bound it, and the binding one
  is (a):

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
* **Release jitter.** With one shared deadline, every parked waiter's timer would
  otherwise fire in the same event-loop tick and recreate the burst. Each waiter
  sleeps :func:`release_delay_seconds`, dispatching uniformly in
  ``[deadline, deadline + remaining]``: the proportional 50-150% band of
  ``helpers.jitter_delay_ms`` shifted to 100-200% so that no waiter can dispatch
  *before* the deadline; the dispersal window equals the gate length itself, so
  there is no new tuned constant. It is never clamped, which is why the
  fail-fast test above is on ``2 * remaining``. The previous form,
  ``min(remaining * (1.0 + random()), park_until - now)``, produced ZERO jitter
  wherever the clamp bound: for a demanded wait at or past the ceiling every
  draw clamped, so all eight waiters' timers fired at exactly ``park_until`` and
  the herd was reassembled intact. Measured over 8 parked waiters, the release
  spread is now 0.16s at ``Retry-After`` 0.2s and 0.46s at a demanded wait of
  half the budget -- the regime that previously released in one tick.
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
* Above the budget there is no throttling at all (see the fail-fast bullet):
  the herd re-tries into the limit and each member spends its own 429. That is
  the accepted cost of never holding a request past the point where the retry
  loop would have given up.
* Because the hold must cover the jitter tail, the largest ``Retry-After`` the
  gate throttles is HALF the budget (15s at the default). Between half and the
  full budget the retry loop still honors the wait while the gate does not --
  the two agree on the number and on what it bounds (the total hold), not on
  who holds longest.
* Release order is the jitter roll, not arrival order: there is no fairness or
  FIFO guarantee between waiters, and a late arrival can dispatch first.
* The deadline is keyed by HOST alone. A 429 earned by one API key, model or
  organization gates every request this process sends to that host, including
  ones drawing on untouched quota.
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


def release_delay_seconds(remaining: float, roll: float) -> float:
    """Seconds a waiter sleeps before dispatching, for ``roll`` drawn from [0, 1).

    Dispatch lands uniformly in ``[deadline, deadline + remaining]`` -- never
    before the deadline, and spread over a window as wide as the gate itself in
    EVERY regime the gate parks in, because :meth:`UpstreamRateGate.wait` only
    parks when the whole window fits inside the budget. Pure, so that dispersal
    is testable without racing event-loop wakeups.
    """
    return remaining * (1.0 + roll)


class UpstreamRateGate:
    """One per-upstream-host deadline map, shared by every wrapped client."""

    def __init__(
        self,
        config: ProxyConfig,
        shutdown_event: Callable[[], asyncio.Event],
        *,
        max_wait_seconds: float | None = None,
    ) -> None:
        self._config = config
        self._shutdown_event = shutdown_event
        # One policy, one number: the gate's willingness to wait IS the retry
        # loop's ``retry_after_budget_ms``. A separate gate constant is what let
        # the two disagree. The override is for tests only.
        self._max_wait_seconds = (
            config.retry_after_budget_ms / 1000.0 if max_wait_seconds is None else max_wait_seconds
        )
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
        """Park until ``host``'s gate expires, unless it is beyond the budget.

        Returns True if shutdown interrupted the wait, matching
        ``HeadroomProxy._wait_for_retry_delay_or_shutdown``.
        """
        shutdown = self._shutdown_event()
        while True:
            deadline = self.deadline(host)
            if deadline is None:
                return False
            remaining = deadline - time.monotonic()
            if 2.0 * remaining > self._max_wait_seconds:
                # Over budget: dispatch now and let the 429 reach the client,
                # the same call ``helpers.overload_retry_delay_ms`` makes for
                # the retry loop. Holding the request (and its inbound
                # concurrency slot) only to hand back that same 429 later is
                # strictly worse. The factor 2 is the jitter band, not a second
                # ceiling: the budget bounds the WHOLE hold, and a waiter's hold
                # is up to 2x the remaining wait (see release_delay_seconds), so
                # every park that starts is guaranteed to fit inside the budget
                # and to keep a full dispersal window.
                return False
            delay = release_delay_seconds(remaining, random.random())
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
