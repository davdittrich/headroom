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
* **AIMD concurrency limiter, the proactive half.** Everything above is
  reactive: it can only act once a 429 has been spent, and above half the
  budget it declines to act at all -- which is where the measured traffic
  lives (276 of 292 retries demanded >= 30s against a 30s budget). So
  :meth:`UpstreamRateGate.admit` also bounds how many requests this process
  has IN FLIGHT to a host at once, additive-increase/multiplicative-decrease:
  a 429 halves the limit (first 429 halves the in-flight count that earned it,
  so no initial limit is guessed), ``ceil(limit)`` consecutive successes buy
  one more slot, the floor is 1 and 529 moves nothing. Until a host has either
  429'd or reported a quota there is NO limit, so a solo agent is a no-op and
  pays nothing. Where the gate parks a request because the upstream said to
  wait, the limiter holds it because this process already has enough
  outstanding; both spend the SAME per-attempt budget, so the limiter adds no
  term to the worst-case latency (``3*B + 2*30s``, unchanged). It does change
  the TYPICAL case: a host that only ever reported a small quota, with no 429
  anywhere, can now hold a request. See :meth:`UpstreamRateGate.admit`.
  Decreases are once per congestion EPISODE, not once per 429: every request of
  a burst that overshot together reports the same overshoot, and applying all
  N halvings would collapse the limit to 1 (8 parallel agents: 8 -> 1, then 28
  serialised successes to climb back).
* **Header seeding, for API-key auth.** API-key responses carry
  ``anthropic-ratelimit-requests-remaining``/``-reset`` on every status;
  subscription/OAuth carries neither. When present, ``remaining`` is a hard
  upper bound on useful concurrency -- issuing more simultaneous requests than
  the quota has left guarantees a 429 -- so it is combined with the AIMD limit
  as a MINIMUM: a generous quota cannot erase what a 429 taught, a scarce one
  still binds, and neither half ratchets. The seed expires at ``-reset``, so a
  stale scarce reading cannot pin a host. ``remaining: 0`` is ignored: it is a
  deadline statement, which the 429 path already owns.

Limitations, deliberate:

* State is process-local and intentionally lost on restart; a fresh process
  re-learns from its next 429. There is no persistence and no cross-process
  coordination. The deadline map is swept of expired hosts on every 429, so a
  host that is 429'd once and never contacted again does not leak an entry.
* Requests already in flight when the gate opens are NOT cancelled. The gate
  governs dispatch only.
* A streaming request frees its concurrency slot when the response HEADERS
  arrive, not when the body finishes: the alternative is wrapping the byte
  stream, which is exactly the buffering this transport promises not to do.
  **The limit is therefore NOT a true in-flight bound for streaming traffic:
  it under-counts concurrency by roughly the body/TTFB ratio, which for the SSE
  responses Claude Code generates is large (tens).** AIMD still converges --
  it measures 429s, not slots -- but do not read "limit 4" as "at most 4
  streams open". ponytail: upgrade path is an ``aclose``-hooked stream wrapper,
  the day the transparency guarantee can afford one.
* The limiter fails OPEN once a request's share of the budget is spent, so
  under a tight limit many waiters can expire in the same tick and dispatch
  together. Dispersing that (each waiter drawing its own fail-open instant)
  was implemented and measured: it costs more in shortened holds than the herd
  costs, 9 -> 17 wasted upstream 429s (median of 5 reps), so it is not shipped.
* The limiter cannot prevent the FIRST burst on subscription/OAuth auth: with
  no quota headers there is nothing to learn from until a 429 has been spent.
  It bounds every burst after that. API-key auth is seeded from the headers and
  can be bounded before the first 429.
* Against a per-window RATE limit whose offered load exceeds capacity, a
  CONCURRENCY limiter cannot help at all -- only waiting past the window does,
  which is the gate's job. Measured: unchanged 24/48 -> 2/26 in that harness.
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
* ``headroom/subscription/client.py`` (around line 109) and
  ``headroom/subscription/codex_rate_limits.py`` (around line 448) each build
  their own ``httpx.AsyncClient`` per poll, against the same upstream hosts
  (``api.anthropic.com`` and ``chatgpt.com`` respectively) that this gate
  governs on the main request path. A 429 from either poll neither opens nor
  observes the gate.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

from .helpers import jitter_delay_ms, retry_after_ms
from .models import ProxyConfig

RATE_LIMIT_STATUS = 429

# Request-quota headers, sent by API-key auth on EVERY response including 200s
# (tests/test_proxy_streaming_ratelimit_headers.py). Subscription/OAuth sends
# none, which is why they may only ever tighten a limit AIMD already found.
REQUESTS_REMAINING_HEADER = "anthropic-ratelimit-requests-remaining"
REQUESTS_RESET_HEADER = "anthropic-ratelimit-requests-reset"

# AIMD parameters. Each is derived, none is tuned:
#
# * ``_DECREASE_FACTOR = 0.5`` -- Chiu & Jain: a distributed control loop with no
#   coordination converges to a fair, stable allocation only with multiplicative
#   decrease, and 1/2 is the standard factor (TCP congestion avoidance) because
#   it reaches the safe region in log2(overshoot) steps. The ticket sets it here;
#   0.7-0.8 is the documented fallback if halving is measured to cost throughput.
# * ``_INCREASE_STEP = 1.0`` -- one in-flight slot is the smallest quantum of
#   concurrency there is. Any larger step is multiplicative growth wearing an
#   additive mask, which breaks the AI half of AIMD.
# * Success threshold ``ceil(limit)`` successes per step -- one full round at the
#   CURRENT limit, i.e. TCP's "+1 per RTT". It is a function of the limit rather
#   than a constant, so there is no third number to tune, and it makes probing
#   automatically gentler the higher the limit already is.
# * ``_MIN_LIMIT = 1.0`` -- the floor is one because zero is a deadlock, and a
#   solo request must always be admitted (see :meth:`UpstreamRateGate.admit`).
_DECREASE_FACTOR = 0.5
_INCREASE_STEP = 1.0
_MIN_LIMIT = 1.0


def _reset_seconds(raw: str | None) -> float | None:
    """Seconds until an ``anthropic-ratelimit-*-reset`` timestamp, or None."""
    if not raw:
        return None
    try:
        reset = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if reset.tzinfo is None:
        reset = reset.replace(tzinfo=timezone.utc)
    return (reset - datetime.now(timezone.utc)).total_seconds()


@dataclass
class _HostLimiter:
    """Per-host concurrency state: what AIMD learned, what the headers said."""

    aimd: float | None = None  # None => never constrained, so never binding
    seed: float | None = None
    seed_until: float = 0.0
    inflight: int = 0
    waiting: int = 0
    successes: int = 0
    generation: int = 0  # bumped by each decrease; see UpstreamRateGate.observe
    slot: asyncio.Event = field(default_factory=asyncio.Event)

    def idle(self, now: float) -> bool:
        """True when this entry carries no information worth keeping."""
        return not (self.inflight or self.waiting or self.limit(now) is not None)

    def limit(self, now: float) -> float | None:
        """Effective limit: the tighter of AIMD and a still-valid header seed.

        A minimum, not a last-writer-wins overwrite, because the two carry
        different information: the seed is the upstream's own statement of
        remaining quota, AIMD is what this process measured. A generous seed
        must not erase what a 429 taught, and a scarce seed must still bind.
        Each half is independently last-writer-wins, so neither ratchets.
        """
        seed = self.seed if self.seed is not None and self.seed_until > now else None
        live = [value for value in (self.aimd, seed) if value is not None]
        return min(live) if live else None


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
        # Second map, one lifecycle: both are swept together in :meth:`observe`.
        # It is a separate dict only because ``_until`` holds a bare float and
        # is written directly by tests; nothing else keeps them apart.
        self._limiters: dict[str, _HostLimiter] = {}

    def concurrency_limit(self, host: str) -> float | None:
        """Effective in-flight cap for ``host``, or None when unconstrained."""
        state = self._limiters.get(host)
        return None if state is None else state.limit(time.monotonic())

    def deadline(self, host: str) -> float | None:
        """Live gate deadline for ``host``, or ``None`` when it is not gated."""
        until = self._until.get(host)
        if until is None:
            return None
        if until <= time.monotonic():
            self._until.pop(host, None)
            return None
        return until

    def observe(
        self, host: str, response: httpx.Response, admitted_generation: int | None = None
    ) -> None:
        """Learn from a response: seed from quota headers, and gate on a 429.

        ``admitted_generation`` is what :meth:`generation` returned when this
        request was admitted. It is what makes the decrease happen once per
        congestion EPISODE rather than once per 429: every request of a burst
        that overshot together reports the same overshoot, and applying all N
        halvings is not AIMD, it is a collapse (8 -> 1 for eight parallel
        agents, then 28 serialised successes to climb back). ``None`` means
        "unstamped", which always counts -- the direct-``observe`` path used by
        tests and by any caller that did not go through :meth:`admit`.
        """
        self._seed_from_headers(host, response)
        if response.status_code != RATE_LIMIT_STATUS:
            # 529 lands here: upstream overload is not account quota, so it
            # neither opens the gate nor moves the concurrency limit.
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

        state = self._limiter(host)
        if admitted_generation is not None and admitted_generation < state.generation:
            return  # same congestion episode: already paid for by the decrease
        # Multiplicative decrease, once per episode. The first 429 has nothing to
        # halve, so it halves the in-flight count that actually earned it -- the
        # observed level that was too much -- which is why no initial limit has
        # to be guessed. ``inflight`` still counts this request: release() runs
        # after.
        base = state.aimd if state.aimd is not None else float(max(1, state.inflight))
        state.aimd = max(_MIN_LIMIT, base * _DECREASE_FACTOR)
        state.successes = 0
        state.generation += 1
        # Same sweep, same trigger: a host with nothing in flight, nobody
        # waiting and no live deadline is quiescent, and AIMD state from a
        # lapsed episode is stale anyway -- the next 429 re-learns it in one
        # step. Bounds the map by "hosts with live activity", as ``_until`` is.
        # ``release`` drops idle entries too, so hosts that never 429 at all
        # cannot accumulate here between sweeps.
        self._limiters = {
            h: s
            for h, s in self._limiters.items()
            if h == host or s.inflight or s.waiting or self._until.get(h, 0.0) > now
        }

    def generation(self, host: str) -> int:
        """Congestion epoch of ``host``, to be passed back to :meth:`observe`."""
        return self._limiter(host).generation

    def _limiter(self, host: str) -> _HostLimiter:
        return self._limiters.setdefault(host, _HostLimiter())

    def _seed_from_headers(self, host: str, response: httpx.Response) -> None:
        """Bound concurrency by the upstream's own remaining request quota.

        Absence of either header is not an error: subscription/OAuth auth sends
        neither, and that path stays pure AIMD. ``-reset`` is what keeps a stale
        ``remaining: 1`` from pinning a host forever -- past its reset the quota
        has refilled, so the seed is simply dropped.
        """
        raw = response.headers.get(REQUESTS_REMAINING_HEADER)
        if raw is None:
            return
        try:
            remaining = float(raw)
        except ValueError:
            return
        if remaining <= 0.0:
            # The seed IS ``remaining``, unclamped: k requests may still be
            # issued this window, so at most k may usefully run at once. k = 0
            # says "none at all", which is a deadline statement the 429 +
            # Retry-After path already owns. Clamping it up to the floor of 1
            # would fabricate a permission the header explicitly denies, and
            # would pin the host at limit 1 for the rest of the window; measured
            # 10 -> 14 wasted 429s in the WU3 harness, because the waiters that
            # fail open at the budget then dispatch as one herd. The boundary is
            # at zero because that is where the header stops being a statement
            # about concurrency, not because 1 and 2 measured better.
            return
        reset_in = _reset_seconds(response.headers.get(REQUESTS_RESET_HEADER))
        if reset_in is None or reset_in <= 0.0:
            return
        state = self._limiter(host)
        state.seed = max(_MIN_LIMIT, remaining)
        state.seed_until = time.monotonic() + reset_in

    async def admit(self, host: str) -> bool:
        """Hold the request until the host's gate AND its limiter allow dispatch.

        Returns True if shutdown interrupted the hold, like :meth:`wait`.

        The limiter half is bounded by what is left of ``retry_after_budget_ms``
        after the park, so it adds no term to the 3*B + 2*30s worst case. Two
        honest qualifications:

        * The BOUND is on the limiter, not on ``admit`` as a whole: ``wait``
          re-loops on a refreshed deadline, so a stream of fresh 429s can hold
          the gate half past B on its own. That is pre-existing ``wait``
          behaviour, not something the limiter adds.
        * The TYPICAL case is not unchanged. Before this limiter, a host with no
          live 429 deadline was never held at all; now a host that has only ever
          reported a small ``requests-remaining`` can hold a request for up to
          the budget without a single 429 having been seen. That is deliberate:
          the alternative is dispatching into a quota the upstream just said is
          nearly gone, whose 429 would carry a ``Retry-After`` past the budget
          that the gate then refuses to park on -- i.e. a client-visible 429
          instead of a hold. It self-terminates early when the seed's ``-reset``
          passes, because the seed stops applying.
        """
        hold_deadline = time.monotonic() + self._max_wait_seconds
        if self.deadline(host) is not None and await self.wait(host):
            return True
        state = self._limiter(host)
        shutdown = self._shutdown_event()
        # Measured and rejected: drawing each waiter's fail-open instant
        # uniformly inside the budget, to disperse the herd that forms when many
        # waiters expire in the same tick. It disperses, but the shorter
        # expected hold costs more than the herd does -- 9 -> 17 wasted upstream
        # 429s, median of 5 reps of the WU3 harness, both arms otherwise
        # identical. The hold stays the full remaining budget.
        while True:
            now = time.monotonic()
            limit = state.limit(now)
            if limit is None or state.inflight < limit:
                break  # includes the solo-agent case: 0 < limit, always
            remaining = hold_deadline - now
            if remaining <= 0.0:
                # Budget spent: dispatch anyway. Same call the gate's park and
                # ``helpers.overload_retry_delay_ms`` make -- a limit must never
                # become an unbounded hang.
                break
            state.slot.clear()
            state.waiting += 1
            # Both waiters are created up front and cancelled in the SAME finally
            # that decrements ``waiting``: a cancelled admit -- routine, every
            # client disconnect and httpx timeout takes this path -- must not
            # leave Event.wait() futures registered in ``slot._waiters`` for the
            # lifetime of the process.
            waiters = (
                asyncio.ensure_future(state.slot.wait()),
                asyncio.ensure_future(shutdown.wait()),
            )
            try:
                await asyncio.wait(waiters, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
            finally:
                state.waiting -= 1
                for task in waiters:
                    task.cancel()
            if shutdown.is_set():
                return True
        state.inflight += 1
        return False

    def release(self, host: str, response: httpx.Response | None) -> None:
        """Free the slot and, on a success, pay into the additive increase."""
        state = self._limiters.get(host)
        if state is None:
            return
        state.inflight = max(0, state.inflight - 1)
        state.slot.set()
        if state.idle(time.monotonic()):
            # O(1) drop of an entry that carries nothing: a host that never 429s
            # and reports no quota leaves no state behind at all, so the map
            # cannot grow with the number of hosts contacted. The sweep in
            # observe() only ever sees hosts that DID push back.
            self._limiters.pop(host, None)
            return
        if response is None or response.status_code >= 400:
            return  # 429/529/errors buy no headroom; only a real success does
        limit = state.aimd
        if limit is None:
            return  # unconstrained: there is nothing to raise
        state.successes += 1
        if state.successes >= math.ceil(limit):
            state.aimd = limit + _INCREASE_STEP
            state.successes = 0

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
        if await self._gate.admit(host):
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
        generation = self._gate.generation(host)  # congestion epoch at admission
        response: httpx.Response | None = None
        try:
            response = await self._inner.handle_async_request(request)
        finally:
            # observe() first: it reads ``inflight``, which must still include
            # this request so the first 429 halves the level that earned it.
            if response is not None:
                self._gate.observe(host, response, generation)
            self._gate.release(host, response)
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
