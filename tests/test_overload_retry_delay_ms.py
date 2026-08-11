"""Direct unit tests for the shared 429/529 overload-retry delay policy.

overload_retry_delay_ms is the single source of truth extracted from the
duplicated blocks in HeadroomProxy._retry_request (server.py) and the
streaming handler (handlers/streaming.py). It is a pure function of values,
not ProxyConfig or a live proxy, so it is testable without constructing
either.
"""

from __future__ import annotations

from unittest.mock import patch

from headroom.proxy.helpers import overload_retry_delay_ms


def _call(status_code: int, retry_after: float | None, **overrides: float | int) -> float | None:
    kwargs: dict[str, float | int] = {
        "retry_after_budget_ms": 5_000,
        "retry_base_delay_ms": 100,
        "retry_max_delay_ms": 10_000,
        "attempt": 0,
    }
    kwargs.update(overrides)
    return overload_retry_delay_ms(status_code, retry_after, **kwargs)  # type: ignore[arg-type]


def test_429_within_budget_uses_retry_after_verbatim() -> None:
    assert _call(429, 3_000, retry_after_budget_ms=5_000) == 3_000


def test_429_over_budget_gives_up() -> None:
    assert _call(429, 6_000, retry_after_budget_ms=5_000) is None


def test_529_clamps_to_retry_max_delay_ms_not_budget() -> None:
    # 529 ignores the 429 budget entirely and clamps to retry_max_delay_ms.
    delay = _call(529, 50_000, retry_after_budget_ms=1, retry_max_delay_ms=10_000)
    assert delay == 10_000


def test_529_below_cap_uses_retry_after_verbatim() -> None:
    assert _call(529, 2_000, retry_max_delay_ms=10_000) == 2_000


def test_no_usable_retry_after_falls_back_to_jitter() -> None:
    with patch("headroom.proxy.helpers.jitter_delay_ms", return_value=42.0) as mock_jitter:
        delay = _call(429, None, retry_base_delay_ms=100, retry_max_delay_ms=10_000, attempt=2)
    assert delay == 42.0
    mock_jitter.assert_called_once_with(100, 10_000, 2)


def test_non_positive_retry_after_treated_as_unusable() -> None:
    # A parsed delay of 0 (or a past HTTP-date floored to 0) is not a usable
    # wait signal — falls back to jittered backoff like an absent header.
    with patch("headroom.proxy.helpers.jitter_delay_ms", return_value=7.0) as mock_jitter:
        delay = _call(529, 0.0)
    assert delay == 7.0
    mock_jitter.assert_called_once()
