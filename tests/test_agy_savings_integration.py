"""WU3 integration test: agy inbox event -> proxy drain -> real dashboard surfaces.

Proves the end-to-end replay path claimed by headroom-4l8: an event emitted by
agy, when drained by the shared proxy, moves the SAME in-memory metrics the
dashboard renders — the token-savings counter (``tokens_saved_total``, the source
of the dashboard token hero) AND the per-project SavingsTracker rows — and does
so exactly once across repeated drains (at-least-once + dedup).

Isolated: constructs a real ``PrometheusMetrics`` + ``SavingsTracker`` in-process,
no network, no live proxy, HOME pinned to a tmp dir. Never runs the broad suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from headroom.proxy import agy_savings_inbox
from headroom.proxy.prometheus_metrics import PrometheusMetrics
from headroom.proxy.savings_tracker import SavingsTracker


def _event(project: str, *, tokens_saved: int, input_tokens: int) -> dict:
    """A minimal-but-complete funnel-kwargs payload for one agy request."""
    return {
        "provider": "anthropic",
        "model": "claude-sonnet",
        "input_tokens": input_tokens,
        "output_tokens": 100,
        "tokens_saved": tokens_saved,
        "latency_ms": 25.0,
        "cached": False,
        "overhead_ms": 1.0,
        "ttfb_ms": 5.0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "cache_write_5m_tokens": 0,
        "cache_write_1h_tokens": 0,
        "uncached_input_tokens": input_tokens,
        "attempted_input_tokens": input_tokens + tokens_saved,
        "project": project,
        "client": "agy",
    }


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    # Belt-and-suspenders: pin every savings sink under tmp so nothing global is touched.
    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(tmp_path / "proxy_savings.json"))
    monkeypatch.setenv("HEADROOM_SAVINGS_EVENTS_PATH", str(tmp_path / "savings_events.jsonl"))
    monkeypatch.setenv("HEADROOM_OTEL_METRICS_ENABLED", "0")
    return tmp_path


async def test_drain_moves_token_hero_and_per_project(isolated_home: Path) -> None:
    tracker = SavingsTracker(path=str(isolated_home / "proxy_savings.json"))
    metrics = PrometheusMetrics(savings_tracker=tracker)

    # Two agy requests in two projects land in the inbox.
    agy_savings_inbox.emit_event(**_event("proj-a", tokens_saved=800, input_tokens=1200))
    agy_savings_inbox.emit_event(**_event("proj-b", tokens_saved=300, input_tokens=500))

    recorded = await agy_savings_inbox.drain_inbox(metrics)
    assert recorded == 2

    # Token hero source: the dashboard reads m.tokens_saved_total (server.py:2685).
    assert metrics.tokens_saved_total == 1100
    # Request-count fidelity: both requests are reflected, not just the savings.
    assert metrics.requests_total == 2

    # Per-project section: the dashboard reads savings_tracker.stats_preview()["projects"].
    projects = metrics.savings_tracker.stats_preview()["projects"]
    assert "proj-a" in projects and "proj-b" in projects
    assert projects["proj-a"]["tokens_saved"] == 800
    assert projects["proj-b"]["tokens_saved"] == 300

    # Inbox drained empty.
    assert not list(agy_savings_inbox.inbox_dir().glob("evt-*.json"))


async def test_redrain_does_not_double_count(isolated_home: Path) -> None:
    tracker = SavingsTracker(path=str(isolated_home / "proxy_savings.json"))
    metrics = PrometheusMetrics(savings_tracker=tracker)

    agy_savings_inbox.emit_event(**_event("proj-a", tokens_saved=800, input_tokens=1200))
    assert await agy_savings_inbox.drain_inbox(metrics) == 1
    # A second drain with nothing new must not re-apply the event.
    assert await agy_savings_inbox.drain_inbox(metrics) == 0

    assert metrics.tokens_saved_total == 800
    assert metrics.requests_total == 1
