"""Tests for the loud ccr->lossless downgrade warning in ``headroom wrap agy``.

TDD: written before implementation -- these MUST fail against the current
``headroom/cli/wrap.py`` (no ``_maybe_warn_agy_ccr_downgrade`` exists yet).

Scope (headroom-svf): when ``headroom wrap agy`` runs with
``HEADROOM_AGY_FR_MODE=ccr`` (the default) but the retrieve MCP could NOT be
wired for this run, the Cloud Code Assist handler
(``headroom.proxy.handlers.gemini._resolve_agy_fr_mode``) silently downgrades
functionResponse compression to ``lossless`` (a no-op), so tool-output
savings collapse to ~0 with no user-visible warning. This must become loud
and actionable, with best-effort cause detection:

* ``mcp`` not importable in *this* (parent) interpreter -> ADVISORY hint to
  install ``headroom-ai[proxy]`` (the agy child is resolved via
  ``shutil.which("headroom")`` and need NOT share this venv, so this is a
  likely-cause hint, not a certainty).
* ``mcp`` importable here -> the failure must be the retrieve handshake ->
  point at proxy.log.

The warning fires ONLY when ccr was requested (default or explicit) AND the
retrieve MCP did not wire.  It must stay silent when retrieve DID wire, or
when the mode was explicitly ``lossless`` (no downgrade occurred).
"""

from __future__ import annotations

import pytest


def _get_fn():
    from headroom.cli.wrap import _maybe_warn_agy_ccr_downgrade

    return _maybe_warn_agy_ccr_downgrade


class TestMaybeWarnAgyCcrDowngrade:
    # ------------------------------------------------------------------
    # Gating: fires only for ccr + not-wired.
    # ------------------------------------------------------------------

    def test_silent_when_retrieve_registered(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("HEADROOM_AGY_FR_MODE", raising=False)
        fn = _get_fn()
        fn(retrieve_registered=True)
        out = capsys.readouterr().out
        assert out == ""

    def test_silent_when_mode_explicitly_lossless(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("HEADROOM_AGY_FR_MODE", "lossless")
        fn = _get_fn()
        fn(retrieve_registered=False)
        out = capsys.readouterr().out
        assert out == ""

    def test_fires_on_default_mode_when_not_wired(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("HEADROOM_AGY_FR_MODE", raising=False)
        fn = _get_fn()
        fn(retrieve_registered=False)
        out = capsys.readouterr().out
        assert "DISABLED" in out
        assert "lossless" in out
        assert "~0" in out

    def test_fires_on_explicit_ccr_when_not_wired(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("HEADROOM_AGY_FR_MODE", "ccr")
        fn = _get_fn()
        fn(retrieve_registered=False)
        out = capsys.readouterr().out
        assert "DISABLED" in out

    def test_invalid_mode_value_treated_as_ccr_default(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Mirrors _resolve_agy_fr_mode's own fallback-to-ccr for garbage values.
        monkeypatch.setenv("HEADROOM_AGY_FR_MODE", "bogus")
        fn = _get_fn()
        fn(retrieve_registered=False)
        out = capsys.readouterr().out
        assert "DISABLED" in out

    # ------------------------------------------------------------------
    # Cause detection: in-parent `mcp` importability probe drives the branch.
    # ------------------------------------------------------------------

    def test_mcp_missing_branch_recommends_proxy_extra(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("HEADROOM_AGY_FR_MODE", raising=False)
        monkeypatch.setattr("headroom.cli.wrap._module_available", lambda name: False)
        fn = _get_fn()
        fn(retrieve_registered=False)
        out = capsys.readouterr().out
        assert "headroom-ai[proxy]" in out
        assert "pip install mcp" in out
        # Advisory caveat: parent-mcp-present/absent doesn't guarantee child state.
        assert "ADVISORY" in out or "likely cause" in out
        assert "proxy.log" not in out

    def test_mcp_present_branch_points_at_proxy_log(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("HEADROOM_AGY_FR_MODE", raising=False)
        monkeypatch.setattr("headroom.cli.wrap._module_available", lambda name: True)
        fn = _get_fn()
        fn(retrieve_registered=False)
        out = capsys.readouterr().out
        assert "proxy.log" in out
        assert "headroom-ai[proxy]" not in out

    def test_probes_mcp_module_name(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("HEADROOM_AGY_FR_MODE", raising=False)
        seen: list[str] = []

        def _fake_module_available(name: str) -> bool:
            seen.append(name)
            return True

        monkeypatch.setattr("headroom.cli.wrap._module_available", _fake_module_available)
        fn = _get_fn()
        fn(retrieve_registered=False)
        assert seen == ["mcp"]
