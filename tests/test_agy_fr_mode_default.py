"""Tests for the ``lossless``-default safety floor of ``_requested_agy_fr_mode``.

Scope (headroom-37g.16, WU1): ``HEADROOM_AGY_FR_MODE`` must default to
``lossless`` -- both when unset and when set to an invalid value -- so no
unrecoverable CCR ``[Retrieve more: hash=...]`` markers ship until voluntary
retrieval is proven wired. ``ccr`` remains available but must be requested
explicitly.
"""

from __future__ import annotations

import pytest

from headroom.proxy.handlers.gemini import _requested_agy_fr_mode


class TestRequestedAgyFrModeDefault:
    def test_unset_defaults_to_lossless(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HEADROOM_AGY_FR_MODE", raising=False)
        assert _requested_agy_fr_mode() == "lossless"

    def test_invalid_value_falls_back_to_lossless(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_AGY_FR_MODE", "xyz")
        assert _requested_agy_fr_mode() == "lossless"

    def test_explicit_ccr_is_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_AGY_FR_MODE", "ccr")
        assert _requested_agy_fr_mode() == "ccr"

    def test_explicit_lossless_is_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_AGY_FR_MODE", "lossless")
        assert _requested_agy_fr_mode() == "lossless"

    def test_normalizes_case_and_whitespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_AGY_FR_MODE", " CCR ")
        assert _requested_agy_fr_mode() == "ccr"
