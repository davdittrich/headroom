"""Tests for `headroom.binaries._path_lookup` resolution order.

Registry (pinned-wheel) tools must prefer the bundled binary in this
interpreter's Scripts/bin dir over whatever happens to be first on PATH, so
headroom always runs the pinned, reproducible build instead of a shadowing
system binary. Non-registry tools keep the original PATH-first order.

Fully hermetic: `shutil.which`, `sys.prefix`, `sys.platform`, `_in_registry`
and `_tool_entry` are all monkeypatched, so no real installed binary or the
host's real PATH is ever consulted.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from headroom import binaries


def _make_exe(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _scripts_subdir(windows: bool) -> str:
    return "Scripts" if windows else "bin"


def _setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    in_registry: bool,
    binary_alias: str | None = None,
    which_map: dict[str, str] | None = None,
    windows: bool = False,
) -> Path:
    """Wire up `binaries` module state for a single `_path_lookup` call.

    Returns the (not-yet-populated) scripts/bin directory the fake
    interpreter prefix would use.
    """
    which_map = which_map or {}
    monkeypatch.setattr(binaries.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(binaries.sys, "platform", "win32" if windows else "linux", raising=False)
    monkeypatch.setattr(binaries, "_in_registry", lambda tool: in_registry)
    monkeypatch.setattr(binaries, "_tool_entry", lambda tool: {"binary": binary_alias or tool})
    monkeypatch.setattr(binaries.shutil, "which", lambda name: which_map.get(name))
    return tmp_path / _scripts_subdir(windows)


# ---------- DoD 1: registry tool in both -> bundled wheel wins ------------ #


def test_registry_tool_in_both_prefers_bundled_wheel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripts_dir = _setup(
        monkeypatch,
        tmp_path,
        in_registry=True,
        which_map={"ast-grep": "/usr/bin/ast-grep"},
    )
    bundled = scripts_dir / "ast-grep"
    _make_exe(bundled)

    result = binaries._path_lookup("ast-grep")

    assert result == bundled


# ---------- DoD 2: registry tool only on PATH -> graceful fallback -------- #


def test_registry_tool_only_on_path_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _setup(
        monkeypatch,
        tmp_path,
        in_registry=True,
        which_map={"ast-grep": "/usr/bin/ast-grep"},
    )
    # scripts dir intentionally left empty/non-existent.

    result = binaries._path_lookup("ast-grep")

    assert result == Path("/usr/bin/ast-grep")


# ---------- DoD 3: registry tool only in scripts dir -> unchanged --------- #


def test_registry_tool_only_in_scripts_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scripts_dir = _setup(monkeypatch, tmp_path, in_registry=True, which_map={})
    bundled = scripts_dir / "ast-grep"
    _make_exe(bundled)

    result = binaries._path_lookup("ast-grep")

    assert result == bundled


# ---------- DoD 4: non-registry tool in both -> PATH order unchanged ------ #


def test_non_registry_tool_in_both_prefers_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripts_dir = _setup(
        monkeypatch,
        tmp_path,
        in_registry=False,
        which_map={"grep": "/usr/bin/grep"},
    )
    shadowed = scripts_dir / "grep"
    _make_exe(shadowed)

    result = binaries._path_lookup("grep")

    assert result == Path("/usr/bin/grep")


# ---------- DoD 5: neither location has the tool -> None ------------------ #


@pytest.mark.parametrize("in_registry", [True, False])
def test_neither_location_has_tool_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, in_registry: bool
) -> None:
    _setup(monkeypatch, tmp_path, in_registry=in_registry, which_map={})

    result = binaries._path_lookup("ast-grep")

    assert result is None


# ---------- DoD 6: alias/`binary` candidate handling ----------------------- #


def test_registry_tool_alias_resolved_via_scripts_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripts_dir = _setup(
        monkeypatch,
        tmp_path,
        in_registry=True,
        binary_alias="sg",
        which_map={},
    )
    aliased = scripts_dir / "sg"
    _make_exe(aliased)

    result = binaries._path_lookup("ast-grep")

    assert result == aliased


def test_registry_tool_alias_resolved_via_path_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _setup(
        monkeypatch,
        tmp_path,
        in_registry=True,
        binary_alias="sg",
        which_map={"sg": "/usr/bin/sg"},
    )

    result = binaries._path_lookup("ast-grep")

    assert result == Path("/usr/bin/sg")


# ---------- DoD 7: Windows path shape (Scripts dir + .exe suffix) --------- #


def test_windows_registry_tool_uses_scripts_dir_and_exe_suffix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripts_dir = _setup(
        monkeypatch,
        tmp_path,
        in_registry=True,
        which_map={"ast-grep": "C:\\Other\\ast-grep.exe"},
        windows=True,
    )
    assert scripts_dir.name == "Scripts"
    bundled = scripts_dir / "ast-grep.exe"
    _make_exe(bundled)

    result = binaries._path_lookup("ast-grep")

    assert result == bundled


def test_windows_non_registry_tool_falls_back_to_scripts_dir_with_exe_suffix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripts_dir = _setup(monkeypatch, tmp_path, in_registry=False, which_map={}, windows=True)
    bundled = scripts_dir / "ast-grep.exe"
    _make_exe(bundled)

    result = binaries._path_lookup("ast-grep")

    assert result == bundled
