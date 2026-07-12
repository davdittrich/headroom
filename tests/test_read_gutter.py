"""Tests for Claude Code Read-tool line-number gutter handling (headroom-510.2).

Claude Code's Read tool prefixes every emitted line with a right-aligned
line-number gutter matching ``^\\s*\\d+[\\t→]`` (a TAB *or* a U+2192
arrow separator; the separator has drifted across Claude Code versions). That
makes the payload invalid source, so both code-aware compressors used to bail
and return the original (0% compression). These tests exercise the shared
``read_gutter`` helper and its wiring into the two compressors.

Detection is by content/shape only — there is NO Claude Code version gating.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from headroom.proxy.interceptors import astgrep
from headroom.transforms import read_gutter
from headroom.transforms.code_compressor import (
    CodeAwareCompressor,
    CodeCompressorConfig,
)
from headroom.transforms.read_gutter import (
    detect_and_strip_gutter,
    reanchor,
    strip_line_gutter,
)

_GUTTER_LINE = re.compile(r"^\s*\d+[\t→]")
_GUTTER_SIG = re.compile(r"^\s*\d+[\t→](?:async def |def |class )")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PAYLOAD_FILE = _REPO_ROOT / "headroom" / "proxy" / "handlers" / "anthropic.py"


# --------------------------------------------------------------------------- #
# Fixtures / helpers                                                          #
# --------------------------------------------------------------------------- #
def _load_valid_python_prefix(max_lines: int = 150) -> str:
    """Return the largest <= max_lines prefix of a real repo file that parses.

    Slicing a real file at an arbitrary line can leave an unclosed block, so we
    walk back until the prefix is valid Python with enough structure to
    compress meaningfully. Uses ``headroom/proxy/handlers/anthropic.py`` per the
    work-unit spec.
    """
    lines = _PAYLOAD_FILE.read_text(encoding="utf-8").splitlines()
    for n in range(min(max_lines, len(lines)), 0, -1):
        candidate = "\n".join(lines[:n])
        try:
            ast.parse(candidate)
        except SyntaxError:
            continue
        if candidate.count("def ") >= 3:
            return candidate
    raise RuntimeError("no valid Python prefix found in payload file")


def _add_gutter(text: str, sep: str = "\t") -> str:
    """Prefix each 1-based line with a right-aligned number + separator.

    Mimics Claude Code's ``cat -n`` style Read output.
    """
    lines = text.split("\n")
    width = len(str(len(lines)))
    return "\n".join(f"{i:>{width}}{sep}{line}" for i, line in enumerate(lines, 1))


def _def_line_indices(stripped: str) -> list[int]:
    """0-based indices of def/class/async def starts in the stripped payload."""
    return [
        i
        for i, line in enumerate(stripped.split("\n"))
        if re.match(r"\s*(?:async def |def |class )", line)
    ]


def _synthetic_matches(stripped: str) -> list[dict]:
    """Build ast-grep-shaped match dicts for each definition line."""
    return [
        {"range": {"start": {"line": idx}, "byteOffset": {"start": idx}}}
        for idx in _def_line_indices(stripped)
    ]


# --------------------------------------------------------------------------- #
# DoD 1 — detect_and_strip_gutter (tab, majority)                             #
# --------------------------------------------------------------------------- #
def test_detect_and_strip_gutter_tab_majority_strips_line_preservingly():
    clean = _load_valid_python_prefix()
    guttered = _add_gutter(clean, sep="\t")
    orig_lines = guttered.split("\n")

    stripped, prefixes, had = detect_and_strip_gutter(guttered)

    assert had is True
    stripped_lines = stripped.split("\n")
    # 1:1 line-preserving.
    assert len(stripped_lines) == len(orig_lines)
    assert len(prefixes) == len(orig_lines)
    # No gutters remain.
    for sl in stripped_lines:
        assert not _GUTTER_LINE.match(sl)
    # Prefixes reproduce the originals byte-for-byte.
    for prefix, sl, orig in zip(prefixes, stripped_lines, orig_lines):
        assert prefix + sl == orig
    # The stripped body is exactly the clean source.
    assert stripped == clean


def test_detect_and_strip_gutter_clean_input_is_identity():
    clean = _load_valid_python_prefix()
    stripped, prefixes, had = detect_and_strip_gutter(clean)
    assert had is False
    assert stripped == clean
    assert prefixes == []


def test_detect_and_strip_gutter_does_not_mutate_input():
    clean = _load_valid_python_prefix()
    guttered = _add_gutter(clean)
    before = guttered
    detect_and_strip_gutter(guttered)
    assert guttered == before


# --------------------------------------------------------------------------- #
# DoD 2 — both separators (tab AND U+2192 arrow)                              #
# --------------------------------------------------------------------------- #
def test_detect_and_strip_gutter_arrow_separator():
    clean = _load_valid_python_prefix()
    guttered = _add_gutter(clean, sep="→")
    orig_lines = guttered.split("\n")

    stripped, prefixes, had = detect_and_strip_gutter(guttered)

    assert had is True
    assert stripped == clean
    # Prefixes must carry the arrow separator, not a tab.
    non_blank_prefixes = [p for p in prefixes if p]
    assert non_blank_prefixes
    assert all(p.endswith("→") for p in non_blank_prefixes)
    for prefix, sl, orig in zip(prefixes, stripped.split("\n"), orig_lines):
        assert prefix + sl == orig


def test_strip_line_gutter_handles_both_separators_and_noop():
    assert strip_line_gutter("  42\tcode") == "code"
    assert strip_line_gutter("  42→code") == "code"
    # No gutter -> unchanged.
    assert strip_line_gutter("    def foo():") == "    def foo():"


# --------------------------------------------------------------------------- #
# DoD 3 — reanchor restores line numbers; markers pass through; deterministic #
# --------------------------------------------------------------------------- #
def test_reanchor_restores_original_line_numbers_and_passes_markers():
    clean = _load_valid_python_prefix()
    guttered = _add_gutter(clean)
    stripped, prefixes, had = detect_and_strip_gutter(guttered)
    assert had is True

    stripped_lines = stripped.split("\n")
    # Simulate a compressor output: a header, two kept verbatim source lines
    # (out of the middle of the file), an elision marker, and a CCR footer.
    kept_a = stripped_lines[5]
    kept_b = stripped_lines[20]
    header = "[headroom: outlined]"
    marker = "    # ... (body elided)"
    footer = "# [123 tokens compressed. hash=deadbeef.]"
    compressed = "\n".join([header, kept_a, marker, kept_b, footer])

    out = reanchor(compressed, stripped, prefixes)
    out_lines = out.split("\n")

    # Header / marker / footer are untouched (no gutter added).
    assert out_lines[0] == header
    assert out_lines[2] == marker
    assert out_lines[4] == footer
    # Kept lines carry their ORIGINAL gutter prefix (line numbers 6 and 21).
    assert out_lines[1] == prefixes[5] + kept_a
    assert out_lines[3] == prefixes[20] + kept_b
    assert out_lines[1] == guttered.split("\n")[5]
    assert out_lines[3] == guttered.split("\n")[20]


def test_reanchor_is_monotonic_forward_scan():
    # Repeated identical lines must map to successive source occurrences.
    stripped = "a\nx\na\nx\na"
    prefixes = ["1\t", "2\t", "3\t", "4\t", "5\t"]
    compressed = "a\na\na"
    out = reanchor(compressed, stripped, prefixes)
    assert out == "1\ta\n3\ta\n5\ta"


def test_reanchor_is_deterministic():
    clean = _load_valid_python_prefix()
    guttered = _add_gutter(clean)
    stripped, prefixes, _ = detect_and_strip_gutter(guttered)
    compressed = "\n".join(stripped.split("\n")[:10])
    assert reanchor(compressed, stripped, prefixes) == reanchor(compressed, stripped, prefixes)


# --------------------------------------------------------------------------- #
# DoD 4 — code_compressor compresses a REAL guttered Python payload           #
# --------------------------------------------------------------------------- #
def _compressor() -> CodeAwareCompressor:
    # CCR disabled: keeps the test hermetic + output free of a stateful footer.
    return CodeAwareCompressor(CodeCompressorConfig(enable_ccr=False))


def test_code_compressor_compresses_guttered_python():
    clean = _load_valid_python_prefix()
    guttered = _add_gutter(clean, sep="\t")

    result = _compressor().compress(guttered, language="python")

    assert result.syntax_valid is True
    assert result.compression_ratio < 1.0
    assert result.compression_ratio < 0.5

    # Kept signature lines must carry their ORIGINAL guttered line numbers:
    # every guttered signature line in the output must be an exact original
    # guttered source line (i.e. the original number is preserved verbatim).
    guttered_line_set = set(guttered.split("\n"))
    sig_lines = [line for line in result.compressed.split("\n") if _GUTTER_SIG.match(line)]
    assert sig_lines, "expected at least one guttered signature line in output"
    assert all(line in guttered_line_set for line in sig_lines)


def test_code_compressor_guttered_arrow_payload_compresses():
    clean = _load_valid_python_prefix()
    guttered = _add_gutter(clean, sep="→")

    result = _compressor().compress(guttered, language="python")

    assert result.syntax_valid is True
    assert result.compression_ratio < 1.0


# --------------------------------------------------------------------------- #
# DoD 5 — astgrep outline / transform emit guttered signatures (hermetic)     #
# --------------------------------------------------------------------------- #
def test_astgrep_build_outline_emits_guttered_signatures():
    clean = _load_valid_python_prefix()
    guttered = _add_gutter(clean, sep="\t")
    stripped, _, _ = detect_and_strip_gutter(guttered)
    matches = _synthetic_matches(stripped)

    outline = astgrep._build_outline(matches, guttered)

    assert outline is not None
    # Every emitted signature line carries a gutter.
    sig_lines = [ln for ln in outline.split("\n") if _GUTTER_SIG.match(ln)]
    assert sig_lines
    guttered_line_set = set(guttered.split("\n"))
    assert all(ln in guttered_line_set for ln in sig_lines)


def test_astgrep_build_outline_keeps_guttered_docstring():
    # A def whose next line is a docstring: the docstring heuristic must strip
    # the gutter before testing for the triple-quote, then emit it guttered.
    source = 'def f():\n    """doc."""\n    return 1\n'
    guttered = _add_gutter(source, sep="\t")
    matches = [{"range": {"start": {"line": 0}, "byteOffset": {"start": 0}}}]

    outline = astgrep._build_outline(matches, guttered)

    assert outline is not None
    assert re.search(r"^\s*1\tdef f\(\):$", outline, re.M)
    # Docstring line kept, and still carries its gutter.
    assert re.search(r'^\s*2\t\s*"""doc\."""$', outline, re.M)


def test_astgrep_transform_outlines_guttered_input(monkeypatch):
    clean = _load_valid_python_prefix()
    guttered = _add_gutter(clean, sep="\t")
    stripped, _, _ = detect_and_strip_gutter(guttered)
    matches = _synthetic_matches(stripped)

    # Hermetic: no ast-grep binary, no tree-sitter for this path.
    monkeypatch.setattr(astgrep.binaries, "resolve", lambda _name: Path("/fake/sg"))
    monkeypatch.setattr(astgrep, "_run_ast_grep", lambda *a, **k: matches)

    interceptor = astgrep.AstGrepReadOutline()
    outline = interceptor.transform("Read", {"file_path": "/repo/anthropic.py"}, guttered)

    assert outline is not None
    assert "outlined by ast-grep" in outline
    sig_lines = [ln for ln in outline.split("\n") if _GUTTER_SIG.match(ln)]
    assert sig_lines


# --------------------------------------------------------------------------- #
# DoD 6 — no-op on clean (non-guttered) code                                  #
# --------------------------------------------------------------------------- #
def test_code_compressor_noop_on_clean_code():
    clean = _load_valid_python_prefix()

    # No gutter is detected, so the gutter path is inert.
    assert detect_and_strip_gutter(clean) == (clean, [], False)

    result = _compressor().compress(clean, language="python")
    assert result.syntax_valid is True
    assert result.compression_ratio < 1.0
    # No signature line should carry a spurious gutter prefix.
    assert not any(_GUTTER_SIG.match(line) for line in result.compressed.split("\n"))


def test_astgrep_transform_clean_code_has_no_gutters(monkeypatch):
    clean = _load_valid_python_prefix()
    matches = _synthetic_matches(clean)

    monkeypatch.setattr(astgrep.binaries, "resolve", lambda _name: Path("/fake/sg"))
    monkeypatch.setattr(astgrep, "_run_ast_grep", lambda *a, **k: matches)

    interceptor = astgrep.AstGrepReadOutline()
    outline = interceptor.transform("Read", {"file_path": "/repo/anthropic.py"}, clean)

    assert outline is not None
    # Clean input -> no gutter prefixes anywhere in the outline body.
    assert not any(_GUTTER_LINE.match(line) for line in outline.split("\n"))
    clean_line_set = set(clean.split("\n"))
    sig_lines = [
        line for line in outline.split("\n") if re.match(r"^(?:async def |def |class )", line)
    ]
    assert sig_lines
    assert all(line in clean_line_set for line in sig_lines)


# --------------------------------------------------------------------------- #
# DoD 7 — determinism (byte-stable output => prompt-prefix cache safe)        #
# --------------------------------------------------------------------------- #
def test_code_compressor_guttered_output_is_byte_stable():
    clean = _load_valid_python_prefix()
    guttered = _add_gutter(clean, sep="\t")
    first = _compressor().compress(guttered, language="python").compressed
    second = _compressor().compress(guttered, language="python").compressed
    assert first == second


# --------------------------------------------------------------------------- #
# DoD 8 — graceful degradation on a near-gutter / sub-majority payload        #
# --------------------------------------------------------------------------- #
def test_sub_majority_gutter_is_treated_as_clean():
    clean = _load_valid_python_prefix()
    lines = clean.split("\n")
    # Gutter only the first ~40% of lines -> below the strict-majority threshold.
    cutoff = int(len(lines) * 0.4)
    width = len(str(len(lines)))
    mixed = "\n".join(
        (f"{i:>{width}}\t{line}" if i <= cutoff else line) for i, line in enumerate(lines, 1)
    )

    stripped, prefixes, had = detect_and_strip_gutter(mixed)
    assert had is False
    assert stripped == mixed
    assert prefixes == []


def test_compress_does_not_crash_on_sub_majority_gutter():
    clean = _load_valid_python_prefix()
    lines = clean.split("\n")
    cutoff = int(len(lines) * 0.4)
    width = len(str(len(lines)))
    mixed = "\n".join(
        (f"{i:>{width}}\t{line}" if i <= cutoff else line) for i, line in enumerate(lines, 1)
    )
    # Must not raise; behaves as today (no gutter detected).
    result = _compressor().compress(mixed, language="python")
    assert result is not None
    assert result.original == mixed


def test_module_exports_public_helpers():
    assert hasattr(read_gutter, "detect_and_strip_gutter")
    assert hasattr(read_gutter, "reanchor")
    assert hasattr(read_gutter, "strip_line_gutter")
