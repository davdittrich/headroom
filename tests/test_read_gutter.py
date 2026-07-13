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

import pytest

from headroom.proxy.interceptors import astgrep
from headroom.transforms import read_gutter
from headroom.transforms.code_compressor import (
    CodeAwareCompressor,
    CodeCompressorConfig,
    CodeLanguage,
    CodeStructure,
)
from headroom.transforms.read_gutter import (
    detect_and_strip_gutter,
    reanchor,
    strip_line_gutter,
)

_GUTTER_LINE = re.compile(r"^\s*\d+[\t→]")
_GUTTER_SIG = re.compile(r"^\s*\d+[\t→](?:async def |def |class )")
# A signature-shaped line WITH or WITHOUT a leading gutter. Used to detect a
# kept signature that LOST its gutter (matches ``_SIG_ANY`` but not ``_GUTTER_SIG``).
_SIG_ANY = re.compile(r"^(?:\s*\d+[\t→])?(?:async def |def |class )")
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


def test_reanchor_is_order_independent_across_reordered_buckets():
    """``reanchor`` must be ORDER-INDEPENDENT. The compressor emits kept lines
    in FIXED BUCKET ORDER (imports → type_definitions → class_definitions →
    function_signatures → top_level_code), NOT source order. A signature that
    bucketing moves EARLIER in the output than it sat in source (e.g. a
    top-level ``def foo`` emitted after a ``class Bar`` that follows it in
    source) must still receive its ORIGINAL gutter. The old monotonic
    never-rewind forward scan skipped past it and dropped the gutter.
    """
    # stripped: ``def foo`` at idx 1, ``class Bar`` at idx 5 (foo BEFORE Bar).
    stripped = "import os\ndef foo():\n    pass\n\nx = 1\nclass Bar:\n    pass"
    prefixes = ["1\t", "2\t", "3\t", "4\t", "5\t", "6\t", "7\t"]
    # compressed lists ``class Bar`` (later source line) BEFORE ``def foo``
    # (earlier source line) — the bucketing reorder.
    compressed = "class Bar:\ndef foo():"
    out = reanchor(compressed, stripped, prefixes)
    # BOTH keep their correct original gutters (6 for Bar, 2 for foo).
    assert out == "6\tclass Bar:\n2\tdef foo():"


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


def test_code_compressor_reordered_signatures_all_keep_gutters():
    """END-TO-END reorder regression: a top-level ``def foo`` BEFORE a
    ``class Bar`` in source, both with long bodies so both are kept as
    signatures. ``_assemble_compressed`` emits the class block
    (class_definitions bucket) BEFORE ``def foo`` (function_signatures bucket),
    reversing source order. EVERY kept signature line in the output must still
    carry its correct original gutter — the count of gutter-bearing signature
    lines must equal the number of kept signatures (not merely "the ones that
    kept a gutter are valid").
    """
    source = (
        "def foo(alpha, beta):\n"
        "    total = 0\n"
        "    total += alpha\n"
        "    total += beta\n"
        "    total += alpha * beta\n"
        "    total += alpha - beta\n"
        "    total += alpha // 3\n"
        "    total += beta // 2\n"
        "    total += alpha % 4\n"
        "    total -= beta % 5\n"
        "    total += alpha + beta + 1\n"
        "    total += alpha + beta + 2\n"
        "    total += alpha + beta + 3\n"
        "    return total\n"
        "\n"
        "\n"
        "class Bar:\n"
        '    NAME = "bar the first configuration entry"\n'
        "    VALUE = 1234567\n"
        "    ITEMS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\n"
        '    OTHER = {"alpha": 1, "beta": 2, "gamma": 3}\n'
        "    FLAG = True\n"
        '    EXTRA = "padding entry number one here"\n'
        '    MORE = "padding entry number two here"\n'
        '    EVEN_MORE = "padding entry number three"\n'
        '    STILL_MORE = "padding entry number four"\n'
    )
    # Sanity: def foo genuinely precedes class Bar in the SOURCE.
    assert source.index("def foo") < source.index("class Bar")

    guttered = _add_gutter(source, sep="\t")
    result = _compressor().compress(guttered, language="python")

    assert result.syntax_valid is True
    assert result.compression_ratio < 1.0

    guttered_lines = guttered.split("\n")
    guttered_line_set = set(guttered_lines)
    out_lines = result.compressed.split("\n")
    # Every signature-shaped line, whether or not it carries a gutter.
    sig_any = [ln for ln in out_lines if _SIG_ANY.match(ln)]
    # The subset that actually carries a gutter.
    guttered_sigs = [ln for ln in sig_any if _GUTTER_SIG.match(ln)]

    # Both ``def foo`` and ``class Bar`` survive as kept signatures.
    assert len(sig_any) >= 2, f"expected >=2 kept signatures, got {sig_any!r}"
    # The reorder must not cost ANY signature its gutter.
    assert len(guttered_sigs) == len(sig_any), (
        f"a kept signature lost its gutter: sig_any={sig_any!r} guttered_sigs={guttered_sigs!r}"
    )
    # Each retained gutter is the ORIGINAL source line, verbatim.
    assert all(ln in guttered_line_set for ln in guttered_sigs)
    # Explicitly: the out-of-order ``def foo`` kept its original line number.
    assert guttered_lines[0] in guttered_sigs


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
    assert hasattr(read_gutter, "reanchor_spans")
    assert hasattr(read_gutter, "strip_line_gutter")


# ========================================================================= #
# headroom-510.3 — exact cross-bucket reanchor via per-element span overlay  #
#                                                                            #
# ``_assemble_compressed`` stays the SOLE assembler (string byte-identical)  #
# but ALSO emits per-element SPANS; ``reanchor_spans`` overlays gutters onto #
# element/source lines by consuming the source-order occurrence map in       #
# SOURCE-ROW order, fixing the duplicate-line swap of the position-          #
# independent ``reanchor`` and removing its spurious separator gutters.      #
# ========================================================================= #
def _degutter(text: str) -> str:
    """Strip any line-number gutter from every line (content recovery)."""
    return "\n".join(strip_line_gutter(line) for line in text.split("\n"))


def _compressor_lowmin() -> CodeAwareCompressor:
    """CCR off + tiny min-token gate so small fixtures actually compress."""
    return CodeAwareCompressor(CodeCompressorConfig(enable_ccr=False, min_tokens_for_compression=1))


# ---- _assemble_compressed: byte-identical string + correct spans ---------- #
def test_assemble_compressed_typed_buckets_string_and_spans():
    # DoD: assembled CONTENT byte-identical to historical bucket-order output;
    # one span per element in ASSEMBLED-LINE index space; source rows intact.
    struct = CodeStructure(
        imports=[(0, "import os"), (1, "import sys")],
        class_definitions=[(5, "class A:\n    pass")],
        function_signatures=[(9, "def f():\n    return 1")],
    )
    assembled, spans = _compressor()._assemble_compressed(struct, CodeLanguage.PYTHON)

    assert assembled == ("import os\nimport sys\n\nclass A:\n    pass\n\ndef f():\n    return 1")
    assert spans == [(0, 0, 1), (1, 1, 2), (5, 3, 5), (9, 6, 8)]
    n_lines = len(assembled.split("\n"))
    assert all(le <= n_lines for _, _, le in spans)  # no span past the trim


def test_assemble_compressed_generic_other_one_span_per_line():
    # DoD: generic/unknown path -> one span per line, row == line index; trailing
    # blank lines trimmed and their spans clipped; output non-empty.
    struct = CodeStructure(other=["alpha", "beta", "", ""])
    assembled, spans = _compressor()._assemble_compressed(struct, CodeLanguage.UNKNOWN)

    assert assembled == "alpha\nbeta"
    assert spans == [(0, 0, 1), (1, 1, 2)]


# ---- reanchor_spans: exact cross-bucket duplicate (RED vs reanchor) ------- #
def test_reanchor_spans_cross_bucket_duplicate_exact_rows():
    # foo BEFORE Bar in source; identical kept line "    pass" (4-sp) in BOTH,
    # emitted class-bucket-first (reversed vs source). Each must get its TRUE row.
    stripped = "def foo():\n    pass\n\nclass Bar:\n    pass"
    prefixes = ["1\t", "2\t", "3\t", "4\t", "5\t"]
    assembled = "class Bar:\n    pass\n\ndef foo():\n    pass"
    spans = [(3, 0, 2), (0, 3, 5)]  # Bar element row3; foo element row0

    out = read_gutter.reanchor_spans(assembled, spans, stripped, prefixes)

    assert out == "4\tclass Bar:\n5\t    pass\n\n1\tdef foo():\n2\t    pass"
    # The position-independent reanchor SWAPS the duplicate rows (the bug).
    assert reanchor(assembled, stripped, prefixes) != out


def test_reanchor_spans_content_byte_identical_and_separators_bare():
    # DoD: gutters are pure prefixes -> degutter recovers assembled bytes; the
    # inter-group blank separator carries NO gutter (documented cleanup).
    stripped = "def foo():\n    pass\n\nclass Bar:\n    pass"
    prefixes = ["1\t", "2\t", "3\t", "4\t", "5\t"]
    assembled = "class Bar:\n    pass\n\ndef foo():\n    pass"
    spans = [(3, 0, 2), (0, 3, 5)]

    out = read_gutter.reanchor_spans(assembled, spans, stripped, prefixes)

    assert _degutter(out) == assembled
    assert out.split("\n")[2] == ""  # separator un-guttered


def test_reanchor_spans_ccr_footer_passes_through_unguttered():
    # DoD: the CCR footer (appended beyond every span) survives unchanged and
    # carries NO gutter.
    stripped = "import os\ndef foo():\n    pass"
    prefixes = ["1\t", "2\t", "3\t"]
    footer = "# [12 tokens compressed. Retrieve more: hash=abc. Expires in 5m.]"
    assembled = "import os\n\ndef foo():\n    pass\n" + footer
    spans = [(0, 0, 1), (1, 2, 4)]

    out = read_gutter.reanchor_spans(assembled, spans, stripped, prefixes)
    out_lines = out.split("\n")

    assert out_lines[-1] == footer
    assert strip_line_gutter(out_lines[-1]) == footer
    assert out_lines[0] == "1\timport os"
    assert out_lines[2] == "2\tdef foo():"
    assert out_lines[3] == "3\t    pass"


def test_reanchor_spans_interspersed_elision_markers_bare():
    # DoD: an elision marker INSIDE a multi-line element passes through
    # un-guttered while the element's kept source lines get exact gutters.
    stripped = "def big():\n    return x"
    prefixes = ["1\t", "2\t"]
    assembled = "def big():\n    # ... (body elided)\n    return x"
    spans = [(0, 0, 3)]

    out = read_gutter.reanchor_spans(assembled, spans, stripped, prefixes)
    out_lines = out.split("\n")

    assert out_lines[0] == "1\tdef big():"
    assert out_lines[1] == "    # ... (body elided)"  # marker un-guttered
    assert out_lines[2] == "2\t    return x"


def test_reanchor_spans_generic_each_line_its_own_row():
    stripped = "alpha\nbeta\ngamma"
    prefixes = ["1\t", "2\t", "3\t"]
    assembled = "alpha\nbeta\ngamma"
    spans = [(0, 0, 1), (1, 1, 2), (2, 2, 3)]

    out = read_gutter.reanchor_spans(assembled, spans, stripped, prefixes)

    assert out == "1\talpha\n2\tbeta\n3\tgamma"
    assert out  # non-empty


def test_reanchor_spans_is_deterministic():
    stripped = "def foo():\n    pass\n\nclass Bar:\n    pass"
    prefixes = ["1\t", "2\t", "3\t", "4\t", "5\t"]
    assembled = "class Bar:\n    pass\n\ndef foo():\n    pass"
    spans = [(3, 0, 2), (0, 3, 5)]
    a = read_gutter.reanchor_spans(assembled, spans, stripped, prefixes)
    b = read_gutter.reanchor_spans(assembled, spans, stripped, prefixes)
    assert a == b


@pytest.mark.xfail(
    reason="pre-existing residual shared with reanchor: a kept line whose only "
    "other source occurrence is inside an ELIDED (non-emitted) body may resolve "
    "to the elided row, since that occurrence is never consumed",
    strict=False,
)
def test_reanchor_spans_elided_body_duplicate_residual():
    # "    return x" occurs at row1 (inside foo's ELIDED body -> NOT emitted) and
    # row4 (kept in bar). occ consumes the elided row1 first -> bar mis-rows.
    stripped = "def foo():\n    return x\n\ndef bar():\n    return x"
    prefixes = ["1\t", "2\t", "3\t", "4\t", "5\t"]
    assembled = "def foo():\n    # ...\n\ndef bar():\n    return x"
    spans = [(0, 0, 2), (3, 3, 5)]
    out = read_gutter.reanchor_spans(assembled, spans, stripped, prefixes)
    assert out.split("\n")[4] == "5\t    return x"  # DESIRED; currently fails


# ---- End-to-end through compress() ---------------------------------------- #
def test_compress_cross_bucket_duplicate_exact_rows_end_to_end():
    # def foo BEFORE class Bar; identical kept "    pass" duplicated across the
    # function_signatures and class_definitions buckets (reversed at assembly).
    src = "def foo():\n    pass\n\n\nclass Bar:\n    pass\n"
    assert src.index("def foo") < src.index("class Bar")
    guttered = _add_gutter(src, sep="\t")

    out = _compressor_lowmin().compress(guttered, language="python").compressed
    clean_out = _compressor_lowmin().compress(src, language="python").compressed

    # Content byte-identity: degutter(guttered_out) == clean compression.
    assert _degutter(out) == clean_out
    lines = out.split("\n")
    guttered_set = set(guttered.split("\n"))
    pass_lines = [ln for ln in lines if ln.endswith("    pass")]
    assert pass_lines and all(ln in guttered_set for ln in pass_lines)
    # Positional exactness (RED against the swapping reanchor): the "    pass"
    # DIRECTLY under each definition must carry THAT definition's body row —
    # foo.pass=2 under def foo (row1), Bar.pass=6 under class Bar (row5).
    bar_i = next(i for i, ln in enumerate(lines) if ln.endswith("class Bar:"))
    foo_i = next(i for i, ln in enumerate(lines) if ln.endswith("def foo():"))
    assert lines[bar_i + 1] == "6\t    pass"
    assert lines[foo_i + 1] == "2\t    pass"


def test_compress_guttered_content_byte_identical_to_clean_real_payload():
    # DoD: on a REAL payload, degutter(compress(guttered)) == compress(clean).
    clean = _load_valid_python_prefix()
    guttered = _add_gutter(clean, sep="\t")
    comp = _compressor()  # enable_ccr=False -> no stateful footer
    guttered_out = comp.compress(guttered, language="python").compressed
    clean_out = comp.compress(clean, language="python").compressed
    assert _degutter(guttered_out) == clean_out


def test_compress_guttered_ccr_footer_has_no_gutter():
    # DoD: CCR footer survives reanchor_spans and carries NO gutter.
    clean = _load_valid_python_prefix()
    guttered = _add_gutter(clean, sep="\t")
    compressor = CodeAwareCompressor(CodeCompressorConfig(enable_ccr=True))
    result = compressor.compress(guttered, language="python")
    footers = [ln for ln in result.compressed.split("\n") if "Retrieve more: hash=" in ln]
    assert footers, "expected a CCR footer"
    for ln in footers:
        assert not _GUTTER_LINE.match(ln)
        assert strip_line_gutter(ln) == ln


def test_compress_guttered_js_export_wrapped_reanchors():
    js = "export function foo() {\n  return 1;\n}\n\nexport function bar() {\n  return 2;\n}\n"
    guttered = _add_gutter(js, sep="\t")
    comp = _compressor_lowmin()
    out = comp.compress(guttered, language="javascript").compressed
    clean_out = comp.compress(js, language="javascript").compressed
    assert out
    assert _degutter(out) == clean_out
    guttered_set = set(guttered.split("\n"))
    assert all(ln in guttered_set for ln in out.split("\n") if _GUTTER_LINE.match(ln))


def test_compress_guttered_go_package_reanchors():
    go = 'package main\n\nimport "fmt"\n\nfunc Foo() int {\n\treturn 1\n}\n'
    guttered = _add_gutter(go, sep="\t")
    comp = _compressor_lowmin()
    out = comp.compress(guttered, language="go").compressed
    clean_out = comp.compress(go, language="go").compressed
    assert out
    assert _degutter(out) == clean_out
    guttered_set = set(guttered.split("\n"))
    assert all(ln in guttered_set for ln in out.split("\n") if _GUTTER_LINE.match(ln))


# ---- PR-A/1 merge exit gate: #1926 C# support x span carrier -------------- #
def test_compress_guttered_csharp_region_header_span_reanchors():
    """Exit gate for merging #1926 (first-class C#) with the span carrier.

    A guttered C# payload whose leading ``#region``/comment banner lands in the
    ``header_code`` bucket AND whose method body is compressible. Proves the
    merge kept C# flowing through the span carrier and that ``header_code`` is
    span-anchored — i.e. retyped from origin's plain ``list[str]`` to
    ``list[tuple[int, str]]`` and emitted via ``emit_bucket``. If it had stayed
    ``list[str]`` + ``parts.extend`` (origin's form), the header would either
    raise ``TypeError`` at ``"\\n".join`` (tuple in a str list) or bypass span
    tracking and lose its gutter — both caught below. Also guards the three
    C#-only append sites (namespace container + ``#if`` opaque regions) that
    origin appended as plain strings; unfixed they crash ``emit_bucket``.
    """
    cs = (
        "#region License\n"
        "// Copyright (c) 2026 Acme. All rights reserved.\n"
        "// Licensed under the MIT License.\n"
        "#endregion\n"
        "\n"
        "namespace Acme\n"
        "{\n"
        "    public class Calc\n"
        "    {\n"
        "        public int Compute(int x)\n"
        "        {\n"
        "            int a = x + 1;\n"
        "            int b = a * 2;\n"
        "            int c = b - 3;\n"
        "            int d = c / 4;\n"
        "            int e = d + 5;\n"
        "            return e;\n"
        "        }\n"
        "    }\n"
        "}\n"
    )
    guttered = _add_gutter(cs, sep="\t")
    comp = CodeAwareCompressor(
        CodeCompressorConfig(min_tokens_for_compression=1, max_body_lines=1, enable_ccr=False)
    )
    result = comp.compress(guttered, language="csharp")
    clean_out = comp.compress(cs, language="csharp").compressed

    # C# genuinely compressed through the span carrier (no crash, ratio < 1).
    assert result.language == CodeLanguage.CSHARP
    assert result.syntax_valid is True
    assert result.compression_ratio < 1.0

    out = result.compressed
    # Byte-exact overlay: gutters are pure prefixes, so degutter recovers the
    # clean compression verbatim (retrieval-safe).
    assert _degutter(out) == clean_out

    # No wrong line numbers: every guttered kept line is an original source line.
    guttered_set = set(guttered.split("\n"))
    assert all(ln in guttered_set for ln in out.split("\n") if _GUTTER_LINE.match(ln))

    # THE GATE: the leading ``#region`` header survived compression AND carries
    # its correct original gutter (source row 0 -> "1\t..."). Proves header_code
    # is span-anchored, not bypassed.
    header_out = [ln for ln in out.split("\n") if "#region License" in ln]
    assert header_out, "header #region must survive compression"
    # Carries a gutter, recovers the exact source text, and is the ORIGINAL
    # source-row-0 line verbatim (pad-agnostic vs _add_gutter's right-align).
    assert _GUTTER_LINE.match(header_out[0])
    assert strip_line_gutter(header_out[0]) == "#region License"
    assert header_out[0] == guttered.split("\n")[0]
    assert header_out[0] in guttered_set
