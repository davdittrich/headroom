"""Detect, strip, and reanchor Claude Code Read-tool line-number gutters.

Claude Code's ``Read`` tool (and equivalents) prefixes every emitted line with a
right-aligned line-number gutter matching ``^\\s*\\d+[\\t→]`` — a run of
leading whitespace, one or more digits, then a single separator that is either a
TAB or a U+2192 RIGHTWARDS ARROW (the separator has drifted across Claude Code
versions). That prefix makes the payload invalid source, so the code-aware
compressors (``proxy/interceptors/astgrep.py`` and
``transforms/code_compressor.py``) fail to parse it and bail to returning the
original — 0% compression.

These helpers let a compressor parse a gutter-stripped copy while re-emitting the
original line numbers on kept (verbatim) lines. Detection is by content/shape
only: there is NO Claude Code version gating anywhere. All functions are pure and
deterministic so the transformed output is byte-stable (prompt-prefix-cache safe).
"""

from __future__ import annotations

import re
from collections import deque

# A gutter is leading whitespace, one or more digits, then a single separator:
# a TAB or a U+2192 arrow. Anchored at the start of a line.
_GUTTER_RE = re.compile(r"^(\s*\d+[\t→])")


def strip_line_gutter(line: str) -> str:
    """Remove a single leading line-number gutter from ``line`` if present.

    No-op when the line carries no gutter. Deterministic; pure.
    """
    return _GUTTER_RE.sub("", line, count=1)


def detect_and_strip_gutter(text: str) -> tuple[str, list[str], bool]:
    """Detect a majority line-number gutter and strip it line-preservingly.

    Returns ``(stripped_text, prefixes, had_gutter)``:

    - ``had_gutter`` is ``True`` iff a *strict* majority (> 0.5) of the
      NON-BLANK lines carry a gutter.
    - When ``True``: every line's gutter prefix is removed (line count preserved
      1:1), ``prefixes[i]`` holds the exact removed prefix string (number +
      separator, byte-for-byte) for line ``i`` — ``""`` for lines without a
      gutter — and ``stripped_text`` reassembles the de-guttered lines.
    - When ``False``: returns ``(text, [], False)`` unchanged. Callers MUST treat
      this as "no gutter present" and preserve their prior behavior exactly.

    Deterministic; pure; does not mutate the input.
    """
    lines = text.split("\n")
    matches: list[re.Match[str] | None] = []
    non_blank = 0
    guttered = 0
    for line in lines:
        m = _GUTTER_RE.match(line)
        matches.append(m)
        if line.strip():
            non_blank += 1
            if m is not None:
                guttered += 1

    had_gutter = non_blank > 0 and guttered * 2 > non_blank
    if not had_gutter:
        return text, [], False

    prefixes: list[str] = []
    stripped_lines: list[str] = []
    for line, m in zip(lines, matches):
        if m is None:
            prefixes.append("")
            stripped_lines.append(line)
        else:
            prefix = m.group(1)
            prefixes.append(prefix)
            stripped_lines.append(line[len(prefix) :])
    return "\n".join(stripped_lines), prefixes, True


def reanchor(compressed: str, stripped: str, prefixes: list[str]) -> str:
    """Re-attach original gutters to kept lines via a content-keyed lookup.

    LEGACY / not used in production: superseded by :func:`reanchor_spans`, which
    the compressor calls for exact cross-bucket anchoring. ``reanchor`` is
    retained only as the position-independent reference/contrast baseline (see
    ``test_reanchor_spans_cross_bucket_duplicate_exact_rows``).

    ``stripped``/``prefixes`` come from :func:`detect_and_strip_gutter`.

    Builds ``occ``: a map from each stripped line's content to a
    :class:`~collections.deque` of its ``(index, prefix)`` occurrences in source
    order. For each compressed line ``cl``, if ``occ[cl]`` is non-empty it
    ``popleft()``s the next source occurrence and emits ``prefix + cl``;
    otherwise the line passes through unchanged (headers, elision markers, CCR
    footers, reflowed text). O(n) overall. Deterministic; pure.

    Correct ONLY for DISTINCT-content kept lines. Byte-identical lines share one
    deque consumed front-first, so it does NOT preserve original line numbers
    when the emitter reorders identical lines relative to source: ``code_compressor``
    emits kept content in FIXED BUCKET ORDER (imports → type_definitions →
    class_definitions → function_signatures → top_level_code), so a byte-identical
    line duplicated across buckets is SWAPPED, and a duplicate whose earlier
    occurrence is ELIDED is front-biased to the earlier gutter.
    :func:`reanchor_spans` fixes both via per-element source-row spans.
    """
    stripped_lines = stripped.split("\n")
    occ: dict[str, deque[tuple[int, str]]] = {}
    for idx, sl in enumerate(stripped_lines):
        prefix = prefixes[idx] if idx < len(prefixes) else ""
        occ.setdefault(sl, deque()).append((idx, prefix))

    out: list[str] = []
    for cl in compressed.split("\n"):
        bucket = occ.get(cl)
        if bucket:
            _, prefix = bucket.popleft()
            out.append(prefix + cl)
        else:
            out.append(cl)
    return "\n".join(out)


def reanchor_spans(
    assembled: str,
    spans: list[tuple[int, int, int]],
    stripped: str,
    prefixes: list[str],
) -> str:
    """Overlay original gutters onto element lines using per-element spans.

    Exact cross-bucket variant of :func:`reanchor`. ``_assemble_compressed``
    emits, alongside the assembled string, one SPAN ``(source_row, line_start,
    line_end)`` per kept element (half-open, indexing ``assembled.split("\\n")``)
    in emission order. Consuming the source-order occurrence map in SOURCE-ROW
    order (spans sorted by ``source_row``) makes identical kept lines duplicated
    across different assembler buckets — whose emission order differs from source
    order — resolve to their TRUE source line numbers instead of being swapped
    (the residual of the position-independent :func:`reanchor`).

    ``stripped``/``prefixes`` come from :func:`detect_and_strip_gutter`.

    Only lines inside a span are gutter candidates, and only when their content
    matches an unconsumed stripped source line (``popleft`` of the next source
    occurrence). Every other line — inter-group blank separators, elision
    markers interspersed within a multi-line element, the CCR footer, and any
    reflowed text — passes through UN-guttered. This also removes the spurious
    source-blank gutters that :func:`reanchor` applied to separators. O(n).
    Deterministic; pure; preserves original line numbers on kept lines.

    Residuals: a leading-comment blob prepended to an element sorts by the
    element's node row (comment lines sit adjacent above the node, so their
    source occurrences are consumed in the right order). A kept line whose only
    OTHER occurrence lies inside an ELIDED body may resolve to the elided row,
    since that occurrence is never emitted/consumed (pre-existing, shared with
    :func:`reanchor`).
    """
    lines = assembled.split("\n")
    n = len(lines)
    occ: dict[str, deque[tuple[int, str]]] = {}
    for idx, sl in enumerate(stripped.split("\n")):
        prefix = prefixes[idx] if idx < len(prefixes) else ""
        occ.setdefault(sl, deque()).append((idx, prefix))

    assigned: dict[int, str] = {}
    for _row, line_start, line_end in sorted(spans, key=lambda s: s[0]):
        for i in range(line_start, min(line_end, n)):
            bucket = occ.get(lines[i])
            if bucket:
                assigned[i] = bucket.popleft()[1]

    return "\n".join(assigned.get(i, "") + lines[i] for i in range(n))
