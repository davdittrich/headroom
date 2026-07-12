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
    """Re-attach original gutters to kept lines via an order-independent lookup.

    ``stripped``/``prefixes`` come from :func:`detect_and_strip_gutter`.

    The compressed output does NOT preserve source order: ``code_compressor``
    emits kept content in FIXED BUCKET ORDER (imports → type_definitions →
    class_definitions → function_signatures → top_level_code), so a kept
    signature that bucketing moves earlier than it sat in source (e.g. a
    top-level ``def`` after a later ``class``) would be scanned-past by a
    monotonic never-rewind pointer and emitted WITHOUT its gutter. To be robust
    to that, this builds ``occ``: a map from each stripped line's content to a
    :class:`~collections.deque` of its ``(index, prefix)`` occurrences in source
    order. For each compressed line ``cl``, if ``occ[cl]`` is non-empty it
    ``popleft()``s the next source occurrence and emits ``prefix + cl``;
    otherwise the line passes through unchanged (headers, elision markers, CCR
    footers, reflowed text). Global (position-independent) lookup handles the
    reordered buckets; the deque makes successive identical lines map to
    successive source occurrences. O(n) overall. Deterministic; pure; preserves
    original line numbers on kept (verbatim) lines.
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
