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
    """Re-attach original gutters to kept lines via a monotonic forward scan.

    ``stripped``/``prefixes`` come from :func:`detect_and_strip_gutter`.
    Maintains a pointer ``p`` into ``stripped``; for each line ``cl`` of
    ``compressed`` it finds the smallest ``j >= p`` with
    ``stripped_lines[j] == cl``. If found, it emits ``prefixes[j] + cl`` and
    advances ``p`` to ``j + 1`` (so repeated identical lines map to successive
    source occurrences). Lines with no match — headers, elision markers, CCR
    footers, reflowed text — pass through unchanged. Deterministic; pure;
    preserves original line numbers on kept (verbatim) lines.
    """
    stripped_lines = stripped.split("\n")
    n = len(stripped_lines)
    out: list[str] = []
    p = 0
    for cl in compressed.split("\n"):
        j = p
        while j < n and stripped_lines[j] != cl:
            j += 1
        if j < n:
            prefix = prefixes[j] if j < len(prefixes) else ""
            out.append(prefix + cl)
            p = j + 1
        else:
            out.append(cl)
    return "\n".join(out)
