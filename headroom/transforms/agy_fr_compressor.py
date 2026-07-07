"""Deterministic, recoverable compression of agy functionResponse leaves.

Moved out of ``GeminiHandlerMixin`` (headroom-37g.36) so the compression
algorithm is unit-testable standalone, without booting the FastAPI app.
Pure move -- no behavior change; ``GeminiHandlerMixin._compress_agy_function_responses``
now delegates to ``compress_function_response_leaves`` below.

---------------------------------------------------------------------------
WU1 (headroom-37g.1): uniform deterministic recoverable compression of agy
functionResponse leaves.

agy's per-turn bulk lives in ``functionResponse`` parts. Those entries carry
non-text parts, so ``_gemini_contents_to_messages`` routes them into
``preserved_indices`` and ``_rebuild_gemini_contents`` restores them verbatim
-- the text compressor never sees them. Only tiny residual text is compressed,
it inflates, the revert guard fires, and tokens_saved collapses to 0 (PR
#1044: "704 -> 718, reverting").

We compress the large STRING leaves inside those parts with a DETERMINISTIC,
IDEMPOTENT, RECOVERABLE transform applied UNIFORMLY to every functionResponse
leaf (historical + tail). Because headroom is an in-flight MITM that never
rewrites agy's LOCAL history, agy re-sends the ORIGINAL bytes each turn; a
deterministic transform (same original -> identical bytes every turn) yields a
byte-stable compressed prefix that re-hits the Cloud Code Assist server-side
cache. Recoverability is mandatory: the model reads functionResponse back as
its own prior tool results, so lossy summaries would corrupt multi-turn
reasoning.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from headroom.cache.compression_store import default_ccr_hash
from headroom.ccr.tool_injection import is_headroom_retrieve_name

logger = logging.getLogger("headroom.proxy")

# Marker shipped in place of a compressed leaf (CCR mode). It carries fixed
# prose plus the 24-hex-char CCR hash (SHA-256(original)[:24], the
# compression_store default), which ``headroom_retrieve`` resolves back to the
# original bytes. Self-describing: it NAMES the ``headroom_retrieve`` tool and
# gives a one-line call-to-expand instruction, so a model that needs the
# compressed detail knows how to fetch it (a marker naming no tool led to 0
# retrieve calls in the WU4 live trial). All-ours single-hash form: the hash
# appears exactly once, in the trailing ``Retrieve more: hash=`` form that
# also matches the existing bracketed marker style / regex
# (parser.CCR_RETRIEVAL_MARKER_RE).
_FR_CCR_HASH_LEN = 24
_FR_CCR_MARKER_PREFIX = (
    "[functionResponse compressed. Call headroom_retrieve to expand. Retrieve more: hash="
)
_FR_CCR_MARKER_TEMPLATE = _FR_CCR_MARKER_PREFIX + "{hash}]"

# Per-leaf floor DERIVED from marker overhead (not a magic 200). Replacing a
# leaf ships the marker in its place, so the net saving is
# ``leaf_tokens - marker_tokens``. Compressing is only worthwhile when that net
# saving exceeds the marker's OWN cost, i.e. ``leaf_tokens > 2 * marker_tokens``.
# We therefore set the floor to ``_FR_MARKER_MIN_RATIO`` times the marker's
# token cost, computed at runtime against the request's tokenizer.
_FR_MARKER_MIN_RATIO = 2

# WU2-A (headroom-37g.17): agy resends full history with ORIGINAL tool outputs
# every turn (it never rewrites local history to hold headroom's markers). The
# compressor above re-hashes and re-compresses the resent cold original on
# every subsequent turn, so a model that already retrieved a hash via
# ``headroom_retrieve`` is forced to re-retrieve it every turn (observed 236x
# thrash). agy has no call_id, so -- unlike the OpenAI/live_zone.rs path,
# which exempts by call_id (``live_zone.rs:2362-2384``) -- the exemption here
# keys on the retrieved HASH itself: any 24-hex-char token found in the args
# of a functionCall that references ``headroom_retrieve`` is treated as
# "already retrieved this turn" and its matching functionResponse leaf is
# left uncompressed.
_RETRIEVE_HASH_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{24}(?![0-9a-f])")


def _scan_hex_hashes(value: Any, hashes: set[str]) -> None:
    """Recursively collect 24-hex-char tokens from every STRING value in ``value``."""
    if isinstance(value, dict):
        for v in value.values():
            _scan_hex_hashes(v, hashes)
    elif isinstance(value, list):
        for v in value:
            _scan_hex_hashes(v, hashes)
    elif isinstance(value, str):
        hashes.update(_RETRIEVE_HASH_RE.findall(value.lower()))


def _requested_agy_fr_mode() -> str:
    """Normalize the REQUESTED functionResponse mode from the environment.

    ``HEADROOM_AGY_FR_MODE`` selects ``lossless`` or ``ccr`` (default);
    unset/invalid values fall back to ``ccr``. Single source of truth
    shared by ``_resolve_agy_fr_mode`` (the downgrade decision) and the
    wrap-agy downgrade warning (``headroom.cli.wrap._maybe_warn_agy_ccr_downgrade``)
    so the two cannot drift.
    """
    mode = (os.environ.get("HEADROOM_AGY_FR_MODE") or "ccr").strip().lower()
    if mode not in ("ccr", "lossless"):
        return "ccr"
    return mode


def _fr_marker_token_floor(tokenizer: Any) -> int:
    """Derive the per-leaf compression floor from the CCR marker overhead.

    A compressed leaf ships the marker in its place, so the net saving is
    ``leaf_tokens - marker_tokens``. We only compress when that saving
    exceeds the marker's own cost (``_FR_MARKER_MIN_RATIO`` x marker).
    """
    sample = _FR_CCR_MARKER_TEMPLATE.format(hash="0" * _FR_CCR_HASH_LEN)
    marker_tokens = tokenizer.count_text(sample)
    return int(max(1, marker_tokens * _FR_MARKER_MIN_RATIO))


def _compress_fr_leaf(
    leaf: str,
    mode: str,
    tokenizer: Any,
    store: Any,
    tool_name: str | None,
) -> str:
    """Deterministically compress a single functionResponse string leaf.

    ``ccr``: cache the ORIGINAL and ship a hash marker. The hash defaults to
    SHA-256(original)[:24] -> an identical original yields identical marker
    bytes every turn (deterministic + cache-coherent). Idempotent: an
    already-compressed marker is returned unchanged.
    ``lossless``: format-native reversible compaction (no marker).
    """
    if mode == "ccr":
        # Idempotency guard: never re-wrap our own marker.
        if leaf.startswith(_FR_CCR_MARKER_PREFIX):
            return leaf
        marker_body_tokens = tokenizer.count_text(
            _FR_CCR_MARKER_TEMPLATE.format(hash="0" * _FR_CCR_HASH_LEN)
        )
        # Default hash = SHA-256(original)[:24] (DETERMINISTIC). Do NOT pass
        # explicit_hash -- determinism must come from the content itself so
        # the same leaf maps to the same marker bytes across turns.
        hash_key = store.store(
            leaf,
            _FR_CCR_MARKER_TEMPLATE,
            original_tokens=tokenizer.count_text(leaf),
            compressed_tokens=marker_body_tokens,
            tool_name=tool_name,
        )
        return _FR_CCR_MARKER_TEMPLATE.format(hash=hash_key)
    # lossless: reversible, deterministic, self-verified smaller-or-unchanged.
    from headroom.transforms.lossless_compaction import compact_lossless

    return compact_lossless(leaf, "text")


def _collect_retrieved_hashes(contents: list[dict]) -> set[str]:
    """Collect CCR hashes the model already retrieved via ``headroom_retrieve``.

    Scans every ``functionCall`` part across ALL of ``contents`` (any
    entry, not just the tail -- agy resends the full history every turn)
    for calls that reference ``headroom_retrieve`` (bare name, or the
    generic MCP dispatch shape e.g. ``call_mcp_tool`` whose args mention
    ``headroom_retrieve``), then recursively pulls every 24-hex-char
    token out of that call's ``args``. See the WU2-A comment above
    ``_RETRIEVE_HASH_RE`` for why this keys on the hash rather than a
    call_id (agy has none).
    """
    hashes: set[str] = set()
    for content in contents:
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            fc = part.get("functionCall")
            if not isinstance(fc, dict):
                continue
            name = fc.get("name", "")
            args = fc.get("args") or {}
            if not (is_headroom_retrieve_name(name) or "headroom_retrieve" in json.dumps(args)):
                continue
            _scan_hex_hashes(args, hashes)
    return hashes


def _walk_fr_compress(
    value: Any,
    mode: str,
    tokenizer: Any,
    store: Any,
    floor: int,
    tool_name: str | None,
    stats: dict[str, int],
    retrieved_hashes: set[str],
) -> Any:
    """Recurse dict/list; compress every string leaf >= ``floor`` in place.

    Non-string scalars and sub-floor leaves are skipped. A leaf whose
    default CCR hash is in ``retrieved_hashes`` is exempt (WU2-A: the
    model already retrieved it this turn; re-compressing it would force
    an endless re-retrieve loop). Mutates containers in place and
    returns ``value`` for convenient reassignment.
    """
    if isinstance(value, dict):
        for k, v in value.items():
            value[k] = _walk_fr_compress(
                v, mode, tokenizer, store, floor, tool_name, stats, retrieved_hashes
            )
        return value
    if isinstance(value, list):
        for i, v in enumerate(value):
            value[i] = _walk_fr_compress(
                v, mode, tokenizer, store, floor, tool_name, stats, retrieved_hashes
            )
        return value
    if isinstance(value, str):
        try:
            leaf_tokens = tokenizer.count_text(value)
            if leaf_tokens < floor:
                return value
            if default_ccr_hash(value) in retrieved_hashes:
                return value  # exempt: model already retrieved this hash (live_zone.rs parity)
            new_leaf = _compress_fr_leaf(value, mode, tokenizer, store, tool_name)
        except Exception:
            # Broad by design: one malformed leaf must not abort the whole
            # walk and strand earlier leaves half-compressed in the shared
            # `contents` object. Reachable from untrusted tool output -- e.g.
            # a lone UTF-16 surrogate makes default_ccr_hash's str.encode()
            # raise UnicodeEncodeError. Leave this one leaf verbatim, keep going.
            logger.warning(
                "agy FR: leaving one functionResponse leaf uncompressed (failed to hash/compress)",
                exc_info=True,
            )
            return value
        if new_leaf != value:
            new_tokens = tokenizer.count_text(new_leaf)
            # Guard: only accept an actual reduction (lossless may no-op).
            if new_tokens < leaf_tokens:
                stats["before"] += leaf_tokens
                stats["after"] += new_tokens
                stats["leaves"] += 1
                return new_leaf
        return value
    # Non-string scalar (int/float/bool/None): skipped, JSON shape preserved.
    return value


def compress_function_response_leaves(
    contents: list[dict],
    mode: str,
    tokenizer: Any,
    store: Any,
) -> tuple[int, int, int]:
    """Uniformly compress functionResponse string leaves across ALL entries.

    Walks every ``contents[]`` entry (historical + tail), every ``parts[]``
    entry, and every ``functionResponse`` part (an entry may carry several),
    recursing into the ``response`` value to compress its large string leaves
    in place. ``functionCall`` parts are never touched; JSON shape and
    functionCall/functionResponse pairing are preserved.

    EXEMPTION: a functionResponse named ``headroom_retrieve`` (bare or
    MCP-namespaced, see ``is_headroom_retrieve_name``) is left untouched.
    That tool's own output is the just-resolved ORIGINAL of a marker the
    model expanded; re-compressing it back into the same marker is a
    self-defeating loop (the OpenAI path already exempts this -- see
    ``headroom_retrieve_call_ids`` in ``live_zone.rs``).

    Returns ``(fr_tokens_before, fr_tokens_after, leaves_compressed)`` over
    the leaves that were actually compressed.
    """
    floor = _fr_marker_token_floor(tokenizer)
    stats: dict[str, int] = {"before": 0, "after": 0, "leaves": 0}
    retrieved_hashes = _collect_retrieved_hashes(contents)
    for content in contents:
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            fr = part.get("functionResponse")
            if not isinstance(fr, dict):
                continue
            response = fr.get("response")
            if response is None:
                continue
            if is_headroom_retrieve_name(fr.get("name")):
                continue
            fr["response"] = _walk_fr_compress(
                response, mode, tokenizer, store, floor, fr.get("name"), stats, retrieved_hashes
            )
    return stats["before"], stats["after"], stats["leaves"]
