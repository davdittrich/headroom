"""Gemini handler mixin for HeadroomProxy.

Contains all Google Gemini API handlers including format conversion utilities.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse

from headroom.cache.compression_store import default_ccr_hash
from headroom.ccr.tool_injection import is_headroom_retrieve_name
from headroom.copilot_auth import build_copilot_upstream_url
from headroom.proxy.auth_mode import classify_client
from headroom.proxy.compression_decision import CompressionDecision
from headroom.proxy.helpers import COMPRESSION_TIMEOUT_SECONDS, extract_tags
from headroom.proxy.outcome import RequestOutcome

logger = logging.getLogger("headroom.proxy")

DEFAULT_CLOUDCODE_API_URL = "https://cloudcode-pa.googleapis.com"
ANTIGRAVITY_DAILY_API_URL = "https://daily-cloudcode-pa.googleapis.com"

# ---------------------------------------------------------------------------
# WU1 (headroom-37g.1): uniform deterministic recoverable compression of agy
# functionResponse leaves.
#
# agy's per-turn bulk lives in ``functionResponse`` parts. Those entries carry
# non-text parts, so ``_gemini_contents_to_messages`` routes them into
# ``preserved_indices`` and ``_rebuild_gemini_contents`` restores them verbatim
# -- the text compressor never sees them. Only tiny residual text is compressed,
# it inflates, the revert guard fires, and tokens_saved collapses to 0 (PR
# #1044: "704 -> 718, reverting").
#
# We compress the large STRING leaves inside those parts with a DETERMINISTIC,
# IDEMPOTENT, RECOVERABLE transform applied UNIFORMLY to every functionResponse
# leaf (historical + tail). Because headroom is an in-flight MITM that never
# rewrites agy's LOCAL history, agy re-sends the ORIGINAL bytes each turn; a
# deterministic transform (same original -> identical bytes every turn) yields a
# byte-stable compressed prefix that re-hits the Cloud Code Assist server-side
# cache. Recoverability is mandatory: the model reads functionResponse back as
# its own prior tool results, so lossy summaries would corrupt multi-turn
# reasoning.
# ---------------------------------------------------------------------------

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


def _resolve_agy_fr_mode() -> str:
    """Resolve the functionResponse compression mode for an agy run.

    ``HEADROOM_AGY_FR_MODE`` selects ``ccr`` (default) or ``lossless``;
    ``lossless`` must be requested explicitly to opt out of savings. When
    ``ccr`` is in effect but the CCR retrieve listener is not wired for this
    run (``HEADROOM_AGY_RETRIEVE_WIRED`` != "1"), we must NOT ship
    unrecoverable markers -- downgrade to ``lossless`` (byte-recoverable / no-op).
    """
    mode = _requested_agy_fr_mode()
    if mode == "ccr" and os.environ.get("HEADROOM_AGY_RETRIEVE_WIRED") != "1":
        return "lossless"
    return mode


class GeminiHandlerMixin:
    """Mixin providing Gemini API handler methods for HeadroomProxy."""

    def _is_cloudcode_antigravity_request(
        self, body: dict[str, Any], headers: dict[str, str]
    ) -> bool:
        """Detect Pi/OpenClaw and agy antigravity requests routed via Cloud Code Assist.

        Detection paths (any one is sufficient):
        - body requestType == "agent"  (Pi/OpenClaw classic)
        - body userAgent == "antigravity"  (Pi/OpenClaw classic)
        - HTTP User-Agent header starts with "antigravity/"  (case-insensitive)
        - body model is an agent-model name (e.g. "gemini-3-flash-agent")
        - agy-shaped body: top-level model + project + request.contents present
        """
        user_agent = headers.get("user-agent", "").lower()
        body_user_agent = str(body.get("userAgent", "")).lower()
        model = str(body.get("model", ""))
        # Agent-model names carry "-agent" suffix (e.g. gemini-3-flash-agent)
        is_agent_model = model.endswith("-agent")
        # agy-shaped body confirmation: top-level project + request with contents.
        # Only meaningful together with is_agent_model; the body shape alone is shared
        # with regular Pi/OpenClaw traffic (CLOUDCODE_BODY has the same structure).
        request_block = body.get("request", {})
        is_agy_agent_body = is_agent_model and (
            bool(body.get("project"))
            and isinstance(request_block, dict)
            and bool(request_block.get("contents"))
        )
        return (
            body.get("requestType") == "agent"
            or body_user_agent == "antigravity"
            or user_agent.startswith("antigravity/")
            or is_agy_agent_body
        )

    def _resolve_cloudcode_base_url(
        self,
        is_antigravity: bool,
        original_host: str | None = None,  # reserved for T2 MITM dispatch; unused here
    ) -> str:
        """Resolve upstream base URL for Pi Cloud Code Assist / Antigravity traffic.

        Resolution order (first match wins):
        1. Antigravity path — env HEADROOM_ANTIGRAVITY_API_URL override, else corrected default.
           ``original_host`` (populated by the MITM CONNECT path in T2) is accepted here so
           the signature is stable; full host-preserving logic is wired in T2.
        2. Reverse-proxy path — ``CLOUDCODE_API_URL`` instance attr or DEFAULT_CLOUDCODE_API_URL.
        """
        if is_antigravity:
            override = os.environ.get("HEADROOM_ANTIGRAVITY_API_URL")
            return override.rstrip("/") if override else ANTIGRAVITY_DAILY_API_URL
        return getattr(self, "CLOUDCODE_API_URL", DEFAULT_CLOUDCODE_API_URL).rstrip("/")

    def _has_non_text_parts(self, content: dict) -> bool:
        """Check if a Gemini content entry has non-text parts.

        Non-text parts include:
        - inlineData: Base64-encoded images/media
        - fileData: File references (URI + MIME type)
        - functionCall: Function calls from model
        - functionResponse: Responses to function calls

        Args:
            content: A single Gemini content entry with 'parts' list.

        Returns:
            True if any part contains non-text data.
        """
        parts = content.get("parts", [])
        for part in parts:
            if any(
                key in part
                for key in ("inlineData", "fileData", "functionCall", "functionResponse")
            ):
                return True
        return False

    def _rebuild_gemini_contents(
        self,
        original_contents: list[dict],
        preserved_indices: set[int],
        preserved_contents: dict[int, dict],
        optimized_contents: list[dict],
    ) -> list[dict]:
        """Interleave preserved (non-text) entries back into optimized_contents at their
        original positions.

        preserved_indices uses original contents[] indices, but optimized_contents uses
        a different (shorter) index space because entries with no text parts were excluded
        from the messages[] sent for compression.  Using orig_idx directly to overwrite
        optimized_contents[orig_idx] corrupts or silently drops entries.

        This method walks original_contents in order, placing each position with either
        the preserved original (for non-text entries) or the next optimized text entry.
        """
        opt_iter = iter(optimized_contents)
        result: list[dict] = []
        for idx, content in enumerate(original_contents):
            had_text = any("text" in p for p in content.get("parts", []))
            if idx in preserved_indices:
                result.append(preserved_contents[idx])
                if had_text:
                    # Entry also produced a message; consume but discard the optimized version
                    next(opt_iter, None)
            else:
                opt_entry = next(opt_iter, None)
                if opt_entry is not None:
                    result.append(opt_entry)
                # else: dropped by compression — omit
        return result

    def _gemini_contents_to_messages(
        self,
        contents: list[dict],
        system_instruction: dict | None = None,
        *,
        include_function_responses: bool = False,
    ) -> tuple[list[dict], set[int]]:
        """Convert Gemini contents[] format to OpenAI messages[] format for optimization.

        Gemini format:
            contents: [{"role": "user", "parts": [{"text": "..."}]}]
            systemInstruction: {"parts": [{"text": "..."}]}

        OpenAI format:
            messages: [{"role": "user", "content": "..."}]

        When include_function_responses is True, functionResponse payloads are
        additionally emitted as ``role="tool"`` messages so waste-signal
        detection can see tool output (#819). That richer list is telemetry-only:
        entries with non-text parts stay in preserved_indices and are restored
        verbatim, so it must never be used as the compression input.

        Returns:
            Tuple of (messages, preserved_indices) where preserved_indices contains
            the indices of content entries that have non-text parts (images, function
            calls, etc.) and should not be compressed.
        """
        messages = []
        preserved_indices: set[int] = set()

        # Add system instruction as system message
        if system_instruction:
            parts = system_instruction.get("parts", [])
            text_parts = [p.get("text", "") for p in parts if "text" in p]
            if text_parts:
                messages.append({"role": "system", "content": "\n".join(text_parts)})

        # Convert contents to messages
        for idx, content in enumerate(contents):
            # Track content entries with non-text parts
            if self._has_non_text_parts(content):
                preserved_indices.add(idx)

            role = content.get("role", "user")
            # Map Gemini roles to OpenAI roles
            if role == "model":
                role = "assistant"

            parts = content.get("parts", [])
            text_parts = [p.get("text", "") for p in parts if "text" in p]

            if text_parts:
                messages.append({"role": role, "content": "\n".join(text_parts)})

            if include_function_responses:
                for part in parts:
                    if "functionResponse" not in part:
                        continue
                    payload = self._function_response_text(part["functionResponse"])
                    if payload:
                        messages.append({"role": "tool", "content": payload})

        return messages, preserved_indices

    @staticmethod
    def _function_response_text(function_response: dict) -> str:
        """Serialize a functionResponse payload for waste-signal parsing."""
        response = function_response.get("response")
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        try:
            return json.dumps(response, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(response)

    def _messages_to_gemini_contents(self, messages: list[dict]) -> tuple[list[dict], dict | None]:
        """Convert OpenAI messages[] format back to Gemini contents[] format.

        Returns:
            (contents, system_instruction) tuple
        """
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Extract as systemInstruction
                system_instruction = {"parts": [{"text": content}]}
            else:
                # Map OpenAI roles to Gemini roles
                gemini_role = "model" if role == "assistant" else "user"
                contents.append({"role": gemini_role, "parts": [{"text": content}]})

        return contents, system_instruction

    async def handle_gemini_generate_content(
        self,
        request: Request,
        model: str,
        upstream_base_url: str | None = None,
        provider_name: str = "gemini",
    ) -> Response | StreamingResponse:
        """Handle Gemini native /v1beta/models/{model}:generateContent endpoint.

        Gemini's native API differs from OpenAI:
        - Input: `contents[]` with `parts[]` instead of `messages`
        - System: `systemInstruction` instead of system message
        - Auth: `x-goog-api-key` header instead of `Authorization: Bearer`
        - Output: `candidates[].content.parts[].text`
        """
        from fastapi import HTTPException
        from fastapi.responses import JSONResponse, Response

        from headroom.proxy.helpers import MAX_REQUEST_BODY_SIZE, _read_request_json
        from headroom.tokenizers import get_tokenizer
        from headroom.utils import extract_user_query

        start_time = time.time()
        request_id = await self._next_request_id()

        # Check request body size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "message": f"Request body too large. Maximum size is {MAX_REQUEST_BODY_SIZE // (1024 * 1024)}MB",
                        "code": 413,
                    }
                },
            )

        # Parse request
        try:
            body = await _read_request_json(request)
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid request body: {e!s}",
                        "code": 400,
                    }
                },
            )

        contents = body.get("contents", [])

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        tags = extract_tags(headers)
        client = classify_client(headers)
        # PR-A5 (P5-49): strip internal x-headroom-* from upstream-bound
        # headers AFTER `_extract_tags` reads them. Memory user-id reads
        # `request.headers` below.
        from headroom.proxy.helpers import _strip_internal_headers, log_outbound_headers

        _pre_strip_count_gem = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        log_outbound_headers(
            forwarder="gemini_generate_content",
            stripped_count=_pre_strip_count_gem,
            request_id=request_id,
        )

        # Memory: Get user ID when memory is enabled. Reads `request.headers`
        # directly because `headers` was stripped of `x-headroom-*` (PR-A5).
        memory_user_id: str | None = None
        memory_request_ctx = None
        if self.memory_handler:
            memory_user_id = request.headers.get(
                "x-headroom-user-id",
                os.environ.get("USER", os.environ.get("USERNAME", "default")),
            )
            # Per-project memory routing (GH #462). Gemini's
            # ``systemInstruction`` field carries the system prompt;
            # ``extract_system_prompt`` doesn't know that shape, so we
            # pull it directly when present and fall back to the
            # request body for OpenAI/Anthropic-shaped payloads.
            from headroom.memory.storage_router import (
                RequestContext as _MemRequestContext,
            )
            from headroom.memory.storage_router import (
                extract_system_prompt as _extract_sys_prompt,
            )

            gemini_sys = body.get("systemInstruction") or body.get("system_instruction") or {}
            sys_text = ""
            if isinstance(gemini_sys, dict):
                parts = gemini_sys.get("parts") or []
                if isinstance(parts, list):
                    for p in parts:
                        if isinstance(p, dict):
                            t = p.get("text")
                            if isinstance(t, str):
                                sys_text += ("\n" if sys_text else "") + t
            if not sys_text:
                sys_text = _extract_sys_prompt(body)

            memory_request_ctx = _MemRequestContext(
                headers=dict(request.headers),
                system_prompt=sys_text,
                base_user_id=memory_user_id,
                project_root_override=(
                    getattr(self.memory_handler.config, "project_root_override", "") or None
                ),
            )

        # Canonical memory-injection gate (parallels Anthropic + OpenAI).
        # Pre-PR-this Gemini's memory site silently ignored
        # `x-headroom-bypass: true`, mutating request bytes under the
        # user's "don't touch my bytes" signal.
        from headroom.proxy.helpers import get_memory_injection_mode
        from headroom.proxy.memory_decision import MemoryDecision
        from headroom.proxy.memory_query import MemoryQuery

        memory_decision = MemoryDecision.decide(
            headers=request.headers,
            memory_handler=self.memory_handler,
            memory_user_id=memory_user_id,
            mode_name=get_memory_injection_mode(),
        )
        memory_decision.apply_to_tags(tags)

        # Rate limiting (use Gemini API key)
        if self.rate_limiter:
            rate_key = headers.get("x-goog-api-key", "default")[:20]
            allowed, wait_seconds = await self.rate_limiter.check_request(rate_key)
            if not allowed:
                await self.metrics.record_rate_limited(provider=provider_name)
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limited. Retry after {wait_seconds:.1f}s",
                )

        # Convert Gemini format to messages for optimization
        system_instruction = body.get("systemInstruction")
        messages, preserved_indices = self._gemini_contents_to_messages(
            contents, system_instruction
        )

        # Store original content entries that have non-text parts before compression
        preserved_contents = {idx: contents[idx] for idx in preserved_indices}

        # Early exit if ALL content has non-text parts (nothing to compress)
        if len(preserved_indices) == len(contents):
            # All content has non-text parts, skip compression entirely
            # Just forward the request as-is
            query_params = dict(request.query_params)
            is_streaming = query_params.get("alt") == "sse" or request.url.path.endswith(
                ":streamGenerateContent"
            )
            if upstream_base_url:
                url = build_copilot_upstream_url(upstream_base_url, request.url.path)
                if is_streaming:
                    url = url.replace(":generateContent", ":streamGenerateContent")
                if request.url.query:
                    url = f"{url}?{request.url.query}"
            else:
                url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:generateContent"
            if "key" in query_params and not upstream_base_url:
                url += f"?key={query_params['key']}"

            if is_streaming:
                if upstream_base_url:
                    stream_url = url
                    separator = "&" if "?" in stream_url else "?"
                    if "alt=" not in request.url.query:
                        stream_url = f"{stream_url}{separator}alt=sse"
                else:
                    stream_url = (
                        f"{self.GEMINI_API_URL}/v1beta/models/{model}:streamGenerateContent?alt=sse"
                    )
                if "key" in query_params and not upstream_base_url:
                    stream_url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:streamGenerateContent?key={query_params['key']}&alt=sse"
                return await self._stream_response(
                    stream_url,
                    headers,
                    body,
                    "gemini",
                    model,
                    request_id,
                    0,
                    0,
                    0,
                    [],
                    tags,
                    0,
                    outcome_provider=provider_name,
                )
            else:
                response = await self._retry_request("POST", url, headers, body)
                total_latency = (time.time() - start_time) * 1000
                total_input_tokens = 0
                output_tokens = 0
                cache_read_tokens = 0
                try:
                    resp_json = response.json()
                    usage = resp_json.get("usageMetadata", {})
                    total_input_tokens = usage.get("promptTokenCount", 0)
                    output_tokens = usage.get("candidatesTokenCount", 0)
                    cache_read_tokens = usage.get("cachedContentTokenCount", 0)
                except (json.JSONDecodeError, ValueError, KeyError, TypeError, AttributeError):
                    pass
                await self._record_request_outcome(
                    RequestOutcome(
                        request_id=request_id,
                        provider=provider_name,
                        model=model,
                        status_code=response.status_code,
                        original_tokens=total_input_tokens,
                        optimized_tokens=total_input_tokens,
                        output_tokens=output_tokens,
                        tokens_saved=0,
                        attempted_input_tokens=total_input_tokens,
                        cache_read_tokens=cache_read_tokens,
                        uncached_input_tokens=max(0, total_input_tokens - cache_read_tokens),
                        total_latency_ms=total_latency,
                        num_messages=len(contents),
                        tags=tags or {},
                        client=client,
                    )
                )
                response_headers = dict(response.headers)
                response_headers.pop("content-encoding", None)
                response_headers.pop("content-length", None)
                response_headers["x-headroom-tokens-before"] = str(total_input_tokens)
                response_headers["x-headroom-tokens-after"] = str(total_input_tokens)
                response_headers["x-headroom-tokens-saved"] = "0"
                response_headers["x-headroom-model"] = model
                if cache_read_tokens > 0:
                    response_headers["x-headroom-cached"] = "true"
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=response_headers,
                )

        # Token counting
        tokenizer = get_tokenizer(model)
        original_tokens = tokenizer.count_messages(messages)

        # Optimization
        transforms_applied: list[str] = []
        waste_signals_dict: dict[str, int] | None = None
        optimized_messages = messages
        optimized_tokens = original_tokens

        _compression_failed = False
        _decision = CompressionDecision.decide(
            headers=request.headers,
            config=self.config,
            usage_reporter=self.usage_reporter,
            messages=messages,
        )
        _decision.apply_to_tags(tags)
        if not _decision.should_compress:
            logger.info(
                f"[{request_id}] Compression skipped: reason={_decision.passthrough_reason}"
            )
        if _decision.should_compress:
            try:
                # Use OpenAI pipeline (similar message format)
                context_limit = self.openai_provider.get_context_limit(model)
                # Richer conversion incl. functionResponse payloads so tool
                # output reaches waste-signal detection (#819); telemetry-only.
                waste_messages, _ = self._gemini_contents_to_messages(
                    contents, system_instruction, include_function_responses=True
                )
                result = await self._run_compression_in_executor(
                    lambda: self.openai_pipeline.apply(
                        messages=messages,
                        model=model,
                        model_limit=context_limit,
                        context=extract_user_query(messages),
                        waste_messages=waste_messages,
                    ),
                    timeout=COMPRESSION_TIMEOUT_SECONDS,
                )
                if result.messages != messages:
                    optimized_messages = result.messages
                    transforms_applied = result.transforms_applied
                    # Use pipeline's token counts for consistency with pipeline logs
                    original_tokens = result.tokens_before
                    optimized_tokens = result.tokens_after
                if result.waste_signals:
                    waste_signals_dict = result.waste_signals.to_dict()
            except Exception as e:
                _compression_failed = True
                logger.warning(f"[{request_id}] Gemini optimization failed: {e}")

        # Guard: if "optimization" inflated tokens, revert to originals
        if optimized_tokens > original_tokens:
            logger.warning(
                f"[{request_id}] Optimization inflated tokens "
                f"({original_tokens} -> {optimized_tokens}), reverting to original messages"
            )
            optimized_messages = messages
            optimized_tokens = original_tokens
            transforms_applied = []

        tokens_saved = original_tokens - optimized_tokens
        optimization_latency = (time.time() - start_time) * 1000

        # Memory: inject context for Gemini requests.
        #
        # PR-B6: memory context auto-injects to the live-zone tail (the
        # latest user message) — never to the system / systemInstruction
        # field. The cache hot zone is sacrosanct (invariant I2). When
        # the memory handler is in ``MemoryMode.TOOL`` its
        # ``search_and_format_context`` returns ``None`` so nothing flows
        # in here.
        if memory_decision.inject:
            # Memory-handler is guaranteed present when inject=True.
            # Add a timeout wrapping (matches Anthropic + Responses) so
            # a slow memory backend can't stall Gemini requests — pre-
            # PR-this Gemini was the only handler without one.
            #
            # The append uses provider="openai" because Gemini reuses
            # OpenAI's user-message content shape after the proxy's
            # gemini-contents → messages → gemini-contents round-trip.
            # That's a real coupling, not a bug — `_append_to_latest_
            # user_tail` only knows two surface shapes; openai matches
            # the post-conversion structure exactly.
            try:
                if self.memory_handler.config.inject_context:
                    memory_context = await asyncio.wait_for(
                        self.memory_handler.search_and_format_context(
                            memory_user_id,
                            optimized_messages,
                            request_context=memory_request_ctx,
                            query=MemoryQuery.from_messages(optimized_messages),
                        ),
                        timeout=(self.config.anthropic_pre_upstream_memory_context_timeout_seconds),
                    )
                    if memory_context:
                        new_messages, bytes_appended = (
                            self.memory_handler._append_to_latest_user_tail(
                                optimized_messages,
                                memory_context,
                                provider="openai",
                            )
                        )
                        if bytes_appended > 0:
                            optimized_messages = new_messages
                            logger.info(
                                f"[{request_id}] Memory: Injected {bytes_appended} chars "
                                f"into latest user message tail for user {memory_user_id} (gemini)"
                            )
                        else:
                            logger.debug(
                                f"[{request_id}] Memory: no eligible user message; "
                                "skipped tail injection (gemini)"
                            )
            except Exception as e:
                logger.warning(f"[{request_id}] Memory injection failed (gemini): {e}")

        # Convert back to Gemini format if optimized
        if optimized_messages != messages:
            optimized_contents, optimized_system = self._messages_to_gemini_contents(
                optimized_messages
            )
            optimized_contents = self._rebuild_gemini_contents(
                contents, preserved_indices, preserved_contents, optimized_contents
            )
            body["contents"] = optimized_contents
            if optimized_system:
                body["systemInstruction"] = optimized_system
            elif "systemInstruction" in body:
                del body["systemInstruction"]

        # Check if streaming requested via query param
        query_params = dict(request.query_params)
        is_streaming = query_params.get("alt") == "sse" or request.url.path.endswith(
            ":streamGenerateContent"
        )

        # Build URL - model is extracted from path. Vertex publisher
        # routes use the request's full path under the Vertex base URL;
        # native Gemini uses the public Gemini API shape.
        if upstream_base_url:
            url = build_copilot_upstream_url(upstream_base_url, request.url.path)
            if is_streaming:
                url = url.replace(":generateContent", ":streamGenerateContent")
            if request.url.query:
                url = f"{url}?{request.url.query}"
        else:
            url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:generateContent"

        # Preserve API key in query params if present
        if "key" in query_params and not upstream_base_url:
            url += f"?key={query_params['key']}"

        try:
            if is_streaming:
                # For streaming, use streamGenerateContent endpoint
                if upstream_base_url:
                    stream_url = url
                    separator = "&" if "?" in stream_url else "?"
                    if "alt=" not in request.url.query:
                        stream_url = f"{stream_url}{separator}alt=sse"
                else:
                    stream_url = (
                        f"{self.GEMINI_API_URL}/v1beta/models/{model}:streamGenerateContent?alt=sse"
                    )
                if "key" in query_params and not upstream_base_url:
                    stream_url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:streamGenerateContent?key={query_params['key']}&alt=sse"

                return await self._stream_response(
                    stream_url,
                    headers,
                    body,
                    "gemini",
                    model,
                    request_id,
                    original_tokens,
                    optimized_tokens,
                    tokens_saved,
                    transforms_applied,
                    tags,
                    optimization_latency,
                    outcome_provider=provider_name,
                )
            else:
                response = await self._retry_request("POST", url, headers, body)
                total_latency = (time.time() - start_time) * 1000

                total_input_tokens = optimized_tokens  # fallback
                output_tokens = 0
                cache_read_tokens = 0
                try:
                    resp_json = response.json()
                    usage = resp_json.get("usageMetadata", {})
                    total_input_tokens = usage.get("promptTokenCount", optimized_tokens)
                    output_tokens = usage.get("candidatesTokenCount", 0)
                    # Gemini returns cachedContentTokenCount for context-cached tokens
                    # These are charged at 10-25% of the input price depending on model
                    cache_read_tokens = usage.get("cachedContentTokenCount", 0)
                except (KeyError, TypeError, AttributeError) as e:
                    logger.debug(
                        f"[{request_id}] Failed to extract cached tokens from Gemini response: {e}"
                    )

                uncached_input_tokens = max(0, total_input_tokens - cache_read_tokens)

                # Eligible-tracking is TODO for Gemini; pass the full
                # pre-compression request size as the fallback denominator.
                # This makes Gemini's contribution to the aggregate
                # active_savings_percent equal its whole-request ratio —
                # not ideal but coherent until per-part live-zone
                # tracking exists for this provider.
                #
                # Gemini reports read-side context-cache only via
                # ``cachedContentTokenCount``. There is no write counter
                # in the Gemini response; cache writes happen out-of-band
                # via the explicit Cache API. cache_write_* fields on the
                # outcome stay at their 0 defaults — the dataclass
                # handles "this provider doesn't have this concept"
                # without per-handler conditionals.
                outcome = RequestOutcome(
                    request_id=request_id,
                    provider=provider_name,
                    model=model,
                    status_code=response.status_code,
                    original_tokens=original_tokens,
                    optimized_tokens=total_input_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=tokens_saved,
                    attempted_input_tokens=total_input_tokens + tokens_saved,
                    cache_read_tokens=cache_read_tokens,
                    uncached_input_tokens=uncached_input_tokens,
                    total_latency_ms=total_latency,
                    overhead_ms=optimization_latency,
                    waste_signals=waste_signals_dict,
                    transforms_applied=tuple(transforms_applied),
                    num_messages=len(body.get("contents", [])),
                    tags=tags or {},
                    client=client,
                )
                await self._record_request_outcome(outcome)

                if tokens_saved > 0:
                    logger.info(
                        f"[{request_id}] Gemini {model}: {original_tokens:,} → {optimized_tokens:,} "
                        f"(saved {tokens_saved:,} tokens)"
                    )
                else:
                    logger.info(f"[{request_id}] Gemini {model}: {original_tokens:,} tokens")

                # Remove compression headers
                response_headers = dict(response.headers)
                response_headers.pop("content-encoding", None)
                response_headers.pop("content-length", None)

                # Inject Headroom compression metrics (for SaaS metering)
                response_headers["x-headroom-tokens-before"] = str(original_tokens)
                response_headers["x-headroom-tokens-after"] = str(optimized_tokens)
                response_headers["x-headroom-tokens-saved"] = str(tokens_saved)
                response_headers["x-headroom-model"] = model
                if transforms_applied:
                    from headroom.proxy.cost import header_safe_transforms

                    response_headers["x-headroom-transforms"] = ",".join(
                        header_safe_transforms(transforms_applied)
                    )
                if cache_read_tokens > 0:
                    response_headers["x-headroom-cached"] = "true"
                if _compression_failed:
                    response_headers["x-headroom-compression-failed"] = "true"

                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=response_headers,
                )
        except Exception as e:
            await self.metrics.record_failed(provider=provider_name)
            logger.error(f"[{request_id}] Gemini request failed: {type(e).__name__}: {e}")
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": "An error occurred while processing your request. Please try again.",
                        "code": 502,
                    }
                },
            )

    def _fr_marker_token_floor(self, tokenizer: Any) -> int:
        """Derive the per-leaf compression floor from the CCR marker overhead.

        A compressed leaf ships the marker in its place, so the net saving is
        ``leaf_tokens - marker_tokens``. We only compress when that saving
        exceeds the marker's own cost (``_FR_MARKER_MIN_RATIO`` x marker).
        """
        sample = _FR_CCR_MARKER_TEMPLATE.format(hash="0" * _FR_CCR_HASH_LEN)
        marker_tokens = tokenizer.count_text(sample)
        return max(1, marker_tokens * _FR_MARKER_MIN_RATIO)

    def _compress_fr_leaf(
        self,
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

    def _collect_retrieved_hashes(self, contents: list[dict]) -> set[str]:
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
                if not (
                    is_headroom_retrieve_name(name)
                    or "headroom_retrieve" in json.dumps(args)
                ):
                    continue
                _scan_hex_hashes(args, hashes)
        return hashes

    def _walk_fr_compress(
        self,
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
                value[k] = self._walk_fr_compress(
                    v, mode, tokenizer, store, floor, tool_name, stats, retrieved_hashes
                )
            return value
        if isinstance(value, list):
            for i, v in enumerate(value):
                value[i] = self._walk_fr_compress(
                    v, mode, tokenizer, store, floor, tool_name, stats, retrieved_hashes
                )
            return value
        if isinstance(value, str):
            leaf_tokens = tokenizer.count_text(value)
            if leaf_tokens < floor:
                return value
            if default_ccr_hash(value) in retrieved_hashes:
                return value  # exempt: model already retrieved this hash (live_zone.rs parity)
            new_leaf = self._compress_fr_leaf(value, mode, tokenizer, store, tool_name)
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

    def _compress_agy_function_responses(
        self,
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
        floor = self._fr_marker_token_floor(tokenizer)
        stats: dict[str, int] = {"before": 0, "after": 0, "leaves": 0}
        retrieved_hashes = self._collect_retrieved_hashes(contents)
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
                fr["response"] = self._walk_fr_compress(
                    response, mode, tokenizer, store, floor, fr.get("name"), stats, retrieved_hashes
                )
        return stats["before"], stats["after"], stats["leaves"]

    async def handle_google_cloudcode_stream(
        self,
        request: Request,
    ) -> StreamingResponse | JSONResponse:
        """Handle Pi/OpenClaw Google Cloud Code Assist and Antigravity streaming requests."""
        from fastapi.responses import JSONResponse

        from headroom.proxy.helpers import _read_request_json
        from headroom.tokenizers import get_tokenizer
        from headroom.utils import extract_user_query

        start_time = time.time()
        request_id = await self._next_request_id()

        try:
            body = await _read_request_json(request)
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid request body: {e!s}",
                        "code": 400,
                    }
                },
            )

        request_payload = body.get("request")
        if not isinstance(request_payload, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Invalid Cloud Code Assist request: missing request payload",
                        "code": 400,
                    }
                },
            )

        model = body.get("model", "unknown")
        contents = request_payload.get("contents", [])
        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        headers.pop("accept-encoding", None)
        tags = extract_tags(headers)
        # Note: streaming handlers delegate to _stream_response, which
        # does its own classify_client. No need to compute here.
        is_antigravity = self._is_cloudcode_antigravity_request(body, headers)
        # PR-A5 (P5-49): strip internal x-headroom-* from upstream-bound headers
        # AFTER `_extract_tags` and `is_cloudcode_antigravity` reads.
        from headroom.proxy.helpers import _strip_internal_headers, log_outbound_headers

        _pre_strip_count_cca = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        log_outbound_headers(
            forwarder="gemini_cloudcode_assist",
            stripped_count=_pre_strip_count_cca,
            request_id=request_id,
        )

        system_instruction = request_payload.get("systemInstruction")
        optimization_system_instruction = None if is_antigravity else system_instruction
        messages, preserved_indices = self._gemini_contents_to_messages(
            contents if isinstance(contents, list) else [], optimization_system_instruction
        )
        preserved_contents = {
            idx: contents[idx]
            for idx in preserved_indices
            if isinstance(contents, list) and idx < len(contents)
        }

        tokenizer = get_tokenizer(model)
        original_tokens = tokenizer.count_messages(messages) if messages else 0
        optimized_messages = messages
        optimized_tokens = original_tokens
        transforms_applied: list[str] = []

        _decision = CompressionDecision.decide(
            headers=request.headers,
            config=self.config,
            usage_reporter=self.usage_reporter,
            messages=messages,
        )
        _decision.apply_to_tags(tags)
        if not _decision.should_compress:
            logger.info(
                f"[{request_id}] Compression skipped: reason={_decision.passthrough_reason}"
            )
        if _decision.should_compress:
            try:
                context_limit = self.openai_provider.get_context_limit(model)
                # Richer conversion incl. functionResponse payloads so tool
                # output reaches waste-signal detection (#819); telemetry-only.
                waste_messages, _ = self._gemini_contents_to_messages(
                    contents, system_instruction, include_function_responses=True
                )
                result = await self._run_compression_in_executor(
                    lambda: self.openai_pipeline.apply(
                        messages=messages,
                        model=model,
                        model_limit=context_limit,
                        context=extract_user_query(messages),
                        waste_messages=waste_messages,
                    ),
                    timeout=COMPRESSION_TIMEOUT_SECONDS,
                )
                if result.messages != messages:
                    optimized_messages = result.messages
                    transforms_applied = result.transforms_applied
                    original_tokens = result.tokens_before
                    optimized_tokens = result.tokens_after
            except Exception as e:
                logger.warning(f"[{request_id}] Cloud Code Assist optimization failed: {e}")

        if optimized_tokens > original_tokens:
            logger.warning(
                f"[{request_id}] Cloud Code Assist optimization inflated tokens "
                f"({original_tokens} -> {optimized_tokens}), reverting to original messages"
            )
            optimized_messages = messages
            optimized_tokens = original_tokens
            transforms_applied = []

        # WU1 (headroom-37g.1): uniform deterministic recoverable compression of
        # agy functionResponse leaves. Runs INDEPENDENT of the text-pipeline
        # revert above so the per-turn tool-output bulk (which lives in preserved
        # functionResponse parts the text compressor never sees) is compressed
        # and counted even when the tiny residual text inflates and reverts.
        fr_before = fr_after = fr_leaves = 0
        if is_antigravity and _decision.should_compress and isinstance(contents, list):
            try:
                fr_mode = _resolve_agy_fr_mode()
                fr_store = None
                if fr_mode == "ccr":
                    from headroom.cache.compression_store import get_compression_store

                    fr_store = get_compression_store()
                fr_before, fr_after, fr_leaves = self._compress_agy_function_responses(
                    contents, fr_mode, tokenizer, fr_store
                )
                if fr_leaves:
                    logger.info(
                        f"[{request_id}] agy functionResponse compression: "
                        f"mode={fr_mode} leaves={fr_leaves} "
                        f"tokens {fr_before}->{fr_after} retrieve_wired="
                        f"{os.environ.get('HEADROOM_AGY_RETRIEVE_WIRED') == '1'}"
                    )
            except Exception as e:
                logger.warning(f"[{request_id}] agy functionResponse compression failed: {e}")

        if optimized_messages != messages:
            optimized_contents, optimized_system = self._messages_to_gemini_contents(
                optimized_messages
            )
            optimized_contents = self._rebuild_gemini_contents(
                contents if isinstance(contents, list) else [],
                preserved_indices,
                preserved_contents,
                optimized_contents,
            )
            request_payload["contents"] = optimized_contents
            if not is_antigravity:
                if optimized_system:
                    request_payload["systemInstruction"] = optimized_system
                elif "systemInstruction" in request_payload:
                    del request_payload["systemInstruction"]
        elif fr_leaves:
            # Text pipeline reverted (or produced no change) but functionResponse
            # leaves were compressed in place. Ship the mutated contents as-is:
            # original structure preserved, only FR string leaves replaced. Avoid
            # the messages<->contents round-trip (which collapses multi-part text).
            request_payload["contents"] = contents

        # Fold the functionResponse leaf delta into the accounting so the saving
        # ships and is recorded even when the text pipeline reverted. FR tokens
        # are disjoint from the text-pipeline counts (messages excludes
        # functionResponse), so this never double-counts the #819 waste path.
        original_tokens += fr_before
        optimized_tokens += fr_after
        tokens_saved = original_tokens - optimized_tokens
        optimization_latency = (time.time() - start_time) * 1000
        base_url = self._resolve_cloudcode_base_url(is_antigravity)
        stream_url = f"{base_url}/v1internal:streamGenerateContent"
        if request.url.query:
            stream_url = f"{stream_url}?{request.url.query}"

        return await self._stream_response(
            stream_url,
            headers,
            body,
            "gemini",
            model,
            request_id,
            original_tokens,
            optimized_tokens,
            tokens_saved,
            transforms_applied,
            tags,
            optimization_latency,
        )

    async def handle_gemini_stream_generate_content(
        self,
        request: Request,
        model: str,
    ) -> StreamingResponse | JSONResponse:
        """Handle Gemini streaming endpoint /v1beta/models/{model}:streamGenerateContent."""
        from fastapi.responses import JSONResponse

        from headroom.proxy.helpers import _read_request_json
        from headroom.tokenizers import get_tokenizer

        start_time = time.time()
        request_id = await self._next_request_id()

        # Parse request
        try:
            body = await _read_request_json(request)
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid request body: {e!s}",
                        "code": 400,
                    }
                },
            )

        contents = body.get("contents", [])

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        tags = extract_tags(headers)
        # Streaming variant — delegates to _stream_response which
        # classifies the client itself from headers.
        # PR-A5 (P5-49): strip internal x-headroom-* before forwarding upstream.
        from headroom.proxy.helpers import _strip_internal_headers, log_outbound_headers

        _pre_strip_count_gem_stream = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        log_outbound_headers(
            forwarder="gemini_stream_generate_content",
            stripped_count=_pre_strip_count_gem_stream,
            request_id=request_id,
        )

        # Token counting
        tokenizer = get_tokenizer(model)
        original_tokens = 0
        for content in contents:
            parts = content.get("parts", [])
            for part in parts:
                if "text" in part:
                    original_tokens += tokenizer.count_text(part["text"])

        optimization_latency = (time.time() - start_time) * 1000

        # Build URL with SSE param
        query_params = dict(request.query_params)
        url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:streamGenerateContent?alt=sse"
        if "key" in query_params:
            url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:streamGenerateContent?key={query_params['key']}&alt=sse"

        return await self._stream_response(
            url,
            headers,
            body,
            "gemini",
            model,
            request_id,
            original_tokens,
            original_tokens,
            0,  # tokens_saved
            [],  # transforms_applied
            tags,
            optimization_latency,
        )

    async def handle_gemini_count_tokens(
        self,
        request: Request,
        model: str,
        upstream_base_url: str | None = None,
        provider_name: str = "gemini",
    ) -> Response:
        """Handle Gemini /v1beta/models/{model}:countTokens endpoint with compression.

        This endpoint counts tokens AFTER applying compression, so users can see
        how many tokens they'll actually use after optimization.

        The request format is the same as generateContent:
            {"contents": [...], "systemInstruction": {...}}
        """
        from fastapi.responses import JSONResponse, Response

        from headroom.proxy.helpers import _read_request_json
        from headroom.tokenizers import get_tokenizer
        from headroom.utils import extract_user_query

        start_time = time.time()
        request_id = await self._next_request_id()

        # Parse request
        try:
            body = await _read_request_json(request)
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid request body: {e!s}",
                        "code": 400,
                    }
                },
            )

        contents = body.get("contents", [])

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        client = classify_client(headers)
        tags = extract_tags(headers)
        # PR-A5 (P5-49): strip internal x-headroom-* before forwarding upstream.
        from headroom.proxy.helpers import _strip_internal_headers, log_outbound_headers

        _pre_strip_count_gem_count = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        log_outbound_headers(
            forwarder="gemini_count_tokens",
            stripped_count=_pre_strip_count_gem_count,
            request_id=request_id,
        )

        # Convert Gemini format to messages for optimization
        system_instruction = body.get("systemInstruction")
        messages, preserved_indices = self._gemini_contents_to_messages(
            contents, system_instruction
        )

        # Store original content entries that have non-text parts before compression
        preserved_contents = {idx: contents[idx] for idx in preserved_indices}

        # Early exit if ALL content has non-text parts (nothing to compress)
        if len(preserved_indices) == len(contents):
            # All content has non-text parts, skip compression entirely
            # Just forward the countTokens request as-is
            if upstream_base_url:
                url = build_copilot_upstream_url(upstream_base_url, request.url.path)
                if request.url.query:
                    url = f"{url}?{request.url.query}"
            else:
                url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:countTokens"
            query_params = dict(request.query_params)
            if "key" in query_params and not upstream_base_url:
                url += f"?key={query_params['key']}"

            response = await self._retry_request("POST", url, headers, body)
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        # Token counting (original)
        tokenizer = get_tokenizer(model)
        original_tokens = tokenizer.count_messages(messages)

        # Apply compression using the same pipeline as generateContent
        transforms_applied: list[str] = []
        optimized_messages = messages

        # countTokens is the one Gemini handler that didn't pull tags
        # out of headers; sibling handlers do and thread them into the
        # outcome. Extract here so apply_to_tags below has a dict to
        # mutate and the outcome at end-of-call inherits the tag.
        tags = extract_tags(request.headers)
        _decision = CompressionDecision.decide(
            headers=request.headers,
            config=self.config,
            usage_reporter=self.usage_reporter,
            messages=messages,
        )
        _decision.apply_to_tags(tags)
        if not _decision.should_compress:
            logger.info(
                f"[{request_id}] Compression skipped: reason={_decision.passthrough_reason}"
            )
        if _decision.should_compress:
            try:
                context_limit = self.openai_provider.get_context_limit(model)
                result = await self._run_compression_in_executor(
                    lambda: self.openai_pipeline.apply(
                        messages=messages,
                        model=model,
                        model_limit=context_limit,
                        context=extract_user_query(messages),
                    ),
                    timeout=COMPRESSION_TIMEOUT_SECONDS,
                )
                if result.messages != messages:
                    optimized_messages = result.messages
                    transforms_applied = result.transforms_applied
            except Exception as e:
                logger.warning(f"[{request_id}] Gemini countTokens optimization failed: {e}")

        # Convert back to Gemini format for the API call
        if optimized_messages != messages:
            optimized_contents, optimized_system = self._messages_to_gemini_contents(
                optimized_messages
            )
            optimized_contents = self._rebuild_gemini_contents(
                contents, preserved_indices, preserved_contents, optimized_contents
            )
            body["contents"] = optimized_contents
            if optimized_system:
                body["systemInstruction"] = optimized_system
            elif "systemInstruction" in body:
                del body["systemInstruction"]

        # Build URL
        if upstream_base_url:
            url = build_copilot_upstream_url(upstream_base_url, request.url.path)
            if request.url.query:
                url = f"{url}?{request.url.query}"
        else:
            url = f"{self.GEMINI_API_URL}/v1beta/models/{model}:countTokens"

        # Preserve API key in query params if present
        query_params = dict(request.query_params)
        if "key" in query_params and not upstream_base_url:
            url += f"?key={query_params['key']}"

        try:
            response = await self._retry_request("POST", url, headers, body)
            total_latency = (time.time() - start_time) * 1000

            # Parse response to get token count
            compressed_tokens = 0
            try:
                resp_json = response.json()
                compressed_tokens = resp_json.get("totalTokens", 0)
            except (json.JSONDecodeError, ValueError) as e:
                logger.debug(f"[{request_id}] Failed to parse Gemini token count response: {e}")

            # Track stats
            tokens_saved = (
                max(0, original_tokens - compressed_tokens) if compressed_tokens > 0 else 0
            )

            # Fallback denominator (see comment on the main gemini
            # record_request site) — pre-comp request size.
            # countTokens is a sizing helper; it never generates output
            # tokens and never touches cache. The funnel handles the
            # "nothing to report" shape with all-zero cache defaults.
            await self._record_request_outcome(
                RequestOutcome(
                    request_id=request_id,
                    provider=provider_name,
                    model=model,
                    status_code=response.status_code,
                    original_tokens=original_tokens,
                    optimized_tokens=compressed_tokens,
                    output_tokens=0,
                    tokens_saved=tokens_saved,
                    attempted_input_tokens=compressed_tokens + tokens_saved,
                    total_latency_ms=total_latency,
                    transforms_applied=tuple(transforms_applied),
                    tags=tags,
                    client=client,
                )
            )

            if tokens_saved > 0:
                logger.info(
                    f"[{request_id}] Gemini countTokens {model}: {original_tokens:,} → {compressed_tokens:,} "
                    f"(saved {tokens_saved:,} tokens, transforms: {transforms_applied})"
                )
            else:
                logger.info(
                    f"[{request_id}] Gemini countTokens {model}: {compressed_tokens:,} tokens"
                )

            # Remove compression headers
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )
        except Exception as e:
            await self.metrics.record_failed(provider=provider_name)
            logger.error(f"[{request_id}] Gemini countTokens failed: {type(e).__name__}: {e}")
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": "An error occurred while processing your request. Please try again.",
                        "code": 502,
                    }
                },
            )
