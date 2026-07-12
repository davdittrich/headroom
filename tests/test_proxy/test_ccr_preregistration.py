"""PR-B redesign (headroom-23t, Defect-2) — deterministic session-lifetime
pre-registration of the ``headroom_retrieve`` tool in CACHE mode.

Mechanism under test: in cache mode the retrieval tool is forwarded on EVERY
request from turn 1 (a pure, stateless function of the client request), so the
forwarded tool list never flips mid-session (zero cache busts) and the skip
gate's cache-mode reason to exist disappears — the mutable tail then compresses
every turn through the already-tested tool-present path.

Seams:
  1. ``ccr_tool_preregistered`` frame-local in ``handle_anthropic_messages``.
  2. ``should_skip_ccr_request_compression`` returns False when preregistered.
  3. ``should_inject_ccr_tool(preregistered=...)``.
  4. ``apply_session_sticky_ccr_tool(force_register=...)``.
  5. call-site threading.

Token mode and non-cache mode MUST stay byte-identical to today (the new
keywords default to False and are only True in cache mode).
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.ccr.tool_injection import CCR_TOOL_NAME, create_ccr_tool_definition
from headroom.proxy.helpers import (
    _reset_session_ccr_tracker_for_test,
    apply_session_sticky_ccr_tool,
    get_session_ccr_tracker,
    serialize_tool_definition_canonical,
    should_inject_ccr_tool,
)
from headroom.proxy.server import ProxyConfig, create_app

_RAW_TRANSCRIPT = "\n".join(f"row {idx}: payload payload payload" for idx in range(80))


@pytest.fixture(autouse=True)
def _reset_tracker():
    _reset_session_ccr_tracker_for_test()
    yield
    _reset_session_ccr_tracker_for_test()


# ─────────────────────────────────────────────────────────────────────────────
# WU-1 — helper primitives (seams 3 & 4), pure-function unit coverage
# ─────────────────────────────────────────────────────────────────────────────


def _reference_should_inject(
    *, configured_inject_tool: bool, frozen_message_count: int, has_compressed_content: bool
) -> tuple[bool, bool]:
    """Byte-for-byte replica of the PRE-change ``should_inject_ccr_tool`` body.

    Used to pin that every ``preregistered=False`` row is identical to today.
    """
    inject_tool = configured_inject_tool
    if inject_tool and frozen_message_count > 0:
        inject_tool = False
    is_marker_override = not inject_tool and has_compressed_content
    return (inject_tool or is_marker_override), is_marker_override


@pytest.mark.parametrize("configured", [True, False])
@pytest.mark.parametrize("frozen", [0, 3])
@pytest.mark.parametrize("has_compressed", [True, False])
def test_should_inject_truth_table_preregistered_false_is_unchanged(
    configured, frozen, has_compressed
):
    """preregistered=False (the default) → byte-equal to pre-change behavior."""
    expected = _reference_should_inject(
        configured_inject_tool=configured,
        frozen_message_count=frozen,
        has_compressed_content=has_compressed,
    )
    got_default = should_inject_ccr_tool(
        configured_inject_tool=configured,
        frozen_message_count=frozen,
        has_compressed_content=has_compressed,
    )
    got_explicit = should_inject_ccr_tool(
        configured_inject_tool=configured,
        frozen_message_count=frozen,
        has_compressed_content=has_compressed,
        preregistered=False,
    )
    assert got_default == expected
    assert got_explicit == expected


@pytest.mark.parametrize("frozen", [0, 5])
@pytest.mark.parametrize("has_compressed", [True, False])
def test_should_inject_preregistered_true_configured_true_overrides_deferral(
    frozen, has_compressed
):
    """preregistered=True + configured=True → (True, False) regardless of frozen/marker.

    Pre-registration overrides the frozen-turn deferral WITHOUT flagging a
    marker-override (is_marker_override False), so the caller does not log a
    #1006 cache-miss override.
    """
    assert should_inject_ccr_tool(
        configured_inject_tool=True,
        frozen_message_count=frozen,
        has_compressed_content=has_compressed,
        preregistered=True,
    ) == (True, False)


@pytest.mark.parametrize("frozen", [0, 5])
@pytest.mark.parametrize("has_compressed", [True, False])
def test_should_inject_preregistered_true_configured_false_falls_through(frozen, has_compressed):
    """preregistered has NO effect without configured_inject_tool.

    (``--no-ccr`` disables pre-registration wholesale.)
    """
    expected = _reference_should_inject(
        configured_inject_tool=False,
        frozen_message_count=frozen,
        has_compressed_content=has_compressed,
    )
    assert (
        should_inject_ccr_tool(
            configured_inject_tool=False,
            frozen_message_count=frozen,
            has_compressed_content=has_compressed,
            preregistered=True,
        )
        == expected
    )


def _ccr_tools(tools):
    return [t for t in (tools or []) if (t.get("name") or "") == CCR_TOOL_NAME]


def test_force_register_injects_without_compression_or_prior_ccr():
    """force_register=True injects even with no fresh compression and no prior CCR."""
    session_id = "prereg-sess-1"
    assert get_session_ccr_tracker().has_done_ccr("anthropic", session_id) is False

    tools, injected = apply_session_sticky_ccr_tool(
        provider="anthropic",
        session_id=session_id,
        request_id="r1",
        existing_tools=[{"name": "lookup", "input_schema": {"type": "object"}}],
        has_compressed_content_this_turn=False,
        force_register=True,
    )
    assert injected is True
    assert len(_ccr_tools(tools)) == 1
    # Golden bytes recorded so a later sticky-replay path would match.
    assert get_session_ccr_tracker().has_done_ccr("anthropic", session_id) is True


def test_force_register_noop_when_client_already_has_tool():
    """Client (MCP) already provided headroom_retrieve → dedupe wins, no double-up."""
    session_id = "prereg-sess-2"
    client_tool = {
        "name": CCR_TOOL_NAME,
        "description": "client-provided",
        "input_schema": {"type": "object", "properties": {}},
    }
    tools, injected = apply_session_sticky_ccr_tool(
        provider="anthropic",
        session_id=session_id,
        request_id="r1",
        existing_tools=[client_tool],
        has_compressed_content_this_turn=False,
        force_register=True,
    )
    assert injected is False
    assert len(_ccr_tools(tools)) == 1  # exactly the client's, not duplicated


def test_force_register_bytes_match_sticky_first_time_golden():
    """Bytes injected via force_register == the sticky first-time golden bytes."""
    # Sticky first-time path (compression drove it).
    sticky_tools, _ = apply_session_sticky_ccr_tool(
        provider="anthropic",
        session_id="sticky-golden",
        request_id="r1",
        existing_tools=None,
        has_compressed_content_this_turn=True,
    )
    sticky_ccr = _ccr_tools(sticky_tools)[0]

    # force_register path.
    forced_tools, _ = apply_session_sticky_ccr_tool(
        provider="anthropic",
        session_id="forced-golden",
        request_id="r1",
        existing_tools=None,
        has_compressed_content_this_turn=False,
        force_register=True,
    )
    forced_ccr = _ccr_tools(forced_tools)[0]

    assert serialize_tool_definition_canonical(forced_ccr) == serialize_tool_definition_canonical(
        sticky_ccr
    )
    # And identical to the canonical constant.
    assert serialize_tool_definition_canonical(forced_ccr) == serialize_tool_definition_canonical(
        create_ccr_tool_definition("anthropic")
    )


def test_force_register_deterministic_across_turns_and_after_reset():
    """Repeated turns produce byte-identical tools, incl. after a tracker reset."""
    session_id = "prereg-determinism"
    client = [{"name": "lookup", "input_schema": {"type": "object"}}]

    def _one_turn():
        tools, _ = apply_session_sticky_ccr_tool(
            provider="anthropic",
            session_id=session_id,
            request_id="r",
            existing_tools=copy.deepcopy(client),
            has_compressed_content_this_turn=False,
            force_register=True,
        )
        return serialize_tool_definition_canonical(_ccr_tools(tools)[0])

    turn1 = _one_turn()
    turn2 = _one_turn()
    turn3 = _one_turn()
    assert turn1 == turn2 == turn3

    # Restart simulation: tracker state evaporates; pre-registration still
    # deterministically produces the same bytes (stateless by construction).
    _reset_session_ccr_tracker_for_test()
    assert get_session_ccr_tracker().has_done_ccr("anthropic", session_id) is False
    assert _one_turn() == turn1


# ─────────────────────────────────────────────────────────────────────────────
# Integration harness (seams 1, 2, 5) — drives the real Anthropic handler
# ─────────────────────────────────────────────────────────────────────────────


class _FakePrefixTracker:
    def __init__(self, frozen_count: int):
        self._frozen_count = frozen_count
        self._cached_token_count = 0
        self._last_original_messages: list[dict] = []
        self._last_forwarded_messages: list[dict] = []

    def get_frozen_message_count(self) -> int:
        return self._frozen_count

    def get_last_original_messages(self):  # noqa: ANN201
        return copy.deepcopy(self._last_original_messages)

    def get_last_forwarded_messages(self):  # noqa: ANN201
        return copy.deepcopy(self._last_forwarded_messages)

    def update_from_response(self, **kwargs):  # noqa: ANN003
        self._cached_token_count = kwargs.get("cache_read_tokens", 0) + kwargs.get(
            "cache_write_tokens", 0
        )
        self._last_original_messages = copy.deepcopy(
            kwargs.get("original_messages", kwargs.get("messages", []))
        )
        self._last_forwarded_messages = copy.deepcopy(kwargs.get("messages", []))
        return None


def _force_compression(monkeypatch) -> None:  # noqa: ANN001
    decision = SimpleNamespace(should_compress=True, passthrough_reason=None)
    decision.apply_to_tags = lambda tags: None
    monkeypatch.setattr(
        "headroom.proxy.handlers.anthropic.CompressionDecision.decide",
        lambda **kwargs: decision,
    )


def _disable_pipeline_extensions(proxy) -> None:  # noqa: ANN001
    proxy.pipeline_extensions.emit = lambda *args, **kwargs: SimpleNamespace(
        messages=kwargs.get("messages"),
        tools=kwargs.get("tools"),
        headers=kwargs.get("headers"),
        metadata=kwargs.get("metadata"),
    )


def _make_client(mode: str, *, ccr_inject_tool: bool = True) -> TestClient:
    config = ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=ccr_inject_tool,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        mode=mode,
    )
    return TestClient(create_app(config))


_COMPRESSED_DELTA = {
    "role": "user",
    # Marker-less on purpose: pre-registration injects the retrieval tool
    # independent of any compression marker, so no fixture marker is needed.
    # A fabricated ``hash=`` would otherwise trip today's #1006 marker-override
    # in TOKEN mode (a pre-existing path, unrelated to pre-registration) and
    # muddy the DoD-6 "token/non-cache unchanged" assertions.
    "content": "[100 items compressed to 10]",
}


def _drive_turn(
    client: TestClient,
    *,
    messages: list,
    tools: list | None,
    prev_original: list | None,
    prev_forwarded: list | None,
    session_id: str = "prereg-session",
    compressed_delta: dict | None = None,
) -> dict:
    """POST one /v1/messages turn; return the FORWARDED body (dict)."""
    proxy = client.app.state.proxy
    _disable_pipeline_extensions(proxy)

    frozen = len(prev_original) if prev_original else 0
    fake_tracker = _FakePrefixTracker(frozen_count=frozen)
    fake_tracker._last_original_messages = copy.deepcopy(prev_original or [])
    fake_tracker._last_forwarded_messages = copy.deepcopy(prev_forwarded or [])
    proxy.session_tracker_store.compute_session_id = (
        lambda request, model, messages, _sid=session_id: _sid
    )
    proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

    captured: dict = {}

    def _fake_apply(**kwargs):
        captured.setdefault("compression_calls", []).append(list(kwargs["messages"]))
        fz = kwargs.get("frozen_message_count") or 0
        # Realistic cache-delta contract: freeze the prefix byte-identical,
        # compress only the tail (indices >= frozen_message_count).
        out = list(kwargs["messages"][:fz]) + [
            copy.deepcopy(compressed_delta if compressed_delta is not None else _COMPRESSED_DELTA)
        ]
        return SimpleNamespace(
            messages=out,
            transforms_applied=["fake:code_aware"],
            timing={},
            tokens_before=400,
            tokens_after=40,
            waste_signals=None,
        )

    proxy.anthropic_pipeline.apply = _fake_apply

    async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
        captured["body"] = body
        captured["transforms"] = None
        return httpx.Response(
            200,
            json={
                "id": "msg_prereg",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {
                    "input_tokens": 40,
                    "output_tokens": 3,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        )

    proxy._retry_request = _fake_retry

    payload = {"model": "claude-sonnet-4-6", "max_tokens": 64, "messages": messages}
    if tools is not None:
        payload["tools"] = tools
    response = client.post(
        "/v1/messages",
        headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
        json=payload,
    )
    assert response.status_code == 200, response.text
    captured["compression_calls"] = captured.get("compression_calls", [])
    return captured


_CLIENT_TOOL = {
    "name": "Bash",
    "description": "Run a shell command",
    "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}},
}


def _tool_names(body: dict) -> list[str]:
    return [t.get("name") for t in body.get("tools", [])]


# ─── DoD 3: gate ─────────────────────────────────────────────────────────────


def test_cache_mode_preregistered_compresses_frozen_turn_and_injects_tool(monkeypatch):
    """Cache mode + client tools + frozen prefix → gate no longer skips.

    The mutable delta is compressed (self-perpetuation broken) AND
    ``headroom_retrieve`` is forwarded even though the client never sent it.
    """
    _force_compression(monkeypatch)
    prev_original = [{"role": "user", "content": "turn1"}]
    prev_forwarded = [{"role": "user", "content": "turn1"}]
    messages = prev_original + [{"role": "user", "content": _RAW_TRANSCRIPT}]

    with _make_client("cache") as client:
        captured = _drive_turn(
            client,
            messages=messages,
            tools=[copy.deepcopy(_CLIENT_TOOL)],
            prev_original=prev_original,
            prev_forwarded=prev_forwarded,
        )

    # Gate did NOT skip: pipeline.apply ran on the delta.
    assert len(captured["compression_calls"]) == 1
    body = captured["body"]
    # Tool pre-registered despite the client never sending it.
    assert CCR_TOOL_NAME in _tool_names(body)
    assert "Bash" in _tool_names(body)
    # Frozen prefix replayed byte-identical; compressed delta appended.
    assert body["messages"][0] == prev_forwarded[0]
    assert body["messages"][-1] == _COMPRESSED_DELTA


def test_cache_mode_tool_less_body_keeps_todays_deferral(monkeypatch):
    """Cache mode + NO client tools → pre-registration OFF → today's skip stands."""
    _force_compression(monkeypatch)
    prev_original = [{"role": "user", "content": "turn1"}]
    prev_forwarded = [
        {"role": "user", "content": "[compressed hash=abc123def456abc123def456]"},
    ]
    messages = prev_original + [{"role": "user", "content": _RAW_TRANSCRIPT}]

    with _make_client("cache") as client:
        captured = _drive_turn(
            client,
            messages=messages,
            tools=None,  # tool-less utility subrequest shape
            prev_original=prev_original,
            prev_forwarded=prev_forwarded,
        )

    body = captured["body"]
    # Deferral holds: no compression call, no tool injected.
    assert captured["compression_calls"] == []
    assert "tools" not in body
    assert CCR_TOOL_NAME not in _tool_names(body)


# ─── DoD 6: token / non-cache byte-identity ──────────────────────────────────


@pytest.mark.parametrize("mode", ["token", "off"])
def test_non_cache_modes_never_preregister_tool(mode, monkeypatch):
    """token + non-cache modes: pre-registration is inert (byte-identical to today).

    A frozen-prefix turn with a client tool list and no fresh markers must NOT
    gain ``headroom_retrieve`` — exactly today's behavior.
    """
    _force_compression(monkeypatch)
    prev_original = [{"role": "user", "content": "turn1"}]
    prev_forwarded = [{"role": "user", "content": "turn1"}]
    messages = prev_original + [{"role": "user", "content": _RAW_TRANSCRIPT}]

    with _make_client(mode) as client:
        captured = _drive_turn(
            client,
            messages=messages,
            tools=[copy.deepcopy(_CLIENT_TOOL)],
            prev_original=prev_original,
            prev_forwarded=prev_forwarded,
        )

    body = captured["body"]
    # No pre-registration outside cache mode: the retrieval tool is absent
    # unless the ordinary (non-preregistration) path injected it — and here it
    # cannot, because the fake compressed delta's markers are historical/absent
    # on the non-cache path.
    assert CCR_TOOL_NAME not in _tool_names(body)
    # The client's own tool is forwarded untouched (present, single copy).
    assert _tool_names(body).count("Bash") == 1


# ─── DoD 4: multi-turn frozen-prefix byte-identity (the cache-safety test) ────


def _strip_cache_control(messages: list) -> list:
    """Drop every cache_control marker (message- and block-level).

    The provider caches on message CONTENT, and ``normalize_message_cache_control``
    legitimately MOVES the single ephemeral breakpoint onto the newest block each
    turn — so raw forwarded bytes shift while the frozen content stays identical.
    Content-identity (this stripped view) is the real cache-safety invariant.
    """
    out = []
    for m in messages:
        m2 = {k: v for k, v in m.items() if k != "cache_control"}
        content = m2.get("content")
        if isinstance(content, list):
            m2["content"] = [
                {k: v for k, v in b.items() if k != "cache_control"} if isinstance(b, dict) else b
                for b in content
            ]
        out.append(m2)
    return out


def _user_block(text: str) -> dict:
    """A block-style user message (the shape that can carry cache_control)."""
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def test_multiturn_cache_mode_tools_and_prefix_byte_identical(monkeypatch):
    """>=4-turn cache-mode session: constant client tools, growing compressible tail.

    Uses block-style content (real Claude Code shape) so a real message-level
    cache_control breakpoint exists. Per turn assert:
      (i)   forwarded body["tools"] bytes identical across turns and contain
            headroom_retrieve from turn 1;
      (ii)  forwarded messages[:frozen] CONTENT-identical to the previous turn's
            forwarded prefix (modulo the single breakpoint normalize moves to the
            newest block — the content-keyed cache invariant);
      (iii) exactly one message-level cache_control breakpoint;
      (iv)  transforms_applied non-empty on a frozen turn with a compressible
            delta (self-perpetuation broken).
    """
    _force_compression(monkeypatch)
    client_tools = [copy.deepcopy(_CLIENT_TOOL)]

    forwarded_tools_bytes: list[bytes] = []
    prev_original: list | None = None
    prev_forwarded: list | None = None
    # Reconstruct the client's growing ORIGINAL transcript across turns.
    original_transcript: list = []

    with _make_client("cache") as client:
        for turn in range(5):
            # Each turn appends one block-style user message (fresh compressible tail).
            original_transcript = original_transcript + [
                _user_block(f"{_RAW_TRANSCRIPT}::turn{turn}")
            ]
            captured = _drive_turn(
                client,
                messages=copy.deepcopy(original_transcript),
                tools=copy.deepcopy(client_tools),
                prev_original=copy.deepcopy(prev_original) if prev_original else None,
                prev_forwarded=copy.deepcopy(prev_forwarded) if prev_forwarded else None,
                compressed_delta=_user_block(f"[compressed::turn{turn}]"),
            )
            body = captured["body"]

            # (i) tools present from turn 1 and byte-stable across turns.
            assert CCR_TOOL_NAME in _tool_names(body)
            assert "Bash" in _tool_names(body)
            forwarded_tools_bytes.append(json.dumps(body["tools"], sort_keys=True).encode())

            # (iii) exactly one message-level cache_control breakpoint.
            breakpoints = _count_message_cache_control(body["messages"])
            assert breakpoints == 1, f"turn {turn}: {breakpoints} breakpoints, expected 1"

            # (ii) frozen-prefix CONTENT identity vs previous forwarded prefix.
            if prev_forwarded is not None:
                frozen_n = len(prev_forwarded)
                assert _strip_cache_control(body["messages"][:frozen_n]) == _strip_cache_control(
                    prev_forwarded
                ), f"turn {turn}: forwarded frozen prefix content diverged from last turn"
                # (iv) a compressible delta existed → compression actually ran.
                assert len(captured["compression_calls"]) == 1

            # Advance the simulated session lineage.
            prev_original = copy.deepcopy(original_transcript)
            prev_forwarded = copy.deepcopy(body["messages"])

    # (i) tools byte-identical on every turn.
    assert len(set(forwarded_tools_bytes)) == 1, "forwarded tool bytes flipped mid-session"


def _count_message_cache_control(messages: list) -> int:
    n = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict) and blk.get("cache_control") is not None:
                    n += 1
        if isinstance(m, dict) and m.get("cache_control") is not None:
            n += 1
    return n


# ─── DoD 5: concurrency (frame-local invariant) ──────────────────────────────


def test_concurrent_cache_and_token_requests_no_cross_contamination(monkeypatch):
    """>=8 threads mixing preregistered-cache-mode and token-mode shapes.

    The pre-registration flag is a coroutine-frame local, never shared state.
    Token-mode outputs must never gain headroom_retrieve from a concurrent
    cache-mode request. Each request drives an ISOLATED client so the per-call
    harness monkeypatches (retry/tracker) cannot race across threads — the only
    shared state exercised is the process-wide production singletons (router,
    session CCR tracker), which is exactly the frame-local invariant under test.
    """
    _force_compression(monkeypatch)
    prev_original = [{"role": "user", "content": "turn1"}]
    prev_forwarded = [{"role": "user", "content": "turn1"}]
    messages = prev_original + [{"role": "user", "content": _RAW_TRANSCRIPT}]

    jobs = [("cache", i) for i in range(8)] + [("token", i) for i in range(8)]
    clients = {job: _make_client(job[0]) for job in jobs}
    for c in clients.values():
        c.__enter__()

    def _run(job: tuple[str, int]) -> tuple[str, list[str]]:
        kind, idx = job
        captured = _drive_turn(
            clients[job],
            messages=copy.deepcopy(messages),
            tools=[copy.deepcopy(_CLIENT_TOOL)],
            prev_original=copy.deepcopy(prev_original),
            prev_forwarded=copy.deepcopy(prev_forwarded),
            session_id=f"{kind}-sess-{idx}",
        )
        return kind, _tool_names(captured["body"])

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(_run, jobs))
    finally:
        for c in clients.values():
            c.__exit__(None, None, None)

    for kind, names in results:
        if kind == "cache":
            assert CCR_TOOL_NAME in names, "cache-mode request lost pre-registration"
        else:
            assert CCR_TOOL_NAME not in names, "token-mode request contaminated by cache-mode"
        assert names.count("Bash") == 1


# ─── DoD 7: spurious retrieve in a never-compressed session ───────────────────


def test_spurious_retrieve_unknown_hash_resolves_via_not_found_path():
    """Pre-registration's one new risk: the model can now SEE headroom_retrieve in
    a session that never compressed, and may call it with an unknown hash.

    Assert that resolves through the EXISTING not-found path (a graceful,
    unsuccessful tool_result) rather than crashing or fabricating content.
    """
    from headroom.ccr.response_handler import CCRResponseHandler

    unknown_hash = "deadbeefdeadbeefdeadbeef"  # valid 24-hex format, never stored
    response = {
        "id": "msg_spurious",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": CCR_TOOL_NAME,
                "input": {"hash": unknown_hash},
            }
        ],
    }

    handler = CCRResponseHandler()
    assert handler.has_ccr_tool_calls(response, "anthropic") is True

    ccr_calls, _other = handler._parse_ccr_tool_calls(response, "anthropic")
    assert len(ccr_calls) == 1

    result = handler._execute_retrieval(ccr_calls[0])  # must not raise
    assert result.success is False
    payload = json.loads(result.content)  # graceful, parseable not-found result
    assert payload["hash"] == unknown_hash
    assert "error" in payload  # explicit miss, not fabricated content
