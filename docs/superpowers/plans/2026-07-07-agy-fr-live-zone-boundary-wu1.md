# agy FR structural live-zone boundary (WU1 / 37g.11) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the agy ccr thrash by excluding the hot recent functionResponse frame from compression — recent tool outputs the model still needs stay verbatim; only cold history compresses.

**Architecture:** Extract two pure functions (`fr_live_zone_start`, `should_compress_leaf`) and thread a `live_zone_start` index into the existing FR leaf-walker so entries at/after the boundary are left verbatim. Prototype in Python (no Rust planner yet). Boundary is structural (latest genuine user-text turn), not a tunable N.

**Tech Stack:** Python 3.11, existing `headroom/proxy/handlers/gemini.py`, pytest.

## Global Constraints

- Branch `agy1044`. Design: `docs/superpowers/specs/2026-07-07-agy-ccr-live-zone-parity-design.md` §4A (design-review-gate PASSED 5/5).
- Lint/type on changed files ONLY: `uvx ruff@0.15.17 check <files>` + `uvx ruff@0.15.17 format --check <files>`; mypy 1.20.2.
- Tests run ISOLATED: `HOME=<scratch>/testhome uv run python -m pytest <file> -q`. NEVER the full/live suite (crashes the local :8787 proxy).
- Boundary is STRUCTURAL turn position — NOT a tunable N, NOT a cache-marker position.
- Do not touch `functionCall` parts, the `is_headroom_retrieve_name` exemption (gemini.py:988), or the marker template. JSON shape + functionCall/functionResponse pairing preserved.
- Frozen floor = 0 for WU1 (request-side `cachedContent` is not parsed in this handler; raising the floor is a later WU).

---

### Task 1: Resolve `live_zone_only` semantics + boundary direction (spike + decision, no code)

**Files:**
- Read: `headroom/transforms/compression_policy.py:70-232`; `crates/headroom-core/src/transforms/live_zone.rs:515-522,618,944`; `crates/headroom-core/src/compression_policy.rs:31-32,219-234`
- Modify (append a short "Implementation note"): `docs/superpowers/specs/2026-07-07-agy-ccr-live-zone-parity-design.md`

**Why this is first:** `CompressionPolicy.live_zone_only` (compression_policy.py:91) means *"downstream MUST NOT modify bytes OUTSIDE the live zone"* (a cache-stability freeze of the cached prefix). 4A needs the **inverse intent**: do NOT compress the HOT recent frame; DO compress cold history. These are different axes — reconcile before coding so the boundary direction is not inverted.

- [ ] **Step 1:** Read the four sources above. Confirm: (a) Anthropic excludes `HOT_ZONE_BLOCK_TYPES` from compression *within* the latest user frame (verbatim hot), and (b) `live_zone_only` is a prefix-freeze for cache stability, orthogonal to (a).
- [ ] **Step 2:** Record the decision in the design doc's implementation note. **Recommended resolution (adopt unless the read contradicts it):** WU1's boundary **unconditionally** excludes the hot frame (entries `>= live_zone_start`) from FR compression — this is the thrash fix and is correct for ALL auth modes (compressing hot content that induces the thrash is never desirable). `live_zone_only` is NOT the lever for hot-exclusion; it is a separate cold-side cache-stability concern deferred to a later WU. So WU1 does **not** consume `policy_for_mode` for the hot boundary; it applies the structural hot-frame exclusion directly. (This corrects the iter-2 "route through live_zone_only" framing, which conflated the two axes.)
- [ ] **Step 3: Commit** the design-doc note.

```bash
git add docs/superpowers/specs/2026-07-07-agy-ccr-live-zone-parity-design.md
git commit -m "docs(agy): WU1 note — hot-frame exclusion is unconditional, distinct from live_zone_only cache-freeze"
```

---

### Task 2: `fr_live_zone_start` pure function (boundary computation)

**Files:**
- Modify: `headroom/proxy/handlers/gemini.py` (add module-level function near the other FR helpers, ~line 106)
- Test: `tests/test_agy_fr_live_zone_boundary.py` (create)

**Interfaces:**
- Produces: `fr_live_zone_start(contents: list) -> int` — index into `contents[]` of the latest genuine USER-TEXT turn (an entry with `role == "user"` whose `parts` contain a `text` part and NO `functionResponse` part). Entries at/after this index are the hot frame (verbatim). Returns `0` when no such turn exists (compress nothing — safest). Tool turns (`role == "user"` carrying only `functionResponse`) are skipped so the current model/tool exchange stays hot.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agy_fr_live_zone_boundary.py
from headroom.proxy.handlers.gemini import fr_live_zone_start


def _user_text(t):
    return {"role": "user", "parts": [{"text": t}]}


def _model_text(t):
    return {"role": "model", "parts": [{"text": t}]}


def _tool_result(name, payload):
    return {"role": "user", "parts": [{"functionResponse": {"name": name, "response": {"content": payload}}}]}


def test_boundary_is_latest_user_text_turn():
    contents = [
        _user_text("read the config"),          # 0
        _model_text("ok, reading"),              # 1
        _tool_result("read_file", "BIG"),        # 2  (tool turn, role=user)
        _user_text("what is KEY_0731?"),         # 3  <- latest genuine user text
        _model_text("checking"),                 # 4
        _tool_result("read_file", "SMALL"),      # 5  (hot tool turn)
    ]
    assert fr_live_zone_start(contents) == 3


def test_tool_turn_is_not_a_user_text_turn():
    contents = [_user_text("go"), _tool_result("read_file", "X")]
    assert fr_live_zone_start(contents) == 0  # only turn 0 is genuine user text


def test_empty_contents_returns_zero():
    assert fr_live_zone_start([]) == 0


def test_no_user_text_returns_zero():
    contents = [_tool_result("read_file", "X"), _model_text("hi")]
    assert fr_live_zone_start(contents) == 0


def test_single_user_text_turn():
    assert fr_live_zone_start([_user_text("only")]) == 0


def test_model_role_fr_entry_ignored_for_boundary():
    # An FR-bearing entry with role=='model' must not be treated as a user turn.
    contents = [_user_text("go"), {"role": "model", "parts": [{"functionResponse": {"name": "x", "response": {}}}]}]
    assert fr_live_zone_start(contents) == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `HOME=$SCRATCH/testhome uv run python -m pytest tests/test_agy_fr_live_zone_boundary.py -q`
Expected: FAIL — `ImportError: cannot import name 'fr_live_zone_start'`

- [ ] **Step 3: Implement**

```python
# headroom/proxy/handlers/gemini.py  (module level, after _resolve_agy_fr_mode)
def fr_live_zone_start(contents: list) -> int:
    """Index of the latest genuine user-text turn in Gemini ``contents[]``.

    Entries at/after this index are the HOT frame the model responds against and
    are kept verbatim; earlier entries are cold history eligible for FR
    compression. A "genuine user-text turn" is ``role == "user"`` with at least
    one ``text`` part and NO ``functionResponse`` part (tool-result turns are
    also role=="user" but must NOT anchor the boundary). Returns 0 when none is
    found (compress nothing — safest, mirrors Anthropic latest_user_message_index
    with a 0 floor).
    """
    latest = 0
    for i, entry in enumerate(contents):
        if not isinstance(entry, dict) or entry.get("role") != "user":
            continue
        parts = entry.get("parts")
        if not isinstance(parts, list):
            continue
        has_text = any(isinstance(p, dict) and "text" in p for p in parts)
        has_fr = any(isinstance(p, dict) and "functionResponse" in p for p in parts)
        if has_text and not has_fr:
            latest = i
    return latest
```

- [ ] **Step 4: Run to verify it passes**

Run: `HOME=$SCRATCH/testhome uv run python -m pytest tests/test_agy_fr_live_zone_boundary.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add headroom/proxy/handlers/gemini.py tests/test_agy_fr_live_zone_boundary.py
git commit -m "feat(agy): fr_live_zone_start — structural hot-frame boundary for FR compression"
```

---

### Task 3: `should_compress_leaf` pure predicate

**Files:**
- Modify: `headroom/proxy/handlers/gemini.py` (module level, next to `fr_live_zone_start`)
- Test: `tests/test_agy_fr_live_zone_boundary.py` (extend)

**Interfaces:**
- Produces: `should_compress_leaf(entry_index: int, live_zone_start: int, leaf_tokens: int, floor: int) -> bool` — True iff the entry is cold (`entry_index < live_zone_start`) AND the leaf meets the token floor (`leaf_tokens >= floor`). Pure arithmetic.

- [ ] **Step 1: Write the failing test**

```python
from headroom.proxy.handlers.gemini import should_compress_leaf


def test_hot_entry_never_compresses():
    assert should_compress_leaf(entry_index=5, live_zone_start=3, leaf_tokens=9999, floor=100) is False


def test_entry_at_boundary_is_hot():
    assert should_compress_leaf(entry_index=3, live_zone_start=3, leaf_tokens=9999, floor=100) is False


def test_cold_entry_above_floor_compresses():
    assert should_compress_leaf(entry_index=2, live_zone_start=3, leaf_tokens=9999, floor=100) is True


def test_cold_entry_below_floor_skips():
    assert should_compress_leaf(entry_index=2, live_zone_start=3, leaf_tokens=50, floor=100) is False


def test_boundary_zero_compresses_nothing():
    assert should_compress_leaf(entry_index=0, live_zone_start=0, leaf_tokens=9999, floor=100) is False
```

- [ ] **Step 2: Run to verify it fails** — `ImportError: should_compress_leaf`
- [ ] **Step 3: Implement**

```python
def should_compress_leaf(
    entry_index: int, live_zone_start: int, leaf_tokens: int, floor: int
) -> bool:
    """True iff a leaf is in cold history AND meets the token floor."""
    return entry_index < live_zone_start and leaf_tokens >= floor
```

- [ ] **Step 4: Run to verify it passes** (5 new tests PASS)
- [ ] **Step 5: Commit**

```bash
git add headroom/proxy/handlers/gemini.py tests/test_agy_fr_live_zone_boundary.py
git commit -m "feat(agy): should_compress_leaf — cold-zone + floor predicate"
```

---

### Task 4: Thread the boundary through `_compress_agy_function_responses` / `_walk_fr_compress`

**Files:**
- Modify: `headroom/proxy/handlers/gemini.py:902-993` (`_walk_fr_compress`, `_compress_agy_function_responses`)
- Test: `tests/test_agy_functionresponse_compression.py` (extend — existing FR test file)

**Interfaces:**
- Consumes: `fr_live_zone_start`, `should_compress_leaf` (Tasks 2-3), existing `_compress_fr_leaf`, `_fr_marker_token_floor`.
- Changes: `_compress_agy_function_responses` computes `live_zone_start = fr_live_zone_start(contents)` once, and passes the current `entry_index` down so leaves in the hot frame are skipped. `_walk_fr_compress` gains an `in_cold_zone: bool` param (True when `entry_index < live_zone_start`); a leaf compresses only when `in_cold_zone` is True (the token-floor check stays inside the walker via `should_compress_leaf`).

- [ ] **Step 1: Write the failing test** (hot config leaf stays verbatim; cold one compresses)

```python
# tests/test_agy_functionresponse_compression.py  (add)
def test_hot_frame_functionresponse_not_compressed(monkeypatch):
    from headroom.proxy.handlers.gemini import GeminiHandlerMixin
    from headroom.tokenizers import get_tokenizer
    from headroom.cache.compression_store import get_compression_store

    BIG = "X" * 8000  # well above the marker token floor
    contents = [
        {"role": "user", "parts": [{"text": "read the config"}]},                     # 0 cold user
        {"role": "user", "parts": [{"functionResponse": {"name": "read_file",         # 1 COLD tool result
                                                          "response": {"content": BIG}}}]},
        {"role": "user", "parts": [{"text": "what is KEY_0731?"}]},                    # 2 latest user text -> boundary
        {"role": "user", "parts": [{"functionResponse": {"name": "read_file",         # 3 HOT tool result
                                                          "response": {"content": BIG}}}]},
    ]
    h = GeminiHandlerMixin()
    tok = get_tokenizer()
    store = get_compression_store()
    h._compress_agy_function_responses(contents, "ccr", tok, store)

    cold_leaf = contents[1]["parts"][0]["functionResponse"]["response"]["content"]
    hot_leaf = contents[3]["parts"][0]["functionResponse"]["response"]["content"]
    assert cold_leaf.startswith("[functionResponse compressed")  # cold -> marker
    assert hot_leaf == BIG                                        # hot -> verbatim
```

- [ ] **Step 2: Run to verify it fails**

Run: `HOME=$SCRATCH/testhome uv run python -m pytest tests/test_agy_functionresponse_compression.py::test_hot_frame_functionresponse_not_compressed -q`
Expected: FAIL — hot_leaf is a marker (current code compresses everything).

- [ ] **Step 3: Implement** — add `entry_index`/boundary threading

```python
# _compress_agy_function_responses (gemini.py:971-993) — replace the loop
    floor = self._fr_marker_token_floor(tokenizer)
    live_zone_start = fr_live_zone_start(contents)
    stats: dict[str, int] = {"before": 0, "after": 0, "leaves": 0}
    for entry_index, content in enumerate(contents):
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        in_cold_zone = entry_index < live_zone_start
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
                response, mode, tokenizer, store, floor, fr.get("name"), stats, in_cold_zone
            )
    return stats["before"], stats["after"], stats["leaves"]
```

```python
# _walk_fr_compress (gemini.py:902) — add in_cold_zone param + gate the leaf
    def _walk_fr_compress(
        self, value, mode, tokenizer, store, floor, tool_name, stats, in_cold_zone
    ):
        if isinstance(value, dict):
            for k, v in value.items():
                value[k] = self._walk_fr_compress(
                    v, mode, tokenizer, store, floor, tool_name, stats, in_cold_zone
                )
            return value
        if isinstance(value, list):
            for i, v in enumerate(value):
                value[i] = self._walk_fr_compress(
                    v, mode, tokenizer, store, floor, tool_name, stats, in_cold_zone
                )
            return value
        if isinstance(value, str):
            leaf_tokens = tokenizer.count_text(value)
            if not should_compress_leaf(0 if in_cold_zone else 1, 1, leaf_tokens, floor):
                # 0<1 (cold) passes the index gate; 1<1 (hot) fails it.
                return value
            new_leaf = self._compress_fr_leaf(value, mode, tokenizer, store, tool_name)
            if new_leaf != value:
                new_tokens = tokenizer.count_text(new_leaf)
                if new_tokens < leaf_tokens:
                    stats["before"] += leaf_tokens
                    stats["after"] += new_tokens
                    stats["leaves"] += 1
                    return new_leaf
            return value
        return value
```

> Note: the `should_compress_leaf(0 if in_cold_zone else 1, 1, ...)` call reuses the pure predicate so the floor + zone logic lives in one tested place. (If a reviewer prefers, pass `entry_index`/`live_zone_start` down explicitly instead of the 0/1 encoding — behaviorally identical; keep whichever the surrounding code reads more clearly.)

- [ ] **Step 4: Run to verify it passes** (new test PASS; then run the whole FR file)

Run: `HOME=$SCRATCH/testhome uv run python -m pytest tests/test_agy_functionresponse_compression.py -q`
Expected: PASS (existing tests + new one). If an existing test compressed a leaf that is now in the hot frame, update that fixture to place the leaf in cold history (index < the latest user-text turn) — the intent of those tests is "large leaf compresses," which still holds in the cold zone.

- [ ] **Step 5: Commit**

```bash
git add headroom/proxy/handlers/gemini.py tests/test_agy_functionresponse_compression.py
git commit -m "feat(agy): exclude hot recent functionResponse frame from FR compression (thrash fix)"
```

---

### Task 5: WU1 cache-invariant test (cold leaf byte-identical across turns)

**Files:**
- Test: `tests/test_agy_functionresponse_compression.py` (extend)

**Interfaces:**
- Consumes: `_compress_agy_function_responses`. Asserts a cold leaf's compressed marker bytes are identical across two independent turns (deterministic `SHA-256(original)[:24]`), preserving WU1's cache invariant.

- [ ] **Step 1: Write the test**

```python
def test_cold_leaf_marker_is_byte_stable_across_turns():
    from headroom.proxy.handlers.gemini import GeminiHandlerMixin
    from headroom.tokenizers import get_tokenizer
    from headroom.cache.compression_store import get_compression_store

    BIG = "Y" * 8000
    def mk():
        return [
            {"role": "user", "parts": [{"functionResponse": {"name": "read_file",
                                                             "response": {"content": BIG}}}]},  # 0 cold
            {"role": "user", "parts": [{"text": "later question"}]},                            # 1 boundary
        ]
    h = GeminiHandlerMixin(); tok = get_tokenizer(); store = get_compression_store()
    a = mk(); b = mk()
    h._compress_agy_function_responses(a, "ccr", tok, store)
    h._compress_agy_function_responses(b, "ccr", tok, store)
    leaf_a = a[0]["parts"][0]["functionResponse"]["response"]["content"]
    leaf_b = b[0]["parts"][0]["functionResponse"]["response"]["content"]
    assert leaf_a == leaf_b and leaf_a.startswith("[functionResponse compressed")
```

- [ ] **Step 2: Run** — Expected PASS (deterministic hash).
- [ ] **Step 3: Commit**

```bash
git add tests/test_agy_functionresponse_compression.py
git commit -m "test(agy): cold-leaf marker byte-stability across turns (WU1 cache invariant)"
```

---

### Task 6: Parity test vs the Anthropic oracle + docstring update

**Files:**
- Modify: `headroom/proxy/handlers/gemini.py:79-90` (`_requested_agy_fr_mode` docstring — `ccr` now means live-zone boundary)
- Test: `tests/test_agy_fr_live_zone_boundary.py` (extend)

**Interfaces:**
- Consumes: `fr_live_zone_start`. Asserts the boundary decision matches the Anthropic rule's INTENT (latest user frame) on an equivalent shape. The Rust oracle is `find_latest_user_message_index` (`live_zone.rs:944`, test `respects_frozen_message_count` :1520); since it is not Python-callable here, the parity test encodes the oracle's expected index for a fixed shape and asserts `fr_live_zone_start` agrees — with a comment naming the Rust reference so a future divergence is caught deliberately.

- [ ] **Step 1: Write the parity test**

```python
def test_boundary_parity_with_anthropic_latest_user_frame():
    # Oracle: Anthropic find_latest_user_message_index (live_zone.rs:944) returns
    # the index of the latest genuine user turn. For this shape the latest user
    # text is at index 2; tool-result turns (role=user + functionResponse) do NOT
    # count, matching HOT_ZONE_BLOCK_TYPES exclusion semantics.
    contents = [
        {"role": "user", "parts": [{"text": "q1"}]},                                     # 0
        {"role": "user", "parts": [{"functionResponse": {"name": "read_file", "response": {}}}]},  # 1
        {"role": "user", "parts": [{"text": "q2"}]},                                     # 2 oracle result
        {"role": "model", "parts": [{"text": "a2"}]},                                    # 3
    ]
    assert fr_live_zone_start(contents) == 2
```

- [ ] **Step 2: Run** — Expected PASS.
- [ ] **Step 3: Update the docstring** (`_requested_agy_fr_mode`, gemini.py:81)

```python
    """Normalize the REQUESTED functionResponse mode from the environment.

    ``HEADROOM_AGY_FR_MODE`` selects ``ccr`` (default) or ``lossless``;
    unset/invalid values fall back to ``ccr``. NOTE: under ``ccr``, FR
    compression now applies a STRUCTURAL live-zone boundary
    (``fr_live_zone_start``) — the hot recent functionResponse frame is kept
    verbatim; only cold history compresses. Single source of truth shared by
    ``_resolve_agy_fr_mode`` and ``headroom.cli.wrap._maybe_warn_agy_ccr_downgrade``.
    """
```

- [ ] **Step 4: Commit**

```bash
git add headroom/proxy/handlers/gemini.py tests/test_agy_fr_live_zone_boundary.py
git commit -m "test(agy): boundary parity vs Anthropic oracle; doc: ccr now means live-zone"
```

---

### Task 7: Quality gates on changed files

- [ ] **Step 1:** `uvx ruff@0.15.17 check headroom/proxy/handlers/gemini.py tests/test_agy_fr_live_zone_boundary.py tests/test_agy_functionresponse_compression.py`
- [ ] **Step 2:** `uvx ruff@0.15.17 format --check <same files>`
- [ ] **Step 3:** `uv run mypy headroom/proxy/handlers/gemini.py` (expect: no new errors)
- [ ] **Step 4:** Run the two touched test files isolated; confirm green.
- [ ] **Step 5:** Close WU1: `bd close headroom-37g.11 --reason "4A structural live-zone boundary landed; hot-frame FR excluded from compression, cold history compresses; unit + cache-invariant + parity tests green."` Then hand to WU2 (37g.12, live-harness acceptance) which proves the thrash is actually stopped on fry.

---

## Self-Review

- **Spec coverage (§4A):** boundary pure fn ✓ (T2), predicate ✓ (T3), wiring ✓ (T4), AuthMode/policy question ✓ resolved as unconditional hot-exclusion (T1 — corrects the "route through live_zone_only" framing after finding the flag is a cache-freeze, not a hot-protect), frozen-floor=0 ✓ (constraint), pure-fn unit cases ✓ (T2/T3), cache-invariant ✓ (T5), parity ✓ (T6), docstring ✓ (T6). The live-harness acceptance (§6) is WU2 (37g.12), not WU1.
- **Placeholders:** none — all steps carry code/commands.
- **Type consistency:** `fr_live_zone_start(list)->int`, `should_compress_leaf(int,int,int,int)->bool`, `_walk_fr_compress(...)` gains one `in_cold_zone: bool` — used consistently across T2/T3/T4.
- **Open item surfaced to the human (not a placeholder):** Task 1's `live_zone_only` reconciliation changes the design's "route through policy_for_mode" line. If the human wants agy hot-exclusion to be auth-mode-conditional after all, that becomes a WU2+ refinement; WU1 ships the unconditional (safe, thrash-killing) boundary.
