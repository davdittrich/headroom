# agy ccr thrash — enforced `headroom_retrieve` + lossless floor (supersedes 37g.8, live-zone-parity, native-recovery)

<!-- date: 2026-07-07 -->
<!-- status: design-review-gate iter2 (iter1 = 5/5 NEEDS_REVISION: 13 blockers folded below). Reuses existing headroom infra per project philosophy. -->

## 0. Prior errors corrected

- **live-zone-parity (4A-old)** refuted: compressed the cold prefix + kept the
  live zone verbatim — inverse of the proven Rust arch (`live_zone.rs:643-683`).
- **native-recovery (3B)** refuted: `gemini.py:958` leaves `functionCall`
  uncompressed, so agy already held the path and `os.walk`ed anyway.
- **"agy is non-compliant"** was concluded from a PASSIVE marker suggestion.
  Enforcement was never attempted. When attempted (below), agy DOES retrieve.

## 1. Problem

Under ccr, agy `functionResponse` outputs compress to markers
(`_FR_CCR_MARKER_TEMPLATE`, gemini.py:64) recoverable via `headroom_retrieve`.
Clean-fry, instrumented: lossless converges (5-7 calls, ~14 s); ccr thrashes 3/3
(120 s timeout, 35-42 calls, no answer) — net-negative.

## 2. Root cause (verified)

ccr's contract is compress + `headroom_retrieve`; it works for Anthropic/OpenAI
because those models call the tool when it is merely offered. agy, given a
passive marker, `os.walk`s instead. The fix must **compel** the call. Gemini's
`toolConfig.functionCallingConfig` (mode=ANY) can — and the `cloudcode-pa
/v1internal` backend honors it (agy itself sends `mode=VALIDATED`).

## 3. Design (mechanism — corrected & reuse-grounded)

The real mechanism is NOT "a surgical toolConfig line" (iter-1 framing). It is:
**force the `call_mcp_tool` dispatcher + inject a routing hint into the live tail
+ constrain the dispatch sub-target + extend the retrieve exemption + snapshot/
restore agy's toolConfig + cap-then-release, all keyed by session state.** Each
piece reuses existing headroom infrastructure (§9).

### 3A. FLOOR (ship first, independent): lossless default

Flip the default in the SHARED source of truth `_requested_agy_fr_mode`
(gemini.py:87: `or "ccr"` → `or "lossless"`). `wrap._maybe_warn_agy_ccr_downgrade`
(wrap.py:945) calls the SAME helper, so parity holds automatically (gate-verified
by the Architect). Update the now-stale "ccr is default" docstrings/comments in
the SAME commit (gemini.py:81,96; wrap.py:81,927). **4A and 4B are mutually
exclusive per run:** lossless ships no markers, so 4B has nothing to force —
enabling enforcement requires re-opting into `ccr`. 4A is a standalone one-liner
ticket, sequenced first.

### 3B. PRIMARY: enforced recovery (only meaningful under `ccr`)

Enforcement is **intrinsic to `ccr`** (default-on when `mode==ccr` AND
`HEADROOM_AGY_RETRIEVE_WIRED==1` — ccr markers are worthless unless reliably
recoverable). `HEADROOM_AGY_FORCE_RETRIEVE` is retained ONLY as an internal
kill-switch/escape-hatch, never a second user knob that can desync from
`FR_MODE`.

**Wire location (corrected):** inject into `request_payload = body["request"]`
(gemini.py:1022), NOT top-level `body`. `toolConfig` is a sibling of
`contents`/`systemInstruction` there; `body` is forwarded to
`/v1internal:streamGenerateContent` (gemini.py:1185).

**Trigger (request-observable v1; response-scan is a deferred enhancement):**
fire when the request carries an **unretrieved, AUTHENTICATED** marker. A marker
hash is authenticated by **store membership** — `CompressionStore.retrieve(hash)
is not None` (compression_store.py:382) — NOT a bare `[a-f0-9]{24}` regex match
(closes the forgeable-unsalted-SHA spoof: planted tool-output hashes that are not
real store keys never trigger forcing). "Retrieved" = a `headroom_retrieve`
result for that hash already present (see the extended exemption below for how
dispatcher-wrapped retrieves are recognized).

**Enforcement action (per forced turn):**
1. **Snapshot** agy's original `request_payload["toolConfig"]` (its `VALIDATED`
   config) into session state (§9 session store), if not already saved.
2. Set `request_payload["toolConfig"] = {"functionCallingConfig": {"mode":
   "ANY", "allowedFunctionNames": ["call_mcp_tool"]}}` — `headroom_retrieve` is
   undeclared (reachable only via the `call_mcp_tool` dispatcher, proven in
   §4B-EVIDENCE), so `call_mcp_tool` is the forceable function.
3. **Inject the routing hint into the live tail** via the existing
   `memory_handler._append_to_latest_user_tail` path (the mechanism 4D/anthropic
   already use), **NOT** `systemInstruction` — the agy path deliberately never
   mutates `systemInstruction` (gemini.py:1157) and it is the cache prefix, so
   mutating it forces a prefix miss every forced turn. The hint is a **named
   constant** beside `_FR_CCR_MARKER_TEMPLATE` (gemini.py:64), a static format
   string interpolating ONLY a store-validated 24-hex hash: "To read compressed
   content, call `headroom_retrieve` via `call_mcp_tool` with hash=<h>; do NOT
   search the filesystem." Ephemeral — appended only on forced turns.
4. Emit telemetry via the existing `log_memory_injection(...)` (helpers.py:448):
   `decision="forced_retrieve_toolconfig"`, plus an `x-headroom-fr-force`
   response header (target hash + consecutive-force count + release reason).

**Dispatch sub-target constraint (SECURITY-CRITICAL):** forcing `call_mcp_tool`
compels *a* dispatcher call, not `headroom_retrieve` specifically; the sub-tool
is model-chosen. With Gmail/Calendar/Drive/shell MCP servers in the user's
config, a forced turn could route to a **state-mutating** tool. Mitigation
(BOTH):
- **Gate:** enable forcing ONLY when `headroom` is the sole registered MCP
  server for the run (checkable at wrap time); otherwise fall back to 4A.
- **Verify on the response stream:** headroom sees the model's response
  functionCall (MITM on cloudcode-pa); if a forced turn's `call_mcp_tool`
  targets any server/tool other than `headroom`/`headroom_retrieve`, treat it as
  a failed force (do NOT let the constraint claim to guarantee retrieve) and
  fall back per the loop-cap.

**Extended retrieve exemption (CRITICAL — fixes the observed proliferation):**
`is_headroom_retrieve_name(fr.name)` (tool_injection.py:37) matches only bare/
`__headroom_retrieve`. A dispatcher-wrapped retrieve reports `name=call_mcp_tool`,
so `_compress_agy_function_responses`'s exemption (gemini.py:988) **misses** and
re-compresses the just-retrieved original → the H2 self-defeating loop returns
(this is the most likely cause of the 26→45 marker growth in the experiment).
Fix: extend the exemption helper to ALSO treat a `call_mcp_tool` functionResponse
whose args/target is `headroom_retrieve` as exempt. **Open fact to resolve before
coding:** confirm whether agy emits the retrieve result under `name=call_mcp_tool`
or an MCP-namespaced inner name — this single fact determines the exemption
predicate.

**Cap-then-release (observable, stateful — reuses session store):**
- **Release** one turn after a `headroom_retrieve` result appears (stateless-
  detectable from contents) so `mode=ANY` never blocks the answer turn; on
  release **restore the snapshotted `toolConfig`** (agy's `VALIDATED`), never
  clobber to `AUTO`.
- **Hard consecutive-force cap N** (a real invariant, not soft): counter keyed by
  the session id (§9). On exceeding N with no progress (no new
  `headroom_retrieve` call-id vs prior turn), STOP forcing and fall back to 4A —
  the DoS/wrong-hash backstop.

### 3C. DECISIVE EXPERIMENT (clean instrumentation, runs the REFINED policy)

The prior fry runs were confounded (mis-targeted function, blind detection,
un-extended exemption, greedy forcing, stale-8787 recovery). Re-run with: the
extended exemption; force `[call_mcp_tool]`+hint; cap-then-release; `--no-proxy`;
per-run whole-process-tree reap; correct `call_mcp_tool(headroom_retrieve)`
detection. Measure: (1) does agy converge to the CORRECT answer
(`VAL-ZEBRA7731-QUASAR-9284`); (2) `P(headroom_retrieve | forced+hint)`; (3)
call count → lossless-like, zero thrash-timeouts, zero filesystem-scan behavior.
**Only if agy still fails under this refined enforcement do we conclude 4A is
permanent.**

### 3D. ALTERNATIVE (if forcing is honored-but-disruptive): proxy auto-rehydration

Proxy injects the blob inline (no model call) via `_append_to_latest_user_tail`;
session-salted HMAC markers, trust discriminator (model-authored regions only),
cap, `x-headroom-fr-expand` telemetry. Gated behind 37g.13 observability.

## 4B-EVIDENCE. Enforcement experiment results (fry, 2026-07-07)

Throwaway `HEADROOM_AGY_FORCE_RETRIEVE` patch (reverted):
1. agy's 22 declared tools include `call_mcp_tool` (the MCP dispatcher) and
   `view_file` (its file tool) — NO `read_file`, and `headroom_retrieve` is NOT
   directly declared. agy sends its own `toolConfig mode=VALIDATED`.
2. Forcing `[headroom_retrieve]` = no-op (undeclared). Forcing `[call_mcp_tool]`
   works but is indirect: ~1/60 routed to retrieve unaided.
3. Force `[call_mcp_tool]` + routing hint + release: `mcp_retrieve_calls`
   climbed 0→1→2→3→4 — agy retrieves repeatedly. Thrash collapsed
   **220 s/142 calls → 23 s/8 calls, clean exit (ec=0)**, compression intact.
4. **UNPROVEN:** only the thrash/call-count collapse was cleanly observed. The
   CORRECT final answer / convergence was NOT cleanly measured (harness bugs;
   `mode=ANY` blocked the answer turn under greedy forcing; the retrieve
   exemption was not extended, so retrieved blobs were re-compressed → marker
   proliferation). §3C re-runs with the refined policy to settle correctness.

**Bottom line:** "agy is retrieval-non-compliant" is REFUTED — with proper
enforcement agy uses `headroom_retrieve`. Enforcement (3B) is the viable primary;
lossless (3A) is the floor, not the only answer. Correctness is gated on §3C.

## 5. Rejected / retired

3B native-recovery (agy held the path, os.walked); 4A-old live-zone (inverted);
37g.8 structural head (passive compliance bet); longer marker text (still
passive). Keep `4eabc716` (H2 exemption) — and EXTEND it for the dispatcher case.

## 6. Security

- **[HIGH] Dispatch escalation** → dual mitigation: sole-MCP-server gate +
  response-stream sub-target verification (§3B). Disqualifying if a forced turn
  can reach a state-mutating third-party MCP tool.
- **[HIGH] Forgeable markers** → authenticate by **store membership**
  (`CompressionStore.retrieve`) before forcing; never a bare regex match.
- **[MED] systemInstruction trust channel** → avoided entirely (hint goes to the
  live tail, not systemInstruction); static template + store-validated hash only;
  no tool-output content enters the hint.
- **[HIGH] os.walk exfil** → reduced by enforced retrieval (scoped store read
  replaces filesystem hunt); gate (7) requires zero broad-root scan behavior
  (behavioral process-tree detection).
- **Measurement (7):** token-ids/lengths only; crash-safe deletion; no raw
  transcript on a ticket/PR. `log_memory_injection` already hashes queries, never
  logs raw content.

## 7. Decision gate (pre-registered, net-security + net-token)

Named owner = the 37g.7 owner (**[fill: name]**). Decided BEFORE §3C runs:
- **Enforcement honored + routed:** `P(headroom_retrieve | forced+hint)` ≥
  **[fill: e.g. 0.9]** across the pinned holdout; the cap is a HARD invariant.
- **Correctness:** 0 regressions vs lossless on pinned fixtures (fixed config
  size/keys/gap + a multi-key/summarize case).
- **Security:** 0 broad-root filesystem-scan behavior; 0 forced turns reaching a
  non-`headroom` MCP tool (disqualifying).
- **Convergence:** call count → lossless-like; zero thrash-timeouts (first-class,
  alongside tokens); wall-clock-to-answer as the user-facing proxy metric.
- **Tokens:** ≥ **[fill: X %]** net reduction (a forced retrieve pays a 1.0x
  fresh-insert on its turn — nets positive only when cold markers are referenced
  **rarely**; state the **break-even reference-rate** and confirm the corpus of
  **[fill: N]** real multi-turn sessions carries that distribution, not a
  rare-reference bias that flatters 4B). 4A already eliminates the thrash, so 4B
  earns its keep ONLY on this number — hypothesized magnitude: **[fill]**.
- **Outcome if unmet:** 4A (default lossless) is permanent — earned by a real
  test, not assumed.

## 8. Acceptance / TDD (RED-first, no live agy)

Discrete units (each its own ticket per "no bundling"):
1. **4A default flip** + `_maybe_warn_agy_ccr_downgrade` parity test.
2. **Extended exemption:** dispatcher-wrapped-retrieve functionResponse is
   exempt from re-compression (the load-bearing correctness fix; unit-test both
   `name=call_mcp_tool` and namespaced-inner-name shapes once §3B's open fact is
   resolved).
3. **Trigger predicate:** unretrieved + STORE-AUTHENTICATED marker; rejects a
   regex-valid-but-not-in-store hash (forgery); does not fire on
   functionResponse-embedded hashes.
4. **toolConfig inject + snapshot/restore:** forced turn sets ANY/[call_mcp_tool];
   release RESTORES the snapshotted VALIDATED (asserts agy's config not
   downgraded to AUTO).
5. **Hint append** to latest-user tail (not systemInstruction), static template +
   validated hash; byte-stability of the marker unchanged.
6. **Cap-then-release state machine:** release-after-retrieve (stateless);
   consecutive-force cap N keyed by session-id; fall back to 4A on cap.
7. **Telemetry:** `log_memory_injection(decision="forced_retrieve_toolconfig")` +
   `x-headroom-fr-force` header assertions.
- **Integration (fry, quota-gated):** §3C, clean instrumentation.

## 9. Reused headroom infrastructure (no reinvention)

| Need | Reused existing infra | Location |
| --- | --- | --- |
| Retrieve exemption | `is_headroom_retrieve_name` (EXTEND for dispatcher) | tool_injection.py:37 |
| Marker auth | `CompressionStore.retrieve(hash)` membership | compression_store.py:382 |
| Session key (cap/snapshot state) | session-id derivation (`x-headroom-session-id` / model+system hash) | prefix_tracker.py:490 |
| Decision telemetry | `log_memory_injection(...)` (hashes queries, logs every cache decision) | helpers.py:448 |
| Tail injection (hint) | `memory_handler._append_to_latest_user_tail` | gemini.py memory path / anthropic.py:1772 |
| Mode single-source-of-truth | `_requested_agy_fr_mode` (flip default) | gemini.py:87 |
| Marker grammar / hash extract | `CCR_RETRIEVAL_MARKER_RE` / tool_injection extractor | parser.py:31 |
| Response header idiom | `x-headroom-*` | asgi.py:59 |

New code is limited to: the extended-exemption predicate, the force/hint/snapshot
policy state-machine, and the dispatch sub-target verification — all wired onto
the above.

## 10. Related

Supersedes 3 prior agy-ccr designs. `37g` epic; `37g.7` (=4A floor); `37g.13`
(=3D observability); `gem`; `r9k`. Verified: gemini.py:64,78-90,93-104,946,958,
988,1022,1157,1185; tool_injection.py:37; compression_store.py:382;
prefix_tracker.py:490; helpers.py:448; wrap.py:945. Research:
[Gemini function-calling](https://ai.google.dev/gemini-api/docs/function-calling)
(mode=ANY forces calls; cloudcode-pa honors toolConfig).
