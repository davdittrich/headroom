# agy ccr thrash — enforced `headroom_retrieve` + lossless floor (supersedes 37g.8, live-zone-parity, native-recovery)

<!-- date: 2026-07-07 -->
<!-- status: design-review-gate PASSED iter2 (all 5 APPROVED: PM/Architect/Designer/Security/CTO; Architect+CTO verified the §9 reuse table is accurate). Non-blocking refinements folded (side-effect-free auth, PrefixCacheTracker cap state, sole-MCP operational note). REVISED post-WU-SPIKE-3 (37g.29): §3B retrieve exemption = RETRIEVE-CALL-SCAN SUPPRESSION (agy analog of live_zone.rs headroom_retrieve_call_ids, keyed by hash since agy has no call_id). Supersedes call-id-correlation AND the eviction-holed store-membership variant. Grounded: fry HEAD lacks the exemption -> Flash thrashes 236x on delayed-reference; the fix = port what OpenAI/Anthropic already do. WU2/37g.17 reconciled. Forcing WUs (4B) NOT closed pending re-test on the WU2-fixed build (prior 'forcing harmful' data was confounded by the missing exemption). -->

<!-- POST-IMPLEMENTATION NOTE (2026-07-08): this is a historical design record. The
implemented FR compressor + retrieve exemption were later EXTRACTED from gemini.py into
`headroom/transforms/agy_fr_compressor.py` (headroom-37g.36), so §9/§10 `gemini.py:NNN`
citations now resolve there. The `"headroom_retrieve" in json.dumps(args)` exemption test
was replaced by the behaviorally-equivalent structural scan `_args_mention_retrieve`
(headroom-37g.35). Behavior contract unchanged; only location + internal mechanism moved. -->

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

## 2.5 STALE PREMISE — measure voluntary retrieval FIRST (plan-review finding)

The "agy ignores headroom_retrieve" premise behind all of §3B enforcement is
**stale and unvalidated for the current marker.** `gemini.py:58` records the
0-retrieve observation came from a WU4 trial with a marker that **named no
tool**; the marker was SINCE fixed (gemini.py:65 now self-describes + names
`headroom_retrieve`). Every enforcement experiment in §4B-EVIDENCE **forced**
retrieval — nobody re-measured whether agy retrieves **voluntarily** under the
fixed marker. Also verified: **4A lossless delivers ~zero savings**
(`compact_lossless` no-ops on typical tool output, gemini.py:936) — i.e. 4A is
only the safety floor, NOT the epic's savings; savings require markers getting
retrieved.

**Therefore, before ANY forcing is built (§3B), run WU-4A.5:** measure the
voluntary (unforced) retrieve-rate + net-tokens under the current marker (with
the §3B retrieve-call-scan exemption in place, else the retrieved blob re-compresses). If
adequate → **ship ccr-default, delete all of §3B forcing** (no coercion surface,
epic delivered). Only if inadequate is the §3B enforcement machinery justified.
This is the cheapest experiment that can retire 4B while delivering savings.

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
hash is authenticated by **side-effect-free store membership** — a bare
`backend.get(hash)`/`hash in store` check, NOT the full
`CompressionStore.retrieve()` path (which fires `record_access`/`_log_retrieval`/
feedback, compression_store.py:373, and would mutate TTL/LRU recency + pollute
the very `P(retrieve|forced)` metric §7 measures) — and NOT a bare `[a-f0-9]{24}`
regex match (closes the forgeable-unsalted-SHA spoof: planted tool-output hashes
that are not real store keys never trigger forcing). Store scoping is
single-user-local per run; deterministic hashes are cross-session-stable, so if
`get_compression_store()` is a process-global singleton the store MUST be
confirmed per-session-or-local-single-user (else salt, as 3D does) to avoid
cross-session hash referencing. "Retrieved" = a `headroom_retrieve`
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
`is_headroom_retrieve_name(fr.name)` (tool_injection.py:25) matches only bare/
`__headroom_retrieve`. A dispatcher-wrapped retrieve reports `name=call_mcp_tool`,
so `_compress_agy_function_responses`'s exemption (gemini.py:988) **misses** and
re-compresses the just-retrieved original → the H2 self-defeating loop returns
(the most likely cause of the 26→45 marker growth in the experiment).

Fix — **exempt retrieved content from re-compression, exactly as the other
clients already do** (parity with `live_zone.rs`; supersedes the earlier call-id
and hash-membership proposals — both empirically refuted). How OpenAI/Anthropic
avoid this loop: they collect the `call_id` of every `headroom_retrieve` call in
the request and **skip compressing any output paired to one**
(`headroom_retrieve_call_ids` → `continue` at `live_zone.rs:2362-2384`). Once the
model retrieves a blob it stays verbatim → the model has it → never re-retrieves →
no thrash, by construction. agy's `gemini.py` has **no such exemption** — that
absence *is* the bug (WU-SPIKE-3/37g.29: fry HEAD without it thrashes Flash-High
236× on a delayed-reference task; with it, the other clients do not).

Port it, adapted to agy's wire format. agy carries **no `call_id`** (id-less;
retrieve is dispatched via `call_mcp_tool` with the target in args — WU-SPIKE/
37g.23), so key the exemption on the **retrieved hash** instead of the call-id.
The retrieve call *names the hash it wants*, and those `functionCall` parts
**persist in agy's resent history** (that is how 236 were counted). Mechanism:
1. Scan `contents[]` once for `functionCall` parts invoking `headroom_retrieve`
   — bare/`__headroom_retrieve` name, OR `name=call_mcp_tool` whose args reference
   `headroom_retrieve` — and collect every 24-hex hash in their args into a
   request-scoped `retrieved_hashes: set[str]`.
2. When about to compress a `functionResponse` string leaf, compute
   `H = default_ccr_hash(leaf)` (the SAME `SHA-256(original)[:24]` the store keys
   on, compression_store.py:325 — extract a shared helper so exemption-key and
   store-key cannot drift) and **exempt (leave verbatim) iff `H ∈ retrieved_hashes`.**

This is the agy analog of `headroom_retrieve_call_ids`, keyed by hash. It is
**request-scoped** — authority comes from the retrieve calls in the request, not
the mutable store — so it is eviction/TTL/salt/id-immune (the fatal flaw of the
refuted store-membership variant). No over-exemption: a leaf whose hash was never
retrieved still compresses (cold-history savings preserved). The existing
name-based exemption (`is_headroom_retrieve_name`, gemini.py:989) is **kept** as a
bare-name fast path; the hash-scan adds the dispatcher-wrapped case. Convergence:
after the first `retrieve(H)`, the leaf hashing to `H` is exempt → verbatim → the
model answers → stops (acceptance: Flash delayed-reference 236 → ~1 retrieve).

**Dual exemption (review finding, agy adversarial):** the hash-scan above exempts
the *resent cold original* (the observed thrash driver — WU-SPIKE-3 saw 4 *stable*
hashes re-retrieved 236×, i.e. cold originals, not nesting envelopes). But the
retrieve *result* itself is a JSON envelope — `json.dumps({"hash": H,
"original_content": C, …})` (`mcp_server.py:441-448,710`), NOT byte-identical to
`C`, so its hash is not in `retrieved_hashes` and the hash-scan cannot catch it.
The other clients exempt the retrieve *output* **content-agnostically** (by call_id
at `live_zone.rs:2384`; by tool-name at `smart_crusher.py:1017`) — but agy has **no
call_id**, and positional pairing is fragile under parallel/heterogeneous
`call_mcp_tool` dispatch. Re-review (agy + architect, 2026-07-07) therefore
**refuted (B)-as-call-id-pairing and confirmed (A) alone is sufficient** for the
observed convergence (the resent cold original is the driver). (B) is also likely
**redundant**: if agy nests the retrieve envelope as a *parsed dict*,
`_walk_fr_compress` recurses to the inner `original_content` leaf (== `C`, hash `H`)
and (A) already exempts it; (B) is load-bearing only in the *monolithic-JSON-string*
case. So **WU2/37g.17 ships (A) alone**; the envelope exemption is **deferred to
37g.30**, evidence-gated on capturing the real retrieve-RESULT wire shape, and — if
needed — keyed on the **envelope-signature** (`response` carries both `hash` and
`original_content` keys), never on call_id or position.

**Cap-then-release (observable, stateful — reuses session store):**
- **Release** one turn after a `headroom_retrieve` result appears (stateless-
  detectable from contents) so `mode=ANY` never blocks the answer turn; on
  release **restore the snapshotted `toolConfig`** (agy's `VALIDATED`), never
  clobber to `AUTO`.
- **Hard consecutive-force cap N** (a real invariant, not soft; N a named
  constant derived from first principles beside `_FR_CCR_MARKER_TEMPLATE`, not
  tuned-to-pass): counter + the toolConfig snapshot live in a session-keyed
  container — `PrefixCacheTracker.get_or_create(session_id)` (prefix_tracker.py:468,
  session id via `compute_session_id`, :481), NOT handler-instance state (the
  iter-1 blocker-4 failure). The snapshot is written ONCE per episode
  ("if not already saved") and never overwritten by an intermediate ANY config.
  On exceeding N with no progress (the target hash's retrieve result still absent
  vs prior turn), STOP forcing and fall back to 4A — the DoS/wrong-hash backstop.

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

- **[HIGH] Dispatch escalation** → the **sole-MCP-server gate is the
  load-bearing containment** (a forced `call_mcp_tool` physically cannot dispatch
  to Gmail/Calendar/Drive/shell if they aren't registered for the run). The
  response-stream sub-target verification is **defense-in-depth only** — it is
  NEW gemini-path functionCall parsing (the existing response parser
  `_record_ccr_feedback_from_response`, streaming.py:519, is Anthropic-shaped), so
  §6 does NOT claim a shipped dual guarantee until that parser exists.
  Disqualifying if a forced turn can reach a state-mutating third-party MCP tool.
  **OPERATIONAL REALITY:** the sole-MCP gate DISABLES 4B for the common real agy
  config (Gmail/Calendar/Drive MCP servers registered — as in this very
  session) → **4A lossless is the effective default there**. Safe, but §7's
  token-ROI corpus MUST reflect that 4B activates only in sole-headroom-MCP runs.
- **[HIGH] Forgeable markers** → authenticate by **side-effect-free store
  membership** (a bare `backend.get`/`hash in store`, NOT `CompressionStore.
  retrieve()` which fires `record_access`/`_log_retrieval`/feedback and would
  pollute the `P(retrieve|forced)` metric) before forcing; never a bare regex
  match.
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
| Marker auth | side-effect-free `backend.get(hash)` membership (NOT retrieve()) | compression_store.py:355/397 |
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
