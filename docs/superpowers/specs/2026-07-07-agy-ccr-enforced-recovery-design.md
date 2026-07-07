# agy ccr thrash — enforced `headroom_retrieve` + lossless floor (supersedes 37g.8, live-zone-parity, native-recovery)

<!-- date: 2026-07-07 -->
<!-- status: design-review-gate pending. Corrects the premature "agy is non-compliant" conclusion: passive marker suggestion != enforcement; enforcement was never attempted. -->

## 0. What changed and why (correcting two prior errors)

- The **live-zone-parity (4A)** design was refuted by code: it compressed the
  cold prefix + kept the live zone verbatim — the inverse of the proven Rust
  arch (`live_zone.rs:643-683` compresses the latest user message, freezes the
  cold prefix), does not stop the cross-turn thrash, and degenerates to lossless
  on the single-turn benchmark.
- The **native-recovery (3B)** design was refuted by code: `gemini.py:958`
  *"functionCall parts are never touched"* — agy's own `functionCall(read_file,
  path=X)` sits **uncompressed in the same request**, so agy **already had the
  path** and chose `os.walk('/')` from root anyway. You cannot fix that by
  editing the marker string to name a path the client already holds and ignored.
- **The error both share, and that this doc corrects:** concluding "agy is
  retrieval-non-compliant" from agy ignoring a **passive** marker *suggestion*.
  That is not the same as agy refusing an **enforced** tool-use constraint.
  **Enforcement was never attempted.** The honest question is not "will agy
  choose to recover" (evidence: no) but "will agy recover when the API
  **compels** it" — untested.

## 1. Problem

Under ccr, agy `functionResponse` outputs are compressed to markers
(`_FR_CCR_MARKER_TEMPLATE`, gemini.py:64) recoverable via the injected
`headroom_retrieve` MCP tool. On a cross-turn retrieval task, ccr thrashes
(clean-fry, instrumented: lossless converges 5-7 calls/~14 s/correct; ccr 3/3
timeout, 35-42 calls, no answer — net-negative).

## 2. Root cause (corrected)

ccr's savings contract is **compress + `headroom_retrieve`**. The proven
Anthropic/OpenAI paths work because those models **call the tool when the
recovery affordance is merely offered**. agy, given the same *passive* affordance,
ignores it and improvises an `os.walk` hunt. But the affordance was only ever a
**suggestion** (marker text + a declared tool the model may choose to call).
headroom has **never forced** the call. Gemini's API supports forcing it.

## 3. Research: how to enforce tool use (grounded)

`toolConfig.functionCallingConfig` on the Gemini `generateContent` request:
- `mode: "AUTO"` (default) — model decides.
- `mode: "ANY"` — model is **constrained to emit only function calls**;
  `allowedFunctionNames: [...]` restricts to a specific set.
- `mode: "NONE"` — no function calls. (Newer `VALIDATED` mode: decide, but
  constrained-decode the call.)

**Confirmed supported on agy's exact backend:** the `cloudcode-pa.googleapis.com
/v1internal:generateContent` endpoint accepts the same `toolConfig.
functionCallingConfig` object; a community transformer targeting that endpoint
maps `tool_choice=required → mode=any` (+ `allowedFunctionNames` for a named
function). So `{mode: "ANY", allowedFunctionNames: ["headroom_retrieve"]}`
**forces** agy's model to emit `headroom_retrieve(...)`.

**Caveat to test (not assume):** some *older* models (Gemini 1.5 Pro) returned a
text+call pair under `ANY` instead of a lone call — forced-calling has been
model-version-dependent. agy runs Gemini 3.x; the experiment (§5) must confirm
it honors forced calling. Provisioning (`SERVICE_DISABLED`) is a separate,
unrelated failure mode.

**Injection point:** headroom already rewrites the agy request body's `tools`
(tool_injection.py:303; `headroom_retrieve` present via the MCP). Setting
`body["toolConfig"]["functionCallingConfig"]` at the same interception is a
surgical add — headroom does not touch `toolConfig` today (verified: zero
matches).

## 4. Design

### 4A. FLOOR (ship now): lossless default, correctly wired

Default agy to lossless until enforced recovery (4B) is proven. **Implement by
flipping the default in the shared source of truth** — `_requested_agy_fr_mode`
(gemini.py:87: `or "ccr"` → `or "lossless"`) — NOT by adding a downgrade inside
`_resolve_agy_fr_mode`, which would make resolve downgrade while
`wrap._maybe_warn_agy_ccr_downgrade` (wrap.py:945) stays silent (the drift the
docstrings forbid, gemini.py:83-85). Flipping the shared default preserves parity
automatically. Safe-by-construction, stops the net-negative thrash immediately.

### 4B. PRIMARY (the fix the evidence actually points to): enforced `headroom_retrieve`

When the model needs a marker's content and has not retrieved it, the proxy
**compels** the call via `toolConfig.functionCallingConfig`.

- **Trigger (observable in the response stream — headroom is MITM on
  cloudcode-pa and sees model output):** the model emits a *misdirected recovery*
  in its response — a native `functionCall` whose args reference a live marker's
  24-hex hash, or an `os.walk`/`glob`/`grep` for a marker string, or a re-read of
  a source now behind a marker. (A simpler, always-safe fallback trigger: a
  request carries an unresolved live marker and the model's latest turn is
  neither a `headroom_retrieve` call nor progress.)
- **Enforcement action:** on the NEXT outbound request, set
  `body["toolConfig"] = {"functionCallingConfig": {"mode": "ANY",
  "allowedFunctionNames": ["headroom_retrieve"]}}`. The model is forced to emit
  `headroom_retrieve(hash=…)`; the wired MCP resolves it; the original content
  returns through the retrieve path; the model converges. **Revert to `AUTO`**
  the following turn so normal tool use resumes.
- **Hash selection:** with a single relevant live marker the model fills the
  obvious hash; with multiple, measure the wrong-hash rate in the experiment and,
  if needed, narrow `allowedFunctionNames` scope or inject a one-line hint naming
  the target hash alongside the forced config.
- **Forcing-loop guard:** cap consecutive forced-retrieve turns; if a forced
  retrieve does not yield progress, stop forcing and fall back to 4A rather than
  loop.
- **Why this beats the rejected approaches:** it uses the EXISTING retrieve
  infrastructure and the model's own call (conversation coherent — no
  proxy-authored injection as in 4D, no re-read of possibly-mutated files as in
  the rejected 3B), and it targets the actual root cause — non-invocation — by
  removing the choice.

### 4B-EVIDENCE. Enforcement experiment results (fry, 2026-07-07)

A throwaway proxy patch (`HEADROOM_AGY_FORCE_RETRIEVE`) exercised enforcement on
the y4q retrieve-forcing task. Findings (each corrects a prior assumption):

1. **agy's tool surface:** 22 declared native tools; **no `read_file`** (it is
   `view_file`); **MCP tools — including `headroom_retrieve` — are reachable ONLY
   via a generic `call_mcp_tool` dispatcher.** `headroom_retrieve` is never a
   directly-declared function. agy sets its own
   `toolConfig.functionCallingConfig.mode=VALIDATED`, so the backend DOES honor
   `toolConfig`.
2. **Forcing the specific tool is impossible; force the dispatcher:**
   `functionCallingConfig.allowedFunctionNames=["headroom_retrieve"]` targets an
   undeclared function → no effect. Forcing `["call_mcp_tool"]` (mode=ANY) works,
   but the dispatcher is too indirect — agy routed to `headroom_retrieve` only
   ~1/60 forced turns (it picked other MCP tools).
3. **Add a routing HINT + a RELEASE policy → agy retrieves reliably:** append a
   one-line system-instruction hint ("call headroom_retrieve via call_mcp_tool
   with the marker hash; do NOT search the filesystem") on forced turns, and do
   NOT force in the turn right after a retrieve. Result: **`mcp_retrieve_calls`
   climbed 0→1→2→3→4 — agy retrieved repeatedly.** The thrash COLLAPSED:
   **220 s / 142 model calls → 23 s / 8 calls, clean exit (ec=0)**, compression
   intact (7.4 KB → 162 tokens).
4. **Remaining gap = convergence policy, NOT agy refusal:** greedy forcing tries
   to retrieve ALL markers (agy needs only the one relevant blob) and `mode=ANY`
   on the answer turn blocks the final text. A cap-then-release policy (stop
   forcing after the needed content is retrieved) is required; tuning it cleanly
   belongs in real implementation, not the throwaway harness.

**Bottom line: "agy is retrieval-non-compliant" is REFUTED.** With proper
enforcement (force `call_mcp_tool` + routing hint + release), agy uses
`headroom_retrieve` and the thrash is eliminated. Enforcement (4B) is the viable
primary mechanism; lossless (4A) is the floor, no longer the only answer.

### 4C. DECISIVE EXPERIMENT (must run before concluding anything)

This is the experiment the prior designs skipped. On clean fry, instrumented
(per-run reap of agy's whole process tree, call-count, behavioral scan
detection):

1. **Enforcement-honored probe:** does agy's Gemini 3.x emit a lone
   `headroom_retrieve` call under `{mode: ANY, allowedFunctionNames:
   [headroom_retrieve]}` on the y4q retrieve-forcing task (not a text+call pair,
   not a refusal)?
2. **Convergence under enforcement:** with the trigger + forced retrieve wired,
   does the ccr run converge (call count → lossless-like, zero thrash-timeouts,
   zero filesystem-scan behavior) and answer correctly?
3. **Only if agy STILL fails under proper enforcement** (refuses, wrong-hash
   loops, or backend ignores the config) do we conclude retrieval is
   unsalvageable → 4A lossless is permanent. **Non-compliance is a measured
   outcome here, never an assumption.**

### 4D. ALTERNATIVE (if enforcement is honored-but-disruptive): proxy auto-rehydration

If forced calling proves too disruptive (e.g. version-dependent text+call, or
forcing derails multi-step flows), the proxy injects the blob inline instead of
forcing the model to ask — reusing `injected_live_zone_tail` /
`_append_to_latest_user_tail` (anthropic.py:1772). Model-authored-only trigger
(scan iff `part.text`@role==model OR `functionCall.args`; never
`functionResponse`), session-salted HMAC hashes + eviction, expansion cap,
`x-headroom-fr-expand` telemetry. Gated behind the observability experiment
(37g.13).

## 5. Rejected / retired

- **Native-recovery markers (3B)** — refuted: agy already holds the uncompressed
  `functionCall` path (gemini.py:958) and os.walked anyway; net-negative vs
  lossless under the cache discount; unsound on a mutable FS (agy edits files →
  re-read returns different bytes).
- **Live-zone boundary (4A-old)** — refuted (§0).
- **37g.8 structural head + needle backstop** — compliance bet the evidence
  (passive-affordance) falsifies; but note enforcement (4B) is a *different*
  lever it never considered.
- **Longer/smarter marker text / system prompt** — still passive; does not
  compel.
- Keep `4eabc716` (H2 re-compression exemption).

## 6. Security

- **`os.walk('/')` exfil (HIGH, carried):** enforced retrieve **reduces** it —
  a forced `headroom_retrieve(hash)` (a scoped, content-addressed store read)
  replaces the filesystem hunt. Gate (7) remains net-security: any broad-root
  scan behavior (behavioral process-tree detection, not filename-signature)
  disqualifies.
- **Forced call safety:** the compelled call is `headroom_retrieve` only
  (`allowedFunctionNames` scoped); validate the model-supplied hash
  charset/length + session scope before the store lookup; the store is
  single-user-local. No new capability.
- **4D injection (if built):** trust discriminator + session-salted scope +
  cap + telemetry as above.
- **Measurement (7):** token-ids/lengths only; crash-safe deletion; never attach
  a raw transcript to a ticket/PR.

## 7. Decision gate (pre-registered, net-security + net-token)

Decided **before** running, **named owner** (align 37g.7):
- **Enforcement honored:** experiment 4C.1 shows agy emits the forced call
  (else → 4A permanent).
- **Correctness:** 0 regressions vs lossless on pinned holdout fixtures (fixed
  config size/keys/gap + a differently-shaped multi-key / summarize case).
- **Security:** 0 broad-root filesystem-scan behavior (disqualifying).
- **Convergence:** call count → lossless-like, zero thrash-timeouts (first-class,
  alongside tokens).
- **Tokens:** ≥ **[pre-registered X %]** net reduction (a forced retrieve pays a
  1.0x fresh insert on its turn; savings come from cold content compressed to
  markers and NEVER referenced — honestly, this nets positive only when
  references are rare, so the corpus must be representative) across ≥
  **[pre-registered N]** real multi-turn agy sessions INCLUDING the cross-turn
  cold-recall case (short-task-only corpus rejected as self-biasing).
- **Outcome if unmet:** **4A (default lossless) is permanent** — the honest exit,
  now *earned* by a real enforcement test rather than assumed.

## 8. Acceptance / TDD

- **Unit (no live agy):** the `toolConfig` injection (given a request + a
  "needs-retrieve" signal, the body gains `functionCallingConfig={mode:ANY,
  allowedFunctionNames:[headroom_retrieve]}`; reverts next turn); the trigger
  predicate (fires on a model-authored hash reference / marker-hunt; NOT on
  functionResponse bytes); the forcing-loop cap; marker byte-stability
  (unchanged, deterministic SHA-256[:24]); 4A default-flip parity with
  `wrap._maybe_warn_agy_ccr_downgrade`.
- **Integration (fry, quota-gated):** experiment 4C; process-tree reap; zero
  thrash-timeouts + zero scan behavior.

## 9. Related

- Supersedes 3 prior agy-ccr designs (diagnosis/37g.8, live-zone-parity,
  native-recovery).
- `headroom-37g` epic; `37g.7` (lossless floor = 4A); `37g.13` (4D observability);
  `gem`; `r9k`.
- Verified: `gemini.py:64,78-90,93-104,946,958`; `tool_injection.py:303`;
  `live_zone.rs:643-683`; `wrap.py:945`. Research:
  [Gemini function-calling modes](https://ai.google.dev/gemini-api/docs/function-calling),
  [Cloud Code Assist toolConfig support](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tools/function-calling).
