# agy ccr thrash — structural live-zone boundary (supersedes 37g.8)

<!-- date: 2026-07-07 -->
<!-- status: SUPERSEDED 2026-07-07 by 2026-07-07-agy-ccr-native-recovery-design.md. Gate PASSED but REFUTED by code-verification: the 4A "compress cold / keep hot" boundary is inverted vs the proven Rust live-zone arch (which compresses the LIVE zone + freezes the cold prefix), does NOT stop the cross-turn thrash (config is cold at query time), and degenerates to lossless on the single-user-turn fry benchmark. See §0 of the native-recovery design. -->

## 1. Problem

Under ccr, agy `functionResponse` tool outputs (file reads, command output) are
compressed to opaque markers by `_compress_agy_function_responses`
(`headroom/proxy/handlers/gemini.py:946`, PR-new — main does not compress FR
parts, it adds them to `preserved_indices`). The marker
(`_FR_CCR_MARKER_TEMPLATE`, gemini.py:64-67:
`[functionResponse compressed. Call headroom_retrieve to expand. Retrieve
more: hash=<24hex>]`) is recoverable only via the injected `headroom_retrieve`
MCP tool. On a cross-turn retrieval task (agy reads a 33 KB config; a later
turn asks for one key's value; the config leaf is a marker by answer time), ccr
does not converge.

**Measured, clean box (fry), instrumented, per-run reap, 2026-07-07:**

- **lossless: converges** — 5-7 model calls, ~12-14 s, correct (100%).
- **ccr (with the shipped H2 fix `4eabc716` present): thrashes 3/3** — 120 s
  timeout, no answer, **35-42 model calls** (6-8× = a token bomb; net-**negative**
  on tokens, ccr's only metric).

## 2. Decisive live evidence

During the thrash the model **ignores `headroom_retrieve` entirely** and spawns
subprocesses running `os.walk('/')` + `glob.glob` to **brute-force-search the
whole filesystem** for the marker hash, reading file contents and writing
`find_hash.py` + `search_results.txt` into the workspace. The hash exists only
in the model's own transcript, so the search is futile and it loops.

Reframe: **a coding agent given an opaque marker for content it needs will not
call an injected retrieve tool — it reverts to native file tools and flails.**
The fix must make needed content **PRESENT**, not **RETRIEVABLE**.

## 3. How the other clients already solve it

The Anthropic/OpenAI paths do not rely on retrieval. They compress under a
**structural live-zone policy** (Rust): the recent content the model responds
*against* stays verbatim; only cold history compresses. The boundary is
**structural and model-independent**, defined by **turn position**, not a
tunable constant:

- `crates/headroom-core/src/transforms/live_zone.rs`: `HOT_ZONE_BLOCK_TYPES`
  (L515-522) are excluded from compression within the latest user frame;
  `compress_anthropic_live_zone` (L618) applies a **frozen-prefix floor** via
  `frozen_message_count` / `find_latest_user_message_index` (L944); test
  `respects_frozen_message_count` (L1505).
- `crates/headroom-core/src/compression_policy.rs`: `live_zone_only` is
  **auth-mode-conditioned** — Subscription `live_zone_only=true`, PAYG/OAuth
  `false` (Python mirror `policy_for_mode`, `compression_policy.py`).
- `headroom/proxy/handlers/anthropic.py`: `tool_results[-5:]`,
  `injected_live_zone_tail`; `frozen_message_count` is already threaded through
  `content_router.py:2066,2193,2233` and `smart_crusher.py:907,920`.

**The gap:** `_compress_agy_function_responses` (gemini.py:946) walks *every*
`contents[]` entry (historical + tail) and compresses every FR leaf ≥ a **token
floor**, with **no recency/turn boundary at all** (gemini.py:661,
streaming.py:1649 both confirm no per-part live-zone tracking is wired for this
provider). The just-read hot config gets compressed while the model still needs
it. That gap *is* the thrash.

## 4. Design

Unifying principle: **content PRESENT, not RETRIEVABLE.**

### 4A. PRIMARY — structural live-zone boundary for the agy FR path

Give `_compress_agy_function_responses` the **same structural boundary** the
other clients use: **protect the latest functionResponse frame + the frozen
prefix; compress only cold history.** This is **not** "reuse existing machinery
for free" — no `compress_gemini_live_zone` planner or pyo3 binding exists (Rust
has only anthropic/openai_chat/openai_responses planners; only
`compress_openai_responses_live_zone` is pyo3-bound, lib.rs:1611). It is
**net-new boundary logic**, prototyped in Python.

**Boundary definition (structural, non-arbitrary — NOT a tunable N, NOT a
cache-marker position):**
- Compute `live_zone_start`: the `contents[]` index of the **latest genuine
  user turn** — the last entry with `role=="user"` that is user-authored text
  (NOT a tool turn). In Gemini `contents[]`, a tool result is `role=="user"`
  with a `functionResponse` part; `fr_live_zone_start` MUST skip those and land
  on the last real user-text turn, so the whole current model/tool exchange
  (functionResponses the model is actively working with) stays in the live zone.
  This mirrors Anthropic's `find_latest_user_message_index`
  (`live_zone.rs:944`); the 4A parity test asserts equality against **that**
  reference fn (the oracle), not against the prototype itself. Where possible,
  reuse the existing latest-user index already computed by
  `_append_to_latest_user_tail` (gemini.py memory path) rather than recomputing,
  to bound drift.
- Optionally raise a **frozen floor** from a frozen-prefix signal. NOTE: the
  request-side `cachedContent` field is **not currently parsed** in the Gemini
  handler (only response-side `cachedContentTokenCount` exists, gemini.py:527).
  So the frozen floor defaults to **0** in WU1; parsing `cachedContent` to raise
  it is a **later refinement WU, not a WU1 dependency** (else-0 keeps 4A
  correct), mirroring `frozen_message_count`.
- `_compress_agy_function_responses` skips (leaves verbatim) every `contents[]`
  entry at index `>= live_zone_start`; compresses only entries below it (cold
  history that has aged past the latest frame). The token floor still applies
  within the cold zone.

**Call-site wiring:** `_compress_agy_function_responses` currently receives
`mode` (the FR compression mode), not an `AuthMode`. The `policy_for_mode`
gating requires the AuthMode; thread it explicitly from the call site
(`handle_google_cloudcode_stream`) rather than overloading `mode`, so
`live_zone_only` resolves per auth mode without conflating the two.

**Auth-mode parity (no divergence):** gate this through the existing
`CompressionPolicy.live_zone_only` / `policy_for_mode` per auth mode, so agy
behaves like the other clients (Subscription = live-zone-only; PAYG/OAuth may
compress outside the live zone) rather than a bespoke agy-only rule.

**Prototype-Python-first, defer Rust (chosen):** implement the boundary in
Python by threading a `live_zone_start` index into the existing
`frozen_message_count`-aware path; validate it stops the thrash on the harness.
Only if gate (7) proves ccr-for-agy worth keeping long-term do we invest in the
single-source-of-truth Rust `compress_gemini_live_zone` + `plan_gemini_*`
planner + pyo3 binding (mirroring `plan_responses_item`, lib.rs:1611). The
Python prototype MUST document the mirrored structural rule to bound drift, and
carry a test asserting parity of the boundary decision with the Anthropic rule
on an equivalent message shape.

**Testable pure function (TDD, no live agy):** extract the boundary decision as
a pure function `fr_live_zone_start(contents) -> int` and a leaf-inclusion
predicate `should_compress_leaf(entry_index, live_zone_start, leaf_tokens,
floor) -> bool`. Unit cases: boundary=0 (compress none), boundary=len (compress
all cold), leaf exactly at the boundary edge, empty `contents[]`,
single-turn session, multiple FR parts in one entry.

**Known limit:** masks — does not cure — **cold recall** (a query about a file
read far below the boundary → still a marker → could still thrash and, per §5,
trigger the filesystem scan). Addressed conditionally by 4B.

### 4B. CONDITIONAL — deterministic auto-expand (cold-recall fallback), fully specified

Built only if 4A proves insufficient for cold recall **and** gate (7) justifies
the cost. All iter-1 blockers folded in:

- **[PRE-REQ EXPERIMENT — observability, gates the whole of 4B]** The trigger
  assumes the marker's 24-hex hash appears in an **observable inbound request
  body** (the model re-emits it in assistant text / a `functionCall` arg). The
  live evidence shows the model's `os.walk` search runs **client-side**, which
  the proxy never sees. Before any 4B implementation, run a capture experiment
  on the fry harness: does a live marker's hash appear in a subsequent inbound
  Gemini request? If it never does, **4B is infeasible and is dropped** — do not
  build against an unobservable trigger.
- **Trust-boundary discriminator (security-critical):** fire the expansion
  **only on model-authored references**. Literal predicate (pin to avoid drift):
  scan **iff** `part.text` where the containing entry `role=="model"`, **OR**
  `part.functionCall.args`; **NEVER** `part.functionResponse.response` (any
  role). This is implementable because Gemini `contents[].parts[]` is a tagged
  union (`functionCall`→role model, `functionResponse`→role user), so the
  discriminator is part-type + role, not a heuristic. The model's own
  `search_results.txt` re-injected via a functionResponse therefore does NOT
  trigger expansion. **Pre-registered branch:** if the PRE-REQ experiment shows
  the hash appears inbound ONLY inside a `functionResponse` (the model cats its
  own search file back), the model-authored rule correctly refuses to fire →
  outcome is **drop 4B, default lossless**, NOT relax the discriminator.
- **Session-scoped store:** bind expansion to the **current session's** live
  markers only. Prefer **session-salted hashes** (`HMAC(session_key, content)`)
  so cross-session collision/dedup is impossible by construction; else an
  explicit session-id tag on store entries + a per-session live-marker set that
  gates both expansion and BM25 `retrieve`. **Eviction trigger (precise):** evict
  a session's live-marker set on the FIRST of — explicit end-of-session signal,
  proxy-client deregistration (the refcount teardown), or an idle TTL — so a
  crashed agy run cannot leave live markers expandable into a later session.
  Validate model-emitted hash charset (`[a-f0-9]`) + length (24) before lookup.
- **Expansion cap:** per-turn cap + cooldown on auto-expansions to bound the
  cache-prefix-invalidation cost and a planted-hash DoS lever.
- **Observability contract (required for a silent stream rewrite):** emit a
  decision record (`decision="injected_fr_auto_expand"`, mirroring
  `injected_live_zone_tail` in openai.py) and an `x-headroom-fr-expand`
  response header with the hash + trigger region, so an operator can tell
  whether/where/why it fired. Add an audit log line per expansion.
- **Costs (why it stays rare):** re-inflates at max context depth and
  invalidates the cache prefix from the injection point; append-only at the
  reference point, never rewrites unrelated history.

### 4C. Retire 37g.8

Drop the structural-summary marker head + in-marker decision instruction +
needle backstop: it bets on marker-instruction compliance the live evidence
falsifies (for a specific-key query the value is not in the head → the model
must still call `headroom_retrieve` → it won't). Keep `4eabc716` (the H2
re-compression exemption is defensively correct; simply not sufficient).

## 5. Security

- **[HIGH] Marker-induced filesystem-content-scan exfil (new, first-class):**
  the opaque marker induces the agent to `os.walk('/')` and read local files
  (`~/.ssh`, `.env`, cloud creds) into `search_results.txt`, which then flows
  **upstream into the transcript sent to Gemini** — a proxy-induced local-secret
  exfil path. This is not merely a convergence/token issue. 4A reduces its
  incidence (hot content stays present, so the model does not flail on it); the
  residual cold-recall trigger is what 4B (or lossless-default) must close.
  **Gate (7) is therefore a net-SECURITY gate, not only net-token: a mechanism
  that still induces filesystem-wide content scans is disqualifying regardless
  of token math.**
- **[HIGH] Auto-expand confused deputy (4B):** closed by the model-authored-only
  discriminator + tool_result-region exclusion above; without it the mitigation
  is unimplementable.
- **[MED] Cross-session bleed:** closed by session-salted hashes or session-id
  scoping + eviction (4B).
- **[MED] Cache-invalidation DoS (4B):** closed by the per-turn expansion cap.
- **Net-token measurement (7) secret handling:** count on **token-ids/lengths**,
  never retained plaintext; if a raw payload must persist, mandate a named
  redaction step, tmpfs/ephemeral storage, and `trap`/`finally`-guaranteed
  deletion (crash-safe). Never attach a raw transcript to a ticket/PR.

## 6. Acceptance criteria (TDD-first)

**Unit (no live agy, RED-first):**
- `fr_live_zone_start` + `should_compress_leaf` pure-function cases (§4A).
- WU1 cache invariant: a cold-history leaf's bytes are **byte-identical across
  two turns** once aged past the boundary (frozen markers stable).
- (If 4B built) trigger unit tests: fires iff a valid-charset/length hash in the
  **current-session** live set appears in a **model-authored** region; rejects
  wrong charset/length; **rejects** tool_result-embedded hashes.

**Integration (live harness, quota-gated):**
- Re-run the clean-fry harness (`fry_run.sh` + `fry_seq.sh`; per-run reap
  **extended to agy's whole process tree**; call-count instrumentation): ccr
  model-call count approaches lossless (~5-7), **zero thrash-timeouts**, and
  **zero filesystem-scan artifacts** (`find_hash.py`/`search_results.txt` never
  written).
- **Frozen anti-overfit holdout fixture** (pinned, not ad-hoc): exact config
  size, key count, and gap-in-turns fixed in-repo; plus a differently-shaped
  case (larger gap, multi-key, summarize-not-retrieve) to prove generalization.

## 7. Decision gate (pre-registered, net-security + net-token)

Decided **before** running, by a **named owner**:
- **Correctness:** 0 regressions vs lossless on the holdout fixtures.
- **Security:** 0 filesystem-scan behavior across the corpus (disqualifying if
  any). Detection must be **behavioral, not filename-signature** — monitoring
  only for `find_hash.py`/`search_results.txt` is under-inclusive (a renamed or
  in-memory scan evades it). Monitor the reaped agy process tree for broad-root
  reads (`os.walk`/`glob` over paths outside the workspace, or `open()` on
  `~/.ssh`/`.env`/cloud-cred paths), in addition to the artifact-file check.
- **Tokens:** ≥ **[pre-registered X %]** net-token reduction (charging 4B's
  cache-prefix-invalidation cost against it) across ≥ **[pre-registered N]**
  representative real multi-turn agy sessions whose shape-mix **includes the
  cross-turn cold-recall failure case** (a short-task-only corpus is rejected as
  self-biasing — 4A saves ~0 there).
- **Outcome if not met:** **default lossless for agy and stop** (concede ccr
  does not pay off for an agent that won't cooperate with retrieval). This is an
  explicit, honest exit, not a failure.

## 8. Rejected alternatives

- **37g.8 structural head + needle backstop** — compliance bet against live
  evidence (§4C).
- **Exempt all re-fetchable file reads** — stops thrash but file reads are the
  bulk of a coding CLI's traffic; guts ccr's savings. Stopgap, not a design.
- **Longer/smarter retrieve prompt** — model-dependent, contradicted by the
  brute-force-search evidence.
- **"Ghost file" (marker as a magic path)** — infeasible: headroom is MITM on
  the LLM stream only; the agent's `cat`/shell runs client-side. The salvageable
  form collapses to 4B auto-expand.
- **Rust `compress_gemini_live_zone` now** — deferred (not rejected) per the
  prototype-Python-first decision; promoted only if gate (7) keeps ccr-for-agy.

## 9. Related

- Supersedes `docs/.../2026-07-06-agy-ccr-thrash-diagnosis-design.md` (37g.8).
- `headroom-37g.8` (retire), `headroom-37g.7` (provisional lossless default,
  separate), `headroom-gem` (thrash umbrella + mechanism evidence),
  `headroom-r9k` (`-p`/`--port` collision).
- Independent adversarial review: agy/Gemini 3.1 Pro (PRESENT not RETRIEVABLE;
  confirmed no Gemini live-zone planner/binding exists; boundary must be
  structural turn-position, not cache-marker or tunable N).
