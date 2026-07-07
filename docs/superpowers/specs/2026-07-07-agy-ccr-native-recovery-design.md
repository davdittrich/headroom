# agy ccr thrash — native-recovery markers + lossless floor (supersedes 37g.8 AND the live-zone-parity design)

<!-- date: 2026-07-07 -->
<!-- status: design-review-gate pending (rewrite after code-verification refuted the live-zone-parity design) -->

## 0. Why this supersedes the prior (gate-passed) design

The `2026-07-07-agy-ccr-live-zone-parity-design.md` (4A: "compress cold history,
keep the hot frame verbatim") passed a 5-agent gate but was **refuted by reading
the proven Rust live-zone implementation**:

- **Direction inverted.** `compress_anthropic_live_zone_with_ccr`
  (`live_zone.rs:643-683`) compresses **only the latest user message** (the live
  zone) and **freezes the cold prefix** (`live_zone.rs:36-46`: indices below
  `frozen_message_count` *"MUST be byte-identical"*). 4A compressed the cold
  prefix and kept the live zone verbatim — the opposite.
- **4A does not stop the thrash.** At query time the referenced content is
  *cold* (below the latest user-text turn) → 4A compresses it → marker → thrash.
  Same `gap==N` overfit it claimed to avoid.
- **4A degenerates to lossless on the actual benchmark.** The fry task is one
  user prompt + a tool loop, so `fr_live_zone_start == 0` → nothing compresses →
  lossless. It would "pass" the convergence test only by being lossless.

The gate verified the *machinery*; no reviewer traced end-to-end behavior on
agy's real turn structure. This rewrite fixes that.

## 1. Problem

Under ccr, agy `functionResponse` tool outputs are compressed by
`_compress_agy_function_responses` (`gemini.py:946`) to opaque markers
(`_FR_CCR_MARKER_TEMPLATE`, gemini.py:64:
`[functionResponse compressed. Call headroom_retrieve to expand. Retrieve more:
hash=<24hex>]`), recoverable only via the injected `headroom_retrieve` MCP tool.
On a cross-turn retrieval task, ccr thrashes.

**Measured, clean box (fry), instrumented:** lossless converges (5-7 calls,
~14 s, correct); ccr (H2 fix present) thrashes 3/3 (120 s timeout, 35-42 calls,
no answer) — net-**negative** on tokens.

## 2. Verified root cause

ccr's savings contract is **compress + `headroom_retrieve`**. It works for the
Anthropic/OpenAI paths because **those models call the retrieve tool.** The live
evidence shows agy does **not**: given an opaque marker it spawns `os.walk('/')`
+ `glob` subprocesses to brute-force-search the filesystem for the hash — a
**misdirected native-tool recovery** (it *wants* to re-read the source, but the
marker gives it a hash to `headroom_retrieve`, not the file to re-read, so it
flails). The H2 re-compression fix is present and does not help; re-compression
was never the staller.

**Reframe:** the marker must offer a recovery path agy **will** use — its own
native tool call. headroom's `HEADROOM_RETRIEVE_SCHEMA` already documents the
fallback *"Content expires after a TTL — if expired, re-run the original command
instead."* Make that the **primary** affordance for agy, naming the exact call.

## 3. Design

Two-part: an immediate safe floor, and a savings mechanism that matches agy's
proven behavior.

### 3A. FLOOR (immediate): extend the existing downgrade from wired→effective

`_resolve_agy_fr_mode` (gemini.py:93-104) already encodes headroom's principle:
*ccr requested but retrieve not **wired** (`HEADROOM_AGY_RETRIEVE_WIRED != 1`) →
downgrade to lossless* (don't ship unrecoverable markers). The proven fact is
that agy's retrieve is **wired but ineffective**. Extend the same principle:
until an effective-recovery mechanism (3B) is validated, **default agy to
lossless.** This stops the net-negative thrash now and is pure existing-pattern
(byte-recoverable, no markers, no thrash). Tracks `headroom-37g.7`.

### 3B. SAVINGS MECHANISM (build + validate): native-recovery markers

Replace the agy FR marker's recovery instruction: instead of *"Call
headroom_retrieve"*, name the **original native tool call to re-run**, derived by
correlating each `functionResponse` with its preceding `functionCall` in
`contents[]` (the `functionCall.args` carry the path/query).

- **Marker form (reproducible sources):**
  `[output compressed — re-run read_file(path="/tmp/config.txt") to see it; or headroom_retrieve(hash=<24hex>)]`.
  Native re-run is primary (agy's instinct); the hash stays as a
  belt-and-suspenders fallback. This is an **enhancement of the existing marker**,
  not a new marker family (avoids the proliferation the earlier gate flagged).
- **Convergence path:** cold marker → model re-runs the named native tool → the
  fresh output lands in the live zone → the proxy keeps the **just-arrived
  (latest-message) tool_result verbatim** (does not re-compress it) → model sees
  the content → converges. This is the loop-breaker 4A lacked: recovery uses the
  tool agy actually invokes, and the re-read result is present.
- **Reproducibility allowlist (correctness + security):** apply native-recovery
  markers ONLY to tool calls known idempotent/reproducible on re-run (`read_file`,
  `cat`, `ls`/`glob` on stable paths — an explicit allowlist, grep-able like
  `HOT_ZONE_BLOCK_TYPES`). Non-reproducible outputs (`date`, `curl`, build/run
  commands, anything stateful) are **NOT** given a native-recovery marker — they
  fall back to lossless (kept verbatim) so the model never re-runs a
  non-reproducing command and gets wrong data. Mirrors the schema's existing
  "re-run … if expired" caveat, made safe by construction.
- **Recency window is a PERFORMANCE knob, not correctness (defuses the
  recent-N overfit):** optionally keep the last K tool_results verbatim to avoid
  re-reads in the common case. Because native-recovery guarantees convergence
  regardless of K (a wrong K just costs one extra re-read round-trip, never a
  thrash), K is tuned for token cost, not correctness. K may be 0 (compress all
  cold reproducible outputs) — the fry experiment sets it.
- **Net-token intuition:** the cached prefix shrinks from the full tool output
  to a ~80-byte marker every turn; the full bytes are re-sent only on the rare
  turn the model actually re-reads. Whether that nets positive vs lossless (whose
  full output sits in the cached prefix every turn at the provider's cache
  discount) is exactly what gate (6) measures.

### 3C. ALTERNATIVE (if 3B's re-read round-trips cost too much): proxy auto-rehydration

If the fry experiment shows agy re-reads too often (round-trip cost > savings),
fall back to proxy-side injection: the proxy detects a reference to a live
marker's hash in a **model-authored** region and injects the blob inline
(reusing the existing `injected_live_zone_tail` / `_append_to_latest_user_tail`
machinery, anthropic.py:1772-1788). Gated on the observability experiment
(`headroom-37g.13`): the `os.walk` is client-side, so this only works if the
hash reaches the proxy in an inbound request. Trust discriminator, session-salted
scope, expansion cap, and telemetry as specified previously. If neither 3B nor
3C nets positive → 3A (lossless) is permanent.

## 4. Experiments (cheap, decisive — before committing to a mechanism)

1. **Native-recovery convergence (fry, primary):** swap ONLY the marker to the
   3B form on the y4q retrieve-forcing task; per-run reap + call-count. Question:
   does agy re-run the named tool and converge (call count → lossless-like, zero
   thrash-timeouts, zero `os.walk` behavior)? If agy *still* `os.walk`s even when
   the marker names the file → native-recovery fails → 3A (lossless) or 3C.
2. **Auto-rehydration observability (37g.13):** does a live marker's hash appear
   in an inbound model-authored region? Only needed if experiment 1 fails or 3B's
   re-reads are too costly.

## 5. Security

- **`os.walk('/')` exfil (carried forward, HIGH):** the opaque marker induces
  filesystem-wide content reads that flow upstream to Gemini. **Native-recovery
  markers REDUCE this** — the model re-reads the *named* file instead of
  searching for a hash. Gate (6) remains net-security: any broad-root
  filesystem-scan behavior (behavioral detection on the reaped process tree, not
  just `find_hash.py`/`search_results.txt` filenames) is disqualifying.
- **Reproducibility trust (3B):** a native-recovery marker only ever names the
  model's OWN prior tool call (re-derived from the `functionCall` the proxy
  already saw) — no new capability, no proxy-authored command. The allowlist
  prevents re-running non-reproducing/stateful calls.
- **Auto-rehydration confused deputy (3C, if built):** model-authored-only
  trigger predicate (scan iff `part.text`@role==model OR `functionCall.args`;
  never `functionResponse`), session-salted HMAC hashes + eviction (end-signal |
  client-dereg | idle-TTL), charset/length validation, per-turn expansion cap,
  decision-record + `x-headroom-fr-expand` header.
- **Measurement (6):** token-ids/lengths only; no retained plaintext; crash-safe
  deletion; never attach a raw transcript to a ticket/PR.

## 6. Decision gate (pre-registered, net-security + net-token)

Decided **before** running, by a **named owner** (align with 37g.7's owner):
- **Correctness:** 0 regressions vs lossless on pinned holdout fixtures (fixed
  config size/keys/gap + a differently-shaped multi-key / summarize case).
- **Security:** 0 broad-root filesystem-scan behavior across the corpus
  (disqualifying).
- **Tokens:** ≥ **[pre-registered X %]** net reduction (charging re-read round-
  trips for 3B, or cache-invalidation for 3C) across ≥ **[pre-registered N]**
  representative real multi-turn agy sessions INCLUDING the cross-turn
  cold-recall case (short-task-only corpus rejected as self-biasing).
- **Outcome if unmet:** **3A (default lossless) is permanent for agy** — the
  honest exit; ccr's compress+retrieve model does not pay off for a
  retrieval-noncompliant client, and lossless strictly dominates the current
  net-negative thrash.

## 7. Acceptance / TDD

- **Unit (no live agy):** functionResponse→functionCall correlation
  (derive the re-run call + args from the preceding `functionCall`); the
  reproducibility allowlist predicate (read_file→native-recovery marker;
  date/curl→lossless); marker byte-stability across turns (deterministic
  `SHA-256[:24]`); the "keep latest-message tool_result verbatim" rule
  (edge cases: no prior functionCall, multiple FR parts, non-allowlisted tool).
- **Integration (fry, quota-gated):** experiment 1 above; process-tree reap;
  zero thrash-timeouts + zero scan behavior.

## 8. Rejected / retired

- **4A live-zone boundary** — refuted (§0): inverted vs proven arch; lossless on
  single-turn, thrash on multi-turn.
- **37g.8 structural-summary head + needle backstop** — compliance bet the
  evidence falsifies.
- **Mirror the proven live-zone arch for agy** — it compresses the live-zone
  tool_result and RELIES on `headroom_retrieve`; that is exactly what thrashes
  for a retrieval-noncompliant client.
- **Longer/smarter retrieve prompt** — model-dependent, contradicted by the
  brute-force-search evidence.
- Keep `4eabc716` (H2 re-compression exemption — defensively correct).

## 9. Related

- Supersedes both prior agy-ccr design docs (2026-07-06 diagnosis / 37g.8, and
  2026-07-07 live-zone-parity).
- `headroom-37g` epic; `37g.7` (lossless default = 3A); `37g.13` (auto-rehydrate
  observability = 3C experiment); `gem` (thrash umbrella); `r9k` (`-p` collision).
- Verified against: `crates/headroom-core/src/transforms/live_zone.rs:36-46,
  515-522,643-683`; `compression_policy.rs:31-38`; `gemini.py:64,93-104,946`;
  `plugins/hermes/headroom_retrieve/__init__.py:19` (schema's "re-run original"
  fallback).
- Independent adversarial review: agy/Gemini 3.1 Pro (PRESENT not RETRIEVABLE;
  aligns with agy's native-tool instinct).
