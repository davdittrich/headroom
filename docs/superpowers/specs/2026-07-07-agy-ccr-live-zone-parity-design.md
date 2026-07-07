# agy ccr thrash — live-zone parity (supersedes 37g.8)

<!-- date: 2026-07-07 -->
<!-- status: design-review-gate pending -->

## Problem

Under ccr, agy `functionResponse` tool outputs (file reads, command output) are
compressed to opaque markers
(`[functionResponse compressed. Call headroom_retrieve to expand. Retrieve
more: hash=<24hex>]`), recoverable only via the injected `headroom_retrieve`
MCP tool. On a cross-turn retrieval task (agy reads a 33 KB config; a later
turn asks for one key's value; the config leaf is a marker by answer time),
ccr does not converge.

**Measured, clean box (fry), instrumented, per-run reap, 2026-07-07:**

- **lossless: converges** — 5-7 model calls, ~12-14 s, correct (100%).
- **ccr (with the shipped H2 fix `4eabc716` present): thrashes 3/3** — 120 s
  timeout, no answer, **35-42 model calls** (6-8× = a token bomb; net-**negative**
  on tokens, ccr's only metric).

## The decisive new evidence (why the prior track was wrong)

During the thrash the model **ignores `headroom_retrieve` entirely.** Instead it
spawns subprocesses running `os.walk('/')` + `glob.glob` to **brute-force-search
the whole filesystem** for the marker's hash string, reading file contents and
writing `find_hash.py` + `search_results.txt` into the workspace. The hash
exists only in the model's own conversation transcript (the content lives only
behind `headroom_retrieve`), so the search is futile and it loops.

Reframe: **a coding agent given an opaque marker for content it needs will not
call an injected retrieve tool — it reverts to its native file tools and
flails.** Therefore the fix must make the needed content **PRESENT**, not
**RETRIEVABLE.** Any marker-based scheme that depends on model compliance with a
retrieve instruction is a bet against live evidence.

## How headroom already solves this for the other clients

The mature Anthropic/OpenAI paths do **not** rely on the model retrieving. They
compress under a **live-zone compression policy** (Rust engine):

- `crates/headroom-core/src/compression_policy.rs` — `live_zone_only`,
  `live_zone_compression_enabled()`; Subscription mode is *"live-zone-only"*,
  PAYG *"can touch outside live zone"*.
- `crates/headroom-core/src/transforms/live_zone.rs` — *"the live zone: the
  blocks the model will emit a response against."*
- `headroom/proxy/handlers/anthropic.py` — *"route exclusively to the live zone
  tail"*, `tool_results[-5:]  # only recent results`, live-zone token
  accounting.

I.e. the recent content the model responds against stays **present (verbatim)**;
only **cold history** compresses — cache-aligned, recency-aware,
model-independent.

**The gap:** the agy Gemini FR path (`_compress_agy_function_responses` in
`headroom/proxy/handlers/gemini.py`) bypasses this — it walks *every*
functionResponse leaf and compresses regardless of recency or the live-zone
policy. The just-read hot config gets compressed while the model still needs it.
That gap *is* the thrash.

## Design

Unifying principle: **content PRESENT, not RETRIEVABLE.**

### (2) PRIMARY — live-zone/recency parity for the agy FR path

Make `_compress_agy_function_responses` respect the **same live-zone boundary**
the Rust `compression_policy` enforces for the other clients: never compress
functionResponse leaves inside the live zone (the recent turns the model emits
its next response against); compress only cold history that has aged out.

- Reuse the existing live-zone concept rather than inventing a new marker shape.
- Turn-boundary slice (a leaf's bytes change at most once, when it ages past the
  boundary) — preserves WU1's cache invariant (old leaves freeze into stable
  markers; the live tail stays verbatim).
- Model-independent, zero compliance bet, minimal LoC (a boundary check in the
  leaf walker), zero new security surface.
- Known limit: masks — does not cure — **cold recall** (ask about a file read
  far outside the live zone → still a marker → could still thrash). Addressed
  conditionally by (3).

### (3) CONDITIONAL — deterministic auto-expand (cold-recall fallback)

Only if (2) proves insufficient for cold recall **and** the net-token gate (5)
justifies the cost:

- **Deterministic, self-signaling trigger:** the failure announces itself — a
  request whose stream references / tool-calls a *live* marker's 24-hex hash (or
  searches for it). This is an exact string match against a known live hash, not
  a semantic guess.
- On trigger: the proxy substitutes the real blob inline into the stream before
  the model sees the next turn (append-only at the reference point; never
  rewrites unrelated history).
- **Costs (why it stays a rare fallback):** re-inflates content at max context
  depth and **invalidates the cache prefix from the injection point.** Fire only
  on the deterministic thrash signal so it is rare enough to keep the cache cost
  bounded.

### (4) KILL 37g.8

Retire the structural-summary marker head + in-marker decision instruction +
needle backstop. It doubles down on the premise the live evidence falsifies
(marker-instruction compliance): for a specific-key query the value is not in
the structural head → the model must still call `headroom_retrieve` → it won't.
A longer instruction is not more persuasive to an agent that ignores the short
one. Keep `4eabc716` (the H2 re-compression exemption is defensively correct;
it is simply not sufficient).

### (5) DECISION GATE — net-token measurement decides 2+3 vs just-lossless

Measure **net tokens** of recency-keep (2) across *real* multi-turn agy
sessions vs lossless. Recency-keep leaves the hottest, largest payloads
verbatim — exactly the content most worth compressing — so on short tasks
savings approach zero. If (2) [+ (3) if built] is **not clearly net-positive vs
lossless**, the honest outcome is: **default lossless for agy and stop** (concede
ccr does not pay off for an agent that won't cooperate with retrieval). This
gate prevents shipping machinery that costs more than it saves.

## Acceptance criteria

- Re-run the clean-fry harness (`fry_run.sh` + `fry_seq.sh`: per-run reap +
  call-count instrumentation, extended to reap agy's whole process tree):
  ccr model-call count approaches lossless (~5-7), **zero thrash-timeouts** on
  the retrieval task.
- Differently-shaped holdout (larger gap, multi-key, summarize-not-retrieve) to
  prove generalization, not benchmark overfit.
- Net tokens across real agy sessions clearly positive vs lossless — else ship
  lossless-default per (5).

## Rejected alternatives

- **37g.8 structural head + needle backstop** — compliance bet against live
  evidence (see (4)).
- **Exempt all re-fetchable file reads from compression** — stops the thrash
  (agent re-reads), but file reads are the bulk of a coding CLI's traffic;
  exempting them guts ccr's savings. Acceptable stopgap, poor as the design.
- **Longer / smarter retrieve prompt** — model-dependent, fragile, contradicted
  by the brute-force-search evidence.
- **"Ghost file" (marker as a magic path the proxy intercepts)** — infeasible:
  headroom is MITM on the LLM stream only; the agent's `cat`/shell runs
  client-side and the proxy cannot materialize a file on the agent's disk. The
  salvageable form collapses to (3) auto-expand.

## Security

- **Auto-expand (3) content-matching surface:** a tool output whose bytes
  contain a 24-hex string colliding with a live marker hash could drive a
  confused-deputy expansion of the wrong/attacker-chosen blob. Bind expansion to
  the current session's live markers only; validate hash charset+length; treat
  tool-output-embedded hashes as untrusted; match only same-session store
  entries.
- **Store trust:** the content-addressed store is a single-user-local
  singleton; document the trust assumption; scope the BM25 `query` and any
  expansion to the session.
- **No raw-transcript exfil:** diagnosis transcripts (raw tool outputs) stay
  local, 0600, gitignored, redacted, deleted after use — never attached to a
  ticket/PR.

## Related

- Supersedes `docs/.../2026-07-06-agy-ccr-thrash-diagnosis-design.md` (37g.8
  mechanism track).
- `headroom-37g.8` (to retire), `headroom-37g.7` (provisional lossless default,
  separate), `headroom-gem` (thrash umbrella + mechanism evidence),
  `headroom-r9k` (`-p`/`--port` collision, independent).
- Independent adversarial review: agy/Gemini 3.1 Pro (concurred: PRESENT not
  RETRIEVABLE; recency-keep primary, auto-expand deterministic fallback,
  lossless honest floor; reject 37g.8).
