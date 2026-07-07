# agy ccr thrash — diagnose-first design

<!-- date: 2026-07-06 -->
<!-- status: design-review-gate iter1 = NEEDS_REVISION (PM/CTO/Security/Designer revise, Architect approved); revisions incorporated below; implementation gated on diagnosis -->

## Problem

Under ccr, agy `functionResponse` tool outputs are compressed to opaque markers
(`[functionResponse compressed. Call headroom_retrieve to expand. Retrieve
more: hash=… ]`). On a HARD
cross-turn retrieval task (read a 33 KB / 1200-line config; a second file then
asks for one specific key's value; the config leaf is a marker by answer time),
measured on a clean box (fry, N=8):

- **ccr: 2/8 correct, 4/8 timeout (thrash), 2/8 silent empty-exit.**
- Timeouts are NOT a deadlock — the model makes 123–393 `streamGenerateContent`
  calls at the normal rate, never converging (correct runs: 57–105). It mostly
  does **not** call `headroom_retrieve` during the thrash.
- **lossless: 100% correct, 11–13 s** (N small). So compression/markers break
  convergence on hard retrieval; ccr is never *wrong* (no silent corruption).

## Key reframe (the bar for any fix)

The thrash is a **token bomb**, not "lower accuracy": 3–7× the model-call count,
each re-sending ~full history → **net-negative on tokens**, which is ccr's only
metric. So the success bar is **"restores convergence,"** not "keeps compression
ratio." On hard-retrieval workloads today, ccr is worse than lossless on ccr's
own goal.

## Why NOT the first-tried fix (don't-compress-recent-N)

`HEADROOM_AGY_FR_KEEP_RECENT` (keep last N leaves uncompressed) is **benchmark
overfitting**: N=2 passed only because this task's gap is 2; any gap > N
reproduces the thrash, and if it "works" the mechanism stays latent and
resurfaces in production where no benchmark watches. It also regresses WU1's
cache-coherence (a leaf's bytes change once as it ages past N) and sacrifices
savings (recent reads are often the biggest). It is dominated by turn-boundary
uncompression (same cost, no magic constant). **Reverted** (was uncommitted).
Empirical run was inconclusive anyway (8/8 silent empty-exit — a third agy
print-mode failure mode that also appears in baseline → the `-p` harness is
itself noisy).

## We are fixing blind on a wiretap

headroom is a MITM proxy — we already capture every byte of every thrash run and
have not read one. Four hypotheses; **two make this a bug, not a design change**:

- **H1 — tool not declared:** `headroom_retrieve` may be absent from the
  outbound **Gemini** `functionDeclarations`. `ccr/tool_injection.py:301` prefers
  sticky injection; "Google handler, legacy paths" use a weaker per-request
  fallback. Tool-not-declared → endless reasoning + ~0 retrieve traffic = exactly
  the measured signature.
- **H2 — self-defeating retrieval:** retrieve output may be **re-compressed next
  turn** on the Gemini path. The OpenAI path exempts it
  (`live_zone.rs:2279`); the Gemini `functionResponse` path is unverified. If not
  exempt: retrieve → 33 KB → next turn a marker again → loop.
- **H3 — marker/format confusion.** **H4 — genuine re-reasoning loop** (our prior
  inference; the 2/8 "info unavailable" exits point at H1/H2 instead).

**Cheapest highest-value experiment:** dump ONE thrash-run transcript (body
logging already exists) + grep one outbound Gemini request for
`headroom_retrieve` in the tool block. Collapses H1–H4 in minutes.

## Design (diagnose-first)

1. **Read the wire** (P1): one thrash transcript; verify (i) `headroom_retrieve`
   in outbound Gemini `functionDeclarations`, (ii) does the model emit the call,
   (iii) is retrieve output exempt from re-compression on the Gemini path.
2. **Provisionally default agy → lossless** (P1, ccr opt-in): safety posture
   while diagnosing; justified by the token-bomb arithmetic; zero risk.
3. **Fix informed by (1):**
   - H1/H2 → **bug fix** (declare the tool on the Gemini path / exempt retrieve
     output from re-compression). Design options below become moot.
   - H3/H4 → **mechanism fix**: deterministic **structural-summary marker head**
     (counts + first/last K lines + key-range, content-hashed so WU1's cache
     invariant holds; pairs with the retrieve `query` BM25 param) ± an
     **append-only needle-expansion** backstop (proxy appends the matching
     excerpt at the tail when a later turn references a rare token from a stored
     blob — never rewrites history, cache fully preserved, model-independent).
4. **Turn-boundary uncompression** supersedes recent-N if a positional lever is
   ever wanted (same cost, principled, no constant).

## Rejected

- ship-a-palliative-now (overfit; token bomb persists at gap > N)
- drop-ccr-for-agy entirely (abandons the savings investment prematurely)
- LLM-summary marker head (nondeterministic → kills WU1 byte-stability)
- non-progress auto-expand as primary (heuristic patching a heuristic; rewriting
  history mid-thrash nukes the prefix cache)

## Design Review Gate — revisions (iteration 1)

5-agent gate: **Architect APPROVED** (and code-confirmed the two bug hypotheses:
**H2** — `_compress_agy_function_responses` (gemini.py:963-980) walks *every*
functionResponse part and `_compress_fr_leaf` (:878) guards only its own marker,
so `headroom_retrieve`'s output is re-compressed into the same marker it expanded
from = self-defeating loop; no live_zone-style name exemption on the Gemini path,
conf 88. **H1** — the Gemini handler makes zero tool-injection calls; declaration
depends on agy's MCP wiring, the Google injector path is a weak uninvoked
fallback, conf 82). PM/CTO/Security/Designer = NEEDS_REVISION. Required changes,
folded into the WUs:

### Diagnosis (37g.6)
- Dump a **timeout** transcript (not correct/empty), **≥2**, to confirm the
  H3/H4 signature; done-criterion = an **evidenced verdict on all four H**, not
  "looked at it". Check **H2 first** (cheapest; Architect rates it the likely bug).
- **[SECURITY BLOCKER T1]** the transcript contains raw tool outputs (repo files,
  secrets). Before any dump: write to a fixed **local-only path, mode 0600, in a
  gitignored dir**; run the existing retrieve-log **secret-redaction** helper over
  it; **delete after diagnosis**; **never** attach the raw transcript to a
  ticket/PR/artifact.

### Provisional lossless default (37g.7)
- Add an explicit **graduation/rollback criterion + owner**: this is an interim
  safety valve, not a permanent default — state the exact condition and ticket
  that flips it back (e.g. "37g.8 fix passes acceptance → restore ccr default").
- **Conditional:** only flip if diagnosis does NOT show H1/H2 is a quick bug fix
  that restores convergence outright (a one-line exemption may make the flip
  unnecessary).
- **Estimate hard-retrieval frequency** in real agy usage so the savings-forfeit
  cost of a blanket flip is known, not assumed (blanket flip also gives up ccr on
  the easy tasks where it already works).

### Mechanism fix (37g.8) — only if diagnosis lands on H3/H4
- **Numeric acceptance criteria** (was qualitative): convergence ≥ lossless
  (zero thrash-timeouts across the benchmark) **AND net tokens < lossless**
  (not just < current-ccr). Ratio is explicitly NOT the metric.
- **Characterize/stabilize the `-p` harness noise floor FIRST**: the silent
  empty-exit failure appears in *baseline* too, so N=8 cannot be a trustworthy
  gate until that confound is quantified/removed. (Pre-req sub-task.)
- **Differently-shaped holdout task** (larger gap, multi-key, or
  summarize-not-retrieve) to prove generalization — the append-only backstop
  triggers on *exactly* this benchmark's rare-token→blob shape, the same
  overfit trap recent-N fell into.
- **H1/H2 and H3/H4 are NOT mutually exclusive**: fixing tool-declaration and
  still thrashing on opaque markers is possible; state the bug fix and the
  mechanism fix as *jointly sufficient*, don't close the mechanism track the
  moment a bug is confirmed.
- Structural-summary head **cache-stability constraints**: derive from the RAW
  stored bytes (slice raw leaf, not a re-serialized form); iterate object keys in
  deterministic order — any tokenizer/dict-ordering dependence reintroduces
  per-turn byte drift and breaks WU1's invariant.
- **Idempotency exemption (this IS the H2 fix — bake it in):** already-marker
  content AND `headroom_retrieve` tool output are EXEMPT from structural-head
  compression. Without this the mechanism fix re-creates the H2 loop.
- **Binary/unstructured content fallback:** first/last-K-lines + key-range is
  line/KV-oriented; for binary/unstructured blobs use a size+mimetype head (or
  keep the existing `<<ccr:hash,base64,size>>` variant) — do not preview binary.
- **Marker proliferation:** `HEADROOM_RETRIEVE_SCHEMA` already documents 3 marker
  shapes; adding a 4th feeds H3 (marker/format confusion). Update the schema
  description in THIS WU (not the deferred 37g.5), and prefer replacing an
  existing shape over adding one.
- **In-marker decision instruction** (cheap anti-H4 insurance): the head must
  spell out the decision procedure, e.g. "if the key you need is in this range,
  call headroom_retrieve(query=<key>); otherwise the value is not here — do not
  loop." Don't leave the model to infer it from format.
- **Tiny content:** define the first-K/last-K dedupe rule when total lines < 2K
  (don't show the same lines twice).

### Security — append-only backstop (37g.8), if built
- **[BLOCKER T2]** the store is a **global unscoped singleton**; a needle
  backstop that auto-appends on rare-token match enables cross-session exfil
  (session A's token matches session B's blob) and is triggered by **untrusted**
  tool output (planted rare token / `<<ccr:…>>` marker → injection-driven pull).
  The backstop index + expansion MUST be **strictly session-scoped** (bind
  entries to a session id; match only same-session hashes) and treat
  tool-output-embedded markers/tokens as untrusted.
- **[T3/T4]** document the single-user-local trust assumption for the global
  content-addressed store; validate the model-emitted `hash` charset+length
  before lookup; scope the BM25 `query` to the session.

## Related

- `headroom-37g` (epic), `headroom-gem` (thrash umbrella), `headroom-y4q`
  (closed: ccr correct/no-silent-degradation; residual = this thrash),
  `headroom-37g.5` (marker consolidation, separately deferred).
- Independent adversarial analysis: fable (Gemini) — token-bomb reframe, H1/H2,
  read-the-wire imperative, structural-summary + append-only backstop ranking.
