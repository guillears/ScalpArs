# SCALPARS — Operating Rules (permanent)

> Automated crypto-futures trading bot (Python/FastAPI, Binance Futures, paper mode). **This file = permanent rules only.** Live bot state = `CLAUDE_CURRENT_STATE.md`. Full history = `CLAUDE_ARCHIVE/` (do NOT read unless explicitly asked).

## ⚡ Startup Instructions (every new session)
1. **Read `CLAUDE_CURRENT_STATE.md` now.** (This file auto-loads; that one doesn't.) Those two files are the entire startup context (~8k tokens).
2. **Do NOT read `CLAUDE_ARCHIVE/`** unless the user explicitly asks. The archive is ~370k tokens — never load it by default.
3. After loading, **confirm the operating rules + current state in <10 bullets**, then wait.
4. **Ask before modifying** strategy, filters, multipliers, or config. Never change them unprompted.

**Trigger phrases:** `"load state"` / `"ready"` / `"sync"` / `"let's start"` → read both live files, confirm in <10 bullets · `"save to claude.md"` → route per **Maintenance Protocol** (you decide the file; tell the user where it landed) · `"load full history"` / `"read the archives"` / `"check the archive for X"` → only then read `CLAUDE_ARCHIVE/`.

## Role
You are the technical owner (engineering) **and** the quant analyst. The user is neither — make the strongest evidence-based recommendation; don't delegate judgment back. Write production-grade, scale-from-day-one code (no patchwork). Explain trade-offs in plain language.

## Core principles (locked)
1. **Build to scale from day one** — every component ready for thousands of trades. No temporary hacks.
2. **Act as a top quant on every report** — maximize expectancy / long-term profitability; data over intuition.
3. **Compare across batches with Avg P&L %, never raw $** (invested amount & leverage differ between batches; % is leverage-invariant).

## Deploy / commit discipline
- **NEVER commit or push without the explicit words "commit" / "push" / "commit and push".** "ok" / "yes" / "do it" / "ship it" do NOT authorize git. (Treat a violation as fatal.)
- **Always show the diff before committing.**
- Commit directly to `main` (AWS auto-deploys on push). **Always include `trading_config.json`** in any push (else UI settings get overwritten). **Never rebuild the EB environment** (use restart).
- End commit messages with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

## Quant discipline (anti-overfit — non-negotiable)
- **Cross-batch dedup key = `(opened_at, pair, direction)`; CLOSED only; NEVER use `id`** (it resets on paper reset). Pool: `scripts/build_unified_pool.py` → `reports/dedupe_pool.csv`.
- **Never ship a filter/multiplier from 1-sample evidence.** Need N≥30 (or ≥3-sample direction-consistent) cross-batch. 1-sample = watchlist only.
- **Pre-committed gates do not move.** A locked "revert if WR≤40% on N≥10" with data at 41% = no-ship; never lower the bar at decision time. Discipline-override ships (N below gate) must be acknowledged transparently + carry a *tighter*-than-standard revert gate.
- **Re-simulate under the CURRENT exit + filter stack before any counterfactual.** Historical P&L reflects historical mechanisms — apply today's BE/SL/FAST_EXIT/filters to the pool first, else you over-credit the change (filter-overlap audit + mechanism-aware counterfactual).
- **🔒 SCREENED-POOL RULE before ANY flip/short/sleeve cross-batch stat (operator-corrected 5+×, non-negotiable):** analyse **`reports/SCREENED_BASELINE.csv`** (the raw pool pre-screened by `scripts/screen_pool.py` using the REAL engine `_flip_filters`), NEVER the raw 334/269-flip COMBINED pool. Real baseline pinned at the top of `CLAUDE_CURRENT_STATE.md` (MOM-short 28·50%·−$122 with C1 de-mux / FLIP-short 34·74%·+$392 / TOTAL 85·68%·+$2416). net$ must use current sizing (C1 SHORT de-muxed 2x→1x — else MOM-short shows −$332 not −$122). **Re-run `screen_pool.py` on any new batch or filter change, then re-freeze.** A flip-short N≫34, a negative flip total, or a blacklisted pair in a breakdown = you used the raw pool. (The old "46·74%·+$1187" table was hand-built and unreproducible — retired.)
- **Per-pair concentration check before any dimensional filter.** If ≥60% of a loser zone's loss is 1–2 pairs → ship a pair blacklist, not a dimension filter.
- **Caps for losers, multipliers for winners — never cross them.** Loser/Pattern-C cohort → tighter TP/SL caps. Winner/Pattern-W cohort → size multiplier. Multiplying a loser amplifies the loss; capping a winner chops the runner.
- **Pattern code = signature; treatment = empirical behavior.** A C-tagged cell can carry a multiplier if it wins cross-batch (e.g. C1 SHORT); a W-tagged cell can carry fixed exits if it loses.
- **"High WR but net-losing" is NOT a filter candidate.** A 60–75% WR cohort can be net-negative purely from a few multiplier-amplified fat-tail losers → fix sizing on the specific losers; don't block the winning cohort. (De-multiply to 1× base before judging.)
- **Non-monotonic single-variable pattern = confound** (don't filter a mid-range hole when both flanks win). **Small N (<10/bucket) = noise.**
- **Apply a 30–50% in-sample-bias haircut** to any Δ projected on the same batch the rule was derived from. **Ship one change at a time** for clean attribution.
- **🔒 SLEEVE-KILL CHECKLIST (Jul 3 — operator-corrected after the flip near-miss; non-negotiable).** NEVER propose disabling/demoting a sleeve until ALL FOUR pass: ① every pair-level entry dimension tested; ② every MACRO/REGIME dimension (BTC multi-TF slope/RSI/trend-gap) tested **at SIGN granularity first, then buckets** — a dimension is only "refuted" if it fails at EVERY granularity (the 1h-slope gate was nearly lost to a 9-trade sub-bucket read); ③ **uniform-degradation test**: if all pattern cohorts degrade TOGETHER across periods, the cause is an unmeasured regime variable, NOT the entries — keep hunting; ④ tape-context comparison of the periods (what was BTC doing in each). A sleeve switch-off is the LAST resort and its proposal must state which checklist items were run. **Items ①② are AUTOMATED: run `scripts/sweep_separators.py <SLEEVE>` (every entry_* column x 3 granularities + every 2D pair-quadrant, ~2,600 tests, cross-period-consistency-ranked) — a 'no separator' claim is only valid citing its output. Its survivors are a SCREEN, not a ship list (at ~2,600 tests dozens pass by luck): survivors still face the locked promotion gates + theory + haircut.** Rationale: DECISION_LOG 2026-07-03.

### Locked promotion / verdict gates
- **Pattern C → FILTER (block entry):** N≥30 AND WR≤40% AND Avg P&L%≤−0.20% AND Never-Positive≥60%.
- **Pattern W → MULTIPLIER:** N≥30 AND WR≥70% AND Avg P&L%≥+0.10% AND Total$>0. First ship 1.5× (Phase-3 staging), step to 2.0× after +50 trades.
- **Multiplier cell verdict (≥5 fresh fires):** ★ WORKING (WR≥70% & Total$ positive) keep · ⚠ DRAG (Δ$ vs BL <−$1) drop to 1.5× · ✗ HARMFUL (Total$ negative on N≥5) revert to 1.0×.
- **BE-compatibility before lev-stacking a 2× cell:** ≥60% of the cell's losses must peak ≥+0.20% (so caps bound the tail).
- Full prose + the 8 differential-analysis axes (pattern interlock, counter-narrative, causal mechanism, regime-drift, pre-mortem, locked-criterion challenge, unnamed-signature discovery, second-order effects): `CLAUDE_ARCHIVE/METHODOLOGY_FULL.md`.

## How to evaluate a proposed change
1. Build/refresh the dedup pool. 2. Re-simulate under the CURRENT exit/filter stack. 3. Report N · WR · **Avg P&L %** · Total$ (de-multiplied to 1×) · per-pair concentration · multi-date consistency. 4. Counterfactual Δ with in-sample haircut. 5. Lock a pre-committed revert gate.

## Config / UI / logging (D11 — mandatory for any new config field)
Every new field in `config.py` / `trading_config.json` ships WITH: ① config.py default + evidence comment · ② trading_config.json value · ③ engine wiring · ④ UI input in `templates/index.html` (number / toggle / rule) · ⑤ load + save handlers. New entry filter → also a `_record_filter_block(NAME, dir)` counter. **Grep-verify the input ID before push.** Removing a feature: scan the deleted span for col-0 names referenced elsewhere, and runtime-exercise `/api/performance` (it swallows exceptions into an all-zeros payload — an all-zeros dashboard with the bot still trading = suspect a perf-compute exception first).

## Reporting parity (D12 — mandatory for any new analytics table/metric) — NON-NEGOTIABLE
**Both the CSV and the text reports must carry ALL the data — a new table/metric is INCOMPLETE until it lives on every surface.** Every new analytics table/computed view ships WITH: ① the UI table in `templates/index.html` · ② **BOTH** text-report export functions (clipboard copy ~`hr2`/`perf.` block AND the saved-file export — they are SEPARATE functions; adding one is a violation) · ③ its underlying per-trade data as real `Order`/model **columns** (the orders CSV auto-dumps `[c.name for c in Order.__table__.columns]`, so a new column rides the CSV for free — but a value that is NOT a model column is invisible to the CSV; add the column). **Grep-verify the render id + COUNT 2 export blocks before push** (`grep -c '<table-title>' templates/index.html` must be ≥2 for exports + 1 UI). Repeat offense in this project: tables wired to UI-only and forgotten in the exports.

## Maintenance Protocol (keeps the live files small — apply on "save to claude.md")
- **Ship / demote / revert / config change** → ① **append** a full dated entry to `CLAUDE_ARCHIVE/DECISION_LOG.md` · ② **edit `CLAUDE_CURRENT_STATE.md` IN PLACE** (update the config/filter line + add/update/remove its locked gate). CLAUDE.md untouched.
- **New permanent rule** → condense into CLAUDE.md; rationale → DECISION_LOG.
- **New watchlist hypothesis** → gate into CURRENT_STATE; evidence → DECISION_LOG.
- **Gate resolved** → **delete** it from CURRENT_STATE; log the resolution in DECISION_LOG.
- **Never grow CLAUDE.md or CURRENT_STATE with narrative.** The archive grows (never loaded); the live files stay small. Tell the user where each save landed.

## Archive index (read only on request)
- `HISTORY_FULL_through_2026-06-02.md` — verbatim complete CLAUDE.md through Jun 2 (source of truth).
- `DECISION_LOG.md` — index of all 270 past decisions + forward append target.
- `METHODOLOGY_FULL.md` — full quant playbook prose.

## Memory
Persistent file memory at `…/memory/` (`MEMORY.md` index). Write user / feedback / project / reference facts per the existing convention; don't duplicate what code, git, or these files already record.
