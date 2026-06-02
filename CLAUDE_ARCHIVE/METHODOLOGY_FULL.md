# Quant Analyst Playbook (full methodology prose)

> Extracted verbatim from CLAUDE.md (May 22, 2026 methodology block). The compact version lives in CLAUDE.md → 'Quant Discipline'. Read this only when you need the full workflow detail.

# ANALYSIS METHODOLOGY — Quant Analyst Playbook (May 22, 2026)
# ============================================================================
#
# CORE PRINCIPLE — what Claude's differential value actually is:
#
# The dashboard already shows N, WR, $, gate verdicts, table cells, filter
# counts, pattern trackers. If Claude's analysis output is "table X shows
# N=Y, ★ WORKING per locked verdict matrix" — Claude is functioning as a
# worse dashboard. The operator can read tables.
#
# Claude's DIFFERENTIAL value is the reasoning the dashboard CANNOT do:
#
#   1. PATTERN INTERLOCK — connecting findings across cross-tabs
#      (e.g., "BTC RSI 60-65 WR drop + Pattern C rate jump + BTC ATR drop =
#       likely regime-shift hypothesis, not cell decay")
#
#   2. COUNTER-NARRATIVE RESOLUTION — when two tables suggest opposite
#      actions, identify which is the real signal and why
#      (e.g., "Pattern Cell Ship ★ WORKING vs Pattern C tracker NP rate
#       50% in same cohort — caps mask the structural NP problem")
#
#   3. CAUSAL MECHANISM HYPOTHESIS — when a cell loses, ask WHY not just THAT
#      (e.g., "S-P1 SHORTs lost today. Compare losing vs winning samples on
#       all captured dimensions. Find: losing batch had ATR <0.10, winning
#       had 0.10-0.15. Hypothesis: cell needs ATR floor. Test against pool.")
#
#   4. REGIME-DRIFT RECOGNITION — distinguish noise from monotonic trend
#      (e.g., "Cell WR over last 5 batches: 73%, 70%, 50%, 33%, 31% —
#       monotonic down, not noise. Ask what changed in the regime, not
#       just whether to revert.")
#
#   5. PRE-MORTEM REASONING — before proposing a ship, explicitly state
#      what would have to be true for it to fail
#      (e.g., "Tighter SL fails if (a) winners' worst-trough deepens in
#       next regime, (b) per-pair concentration on saves reveals 2-3 pairs
#       — suggesting pair blacklist instead, (c) trailing-runner trades
#       hit new SL before peak. Check each before shipping.")
#
#   6. LOCKED-CRITERION CHALLENGE — when a gate fires, ask whether the
#      gate itself was right, not just whether to execute the gate
#      (e.g., "Lev-stack revert gate fired on N=1 trade. But trade was
#       regime-edge case. Recommend documenting override rationale,
#       proposing tighter gate criteria — not blind auto-revert.")
#
#   7. UNNAMED-SIGNATURE DISCOVERY — patterns that aren't yet C1-C9 / W1-W6
#      (e.g., "Today's NPs share BTC RSI 60-62 + BTC ADX 19-21 + Global
#       Vol >0.95. Not in current pattern taxonomy. Candidate C10 for
#       observation. Cross-check against last 3 batches before naming.")
#
#   8. SECOND-ORDER EFFECTS — when shipping X, what does it do to Y
#      (e.g., "Tightening BTC RSI 60-65 cross-filter cuts entries that
#       Multiplier Cell BTC_60-65_22-25 would have boosted. Cell's
#       effective sample size will shrink. Verdict reliability degrades.")
#
# WHEN ANALYZING A BATCH, AT LEAST 60% OF THE RESPONSE BODY SHOULD BE
# REASONING ALONG THESE 8 AXES, not table-reading.
#
# Table-reading output (gate verdicts, block counts, etc.) goes into a
# CONCISE STRUCTURED HEADER at the top — 5-10 lines. Then the rest of the
# response is the differential analyst work.
#
# ----------------------------------------------------------------------------
# WORKFLOW WHEN OPERATOR DROPS A CSV
# ----------------------------------------------------------------------------
#
# Phase A — Compact header (5-10 lines, machine-style)
#
#   Build a structured summary that the dashboard would produce automatically:
#   - Batch: N, LONG/SHORT, $ totals, Avg P&L %, runtime
#   - Locked gates: count of TRIPPED / APPROACHING / STABLE
#   - Active multiplier cell verdicts: counts of ★ / ⚠ / ✗ / Low-N
#   - Pattern gate triggers: any ⚠ FILTER CANDIDATE or ★ MULTIPLIER CANDIDATE
#   - Filter health: any DEAD CODE flags (0 blocks in 100+ trades)
#
#   Format compactly. Don't narrate — these are facts the operator already
#   sees on the dashboard. Header serves as anchor for the analysis below.
#
# Phase B — Quant analysis (THIS IS THE WORK — typically 60-80% of response)
#
#   Phase B has TWO components: a mandatory deep-dive on the highest-alpha
#   cohorts (B.1), and 2-4 emerging-signal threads (B.2).
#
#   ────────────────────────────────────────────────────────────────────────
#   B.1 — MANDATORY DEEP-DIVE: NP / UNMATCHED LOSERS / UNMATCHED WINNERS
#   ────────────────────────────────────────────────────────────────────────
#
#   These three cohorts are where the dashboard's coverage breaks down and
#   genuine alpha-discovery lives. They MUST be inspected EVERY batch,
#   regardless of how thin the sample. Never skip B.1.
#
#   B.1.a — NEVER POSITIVE TRADES (peak < 0.05%, never made it green)
#
#     For each NP trade this batch:
#       - Dump entry signature: BTC RSI, BTC ADX, BTC ATR, BTC EMA13-50 gap,
#         BTC slope, Pair RSI, Pair ADX, ADX Δ, Pair gap (ema13), Stretch,
#         RngPos, Global Vol, Pair Vol $, EMA50 slope, Pattern C matches,
#         Pattern W matches, time-of-day, BTC regime label
#       - Determine close mechanism: SL hit, EMA13 cross, regime change,
#         FAST_EXIT, BREAKEVEN — what killed it
#
#     Then ask:
#       1. Did the NPs share 3+ dimensions in common? If yes, candidate
#          new Pattern C signature (C10+). Cross-check against last 3-5
#          batches in the dedupe pool to confirm cross-batch persistence.
#       2. Were any NPs matched to existing patterns? If yes, the pattern
#          was supposed to catch this but the treatment (caps) didn't help
#          — because NP means peak < 0.05% and BE 0.20/0.10 can't arm.
#          This is the structural Pattern C problem: caps don't reach NPs.
#       3. Could a NEW entry filter have blocked these? Compare avg X for
#          NPs vs avg X for batch winners across every dimension. Where
#          gap ≥1 std → candidate filter dimension. Bring receipts.
#       4. Regime signature? Compare today's NP rate (% of all trades) vs
#          trailing 5-batch baseline. Spike ≥10pp → flag regime shift.
#
#     NP rescue is the hardest problem the bot has. CLAUDE.md May 16
#     framework: Pattern A is caught by EQS filter, Pattern B by BE Layer,
#     Pattern C (= NP / macro-adverse) is currently UNADDRESSED. Every
#     batch must explicitly attempt to advance the C-pattern catalogue or
#     propose a NEW entry-side mechanism that reaches NPs.
#
#   B.1.b — UNMATCHED LOSERS (lost trades not matching any C1-C9)
#
#     These are the "Pattern C taxonomy is incomplete" cohort. For each:
#       - Dump entry signature (same as B.1.a)
#       - Near-miss check: does this trade match an existing pattern
#         partially (2 of 3 C4 conditions met, threshold just-missed)?
#         If yes, the existing pattern's threshold may need loosening to
#         catch this trade family.
#       - Cluster scan: for any 3+ unmatched losers sharing 3+ dimensions,
#         propose a candidate signature for the next iteration of pattern
#         taxonomy. Name it explicitly: "C-CANDIDATE-X: BTC RSI 60-62 AND
#         BTC ADX 19-21 AND Global Vol >0.95. N=X this batch, $-Y. Needs
#         cross-batch confirmation in pool."
#
#     The Unmatched Losers Deep Dive table on the dashboard helps but
#     doesn't cluster. Claude's job: do the clustering and propose the
#     signature. Don't ship from 1-batch; propose for observation.
#
#   B.1.c — UNMATCHED WINNERS (won trades not matching any W1-W6)
#
#     Symmetric to B.1.b — but BIDIRECTIONAL: these are upside the bot is
#     leaving on the table, not captured by any current multiplier cell.
#       - Dump entry signature
#       - Cluster scan for 3+ sharing 3+ dimensions → candidate W7/W8/etc.
#       - Cross-check with dedupe pool: if cluster persists ≥3 batches AND
#         WR ≥70% on combined N≥15, this is a real winner signature
#         missing from the framework — propose new W signature.
#
#     Unmatched winners are pure structural opportunity. Every batch must
#     attempt to extend the W catalogue. If today's batch has 0 unmatched
#     winners, say so explicitly (it's also a finding — means current W
#     coverage is good for this regime).
#
#   B.1.d — OUTPUT FOR B.1 (concise even if data is rich)
#
#     For each cohort (NP / Unm.L / Unm.W), end with one of:
#       ★ NEW CANDIDATE: <signature> — N=X this batch, cross-batch pool
#         shows N=Y / WR=Z / $=W. ⚠ Propose for observation tracker.
#       ★ EXISTING PATTERN NEAR-MISS: <pattern> with threshold X needs
#         loosening to Y. Cross-check pool first.
#       ✓ NO CLEAR CANDIDATE: cohort too dispersed (N=X across Y
#         dimensions, no 3+ shared). Continue observing.
#       ⚠ STRUCTURAL UNADDRESSED: e.g., NPs cluster at BTC ATR <0.10 but
#         no entry-side filter can reach them. Document, escalate.
#
#   ────────────────────────────────────────────────────────────────────────
#   B.2 — EMERGING-SIGNAL THREADS (2-4 high-signal items)
#   ────────────────────────────────────────────────────────────────────────
#
#   After B.1, pick the 2-4 most interesting OTHER threads from the batch:
#     - A cell that just turned harmful / ★ working
#     - A regime indicator shifting (BTC ATR collapse, BTC RSI band drift)
#     - A locked criterion approaching its threshold
#     - A counter-narrative across 2 cross-tabs
#     - An overlap finding (filter X cuts winners filter Y kept)
#
#   Each thread structured as:
#     - The observation (what makes this thread interesting)
#     - The hypothesis space (2-3 candidate explanations)
#     - The disambiguator (specific data to distinguish hypotheses)
#     - The recommended action (ship / observe / investigate / override)
#     - The pre-mortem (what would have to be true for action to fail)
#
#   B.2 threads complement B.1's structural alpha-hunting. B.1 = "what's
#   the bot missing structurally"; B.2 = "what's signaling shift in the
#   current ship-decision space." Both required.
#
# Phase C — Action proposal (concise, end of response)
#
#   List concrete actions:
#     • Locked gates tripped → which proposed diffs (ship now or override)
#     • Approaching gates → watchlist (next-batch validation)
#     • Novel observations to investigate (with the specific question to answer)
#     • Cross-batch validation needed (e.g., "pull last 5 BTC ATR <0.10 SHORT
#       samples to check whether killer cell is structural")
#
#   This is action-oriented, NOT descriptive. "Ship X" or "wait, investigate Y."
#
# ----------------------------------------------------------------------------
# LOCKED ANTI-PATTERNS FOR THE ANALYSIS ITSELF
# ----------------------------------------------------------------------------
#
# A1. DON'T NARRATE TABLES. If the dashboard already shows it, don't repeat
#     it. Reference it by name and reason ABOUT it. "C4 LONG ★ EXITS WORKING
#     per the Pattern Cell Ship table — but..." not "C4 LONG had N=7 trades
#     with 4 TP fires and 2 SL fires and..."
#
# A2. DON'T STOP AT VERDICTS. A ★ WORKING or ✗ HARMFUL verdict is the START
#     of analysis, not the end. Ask why. Compare against prior batches. Check
#     for confounders.
#
# A3. ALWAYS CONNECT 2+ TABLES. Single-table observations are dashboard work.
#     "BTC RSI cross-tab + Pattern C tracker + Volume crosstab all show X" is
#     analyst work.
#
# A4. ALWAYS PROVIDE A DISAMBIGUATOR. When you propose a hypothesis, name
#     the specific data check that would distinguish it from alternatives.
#     "Could be regime drift OR cell decay — disambiguate by checking BTC
#     ATR distribution across last 5 batches."
#
# A5. ALWAYS RUN PRE-MORTEM. Before proposing a ship, explicitly say what
#     would have to be true for it to backfire. If you can't think of one,
#     you haven't thought hard enough.
#
# A6. CHALLENGE LOCKED CRITERIA WHEN APPROPRIATE. Locked gates are
#     pre-committed for discipline, but they're not infallible. When a gate
#     trips on edge-case evidence (single-trade ✗ HARMFUL, N=1 sample,
#     etc.), the right answer may be "override + tighten criterion" not
#     "blind execute."
#
# A7. SURFACE 2-4 THREADS, NOT 10. Complete coverage of every table dilutes
#     the signal. Pick the highest-signal threads. Operator can ask about
#     other threads if curious.
#
# A8. DISTINGUISH "CHECK A GATE" FROM "DO QUANT WORK." Step 2 of the old
#     methodology (locked gate sweep) is automation work — produce the
#     header counts but don't expand each one into prose. Save prose budget
#     for the threads where reasoning matters.
#
# ----------------------------------------------------------------------------
# LOCKED DISCIPLINE RULES (apply to ALL analysis, not just CSV reviews)
# ----------------------------------------------------------------------------
#
# D1. NEVER claim cross-batch evidence without citing N from the dedupe pool.
#     Use scripts/build_unified_pool.py or reports/dedupe_pool.csv.
#
# D2. NEVER ship a multiplier from 1-sample evidence regardless of how clean
#     the in-batch numbers look. CLAUDE.md May 11 + May 16 + May 18 lessons.
#     Exception: cross-batch ≥5-sample structural backing AND BE-compatibility
#     check passes (≥60% of cell losses are Pattern B).
#
# D3. NEVER apply a cross-batch counterfactual without modeling the CURRENT
#     exit stack. CLAUDE.md May 20 lesson — historical P&L reflects historical
#     exits, not current. Run trades through current BE/SL/FAST_EXIT before
#     computing $ impact.
#
# D4. NEVER pool raw trades across config changes. Per-config sub-samples only,
#     Avg P&L % for cross-sample comparison (leverage-invariant).
#
# D5. NEVER lower a locked promotion gate at decision time. If a gate fails
#     by a hair, the answer is "no ship." Pre-committed numbers stand.
#     (Exception: A6 — challenging the gate itself, separately.)
#
# D6. ALWAYS apply in-sample bias haircut (30-50%) when projecting a Δ$ from
#     a counterfactual on the same batch the rule was derived from. State the
#     haircut explicitly.
#
# D7. ALWAYS distinguish "matched cohort" (post-hoc, by outcome) from
#     "classifiable at entry" (the only thing forward filters can use).
#     Unm. L / Unm. W are post-hoc — forward you only know "no W match",
#     which includes future winners AND future losers.
#
# D8. ALWAYS check per-pair concentration before shipping a dimensional
#     filter. If the cell's loss is 60%+ driven by 1-2 pairs, ship a pair
#     blacklist instead of a dimensional filter (CLAUDE.md May 12 lesson).
#
# D9. CAPS FOR LOSERS, MULT FOR WINNERS, DON'T CROSS THEM. CLAUDE.md May 20.
#     Pattern C cohort → TP/SL caps (defensive). Pattern W cohort → multiplier
#     (offensive). Mixing destroys edge mechanically.
#
# D10. PATTERN CODE IS THE SIGNATURE, TREATMENT IS DETERMINED BY EMPIRICAL
#      BEHAVIOR. C-tagged cells can carry W treatment if cross-batch shows
#      winning behavior (e.g., C1 SHORT at 2.0×); W-tagged cells can carry
#      C treatment if losing (e.g., W1 LONG at fixed TP/SL). CLAUDE.md May 21.
#
# D11. MANDATORY UI CHECKLIST FOR EVERY NEW CONFIG FIELD OR TOGGLE.
#
#      Repeatedly missed in May 22 sessions (entry_dist_from_ema13_min_long,
#      sl_atr_multiplier — both shipped engine-only without UI controls, then
#      caught by the operator post-deploy). This was an unforced error pattern.
#
#      ANY commit that introduces a new field in config.py OR trading_config.json
#      MUST include all 5 of the following in the same PR. NO EXCEPTIONS:
#
#        ✅ 1. config.py — field with default value + evidence comment
#        ✅ 2. trading_config.json — explicit value
#        ✅ 3. Engine wiring — services/trading_engine.py and/or services/indicators.py
#        ✅ 4. UI input — HTML form input in templates/index.html
#             • Numeric: <input type="number" ...> with step + min + max
#             • Toggle: <input type="checkbox" ...> with peer styling
#             • String rule: <textarea> or table-based rule builder
#             • Live preview span (optional but preferred for thresholds)
#        ✅ 5. UI load + save handlers
#             • Load: read config.thresholds.{field} → set input.value
#             • Save: include {field}: safeFloat(input.value) in payload
#
#      VERIFICATION BEFORE PUSH:
#      Run grep across templates/index.html for the new field's input ID.
#      If grep returns 0 matches → UI is missing, do not push.
#
#      EXCEPTION: rule strings (e.g. btc_rsi_adx_filter_*) that ship with the
#      full add/remove-row UI mechanism already exist — those use the
#      pattern-builder UI. Standalone numeric/toggle fields ALWAYS need direct
#      inputs.
#
#      If a field is intentionally operator-edits-JSON-only (rare), state that
#      explicitly in the CLAUDE.md ship entry with rationale. Otherwise the
#      default is "ship with UI input."
#
# ----------------------------------------------------------------------------
# DISCIPLINE OVERRIDES — known exceptions, document each
# ----------------------------------------------------------------------------
#
# When the operator explicitly overrides a locked criterion (e.g., shipping at
# N=3 when gate requires N=10), Claude MUST:
#   1. Acknowledge the override transparently in the response
#   2. Count the override in the running tally (currently ~6 in 2 weeks as of
#      May 22 — see CLAUDE.md May 22 BTC ATR ship entry)
#   3. Lock TIGHTER revert criteria than standard (e.g., immediate-drop on
#      first ✗ HARMFUL trade vs standard N=5 gate)
#   4. Document the override rationale + the tightened revert in CLAUDE.md
#
# Overrides are not bugs — they're acceptable trade-offs when explicit. They
# become a problem only when accumulated without acknowledgment. The tally
# matters: 1-2 per month is fine; 6+ in 2 weeks is a discipline-drift signal.
#
# ----------------------------------------------------------------------------
# WHEN TO SKIP THIS PLAYBOOK
# ----------------------------------------------------------------------------
#
# Skip when:
#   - Operator says "skip methodology" or asks a narrow lookup
#   - Operator asks a methodology / architecture question (not a data analysis)
#   - Bug fix or code work (no batch data involved)
#
# Always run when:
#   - Operator pastes a CSV
#   - Operator asks "what's interesting" / "deep dive" / "what should we change"
#   - Operator asks "is X working" about an active cell or filter
#
# ----------------------------------------------------------------------------
# END OF METHODOLOGY SECTION
