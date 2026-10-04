# AUDIT — ⚖️ Sleeve Sizing table (uncommitted diff) — 2026-10-04

Independent audit of the uncommitted diff in `templates/index.html`, `main.py` and the new `tests/test_sleeve_sizing_ui.py`.
Scope excluded: `scripts/*`, `reports/*`. Nothing was committed or pushed, the live bot API was never called, and `services/*` was not edited.

**Verdict: SHIP.** No blocking defect. Two low-severity fixes are recommended (D1, D2), and either can land in the same commit.

## Method
- **Dynamic, real page JS.** A local mock server (scratchpad) served HEAD `index.html` (`/before`) and the working copy (`/after`). Its `/api/config` returned `load_trading_config().model_dump()` (the exact `GET /api/config` shape) and `/api/balance` was stubbed. In the browser pane, `fetch` was wrapped so that `PUT /api/config` was captured instead of sent. Then the page's own `loadConfig()` → `saveConfig()` ran, for (a) the real config and (b) a perturbed config: all 17 sleeve booleans flipped, all 37 sizing multipliers set to distinctive non-default values, and `manual_max_open_positions` = 3.
- **Python reference.** It used the real `TradingEngine.calculate_position_size` (unbound, fake `self`), plus the per-sleeve cell rules from `open_position` (trading_engine.py 9060-9450) and the liquidity block (hard ceiling + gross cap). It was compared row by row against the page's `_ssBase/_ssRowSpecs/_ssSize`.

## Results

| # | Item | Result | Evidence |
|---|------|--------|----------|
| 1 | Save payload unchanged | **PASS** | Real config: 727 → 733 keys, 0 dropped, 0 changed values or types. The 6 added keys are `thresholds.bull_long_{enabled,size_mult,lev_mult}` and `thresholds.bounce_long_{enabled,size_mult,lev_mult}`, all intended. The perturbed config gives the same result. The server also merges (`PUT /api/config` does `current.update(update)`), so even a dropped key could not reset a live setting. |
| 2 | Load → save round-trip | **PASS** | All 54 sizing/toggle keys and `investment.manual_max_open_positions` reproduce the loaded value exactly, in both the real and perturbed configs. That includes default-true checkboxes flipped to false (`flip_entry`, `bullrun`, `bearrun`) and `bull_long_enabled=true`. `surge_long/short_enabled` are absent from the payload by design (sent only when toggled), the same as before. Across all thresholds, round-trip drift is 5 keys before and the same 5 after (pre-existing format normalisation, e.g. `flip_entry_sources` `1.0`→`1`). Nothing new. |
| 3 | IDs | **PASS** | No duplicate ids, no ids removed. 9 ids added (6 inputs + `sleeve-sizing-body`, `ss-basis`, `ss-totals`). No `getElementById`, `querySelector('#…')` or `_ssNum(…)` points to a missing id, and there is no orphan `label for=`. All 49 moved inputs keep the same type, value, checked, step, min and max. No DOM-relative access (`closest`, `parentElement`, siblings) to any moved id. No `<form>` on the page. The ⑤ `lev-sched-*` and `res-sched-*` classes exist. |
| 4 | No trading-logic change | **PASS** | `services/*`, `config.py` and `trading_config.json` are unmodified (git). `_reserve_split` HEAD vs working tree over a 46,080-case grid (4 reserve modes × reserve value × fee usd/pct/hours × burn × maturity × BNB held incl. None × balance $0–$300k × deployed margin × schedule): **0 mismatches**. `/api/balance` additions: paper `balance + used_margin` uses the same operands as existing lines. Live uses `float(usdt_free or 0)` + `o.investment or 0`. `_fee_reserve_burn_leg()` runs the identical code `_reserve_split` already ran on the same request, so it adds no new failure mode. `sizing_equity` equals the engine's `total_portfolio` (available + DB open margin, engine 8682-8689) in both modes. |
| 5 | Preview correctness | **PASS** (one display edge, D1) | 150 rows compared (25 rows × $3k/$30k/$300k × real and perturbed configs): **0 mismatches** on cell inv/lev, margin, leverage, notional and skip. The perturbed config exercised the hard ceiling ($40k), gross cap 6×, the leverage schedule 25/15/8, 10 % reserve + 2.5 % fee, min size 120, Inv 0 fall-throughs (CROSS_OB → UNMATCHED, fade/CALM3D → default), the SURGE 0 → 0.05 floor, and the half-even cases 6.5→6, 12.5→12, 7.5→8, 3.5→4. Sleeves really open at STRONG_BUY (trading_engine 5927/6050/7228/7318/7595/7700/8044). Stop sources were checked against the engine: SURGE_SHORT = STRONG_BUY stop_loss (trading_engine 1044) and FRENZY = `frenzy_stop_pct`. |
| 6 | config.py / trading_config.json | **PASS / N/A** | No keys added. `bull_long_*` and `bounce_long_*` already existed (config.py 1748-1775, JSON 284-297, engine 6045/7695/9173/9198). D11 parity is now complete (default + JSON + engine + UI + load + save). Existing values are untouched: the JSON is unmodified and the round-trip is exact, so the first save sends back whatever the live server holds. |
| 7 | Tests + syntax | **PASS** | `venv/bin/pytest tests/ -q` → **573 passed**. `node --check` passes on all 5 inline scripts, before and after. |
| 8 | Reviews | done | See below |

## Defects (none blocking)

**D1 (low, display only):** `templates/index.html:20746`, `_ssPyRound` treats `|d−0.5|<1e-9` as an exact half, but Python `round()` uses the exact float value. A sweep of 375k (lev 1..125 × mult 0.001..3) gives 49 cases where the preview is ±1× off Python. In realistic ranges (lev ≤ 50, mult ≤ 2) there are only 3: 45×0.7, 50×1.09, 50×1.15. Live uses 20×, so it is unreachable today. **Fix:** `if (d === 0.5) return (f % 2 === 0) ? f : f + 1;` (JS doubles match Python's arithmetic exactly).

**D2 (low, footgun on disabled sleeves):** `templates/index.html:23929`, the save accepts `0` for `bull_long_lev_mult` and `bounce_long_lev_mult` (`x >= 0`). The engine reads 0 as 1.0 (`or 1.0`, trading_engine 9174/9199), so typing 0 for Bounce-long lev means **20×**, not the 1× observation the 0.05 default implies. The input's `min="0.05"` is not enforced. Both sleeves are OFF, so there is no live impact. **Fix:** for the two lev keys, `x > 0 ? x : _d` (0 → shipped default), or floor at 0.05. This mirrors the FRENZY "0 → floor, never 20×" rule.

## Review ① — caveman
- index.html:20746: tolerance half-even ≠ Python round at k+0.5±ε. Use `d === 0.5`.
- index.html:23929: lev mult 0 saved → engine 20×. Clamp `x>0` for lev keys.
- index.html:20975-20977: document-wide `input`/`change` listeners re-render on every keystroke anywhere. Measured 1.4 ms/render, acceptable. Scoping to the settings panel is optional.
- index.html:20901: bull-long stop read from `_cachedConfig` (no input), stale until reload. Fine, since there is no UI field for it.
- main.py:1056/1121: `_fee_reserve_burn_leg()` evaluated twice per request (inside `_reserve_split` + directly). Trivial cost; could reuse.
- Preview assumes STRONG_BUY confidence and an empty book. Both are disclosed in the footnote and basis line; OK.

## Review ② — deep
- **Save path:** ids unchanged, so the load/save handlers are byte-identical for the 49 moved inputs. The only handler changes are additive: a bull/bounce load block after the existing load and 6 new save keys. Proven dynamically (items 1–2). An old dashboard tab saving after deploy omits the new keys, and the server's merge keeps them.
- **Engine parity:** every row's cell semantics were re-derived from `open_position`: absolute-assign vs momentum clamp (floor 0.5), spike/CALM3D with no hard cap, CROSS_OB/ADX-surge Inv 0 fall-through, flip registry × NEGDI/TG take-the-max with lev only when > 1, bear-run lev `or 0.05`, SURGE/FRENZY explicit 0 → floors, FRENZY strong lev. The 150-row numeric match confirms them.
- **Server:** the helper extraction is behaviour-identical (46k-case equivalence + existing reserve tests green). The new response fields cannot raise beyond what the same request already evaluates.
- **Not modelled (disclosed):** the ① per-pair 24h-volume cap, ⑤ exchange leverage brackets, and gross room used by open positions. The preview is an upper bound for a next fill on an empty book.
- **D12 (reporting parity):** N/A. This is a settings table with a live preview, not an analytics table or per-trade metric.

## Ship verdict
**SHIP.** No save/load regression, no trading-logic change, preview numerically equal to the engine. Recommend applying D1 and D2 (two one-line edits) before the commit. Re-run pytest afterwards, and run the dual-review gate per CLAUDE.md before the operator authorizes the commit.
