# SCALPARS Go-Live Audit — 2026-10-14 (read-only)

## Verdict: NO-GO today. It becomes GO WITH CONDITIONS once the must-fix list below is closed.

In plain English: the core live machinery works and much of it has already run with real money in August. That covers market and limit orders, fill-price repair, the exit retry queue, the resting exchange safety stop (backstop), cancel-before-close, the ban guard and the reset guard. Paper and live now use the same stops, and the WILLY stop and leverage are in a safe place. But three of the seven items you locked as "do all before going live" (CURRENT_STATE line 3) are still open:
- ③ maintenance-margin-aware liquidation check.
- ④ margin-mode decision. This one contradicts an earlier operator decision.
- ⑦ per-order backstop % stored and re-placed when a stop changes.

Four structural gaps from the 2026-08-11 audit are also still open:
- Nothing detects a Binance position that has no matching row in the bot's database.
- The paper→live switch is a bare flag flip.
- The bot has no automatic daily-loss kill switch.
- The risk budget is still the paper one: cross margin, 20× sleeves, gross 25×, up to 10 positions.

Under your own rule ("BEFORE GOING LIVE (do all)") that is a NO-GO. None of these is large: about 1–2 days of build plus a written runbook. After that, start small (see §4).

Paper journal health for B21 is clean: no crashes, no gap in the 5-min scans, every fill sized to today's rules, every exit consistent with its configured exit. The one open item is a FRENZY_WIDE refusal on WAL at 06:05 whose cause is not logged in the journal.

## Checklist

| # | Item | Status | Evidence |
|---|---|---|---|
| ① | Paper-only branches (`is_paper`) in exit/sizing code | **DONE** | Every `is_paper_mode` branch in `services/trading_engine.py` is order plumbing or bookkeeping. Examples: live order :11126, backstop place :11319, cancel-before-close :12632, live close retry :12658, sweep :15219, retry queue :15242, M2 repair :15298, paper balance/BNB :11590/:13340. `_rh_backstop_floor` ignores `is_paper` (:44-51). `manual_backstop_stop_floor` ignores it too (:596-601). FRENZY exits pass no hidden floor (:14340-14345). `frenzy.py`, `surge.py` and `recovery_hold.py` have no `is_paper` at all. Remaining intended divergences: `book_changed_refusal` is paper-only (:400-409); the WILLY hold race refuses in paper but books in live (:11364-11372); live-only LEVERAGE_BLOCKED skip (:16040). |
| ② | Per-sleeve stop parity, with the resting backstop beyond the bot stop by ≥0.3 % | **DONE (at today's config)** | `order_backstop_pct` (:560-593), `bot_own_stop_pct` (:514-527). See the per-sleeve table below. The smallest gap is 0.39 % (recovery hold −2.2 vs 2.5). |
| ③ | Maintenance margin cached / leverage capped so liquidation sits beyond the backstop | **NOT DONE** | `get_leverage_brackets` keeps only `(maxNotional, maxLeverage)` and drops `maintMarginRatio`/`cum` (`binance_service.py:879-885`). The liquidation distance uses a fixed 0.5 % allowance, `MANUAL_MAINT_MARGIN_PCT` (`trading_engine.py:464-470`). Binance first-tier maintenance margin is roughly 1/(2×max leverage): 2.5 % on a max-20× pair. So on such a pair at 20× isolated, liquidation is about 2.5 % away, the same distance as the 2.5 % backstop. The bracket cap does cap leverage to the pair maximum (`leverage_bracket_cap_enabled=true`; PARTI was capped to 10× in B21). |
| ④ | ISOLATED margin set per pair | **NOT DONE — needs a decision first** | The code never sets margin mode: no `marginType` / `set_margin_mode` anywhere in `services/` or `main.py`; it only calls `set_leverage` (`binance_service.py:888`). **Conflict:** DECISION_LOG l.3379 (Aug-22, operator) = "stay CROSS, isolated WITHDRAWN". CURRENT_STATE l.3 (Oct-12, DL 271) = "set ISOLATED (must)". Pick one before live. If you pick ISOLATED, ③ becomes binding. |
| ⑤ | WILLY stop never 0 | **DONE in config, not guarded** | `trading_config.json` `frenzy_willy_stop_pct = 4.0`. The engine treats 0 as "no stop" (`trading_engine.py:523-524`, `frenzy.py:555-558`), and nothing in the code or UI prevents 0. At 0, live would stop at the 2.5 % backstop while paper would have no stop. |
| ⑥ | WILLY backstop gap at its leverage | **DONE (via 10×)** | `frenzy_willy_lev_mult = 0.5` (10×). Backstop = min(4 + 0.5, 0.92×(10−0.5)) = **4.5 %**. Bot stop −4 % net is about −3.91 % price, so the gap is **0.59 %** (≥ 0.3). Re-check if WILLY leverage is ever raised: at 20× the gap is 0.23 %, at 25× it fails. |
| ⑦ | Placed backstop % stored per order; re-place on stop change | **NOT DONE** | `models.py:39` has only `backstop_algo_id`, no % or trigger column. The sweep heals only rows whose id is missing (`trading_engine.py:15219-15238`). It never re-prices a live stop after a config change and never checks that the resting algo order still exists. |
| P1 | Paper→live switch | **PARTIAL** | `POST /api/paper-mode` → `set_paper_mode` flips a flag and saves (`main.py:816-820`, `trading_engine.py:3248-3256`). It does not refuse with open paper rows (they freeze), does not check keys, and does not reconcile. If the bot is running, the next scan places real orders. The setting is stored in the BotState database, not read from `trading_config.json` `paper_trading` (:2890-3010). |
| P2 | API keys | **OK (operator verify)** | Keys come from env (`config.py:3122-3123`); `.env` is git-ignored and was never committed. The keys work on prod: the bracket table loaded in paper (PARTI `bracket_capped=True`). Operator must verify the key permissions: futures + spot trade + universal transfer (the BNB auto-swap moves USDT futures↔spot, `binance_service.py:292+`), **withdrawals OFF**, IP-whitelisted to EB. Dashboard login is enforced (`main.py:541-559`); set `LOGIN_PASSWORD` on EB. |
| P3 | Leverage set + mismatch | **OK** | `set_leverage` runs before every entry, market (:930-940) and limit (:1196-1205). A mismatch blocks the pair, and the scan skips it (`trading_engine.py:16040`). |
| P4 | Precision / min notional | **OK (market) / PARTIAL (limit)** | Market entries round the amount and refuse below min amount/cost (`binance_service.py:942-960`). Maker limit entries have no client-side min-notional check, so the exchange validates. |
| P5 | Order type | **OK** | Maker entry is on (`maker_entry_enabled`, 20 s, 1 tick), with taker top-up/fallback and signal re-validation (`trading_engine.py:5270-5445`). Partial maker fills ≥95 % are accepted; below that the rest is topped up and the booking uses the actual filled size (:11380-11390). Maker exits are off (`maker_exit_enabled=false`, which avoids the Aug-11 B4 leak paths). FRENZY / urgent exits are taker. |
| P6 | Exchange backstop place / cancel / re-place | **OK** | Placed after the fill (:11319-11336); `closePosition`, MARK_PRICE, `priceProtect`, Algo API, with a -4130 orphan clear-and-retry (`binance_service.py:1021-1063`). Cancelled before every close (:12632-12649). Re-placed if the close fails (:12797-12812). Per-scan sweep for missing ids (:15219). The reconciler classifies BACKSTOP_STOP vs orphan-cancel (`main.py:2758-2781`). `broker_backstop_enabled=true`. Algo endpoints exist in ccxt 4.5.34 (requirements pin). |
| P7 | Restart reconciliation | **PARTIAL** | Runs every 60 s, live only (`main.py:154`). It closes database rows that are absent on Binance as EXTERNAL_CLOSE, skipping rows under 60 s old and stale close intents (`main.py:2644-2810`). **One-directional:** a Binance position without a database row (fill booked, then the database insert died — the "ghost PROM" class) is never detected. Bot opens have no flatten-on-booking-failure; only MANUAL does (:12146-12171). No user-data stream, so liquidations show up ≤60 s late as EXTERNAL_CLOSE. |
| P8 | Rejections / errors | **OK** | A None from the order is logged and skipped, with OPEN_FAILED_* counters per sleeve, including MOMENTUM (:17907). A fill-price-0 response is repaired by re-fetch, ticker and per-scan M2 repair (`binance_service.py:976-1000`, `trading_engine.py:15298`). Exits retry 3×, check whether the position is gone, then go to the retry queue (:12667-12820). -2019 / -4120 have no special handling: they fall into the generic None path, and -4120 no longer applies because the Algo API is used. |
| P9 | Rate limits / bans | **OK** | `_check_ban` on private order calls, which sleep out a ban (`binance_service.py:139-155`). The backstop is the safety net during a ban. Minor: `get_open_positions` does not call `_detect_ban` (:1268-1293). |
| P10 | Kill switch / max loss | **NOT PRESENT** | No daily-loss or drawdown halt in code. LITE's "> 25 % of book in a UTC day" is a manual revert rule (`config.py:1395`). `auto_kill_enabled=false`. |
| P11 | Position / gross caps | **OK, but sized for paper** | `max_open_positions=4`, but `redeploy_leftover_enabled=true` raises the ceiling to `max_open_positions_hard=10` (:9839-9844). The binding cap is `max_gross_leverage=25` on cross margin. |
| P12 | $100 minimum vs Binance min notional | **OK** | `min_investment_size` defaults to 100 (`config.py:2782`; not in the JSON). FRENZY LONG 1× is about $145 notional, momentum long 1× about $700. Both clear Binance's $5–$20 alt minimum. BTCUSDT's $100 minimum is met only while margin × leverage ≥ $100. |

### ② Per-sleeve stops at today's config

Backstop = `order_backstop_pct`. The liquidation allowance uses the code's 0.5 % assumption; see ③.

| Sleeve | Lev | Bot stop (net) | Backstop | Gap (price) | Status |
|---|---|---|---|---|---|
| FRENZY_LONG | 1× (0.05) | −3.0 | 3.5 % | ≈0.59 | ok |
| FRENZY_WIDE | 1× (0.05) | −3.0 | 3.5 % | ≈0.59 | OFF since DL 291 |
| FRENZY_LITE | 10× (0.5) | −3.0 | 3.5 % (liquidation cap 8.74 % not binding) | ≈0.59 | ok |
| FRENZY_WILLY | 10× (0.5) | −4.0 | 4.5 % | ≈0.59 | ok |
| Momentum long | 1× (`momentum_long_lev_mult` 0.05) | −0.7, ATR-widened to −1.2 | 2.5 % | ≥1.3 | ok |
| Momentum short / flips | 20× | −0.7…−1.2 | 2.5 % | ≥1.3 | ok; 20× isolated on a max-20× pair puts liquidation at the backstop (③) |
| Recovery hold (any) | as opened | −2.2 (`_rh_backstop_floor`) | 2.5 % | ≈0.39 | tightest; ok |
| Spike fade / chase | 20× cap (bracket-capped) | −1.5 / −1.2 | 2.5 % | ≥1.0 | ok |
| Bull-run / SURGE long | 20× | −0.7…−1.2 + trail | 2.5 % | ≥1.3 | ok |
| MANUAL | as chosen | own stop | own + 0.5, open refused if gap < 0.3 | ≥0.3 by construction | **never run live** |

## Paper journal + orders health (B21, 2026-10-12 16:00 → 10-14 17:24 UTC)

**Decision journal** (`scalpars_decisions_paper_2026-10-14_17-24-45.csv`, read in chunks):
- Events: FAILS 49,104 · BOOK 27,146 · BLOCK 11,317 · SCAN 593 · ADMIT 40 · OPEN 20 · POSITION 17 · WILLY 14.
- Scan cadence is a steady 300 s median with **no gap > 10 min**, so there was no visible outage or restart gap.
- **Zero** OPEN_FAILED_*, BACKSTOP_*, BOOK_CHANGED, BRACKET_CAP_SKIP, or size-related refusals. Note the journal does not record exceptions; those live only in the EB logs.
- One anomaly: **`FRENZY_WIDE_OPEN_REFUSED` WALUSDT 2026-10-14 06:05**, after the DL 289 deploy. The refusal reason is generic ("slots / balance / cooldown / price moved"); WAL's LITE trade had closed at 06:02. DL 289's revert gate fires if a LONG/WIDE setup is refused *for size*, so grep the EB log `[FRENZY_WIDE] WALUSDT` at 06:05 to confirm it was not size. Since DL 291 turned WIDE off this only matters as evidence, but the same gate covers LONG.
- WILLY: 7 arms, 4 opens, 1 red-bar expiry, 2 turnover-unread (fail-open).

**Orders** (`scalpars_orders_paper_2026-10-14_17-24-40.csv`, 17 closed, ids 1–17 after the reset):
- Sizing matches the rules:
  - After DL 289 (≥ 05:30 UTC): FRENZY_WIDE BAT $142 at 1× (inv 0.2 / lev 0.05) ✓; FRENZY_LONG WAL ×2 at $145 / 1× ✓; LITE ×2 at $710 / 10× ✓; momentum short TIA $714 / 20× ✓.
  - Before DL 289: FRENZY_LONG BAT 2× / $760, WIDE BAT 1× / $704, momentum longs at 10× (LINK, ARB 1.5× UNMATCHED, SUI CALM3D). This is correct for the old config.
  - **No momentum long has filled since DL 289**, so the 1× momentum long is still unverified on a real fill.
- Exits match the configuration:
  - LITE TP +3.00 / SL −3.0x; LONG / WIDE TP +1.51 / SL −3.00; WILLY TP +1.25; momentum STOP_LOSS L1 at −0.69 / −0.75; RUNNER_TRAIL on shorts / fade.
  - No stop filled worse than −3.02 %.
- Slippage (paper = decision → order drift, no order-book impact): four entries with |slip| ≥ 0.30 %, all on thin FRENZY pairs:
  - BTW LITE +0.30 and WAL LITE +0.40, both adverse at 10×, so each costs about 3–4 % of margin before the trade starts.
  - BTW WILLY −0.53 and WAL LONG −0.39, both favourable.
  - Plus ZEC TAKER_FALLBACK +0.30.
  - Live will add order-book impact on top. Watch `entry_slippage_pct` per sleeve from the first live fill.
- Exit slippage is blank (paper). Backstop id is blank (paper, expected).

## Must-fix before the first live order
1. **④ Decide CROSS vs ISOLATED** and record it. My recommendation is ISOLATED for the first live phase:
   - With roughly 25× gross allowed, cross means one gapped alt can drain the wallet.
   - The Aug-22 cross argument (cross absorbs wicks at 20×) mostly goes away if you also de-lever (see §4).
   - If you choose ISOLATED, build the `marginType` set + verify step at open, refusing the order on failure.
2. **③ Cache `maintMarginRatio`/`cum` from the brackets** and use the real per-pair liquidation distance. Clamp leverage (or refuse) when the backstop would not sit inside liquidation. One pure function + tests.
3. **⑦ Store the placed backstop % / trigger on each order** (new `Order` column; D11/D12 parity), and add a sweep that re-places when config changes the distance. Until this exists: change stops only with a flat book.
4. **Inverse reconciler:** a Binance position with no OPEN database row → CRITICAL + flatten (or adopt), and flatten when booking fails after a bot fill, exactly like MANUAL does at :12146.
5. **Safe mode switch:** `/api/paper-mode false` refuses while any paper row is OPEN or the bot is running, then runs the key check, `positionRisk` read, reconcile, and leverage (and margin-mode) sweep.
6. **⑤ Guard:** reject `frenzy_willy_stop_pct ≤ 0` (and `frenzy_stop_pct ≤ 0`) while `broker_backstop_enabled`, in the config validator and the UI.
7. **Daily-loss kill switch** (D11 field): pause new entries when the UTC-day realized loss exceeds X % of equity. Exits keep running.
8. **Written runbook:**
   1. Commit and deploy the finished DL 291, with the regression suite green. The working tree has uncommitted engine changes, so live code must equal reviewed code.
   2. Pause the bot.
   3. Flat paper book.
   4. Confirm `[BRACKETS] loaded` and the key permissions.
   5. Check the account is in one-way position mode (the code assumes it, `trading_engine.py:15308`) and Multi-Assets mode is off.
   6. Flip to live.
   7. Seed at minimum size.
   8. Watch one full open→close with `[BACKSTOP_PLACED]` / `[BACKSTOP_CANCELED]`.
   9. Do one deliberate restart with a position open and watch the sweep / reconcile.
   10. Then enable sleeves one at a time.

## Open risks the operator must decide (§4)
- **Leverage schedule:**
  - Today momentum shorts, flips, spike, bull-run and SURGE open at **20×**, LITE / WILLY at 10×.
  - At about $2.9k that is a $14k notional per short, about 5× equity on one position. A −2.5 % backstop fill costs about 12 % of equity.
  - Recommended first phase (same as the Aug-11 audit): ≤ 5–10× on every sleeve, `max_gross_leverage` 25 → about 8, `redeploy_leftover_enabled` off (true cap = 4 slots).
- **Sleeves to enable live, day 1:**
  - Momentum long (1×) and FRENZY_LONG blocker (1×) are low-risk.
  - LITE (10×) and WILLY (10×) are the money sleeves but also carry the tail: −3 / −4 at 10× is 30–40 % of the slot.
  - Keep MANUAL off live until the min-size watched test is done.
  - Keep SURGE_SHORT off (already off).
  - Consider holding spike fades (thin-book knives, B5 in the Aug-11 audit) until live slippage is measured.
- **Sizing for a live balance:**
  - Sizing is an equal split of the live wallet.
  - Below about $2,050 sizing equity the FRENZY_LONG margin is clamped up to $100, so a small seed makes LONG relatively larger.
  - At about $1k, slots are about $250. Momentum shorts at 20× would then be about $5k notional each.
- **Reserve:** `reserve_mode=schedule` is inert below $10k, so the whole balance is tradeable. Decide whether to hold back a manual reserve for the first weeks.
- **First-day limits:** suggest 1–2 slots, minimum sizes, the operator watching, a hard stop-trading rule (for example −10 % of equity on the day = pause), and a daily live log review per the LIVE LOG-REVIEW PROTOCOL (`[BACKSTOP_PLACED]` on every open, `BACKSTOP_*_FAILED` = 0, any BACKSTOP_STOP = investigate the outage).
- **Paper→live realism:**
  - Paper has no order-book slippage, optimistic maker fills, no funding in P&L, and no liquidation.
  - Pairs blocked live for a leverage mismatch were tradeable in paper.
  - Judge the first live weeks on `entry_slippage_pct` / `exit_slippage_pct` per sleeve, not on paper expectations.

## Docs housekeeping (no action taken)
- CURRENT_STATE l.316 still says the broker backstop is "DORMANT (`broker_backstop_enabled=false`)", but `trading_config.json` has it `true`. That line is stale.
- The Aug-22 CROSS decision (DECISION_LOG l.3379) and the Oct-12 "set ISOLATED" must-do contradict each other. Resolve and log in DECISION_LOG.
