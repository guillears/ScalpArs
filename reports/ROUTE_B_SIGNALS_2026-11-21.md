# 📡 Route B — signal-level dataset (2026-11-21)

**Short version.** Every signal the bot journaled since Route B recording began (2026-11-19 23:23 UTC) is one row: **263 signals** (187 possibilities, 107 market 15-min windows) across 6 sleeves, 262 with the order book recorded at the signal (100 %), 197 priced under today's exit (187 final). Window: 45.6 h, one BTC regime. **Everything here is descriptive; N is tiny; nothing is a ship input.** H1–H6 trigger outcomes are stored but deliberately NOT shown (pre-registration first).

Research only: no engine / config / recorder change, S3 read-only. Code: `scripts/route_b_signals/` (usage in `build.py`'s docstring). Datasets: `reports/ROUTE_B_SIGNALS_data/` (gitignored).

## 1 · Inputs and the exits priced

| item | value |
|---|---|
| Route B per-minute summaries | 46 hourly files · data to 2026-11-21 21:00 UTC |
| Decision journal | 8 exports read · last journal row 2026-11-21T13:10:00 UTC (signals after it are invisible until the next export; taken fills are added from the orders export: 1) |
| Probe fills dropped | 0 |
| Costs | taker 0.045 % (live config) + modelled slip per taker leg (recorded book, $6k walk) · maker TP 0.018 % (fills only 0.05 % through the level) · real settled funding (fapi /fundingRate) |
| Entry | the open of the first minute starting ≥ signal + 1 s. EXH / DIST signal = the 15m bar CLOSE, so blocked / refused rows enter at bar close + 60 s, while the bot's real fills land ~30–40 s after the close (blocked rows enter ~20–30 s later than a real fill would) · H1–H6: the open of minute m + 2 after trigger minute m · taken signals: the REAL fill price (flagged real_fill; walk starts the minute after the fill; paper drift inside the fill) |
| Taken fill minute | walked PESSIMISTICALLY: open = the fill price, the minute's whole adverse extreme kept (it may come after the fill), its favourable extreme capped at the better of fill and close (out_fill_minute) |
| Stop fill | a minute opening beyond the stop fills at the open; otherwise at the WORSE of the level and the minute's close (pessimistic on purpose — the bot polls every few seconds, real stops land between the level and this price) |


Exits read from the live `trading_config.json` at run time (EXHAUSTION TP cap is 7 since DL 609 — the journaled exh_tp_pct = 10 on these rows is the pre-609 level and is NOT used; the journaled ATR is):

| sleeve | exit |
|---|---|
| FLATBASE_SHORT | TP −5 % maker · stop +10 % incl. funding paid · 72 h |
| EXHAUSTION_SHORT | TP min(20×ATR, 7) % maker · SL min(1.5×ATR, 6) % · 24 h (ATR: journal stamp, else Wilder ATR14 on 15m closed ≤ the signal) |
| DISTRIBUTION_SHORT | TP min(20×ATR, 10) % maker · SL min(1.5×ATR, 6) % · 24 h (ATR: journal stamp, else Wilder ATR14 on 15m closed ≤ the signal) |
| FRENZY_WILLY_HT | TP +6 % net maker · stop −4 % net · 24 h |
| FRENZY_WILLY | NOT PRICED — plain WILLY V2b exit (spike-high target, late TP from min 30, runner trail on falling coins) needs inputs not journaled |


## 2 · Coverage — signals per sleeve × status

possibilities = one per sleeve × coin per holding period (a later signal inside a possibility is flagged `poss_dup`, not dropped) · book / mark at signal = the recorded state existed at the state minute (no look-ahead) · priced = an outcome under today's exit · final = the path finished.

| sleeve | status | block_class | signals | possibilities | sole_block | windows15 | coins | book_at_signal | mark_at_signal | priced | final |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DISTRIBUTION_SHORT | BLOCKED:DIST_NOT_VSW | core_rule | 69 | 44 | 0 | 44 | 30 | 99 % | 100 % | 100 % | 97 % |
| DISTRIBUTION_SHORT | BLOCKED:DIST_PAIR_HELD | capacity | 6 | 0 | 0 | 6 | 4 | 100 % | 100 % | 100 % | 83 % |
| DISTRIBUTION_SHORT | BLOCKED:DIST_POS_CHG | filter | 4 | 4 | 2 | 4 | 4 | 100 % | 100 % | 100 % | 100 % |
| DISTRIBUTION_SHORT | BLOCKED:DIST_RSI1H | filter | 7 | 5 | 7 | 7 | 5 | 100 % | 100 % | 100 % | 86 % |
| DISTRIBUTION_SHORT | BLOCKED:DIST_VOL24_LOW | filter | 28 | 18 | 0 | 17 | 12 | 100 % | 100 % | 100 % | 93 % |
| DISTRIBUTION_SHORT | OPENED | taken | 3 | 3 | 0 | 3 | 3 | 100 % | 100 % | 100 % | 100 % |
| EXHAUSTION_SHORT | BLOCKED:EXH_NO_SPOT | filter | 33 | 32 | 0 | 30 | 14 | 100 % | 100 % | 100 % | 97 % |
| EXHAUSTION_SHORT | BLOCKED:EXH_PAIR_HELD | capacity | 2 | 0 | 0 | 2 | 2 | 100 % | 100 % | 100 % | 100 % |
| EXHAUSTION_SHORT | BLOCKED:EXH_SPOT_SHARE | filter | 5 | 5 | 0 | 5 | 5 | 100 % | 100 % | 100 % | 100 % |
| EXHAUSTION_SHORT | BLOCKED:EXH_SSC_OIVOL | filter | 3 | 3 | 3 | 3 | 2 | 100 % | 100 % | 100 % | 100 % |
| EXHAUSTION_SHORT | BLOCKED:EXH_VOL24_LOW | filter | 5 | 5 | 0 | 5 | 4 | 100 % | 100 % | 100 % | 100 % |
| EXHAUSTION_SHORT | BLOCKED:EXH_VOL_DECAY | filter | 10 | 7 | 0 | 9 | 6 | 100 % | 100 % | 100 % | 90 % |
| EXHAUSTION_SHORT | OPENED | taken | 3 | 3 | 0 | 3 | 2 | 100 % | 100 % | 100 % | 100 % |
| EXHAUSTION_SHORT | OPENED_UNLINKED | taken | 1 | 1 | 0 | 1 | 1 | 100 % | 100 % | 100 % | 0 % |
| FLATBASE_SHORT | BLOCKED:NOT_FLAT | core_rule | 23 | 19 | 0 | 21 | 19 | 100 % | 100 % | 0 % | 0 % |
| FLATBASE_SHORT | BLOCKED:VOL24_LOW | unpriceable | 24 | 4 | 0 | 10 | 4 | 100 % | 100 % | 0 % | 0 % |
| FLATBASE_SHORT | OPENED | taken | 1 | 1 | 0 | 1 | 1 | 100 % | 100 % | 100 % | 100 % |
| FLATBASE_SHORT | REFUSED:PAIR_HELD | capacity | 1 | 0 | 0 | 1 | 1 | 100 % | 100 % | 100 % | 0 % |
| FLATBASE_SHORT | REFUSED:SHADOW_TURNOVER | filter | 1 | 1 | 0 | 1 | 1 | 100 % | 100 % | 100 % | 100 % |
| FRENZY_WIDE | REFUSED:SLEEVE_OFF | capacity | 5 | 5 | 0 | 5 | 5 | 100 % | 100 % | 0 % | 0 % |
| FRENZY_WILLY | BLOCKED:PAIR_HELD | capacity | 1 | 0 | 0 | 1 | 1 | 100 % | 100 % | 0 % | 0 % |
| FRENZY_WILLY | BLOCKED:WILLY_FAILED_SPIKE | filter | 1 | 1 | 1 | 1 | 1 | 100 % | 100 % | 0 % | 0 % |
| FRENZY_WILLY | BLOCKED:WILLY_GAP+WILLY_FAILED_SPIKE | filter | 2 | 2 | 0 | 2 | 2 | 100 % | 100 % | 0 % | 0 % |
| FRENZY_WILLY | REFUSED:A_NOMISS_MISSED | core_rule | 8 | 8 | 0 | 7 | 8 | 100 % | 100 % | 0 % | 0 % |
| FRENZY_WILLY | REFUSED:RED_EXPIRED | core_rule | 2 | 1 | 0 | 2 | 2 | 100 % | 100 % | 0 % | 0 % |
| FRENZY_WILLY_HT | BLOCKED:WILLY_FAILED_SPIKE | filter | 7 | 7 | 7 | 6 | 7 | 100 % | 100 % | 100 % | 100 % |
| FRENZY_WILLY_HT | OPENED | taken | 8 | 8 | 0 | 8 | 5 | 100 % | 100 % | 100 % | 100 % |


## 3 · Data-quality flags

| flag | value |
|---|---|
| out_entry_slip_src | route_b_book: 118 · default: 64 · in_fill_price: 15 |
| out_exit_slip_src | route_b_book_at_exit: 93 · maker_tp: 45 · default: 43 · route_b_book_at_signal: 16 |
| out_entry_src | next_minute_open: 182 · real_fill: 15 |
| out_atr_src | journal: 138 · computed_15m: 41 · –: 18 |
| depth within 0.5 % is a LOWER bound (20 levels reach less) | 58 % of signals with a book |
| depth within 1 % is a LOWER bound (20 levels reach less) | 82 % of signals with a book |
| $6k cost not computable (20 levels too shallow) | 41 % of signals with a book |
| ATR check: computed 15m Wilder ATR vs the journal stamp | N 138 · median |rel diff| 0.0 % · max 0.1 % |
| funding history fetched | 188 of 197 priced |
| not final (path not finished · funding not fetched · unlinked taken fill) — re-priced next run | 10 |
| taken signals whose fill is in no orders export yet (OPENED_UNLINKED: out of TAKEN / parity, never final) | 1 |
| fills closed by the operator / backstop (fill_operator_closed: out of parity) | 0 |
| filter blocks: sole (only failing gate, all scored) · also blocked by another gate · gates unscored (fail-fast engine) | 20 · 40 · 62 of 106 |
| EXH_SSC_OIVOL not reached on earlier EXH blocks (the engine reads OI only after every other gate passed) → an earlier EXH block can never be a clean single-filter row: the EXH sole set is STRUCTURALLY limited to SSC_OIVOL blocks | 53 |
| block classes | filter: 106 · core_rule: 102 · unpriceable: 24 · taken: 16 · capacity: 15 |
| PAIR_HELD rows (capacity: joined to the holding position's possibility, not a filter counterfactual) | 10 |
| EXH_WIDE_READ_CAP counters (shortlist coins not read: no episode, not signals) | 67 |
| FLATBASE episodes still TRACKING (no final status yet) | 2 |
| BLOCK gates not mapped to a sleeve | none |
| core rule not met (DIST_NOT_VSW pre-screen, FB NOT_FLAT, FB no-confirm, WILLY no entry bar) | 102 rows — keep them apart from real signals in any read |
| Binance API stopped (418/429) | False |


## 4 · Parity — the walker vs the booked P&L of the taken fills

Same entry (the real fill), today's exit, recorded costs and real funding. A large Δ means the walker or the exit spec is wrong for that sleeve (the booked P&L reflects the exit the fill RAN, e.g. EXH at TP 10 before DL 609, HT at TP 5 before the +6 change) — read Δ with the exit era in mind.

| sleeve | pair | opened | booked_reason | walker_reason | booked_net_pct | walker_net_pct | delta | walker_funding_pct |
|---|---|---|---|---|---|---|---|---|
| FRENZY_WILLY_HT | BLESSUSDT | 2026-11-20T07:10 | STOP_LOSS | STOP_LOSS | -4.90 | -5.09 | -0.19 | +0.00 |
| FRENZY_WILLY_HT | EVAAUSDT | 2026-11-20T07:50 | FRENZY_TP | TP | +4.97 | +5.97 | +1.00 | -0.03 |
| FRENZY_WILLY_HT | CYBERUSDT | 2026-11-20T08:30 | FRENZY_TP | TP | +5.02 | +6.00 | +0.98 | +0.00 |
| DISTRIBUTION_SHORT | CYBERUSDT | 2026-11-20T09:00 | FRENZY_TP | TP | +9.95 | +9.94 | -0.01 | +0.00 |
| FRENZY_WILLY_HT | EVAAUSDT | 2026-11-20T10:25 | FRENZY_TP | TP | +5.00 | +6.00 | +1.00 | +0.00 |
| FRENZY_WILLY_HT | NAORISUSDT | 2026-11-20T10:35 | STOP_LOSS | STOP_LOSS | -4.04 | -4.18 | -0.13 | -0.02 |
| DISTRIBUTION_SHORT | KERNELUSDT | 2026-11-20T12:30 | STOP_LOSS | STOP_LOSS | -6.11 | -6.19 | -0.08 | +0.00 |
| FRENZY_WILLY_HT | EVAAUSDT | 2026-11-20T12:55 | FRENZY_TP | TP | +5.00 | +6.00 | +1.00 | +0.00 |
| EXHAUSTION_SHORT | KERNELUSDT | 2026-11-20T14:15 | FRENZY_TP | TP | +9.95 | +6.94 | -3.02 | +0.00 |
| EXHAUSTION_SHORT | RAREUSDT | 2026-11-20T20:00 | STOP_LOSS | TP | -7.42 | +6.94 | +14.35 | +0.00 |
| FLATBASE_SHORT | KERNELUSDT | 2026-11-20T20:00 | STOP_LOSS | STOP_LOSS | -11.69 | -11.17 | +0.52 | -1.60 |
| FRENZY_WILLY_HT | EVAAUSDT | 2026-11-20T23:20 | STOP_LOSS | STOP_LOSS | -4.08 | -4.14 | -0.06 | -0.04 |
| FRENZY_WILLY_HT | ENSOUSDT | 2026-11-21T02:30 | STOP_LOSS | STOP_LOSS | -3.95 | -4.39 | -0.44 | +0.00 |
| DISTRIBUTION_SHORT | BLESSUSDT | 2026-11-21T09:30 | STOP_LOSS | STOP_LOSS | -6.03 | -6.19 | -0.16 | +0.00 |
| EXHAUSTION_SHORT | RAREUSDT | 2026-11-21T13:45 | STOP_LOSS | STOP_LOSS | -4.95 | -5.47 | -0.51 | +0.00 |


## 5 · Existing sleeves — taken vs blocked / refused (DESCRIPTIVE · N-tiny · one regime · NOT a ship input)

Possibilities only, priced final rows. TAKEN = linked fills; blocked rows ONLY where the named gate was the SOLE failing filter and every filter was scored (the engine stops at the first failing gate, so most blocked rows are multi-gate or partly unscored — see section 3); capacity rows (PAIR_HELD) are not filter counterfactuals. Blocked rows are priced as if entered at the next-minute open with today's exit — they are what each gate kept the bot out of in THIS window, nothing more. Window units decide any later read (the locked gates: ≥ 15 blocked, ≥ 8 windows, window-clustered bootstrap); no row here meets them.

| sleeve | group | N | windows15 | coins | WR | avg_net_pct | median_net_pct | TP | STOP | TIME | avg_mfe_pct | avg_mae_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DISTRIBUTION_SHORT | BLOCKED:DIST_POS_CHG (sole) | 2 | 2 | 2 | 50 % | +1.79 | +1.79 | 1 | 1 | 0 | +5.97 | -5.35 |
| DISTRIBUTION_SHORT | BLOCKED:DIST_RSI1H (sole) | 5 | 5 | 4 | 20 % | -3.53 | -5.59 | 1 | 4 | 0 | +6.00 | -4.36 |
| DISTRIBUTION_SHORT | TAKEN | 3 | 3 | 3 | 33 % | -0.82 | -6.19 | 1 | 2 | 0 | +4.21 | -5.45 |
| EXHAUSTION_SHORT | BLOCKED:EXH_SSC_OIVOL (sole) | 3 | 3 | 2 | 67 % | +2.32 | +5.21 | 2 | 1 | 0 | +5.46 | -3.10 |
| EXHAUSTION_SHORT | TAKEN | 3 | 3 | 2 | 67 % | +2.80 | +6.94 | 2 | 1 | 0 | +5.03 | -2.88 |
| FLATBASE_SHORT | TAKEN | 1 | 1 | 1 | 0 % | -11.17 | -11.17 | 0 | 1 | 0 | +0.37 | -9.91 |
| FRENZY_WILLY_HT | BLOCKED:WILLY_FAILED_SPIKE (sole) | 7 | 6 | 7 | 14 % | -2.90 | -4.22 | 1 | 6 | 0 | +3.82 | -5.33 |
| FRENZY_WILLY_HT | TAKEN | 8 | 8 | 5 | 50 % | +0.77 | +0.91 | 4 | 4 | 0 | +5.43 | -3.58 |


## 6 · New-sleeve triggers H1–H6 — COUNTS AND COVERAGE ONLY (outcomes stored, not read)

Definitions: Route B report §4, made causal (each coin's 99th pct / median from rows BEFORE the trigger minute, so counts differ from §4's whole-sample counts). One possibility per coin per 60 min (H6: 24 h), alts only. The forward returns (5 / 15 / 60 / 240 min) and the TP/SL grid are in `triggers_H.csv.gz`; open them only after the PREREG file is frozen.

| trigger | direction | possibilities | coins | windows15 | per_day | windows_per_day | book_at_trigger | full_240m_path |
|---|---|---|---|---|---|---|---|---|
| H1_liq_long_cascade | LONG | 71 | 46 | 17 | 37.4 | 8.9 | 100 % | 100 % |
| H2_liq_short_squeeze | SHORT | 36 | 20 | 25 | 18.9 | 13.2 | 100 % | 94 % |
| H3_imb_ask_heavy_thin | SHORT | 64 | 41 | 55 | 33.7 | 28.9 | 100 % | 94 % |
| H3_imb_bid_heavy_thin | LONG | 69 | 32 | 57 | 36.3 | 30 | 100 % | 88 % |
| H4_ask_vacuum | LONG | 122 | 62 | 79 | 64.2 | 41.6 | 100 % | 98 % |
| H4_bid_vacuum | SHORT | 146 | 73 | 80 | 76.8 | 42.1 | 100 % | 92 % |
| H5_taker_buy_climax | SHORT | 122 | 77 | 85 | 64.2 | 44.7 | 100 % | 88 % |
| H5_taker_sell_climax | LONG | 203 | 108 | 81 | 106.8 | 42.6 | 100 % | 98 % |
| H6_funding_ge_+0.1pct | SHORT | 67 | 65 | 39 | 35.3 | 20.5 | 7 % | 99 % |
| H6_funding_le_-0.5pct | LONG | 30 | 26 | 26 | 15.8 | 13.7 | 43 % | 100 % |


## 7 · Daily run · known gaps

- Run: `venv/bin/python -I scripts/route_b_signals/build.py` after the day's dashboard exports are downloaded (incremental: final outcomes carried over, censored ones re-priced, new S3 hours / klines days fetched). `--full` re-prices everything.
- **Journal lag:** signals exist only up to the newest decisions export; the bot does not push its journal to S3.
- **Plain FRENZY_WILLY not priced** (V2b spike-high target / late TP / runner need inputs not journaled); FRENZY_WIDE is off.
- **Unpriced by design:** FB NOT_FLAT / VOL24_LOW and WILLY RED_EXPIRED / NOMISS_MISSED have no entry bar.
- **Taken fills** walk from the minute after the fill (the rest of the fill minute is not walked); paper fills carry paper drift.
- **1-minute bars:** a minute touching both levels books the stop; intraminute order unknown (conservative).
- **20-level book:** depth / cost on fine-tick coins are lower bounds or missing (the default 0.10 % slip is then used and flagged).
- **Liquidations are a sample** (≤ 1 per coin per second); the burst baseline needs ≥ 20 prior liquidation minutes per coin.
- **One regime, ~2 days**: every table is descriptive until the pre-registered weekly scan (≈ 2026-12-10).
