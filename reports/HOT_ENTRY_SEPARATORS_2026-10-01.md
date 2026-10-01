# 🔥 Hot-state entries — what separates target-first from stop-first?

49,992 long entries (30 s apart) in 1,307 episodes · 196 pairs · 233 days. Target-first rate, episode-weighted: **Jan–Apr 54.0 % · May–Sep 51.3 %** (needs 72 % to pay at +0.59 / −1.11 with 0.11 % costs).

Shuffle null (episodes' outcomes reassigned at random): best spread any feature shows by luck on Jan–Apr = 11.7 points.

| Feature | target-first % by quintile, Jan–Apr (low → high) | May–Sep | best end same? | lift Jan–Apr / May–Sep | candidate |
|---|---|---|---|---|---|
| worst_dip_60s | 61 · 59 · 57 · 53 · 52 | 63 · 59 · 55 · 54 · 51 | yes (Q1) | +7 / +12 | — |
| range_60s | 52 · 53 · 57 · 59 · 61 | 51 · 54 · 55 · 59 · 63 | yes (Q5) | +7 / +12 | — |
| pullback_from_5m_high | 62 · 57 · 54 · 55 · 57 | 65 · 58 · 54 · 54 · 55 | yes (Q1) | +8 / +14 | — |
| ret_300s | 59 · 54 · 53 · 52 · 56 | 61 · 54 · 51 · 55 · 55 | yes (Q1) | +5 / +10 | — |
| ret_120s | 60 · 55 · 55 · 53 · 58 | 61 · 55 · 53 · 53 · 57 | yes (Q1) | +6 / +10 | — |
| ret_60s | 59 · 54 · 53 · 56 · 57 | 61 · 53 · 53 · 54 · 57 | yes (Q1) | +5 / +10 | — |
| ret_30s | 57 · 56 · 54 · 55 · 58 | 60 · 52 · 54 · 54 · 58 | no | +4 / +8 | — |
| atr_5m | 54 · 56 · 57 · 56 · 57 | 51 · 51 · 54 · 56 · 58 | no | +3 / +6 | — |
| pos_300s | 57 · 56 · 53 · 54 · 57 | 60 · 56 · 54 · 53 · 55 | yes (Q1) | +3 / +8 | — |
| stretch | 54 · 54 · 56 · 54 · 56 | 51 · 53 · 57 · 56 · 57 | yes (Q5) | +2 / +6 | — |
| ret_5s | 57 · 55 · 57 · 56 · 56 | 60 · 54 · 54 · 55 · 57 | no | +3 / +8 | — |
| higher_low | 56 · 53 · 56 · 57 · 58 | 58 · 52 · 54 · 56 · 58 | yes (Q5) | +4 / +7 | — |
| ret_15s | 56 · 56 · 56 · 55 · 58 | 57 · 53 · 55 · 55 · 58 | yes (Q5) | +4 / +7 | — |
| chop_60s | 54 · 54 · 58 · 56 · 58 | 53 · 54 · 54 · 55 · 58 | yes (Q5) | +4 / +7 | — |
| secs_since_5m_high | 56 · 54 · 55 · 56 · 56 | 53 · 52 · 56 · 55 · 57 | yes (Q5) | +2 / +5 | — |
| secs_in_episode | 57 · 54 · 57 · 56 · 54 | 56 · 54 · 54 · 52 · 52 | no | +3 / +5 | — |
| vol_vs_week | 53 · 53 · 55 · 57 · 55 | 50 · 52 · 52 · 53 · 55 | no | +2 / +3 | — |
| vol24_log | 54 · 55 · 54 · 52 · 59 | 55 · 51 · 53 · 51 · 51 | no | +5 / +3 | — |
| ret_24h | 56 · 56 · 57 · 54 · 57 | 52 · 54 · 55 · 55 · 55 | no | +3 / +4 | — |
| accel | 56 · 54 · 56 · 56 · 59 | 56 · 53 · 55 · 55 · 57 | yes (Q5) | +5 / +6 | — |
| hour_utc | 53 · 54 · 54 · 56 · 52 | 52 · 51 · 50 · 53 · 50 | no | +2 / +2 | — |
| ret_72h | 50 · 55 · 54 · 55 · 56 | 49 · 53 · 51 · 51 · 52 | no | +2 / +1 | — |
| rsi_5m | 54 · 55 · 57 · 56 · 61 | 53 · 54 · 55 · 56 · 56 | no | +6 / +5 | — |
| pos_60s | 55 · 55 · 56 · 57 · 56 | 57 · 54 · 54 · 56 · 56 | no | +3 / +5 | — |
| hot_secs_so_far | 59 · 59 · 60 · 58 · 51 | 57 · 58 · 56 · 55 · 57 | no | +6 / +6 | — |
| up_share_30s | 54 · 56 · 54 · 59 · 55 | 55 · 56 · 56 · 53 · 55 | no | +5 / +4 | — |
| speed | 54 · 54 · 56 · 58 · 56 | 56 · 55 · 55 · 55 · 56 | no | +4 / +4 | — |

## Candidates on the UNSEEN half (May–Sep)

| Rule (edge from Jan–Apr) | entries | episodes | days | target first | net % / trade | by day [95 %] |
|---|---|---|---|---|---|---|
| (no filter) | 29,186 | 825 | 142 | 51.3% | -0.348 | -0.347 [-0.381, -0.313] |
