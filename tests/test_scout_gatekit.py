"""🧰 scripts/scout_gatekit.py (DECISION_LOG 331): DL-number deploy lookup (multi-number commit tags) and the shared operator-closed
filter; plus the DL 284 LITE TP gate re-based by DECISION_LOG 333 (scout_frenzy_exits.ltg_read). No network, no git dependency (synthetic
commit logs)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_gatekit as GK  # noqa: E402

LOG = [  # (sha, unix seconds, full message) oldest first
    ("a" * 40, 1000, "🔥 FRENZY bearish-day block (DECISION_LOG 250)\n\nbody"),
    ("b" * 40, 2000, "🌊 gate OFF · scout GVOL_SPLIT (DECISION_LOG 261, 262, 263)"),
    ("c" * 40, 3000, "🔁 FRENZY_LITE_RETRY sleeve · LONG stop −4 (DECISION_LOG 316, 317)"),
    ("d" * 40, 4000, "later commit mentioning (DECISION_LOG 263) again in the subject"),
    ("e" * 40, 5000, "📦 misc\n\nfollow-up body text: DL 400 & 401"),
]


@pytest.mark.parametrize("text,nums", [
    ("(DECISION_LOG 261, 262, 263)", [261, 262, 263]),
    ("(DECISION_LOG 263)", [263]),
    ("(DECISION_LOG 316, 317)", [316, 317]),
    ("x (DECISION_LOG 281) · y (DECISION_LOG 280) · z (DECISION_LOG 282)", [281, 280, 282]),
    ("DL 306: reader", [306]),
    ("(DECISION_LOG 224, 230, 231, 232)", [224, 230, 231, 232]),
    ("no tag here 263", []),
    ("(DL 329–336)", [329, 330, 331, 332, 333, 334, 335, 336]),                    # review fix: en-dash range
    ("DECISION_LOG 329-331 · x", [329, 330, 331]),                                 # hyphen range
    ("DL 329 – 331, 340", [329, 330, 331, 340]),                                   # range then a list item
    ("(DECISION_LOG 281, 3 legs)", [281]),                                         # a trailing small number is prose
    ("DL 316 & 2 more", [316]),
    ("(DECISION_LOG 99, 100)", [99, 100]),                                         # growing magnitude stays in the list
    ("DL 300-100", [300, 100]),                                                    # backwards: not a range → two numbers
    ("DL 100–400", [100, 400]),                                                    # too wide for a range → two numbers
])
def test_parse_dl_numbers(text, nums):
    assert GK.parse_dl_numbers(text) == nums


def test_multi_number_tag_resolves():
    assert GK.find_dl_commit(263, LOG) == ("b" * 40, 2000)          # inside a list — and the OLDEST commit wins
    assert GK.find_dl_commit(262, LOG)[1] == 2000
    assert GK.find_dl_commit(316, LOG)[1] == 3000 and GK.find_dl_commit(317, LOG)[1] == 3000
    assert GK.dl_commit_ts(250, LOG) == 1000
    assert GK.tag_commit_ts("(DECISION_LOG 263)", LOG) == 2000       # drop-in for the old literal --grep readers
    assert GK.tag_commit_ts("grep:(DECISION_LOG 316)", LOG) == 3000
    assert GK.find_dl_commit(400, LOG)[1] == 5000                    # body-only mention as the fallback


def test_missing_dl_is_not_evaluable(capsys):
    assert GK.dl_commit_ts(999, LOG) is None
    assert GK.dl_deploy_ms(999, 10, label="TEST_GATE_999", commits=LOG) is None
    err = capsys.readouterr().err
    assert "NOT EVALUABLE" in err and "TEST_GATE_999" in err and "999" in err
    assert GK.tag_commit_ts("(no such commit — selftest)", LOG) is None   # a sentinel without a DL number → None, no lookup
    assert GK.dl_deploy_ms(263, 10, commits=LOG) == 2000 * 1000 + 600_000


def test_real_repo_ec98aea():
    """the motivating miss: the GVOL gate-off commit ec98aea is titled '(DECISION_LOG 261, 262, 263)'."""
    hit = GK.find_dl_commit(263)
    if not GK.git_commits():
        pytest.skip("git unavailable")
    assert hit is not None and hit[0].startswith("ec98aea")


@pytest.mark.parametrize("strategy,reason,want", [
    ("FRENZY_LITE_RETRY", "MANUAL", True),          # closed from the dashboard
    ("FRENZY_LITE", "EXTERNAL_CLOSE", True),        # closed in the Binance app
    ("FRENZY_WILLY", "MANUAL_CLOSE", True),         # paper-era manual close
    ("MOMENTUM", "MANUAL_TP L1", True),
    ("MANUAL", "EXTERNAL_CLOSE", False),            # an operator ENTRY is never a bot fill
    ("MANUAL", "MANUAL", False),
    ("FRENZY_LITE", "FRENZY_TP", False),
    ("FRENZY_LONG", "STOP_LOSS", False),
    ("FRENZY_LONG", "OVERRIDE_TP", False),          # operator-edited level, the BOT executed it
    ("SPIKE_FADE", "STOP_LOSS L1", False),
    ("FRENZY_LITE", None, False),
    ("FRENZY_LITE", float("nan"), False),
])
def test_is_operator_closed(strategy, reason, want):
    assert GK.is_operator_closed(strategy, reason) is want


B23_ROWS = pd.DataFrame([
    dict(opened_at="2026-10-20T14:50:10", pair="AUSDT", entry_strategy="FRENZY_LITE_RETRY", close_reason="MANUAL", pnl_percentage=0.025),
    dict(opened_at="2026-10-20T22:10:09", pair="ICNTUSDT", entry_strategy="FRENZY_LITE", close_reason="EXTERNAL_CLOSE", pnl_percentage=-0.634),
    dict(opened_at="2026-10-19T22:30:09", pair="BRUSDT", entry_strategy="FRENZY_LITE_RETRY", close_reason="FRENZY_TP", pnl_percentage=2.0),
    dict(opened_at="2026-10-20T16:59:46", pair="IOSTUSDT", entry_strategy="MANUAL", close_reason="EXTERNAL_CLOSE", pnl_percentage=1.9),
])


def test_split_and_line_b23():
    kept, ex = GK.split_operator_closed(B23_ROWS)
    assert set(ex.pair) == {"AUSDT", "ICNTUSDT"} and set(kept.pair) == {"BRUSDT", "IOSTUSDT"}
    line = GK.operator_closed_line(ex)
    assert line.startswith("operator-closed: 2 — excluded") and "AUSDT" in line and "ICNTUSDT" in line and "EXTERNAL_CLOSE" in line
    assert GK.operator_closed_line(ex.iloc[0:0]) == "operator-closed: 0 — excluded"
    assert list(GK.reason_mask(B23_ROWS.close_reason)) == [True, True, False, True]   # reason-only mask (callers drop MANUAL entries first)


def test_b23_export_rows_excluded():
    """the archived B23 export: the two known operator-closed bot fills are excluded, nothing else from the bot sleeves."""
    p = os.path.join(ROOT, "reports", "BASELINE23_batch1016-1020_LIVE_orders.csv")
    if not os.path.exists(p):
        pytest.skip("B23 archive not present")
    d = pd.read_csv(p, low_memory=False, usecols=["opened_at", "pair", "entry_strategy", "close_reason"])
    _, ex = GK.split_operator_closed(d)
    got = {(str(r.opened_at)[:16], r.pair, r.entry_strategy, r.close_reason) for r in ex.itertuples()}
    assert ("2026-10-20T14:50", "AUSDT", "FRENZY_LITE_RETRY", "MANUAL") in got
    assert ("2026-10-20T22:10", "ICNTUSDT", "FRENZY_LITE", "EXTERNAL_CLOSE") in got
    assert all(es != "MANUAL" for _, _, es, _ in got)


def test_lite_tp_gate_rebased_333():
    import scout_frenzy_exits as FX
    base = dict(direction="LONG", status="CLOSED")
    rows = [dict(base, opened_at="2026-10-15T20:00:00", pair="OLDUSDT", entry_strategy="FRENZY_LITE", pnl_percentage=3.0, peak_pnl=3.1,
                 close_reason="FRENZY_TP"),                                                     # before the DL 301 deploy → out
            dict(base, opened_at="2026-10-20T14:50:10", pair="AUSDT", entry_strategy="FRENZY_LITE_RETRY", pnl_percentage=0.02, peak_pnl=1.36,
                 close_reason="MANUAL"),                                                        # operator-closed → out, listed
            dict(base, opened_at="2026-10-20T22:10:09", pair="ICNTUSDT", entry_strategy="FRENZY_LITE", pnl_percentage=-0.63, peak_pnl=1.70,
                 close_reason="EXTERNAL_CLOSE")]
    for i in range(20):   # 10 TP +2 winners (peak 2.2 → CF +1.5) and 10 stops at −3 whose peak reached +1.6 (CF +1.5 saves them)
        win = i % 2 == 0
        rows.append(dict(base, opened_at=f"2026-10-{16 + i // 8:02d}T{(i % 8) * 3:02d}:00:00", pair=f"P{i}USDT",
                         entry_strategy="FRENZY_LITE_RETRY" if i % 3 == 0 else "FRENZY_LITE",
                         pnl_percentage=2.0 if win else -3.0, peak_pnl=2.2 if win else 1.6, close_reason="FRENZY_TP" if win else "STOP_LOSS"))
    o = pd.DataFrame(rows)
    r = FX.ltg_read(o, "2026-10-15T21:36:00")
    assert r["n_closed"] == 20 and len(r["opx"]) == 2 and "OLDUSDT" not in set(r["first"].pair)
    assert abs(r["actual"] - (-0.5)) < 1e-9 and abs(r["cf"] - 1.5) < 1e-9 and r["state"] == "to_15"
    txt = "\n".join(FX.ltg_lines(r, "2026-10-15T21:36:00"))
    assert "N 20/20 closed" in txt and "→ LITE to +1.5" in txt and "operator-closed: 2 — excluded" in txt
    # LITE C beats the CF (no saves: losers never reached +1.5) → hold
    o2 = o.assign(peak_pnl=np.where(o.pnl_percentage < 0, 0.5, o.peak_pnl))
    assert FX.ltg_read(o2, "2026-10-15T21:36:00")["state"] == "hold"
    # an open fill inside the first 20 → collecting
    o3 = o.copy()
    o3.loc[o3.pair == "P3USDT", "status"] = "OPEN"
    r3 = FX.ltg_read(o3, "2026-10-15T21:36:00")
    assert r3["state"] == "collecting" and r3["n_closed"] == 19
    assert "collecting (19/20 closed)" in "\n".join(FX.ltg_lines(r3, "2026-10-15T21:36:00"))
