"""📋 scripts/batch_review_checklist.py (operator 2026-10-23: "pass through ALL items, nothing missing"): every plan section / checkbox,
every scout script, every CURRENT_STATE item (no keyword filter — the numbered watchlist included), every referenced script, every persisted
gate state and every scout-report section land in the checklist. Regression asserts for the review findings (the numbered watchlist and the
plan's standing sections were missed by the first version). No network."""
import glob
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import batch_review_checklist as B  # noqa: E402

STATE_TXT = """# SCALPARS
> **👁 SOME_LINE (observe-only, DECISION_LOG 1):** text
## Active watchlist & locked revert gates
> 14. HIATR·LIQ **29/30** — at read
> 57l. **REARM entry-age cap** — every review
> - **Revert gate (either pass):** x
- **REDEPLOY_OPEN must stay 0**
1. momentum_short_btc_atr_min → revert
> 2. ~~GAPFLAT~~ ✅ RESOLVED
plain prose line that is not an item
> 11b-0. **🛑 SPIKE_CHASE DISABLED** — x
> ### 📋 ACTIVE WATCHLIST — NEXT BATCH
- **OPEN gate:** a long description where the idea was later refuted and superseded
- 💰 **Fee reserve** note
① **Momentum core**
  - **PREREQ** nested
- PAIR_35-40_30-35 SHORT = 1.0× (demoted)
"""
PLAN_TXT = """# Plan
## ⭐ Operator priorities
① **FIRST item** — x
②k **SECOND item** — y
③ **Then EVERY gate** — z
## 7. Sizing — LAST
- [ ] **size check**
### 🛡 SURGE_LONG EMA-gap filter (DECISION_LOG 287) — read at every batch review
## 🔬 WHEN yr6 IS DONE
- [ ] **conditional item**
"""


def test_state_items_capture_every_item_shape_and_bucket_closed_lines():
    rows = B.state_items(STATE_TXT)
    txt = [t for _, _, t, _ in rows]
    assert any("HIATR" in t for t in txt) and any("REARM entry-age cap" in t for t in txt)
    assert any("Revert gate (either pass)" in t for t in txt) and any("REDEPLOY_OPEN" in t for t in txt)
    assert any("momentum_short_btc_atr_min" in t for t in txt) and any("SOME_LINE" in t for t in txt)
    assert not any("plain prose" in t for t in txt)
    closed = [t for _, _, t, c in rows if c]
    assert len(closed) == 1 and "GAPFLAT" in closed[0]
    assert {s for _, s, _, _ in rows} >= {"SCALPARS", "Active watchlist & locked revert gates"}


def test_plan_priority_sections_and_checkboxes_with_conditional_tag():
    assert [k for k, _ in B.plan_items(PLAN_TXT)] == ["①", "②k", "③"]
    ps = B.plan_sections(PLAN_TXT)
    assert ("checkbox", "7. Sizing — LAST", "**size check**", False) in ps
    assert any(k == "section" and "SURGE_LONG EMA-gap" in s for k, s, _, _ in ps)
    assert any(k == "checkbox" and c for k, _, _, c in ps)                        # yr6 section → conditional


def test_real_sources_regressions():
    """the first version missed the numbered watchlist and the plan's standing sections — they must be present now."""
    st = " ".join(t for _, _, t, _ in B.state_items())
    assert "HIATR" in st and "Fresh tight-vs-actual" in st
    secs = " ".join(s for _, s, _, _ in B.plan_sections())
    assert "7. Sizing" in secs and "ML_MARKET_TURN" in secs and "SURGE_LONG EMA-gap" in secs


def test_every_scout_is_listed_with_its_runner_label():
    rows = {n: w for n, w, _ in B.scouts()}
    files = {os.path.basename(f)[:-3] for f in glob.glob(os.path.join(ROOT, "scripts", "scout_*.py"))} - B.HELPERS
    assert set(rows) == files
    assert rows["scout_features"].startswith("library") and rows["scout_features_parity"] == "parity check"
    assert rows["scout_willy_exit_v2"] == "opportunity scout" and rows["scout_long_stop4"].startswith("via ")


def test_referenced_scripts_and_gate_states():
    rs = {s: e for s, e, _ in B.referenced_scripts()}
    assert "build_master_pool.py" in rs and rs["build_master_pool.py"] is True
    assert not any(s.startswith("scout_") for s in rs)
    gs = B.gate_states()
    assert gs and all(isinstance(f, str) for f, _ in gs)


def test_missing_sources_are_visible_rows_not_crashes(tmp_path, monkeypatch):
    for k in ("PLAN", "STATE", "OPP", "SCOUT_REPORT", "REVERT_GATES"):
        monkeypatch.setattr(B, k, str(tmp_path / f"missing_{k}.md"))
    txt, n = B.build("X")
    assert txt.count("✗ source missing") >= 4 and n["plan"] == 0 and n["state_open"] == 0


def test_main_writes_the_file(tmp_path, monkeypatch):
    out = tmp_path / "c.md"
    monkeypatch.setattr(B, "OUT", str(out))
    B.main(["--batch", "TEST"])
    txt = out.read_text(encoding="utf-8")
    assert txt.startswith("# 📋 Batch review checklist — TEST")
    for h in ("## D. Method", "## A. ", "## A2. ", "## B. ", "## C. ", "## C2. ", "## E. ", "## F. ", "## G. "):
        assert h in txt
    with pytest.raises(SystemExit):
        B.main(["--batch"])                                                     # argparse, not an IndexError


def test_review2_shapes_false_close_quote_heading_and_nested_gates(tmp_path, monkeypatch):
    rows = B.state_items(STATE_TXT)
    by = {t[:20]: (sec, c) for _, sec, t, c in rows}
    txt = [t for _, _, t, _ in rows]
    assert any("SPIKE_CHASE DISABLED" in t for t in txt)                                       # 11b-0. numbering
    assert any("Fee reserve" in t for t in txt) and any("Momentum core" in t for t in txt) and any("PREREQ" in t for t in txt)
    assert any("PAIR_35-40_30-35" in t for t in txt)
    op = [(s, c) for _, s, t, c in rows if t.startswith("**OPEN gate")][0]
    assert op == ("📋 ACTIVE WATCHLIST — NEXT BATCH", False)                                  # '> ###' sets the section · late 'refuted' stays open
    gj = tmp_path / "g.json"
    gj.write_text('{"gates": {"LOADX": {"progress": "3/10"}, "HEAT": {}}, "tick_tries": {"A": 1}, "updated_utc": "x"}', encoding="utf-8")
    monkeypatch.setattr(B, "REVERT_GATES", str(gj))
    keys = [k for f, k in B.gate_states() if f == "SCOUT_REVERT_GATES.json"]
    assert len(keys) == 2 and keys[0].startswith("gate HEAT") and keys[1].startswith("gate LOADX") and "3/10" in keys[1]   # no tick_tries noise


def test_review3_closed_heads_bold_items_joined_headings_and_bare_script_names():
    st = """> **👁 FRENZY LEVERAGE WATCH — ✗ SUPERSEDED 2026-10-12 by DECISION_LOG 280** x
> **⚡ SURGE_LONG + SURGE_SHORT — LIVE, normal size** """ + "x" * 200 + """ SUPERSEDED later
> 5. ~~OLD RULE~~ REVERTED
> **~~WAKEUP~~ STRUCK** y
> 57l. **REARM entry-age cap** — open
"""
    rows = {t[:25]: c for _, _, t, c in B.state_items(st)}
    closed = {k for k, c in rows.items() if c}
    assert any(k.startswith("**👁 FRENZY LEVERAGE") for k in closed) and any("OLD RULE" in k for k in closed)
    assert any("WAKEUP" in k for k in closed)
    assert not any("SURGE_LONG" in k for k in closed) and not any("REARM" in k for k in closed)   # late prose never closes
    plan = """## ⭐ Operator priorities
- [ ] **in the priority block (A covers it)**
## 0. Method
**bold in §0 (folded into D)**
## 🏁 NEXT BATCH = B22 (B21 archived)
## +3 / −3 (own TP)
**🔴 IMPORTANT — WILLY time exit, read at EVERY batch review**
"""
    ps = B.plan_sections(plan)
    secs = [s for k, s, _, _ in ps if k == "section"]
    assert len(secs) == 1 and "NEXT BATCH = B22" in secs[0] and "own TP" in secs[0]                # wrapped heading joined
    assert any(k == "bold item" and "IMPORTANT" in t for k, _, t, _ in ps)
    assert not any("priority block" in t or "§0" in t for _, _, t, _ in ps)
    names = {s for s, _, _ in B.referenced_scripts()}
    assert {"study_willy_sl_cap_grid.py", "study_willy_late_tp_grid.py"} & names or not os.path.exists(os.path.join(ROOT, "scripts", "study_willy_sl_cap_grid.py"))
