"""🧾 (DECISION_LOG 476) investor monthly statement — services/investor_statement.py (pure money math + one-page PDF) and its wiring.
Hermetic: a synthetic investor (tests/fixtures/statement_fixture.py), no DB, no network."""
import os
import re
import sys
from datetime import date, datetime, timedelta

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from services import investor_statement as S  # noqa: E402
from fixtures.statement_fixture import ledger, orders, snaps  # noqa: E402

NOW = datetime(2026, 12, 2, 17, 5)


def _st(month="2026-11", led=None, sn=None, od=None, now=NOW, paper_now=True, name="Guille"):
    return S.compute_statement(name, ledger() if led is None else led, snaps() if sn is None else sn,
                               orders() if od is None else od, month, now, paper_now)


def _pages(pdf: bytes) -> int:
    return pdf.count(b"/Type /Page") - pdf.count(b"/Type /Pages")


def test_month_balance_adds_up_and_months_sum_to_the_total():
    for m in ("2026-10", "2026-11"):
        st = _st(m)
        assert st.end_balance == pytest.approx(st.start_balance + st.deposits - st.withdrawals + st.result, abs=1e-6)
        assert sum(d.pnl for d in st.days) == pytest.approx(st.result, abs=1e-6)          # day P&L excludes deposits
    nov = _st("2026-11")
    assert sum(r["result"] for r in nov.month_rows) == pytest.approx(nov.total_profit, abs=1e-6)
    assert nov.month_total["end"] == pytest.approx(nov.end_balance)
    assert nov.total_profit == pytest.approx(nov.end_balance - 10100.0, abs=1e-6)
    oct_ = _st("2026-10")
    assert oct_.start_balance == 0 and oct_.deposits == 10100.0 and oct_.first == date(2026, 10, 23)
    assert oct_.ret_pct == pytest.approx(oct_.result / 10100.0 * 100)
    assert nov.ret_pct == pytest.approx(nov.result / nov.start_balance * 100)
    assert nov.start_balance == pytest.approx(oct_.end_balance)


def test_day_returns_are_time_weighted_and_the_join_day_uses_the_buy_nav():
    st = _st("2026-10")
    sn = snaps()
    assert st.days[0].ret == pytest.approx(sn[date(2026, 10, 23)] / 1.0 - 1)              # bought at NAV 1.00
    d26 = [d for d in st.days if d.day == date(2026, 10, 26)][0]
    assert d26.ret == pytest.approx(sn[date(2026, 10, 26)] / sn[date(2026, 10, 25)] - 1)   # the $7,100 deposit does not move it
    assert d26.flow == 7100.0 and abs(d26.pnl) < abs(d26.balance - d26.flow)


def test_daily_compound_return_and_drawdown():
    flat = {date(2026, 10, 22) + timedelta(days=i): 1.01 ** i for i in range(40)}
    led = [dict(type="DEPOSIT", amount=1000.0, nav_at_time=1.0, shares_delta=1000.0, created_at=datetime(2026, 10, 21, 12))]
    st = S.compute_statement("X", led, flat, [], "2026-11", NOW, False)
    assert st.dcr_pct == pytest.approx(1.0, abs=1e-9) and st.max_dd_pct == 0 and st.positive_days == len(st.days) == 30
    st = _st("2026-10")
    assert st.max_dd_pct < -40 and st.max_dd_usd < 0
    assert abs(st.max_dd_usd) < 7100                                                       # the drop's $ never counts the deposit
    assert st.best.ret >= max(d.ret for d in st.days if d.ret is not None) - 1e-12
    assert st.worst.ret <= min(d.ret for d in st.days if d.ret is not None) + 1e-12


def test_flows_withdrawals_transfers_and_adjust_signs():
    assert S.signed_flow(dict(type="WITHDRAW", amount=50)) == -50
    assert S.signed_flow(dict(type="TRANSFER_OUT", amount=20)) == -20 and S.signed_flow(dict(type="TRANSFER_IN", amount=20)) == 20
    assert S.signed_flow(dict(type="ADJUST", amount=30, shares_delta=-1)) == -30 and S.signed_flow(dict(type="ADJUST", amount=30, shares_delta=2)) == 30
    n = snaps()[date(2026, 11, 9)]
    extra = [dict(type="WITHDRAW", amount=1000.0, nav_at_time=n, shares_delta=-1000.0 / n, created_at=datetime(2026, 11, 10, 15)),
             dict(type="DEPOSIT", amount=500.0, nav_at_time=n, shares_delta=500.0 / n, created_at=datetime(2026, 11, 10, 18))]
    st = _st("2026-11", led=ledger(extra))
    assert st.withdrawals == 1000.0 and st.deposits == 500.0
    assert st.end_balance == pytest.approx(st.start_balance - 500.0 + st.result, abs=1e-6)
    assert st.total_withdrawn == 1000.0 and st.total_deposited == 10600.0
    base = _st("2026-11")
    assert st.dcr_pct == pytest.approx(base.dcr_pct) and st.max_dd_pct == pytest.approx(base.max_dd_pct)   # flows never move returns


def test_months_preliminary_and_errors():
    assert S.statement_months(ledger(), NOW) == ["2026-10", "2026-11", "2026-12"]
    st = _st("2026-12")
    assert st.preliminary and st.first == date(2026, 12, 1) and st.last == date(2026, 12, 2)
    assert not _st("2026-11").preliminary
    with pytest.raises(S.StatementError):
        _st("2026-09")
    with pytest.raises(S.StatementError):
        _st("2027-01")
    with pytest.raises(S.StatementError):
        S.compute_statement("X", [], snaps(), [], "2026-11", NOW, True)


def test_simulated_banner_follows_the_trades_and_the_mode():
    assert _st("2026-11").simulated
    live = [dict(o, is_paper=False) for o in orders()]
    st = _st("2026-11", od=live, paper_now=False)
    assert not st.simulated and not S.disclaimer_text(st).startswith("Results shown are simulated")
    assert S.disclaimer_text(_st("2026-11")).startswith("Results shown are simulated")
    assert _st("2026-11", od=live, paper_now=True).simulated is False                     # a finished live month stays live
    assert _st("2026-12", od=[], paper_now=True).simulated                                # no trades yet, bot in paper


def test_text_names_the_month_and_the_first_month_omits_since_joining():
    nov = S.summary_text(_st("2026-11"))
    assert "in November" in nov and "since your first deposit on Oct 23, 2026" in nov and "In November," in nov
    assert "during the month" not in nov
    octt = S.summary_text(_st("2026-10"))
    assert "since your first deposit" not in octt and _st("2026-10").month_rows == []
    assert S.money(-1671.48) == "−$1,671.48" and S.money(450, sign=True) == "+$450.00" and S.pct(-16.48) == "−16.5 %"
    for forbidden in ("account holder", "fund", "member", "NAV", "ownership"):
        assert forbidden not in (S.DISCLAIMER + S.ABOUT + S.DISCLAIMER_PAPER).replace("refund", "")
    assert "within 10 days" in S.DISCLAIMER and "00:00 UTC-3" in S.DISCLAIMER


def test_pdf_is_one_page_even_with_many_months_flows_and_a_long_name():
    pdf = S.render_pdf(_st("2026-11"))
    assert pdf[:5] == b"%PDF-" and _pages(pdf) == 1
    # 15 months of history, a withdrawal + deposits, a long name → the table folds into "Earlier" and still fits one page
    sn = {date(2026, 1, 1) + timedelta(days=i): 1.0 + 0.002 * ((i * 7) % 11 - 5) for i in range(500)}
    led = [dict(type="DEPOSIT", amount=2000.0, nav_at_time=1.0, shares_delta=2000.0, created_at=datetime(2026, 1, 3, 12))]
    for k in range(1, 15):
        t = datetime(2026, 1, 3, 12) + timedelta(days=30 * k)
        nav = sn[S.day_utc3(t) - timedelta(days=1)]
        amt = -300.0 if k % 5 == 0 else 250.0
        led.append(dict(type="WITHDRAW" if amt < 0 else "DEPOSIT", amount=abs(amt), nav_at_time=nav, shares_delta=amt / nav, created_at=t))
    st = S.compute_statement("Maximiliano Alejandro de la Torre Fernández", led, sn, orders(), "2027-03", datetime(2027, 4, 2, 12), True)
    assert len(st.month_rows) == S.MAX_MONTH_ROWS and "–" in st.month_rows[0]["label"]
    assert sum(r["result"] for r in st.month_rows) == pytest.approx(st.total_profit, abs=1e-6)
    pdf = S.render_pdf(st)
    assert _pages(pdf) == 1
    assert S.file_name("Maximiliano Fernández", "2027-03") == "scalpars_statement_Maximiliano_Fern_ndez_2027-03.pdf"


def test_fonts_ship_with_the_app():
    for _, fn in S._FONTS.values():
        assert os.path.getsize(os.path.join(S.FONT_DIR, fn)) > 100_000
    assert os.path.exists(os.path.join(S.FONT_DIR, "OFL-IBMPlexSans.txt"))
    assert re.search(r"^reportlab==", open(os.path.join(ROOT, "requirements.txt")).read(), re.M)


def test_wiring_routes_button_and_no_gzip_on_pdf():
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert '@app.get("/api/investors/{investor_id}/statements")' in main
    assert '@app.get("/api/investors/{investor_id}/statement/{month}.pdf")' in main
    assert '".pdf"' in main[main.index("COMPRESSED_SUFFIXES"):main.index("COMPRESSED_SUFFIXES") + 300]
    assert "asyncio.to_thread(IS.render_pdf" in main and 'Order.status == "CLOSED"' in main
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert 'onclick="openStatementMenu(${inv.id}, this)"' in html and "async function downloadStatement(" in html
    assert "/statement/${month}.pdf" in html and "/statements`" in html


# ── deep review 476 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def _rebuild_ledger():
    """the production shape: a $10,100 FOUNDING row on Oct 23, then the DL 422 rebuild (run Oct 30) restating it as $3,000 opening
    capital + the Oct 26 $7,100 top-up — both ADJUST rows stamped at the RUN time."""
    sn = snaps()
    n25 = sn[date(2026, 10, 25)]
    run = datetime(2026, 10, 30, 20, 0)
    return [dict(type="FOUNDING", amount=10100.0, nav_at_time=1.0, shares_delta=10100.0, created_at=datetime(2026, 10, 23, 15, 0), note=None),
            dict(type="ADJUST", amount=7100.0, nav_at_time=1.0, shares_delta=-7100.0, created_at=run,
                 note=f"{S.REBUILD_MARK}: opening capital $3,000.00 at NAV 1.00 (was $10,100.00)"),
            dict(type="ADJUST", amount=7100.0, nav_at_time=n25, shares_delta=7100.0 / n25, created_at=run,
                 note=f"{S.REBUILD_MARK}: paper top-up 10-26 14:00 UTC, equity before $2,067.00")]


def test_rebuild_rows_are_redated_and_adjust_is_never_a_withdrawal():
    from services.nav_flows import REBUILD_MARK
    assert S.REBUILD_MARK == REBUILD_MARK
    reb = _st("2026-10", led=_rebuild_ledger())
    ref = _st("2026-10")                                                                    # the same money booked as two deposits
    assert reb.withdrawals == 0 and reb.deposits == pytest.approx(10100.0) and reb.total_withdrawn == 0
    assert reb.end_balance == pytest.approx(ref.end_balance) and reb.result == pytest.approx(ref.result)
    for a, b in zip(reb.days, ref.days):                                                    # every day valued as if booked on time
        assert a.balance == pytest.approx(b.balance) and a.pnl == pytest.approx(b.pnl)
    assert reb.best.day == ref.best.day and reb.best.pnl == pytest.approx(ref.best.pnl)
    assert S.dep_wd([dict(type="ADJUST", amount=50, shares_delta=-1)]) == (-50, 0)


def test_holdings_from_before_the_nav_history_and_mixed_paper_live_month():
    sn = {d: v for d, v in snaps().items() if d >= date(2026, 11, 1)}                       # NAV history restarts Nov 1
    st = _st("2026-11", sn=sn)
    assert st.start_balance > 0 and st.days[0].ret == pytest.approx(0.0)                    # valued at the first known close
    od = [dict(o, is_paper=o["closed_at"] < datetime(2026, 11, 15)) for o in orders()]
    st = _st("2026-11", od=od, paper_now=False)
    assert st.simulated and st.mixed_until is not None
    assert S.disclaimer_text(st).startswith("Results up to Nov 14, 2026 are simulated") and "live after" in S.banner_bits(st)[0]
    early = [dict(closed_at=datetime(2026, 10, 23, 4, 0), pnl=-50.0, pnl_percentage=-3.0, is_paper=True)]   # Oct 23 before the deposit
    assert _st("2026-10", od=early).trades == 0


def test_endpoint_turns_a_layout_overflow_into_a_clear_error():
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    seg = main[main.index("async def investor_statement_pdf("):]
    seg = seg[:seg.index("\n\n\n")]
    assert "except RuntimeError" in seg and "note=e.note" in main
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert "window._stmtTok" in html
