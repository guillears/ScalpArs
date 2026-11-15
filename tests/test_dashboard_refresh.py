"""📱 Nov-15 (operator: the phone's total portfolio value lagged after a close / needed a reload) — dashboard refresh wiring."""
import pathlib

HTML = (pathlib.Path(__file__).resolve().parents[1] / "templates/index.html").read_text()


def test_stale_inflight_run_never_blocks_forever():
    i = HTML.index("const _rfGuard = (key, fn) => {")
    body = HTML[i:HTML.index("window._refreshMoneyNow", i)]
    assert "_cur.t >= window._rfHiddenAt && Date.now() - _cur.t < _RF_STALE_MS" in body   # a run from before the page slept stops blocking
    assert "if (window._rfInflight[key] === _tok) window._rfInflight[key] = false;" in body   # only its own run clears the flag
    assert "const _RF_STALE_MS = 90000;" in HTML                                          # long backstop only: a slow server never stacks runs
    assert "if (document.hidden) window._rfHiddenAt = Date.now(); else _onPageBack();" in HTML
    assert "window.addEventListener('pagehide', () => { window._rfHiddenAt = Date.now(); });" in HTML


def test_money_cards_refresh_on_a_close():
    assert "window._refreshMoneyNow = () => _rfGuard('money', () => Promise.allSettled([loadBalance(), loadInvestors()]));" in HTML
    oo = HTML[HTML.index("async function loadOpenOrders()"):]
    oo = oo[:oo.index("window._openOrdersById = {};")]
    assert "[..._prev].some(id => !_ids.has(id))" in oo and "window._refreshMoneyNow()" in oo   # an open trade vanished
    cc = HTML[HTML.index("async function checkClosedOrders()"):]
    cc = cc[:cc.index("}\n", cc.index("loadClosedOrders()) _lastClosedLiteSig = sig"))]
    assert cc.index("window._refreshMoneyNow()") < cc.index("await loadClosedOrders()")


def test_page_return_refreshes_including_ios_bfcache():
    assert "if (Date.now() - _pageBackAt < 1000) return;" in HTML                          # one refresh per return (iOS fires both)
    assert "window.addEventListener('pageshow', (ev) => { if (ev.persisted) _onPageBack(); });" in HTML
