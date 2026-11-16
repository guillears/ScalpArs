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


def test_open_list_reloads_when_the_server_count_disagrees():
    """📱 Nov-16: the card showed 1 open while the Open Orders list stayed empty until a reload."""
    i = HTML.index("const _bo = Number(data.open_orders_count || 0)")
    blk = HTML[i:i + 900]
    assert "_tot !== window._ooPrevIds.size && window._ooRefreshNow" in blk and "window._ooRefreshNow()" in blk


def test_frozen_flush_timer_never_blocks_and_page_return_flushes():
    s = HTML[HTML.index("function _scheduleHTMLFlush()"):]
    s = s[:s.index("_touchGate.timer = setTimeout(_flushPendingHTML, delay);")]
    assert s.index("_touchGate.due < now - 1000") < s.index("if (_touchGate.timer && _touchGate.due <= due) return;")
    pb = HTML[HTML.index("const _onPageBack = () => {"):]
    assert "_flushPendingHTML(); } } catch (e) {}" in pb[:900]


def test_money_cards_every_5s_and_on_an_open():
    assert "window._refreshMoneyNow();\n            }, 5000);" in HTML
    oo = HTML[HTML.index("async function loadOpenOrders()"):]
    assert "[..._ids].some(id => !_prev.has(id))" in oo[:2500]


def test_hung_request_gives_up_and_money_has_one_owner():
    """deep review (Nov-16): a request hung on a dead mobile connection held the 'oo' guard → the list loads time out; the money
    cards are refreshed only under the 'money' key (no duplicate balance / investors requests from batch10)."""
    assert "function _fetchT(url, ms)" in HTML and "c.abort()" in HTML
    for url, ms in (("/api/orders/open", 10000), ("/api/investors", 15000), ("/api/balance", 15000)):
        assert f"await _fetchT(`${{API_BASE}}{url}`, {ms})" in HTML
    b = HTML[HTML.index("// Auto-refresh every 10 seconds"):]
    b = b[:b.index("}, 10000);")]
    assert "loadBalance()" not in b and "loadInvestors()" not in b
    assert "if (_rfGuard('oo', loadOpenOrders)) _ooLast = Date.now();" in HTML
    assert "return true;   // (Nov-16 review) a run actually started" in HTML
