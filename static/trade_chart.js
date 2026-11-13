/* 📈 TRADE CHART — click a Closed Orders row → a pop-up with the trade on candles (DECISION_LOG 536, operator 2026-11-12).
 *
 * SELF-CONTAINED + EASY ROLLBACK: everything lives in this file. The dashboard hooks are three one-liners in templates/index.html,
 * each tagged "📈 (536) trade chart":
 *   ① <script src="/static/trade_chart.js?v=1" defer>   ② data-order-id on each Closed Orders <tr>   ③ window._closedOrdersById = …
 * Rollback = `git revert` of the DL 536 commit, or delete this file + those three lines. Nothing in the bot uses it.
 *
 * DATA: built on the spot, nothing stored. On click the BROWSER asks Binance's PUBLIC futures candles (fapi.binance.com/fapi/v1/klines,
 * CORS-open, 2 weight per request (limit ≤ 500) of the 2,400 / min IP budget — the viewer's IP, never the bot's) for the pair and BTC around the
 * trade (limit = the bars actually needed, ≤ ~400 → 2 weight); cached for this page session only. The trade's own numbers come from the Closed Orders data the dashboard already loaded.
 * Display only — never calls the bot API, never changes anything.
 */
(function () {
  'use strict';
  const API = 'https://fapi.binance.com/fapi/v1/klines';
  const IV = { '1m': 60, '5m': 300, '15m': 900, '1h': 3600 };
  const EXIT_C = '#fbbf24';                 // 📈 exit line / arrow colour (amber — operator 11-13)
  const AFTER_MIN = 45;                      // the bot's own post-exit record window
  const cache = new Map();
  let state = null;                          // { o, iv, rows }
  let seq = 0;                               // review 536: every load has a token — a late answer of an older load never paints
  let lastFocus = null;
  const MAX_BARS = 1500;                     // Binance's per-request cap

  // ── pure helpers (exported on window.TradeChart for the tests) ──
  function parseUtc(s) {
    if (!s) return null;
    let x = String(s).trim().replace(' ', 'T').replace(/(\.\d{3})\d+/, '$1');   // review 536: microseconds → milliseconds (every browser parses 3 digits)
    if (!/[zZ]$/.test(x) && !/[+-]\d{2}:\d{2}$/.test(x)) x += 'Z';
    const t = Date.parse(x);
    return Number.isFinite(t) ? Math.floor(t / 1000) : null;
  }
  function pickInterval(openS, closeS) {           // ≤ ~400 bars over the window shown
    const span = (closeS - openS) / 60 + 30 + AFTER_MIN;
    if (span <= 400) return '1m';
    if (span <= 2000) return '5m';
    if (span <= 6000) return '15m';
    return '1h';
  }
  function windowFor(openS, closeS, iv) {          // 30 min (or 20 bars) before → 45 min (or 20 bars) after
    const step = IV[iv];
    const t0 = Math.floor((openS - Math.max(30 * 60, 20 * step)) / step) * step;
    const t1 = Math.ceil((closeS + Math.max(AFTER_MIN * 60, 20 * step)) / step) * step;
    return [t0, t1];
  }
  function takerPct() {                            // the engine estimates the exit fee at the TAKER rate (review 536 fix): the dashboard's own fee field, else 0.045
    try {
      const v = parseFloat(String(document.getElementById('config-taker-fee')?.value ?? '').replace(',', '.'));
      return Number.isFinite(v) && v > 0 && v < 1 ? v : 0.045;
    } catch (_) { return 0.045; }
  }
  function feePct(o) {                             // the fee % the net P&L carries: this trade's ACTUAL entry fee % + the exit estimated at the taker rate
    const n = Number(o.notional_value) > 0 ? Number(o.notional_value) : Number(o.investment) * Number(o.leverage);
    const f = Number(o.entry_fee), t = takerPct();
    return (n > 0 && o.entry_fee != null && Number.isFinite(f) && f > 0) ? f / n * 100 + t : 2 * t;   // missing / 0 entry fee → taker both legs
  }
  function priceAtPct(o, pct) {                    // NET P&L % (best / worst in trade) → the PRICE it was reached at: the fees added back
    const e = Number(o.entry_price);
    if (!(e > 0) || pct == null || !Number.isFinite(Number(pct))) return null;
    const g = Number(pct) + feePct(o);             // gross move from the entry (operator 11-12: the worst line sat 0.09 % below the real low)
    return o.direction === 'SHORT' ? e * (1 - g / 100) : e * (1 + g / 100);
  }

  function barsFor(t0, t1, iv) { return Math.floor((t1 - t0) / IV[iv]) + 1; }
  function klines(symbol, iv, t0, t1) {
    const key = `${symbol}|${iv}|${t0}|${t1}`;
    if (cache.has(key)) return cache.get(key);           // review 536: the PROMISE is cached → two loads in flight share one request
    const lim = Math.min(MAX_BARS, barsFor(t0, t1, iv));
    const u = `${API}?symbol=${encodeURIComponent(symbol)}&interval=${iv}&startTime=${t0 * 1000}&endTime=${t1 * 1000}&limit=${lim}`;
    const p = (async () => {
      const r = await fetch(u, { credentials: 'omit' });
      if (r.status === 418 || r.status === 429) throw new Error('Binance is rate-limiting this browser — try again in a minute');
      if (r.status === 400) throw new Error(`Binance has no futures market ${symbol} (delisted or not a futures pair)`);
      if (!r.ok) throw new Error(`Binance answered ${r.status} for ${symbol}`);
      return (await r.json()).map(k => [Math.floor(k[0] / 1000), +k[1], +k[2], +k[3], +k[4], +k[5]]);
    })();
    cache.set(key, p);
    p.catch(() => cache.delete(key));                    // errors are never cached
    return p;
  }

  // ── pop-up ──
  function el(tag, cls, html) { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
  function esc(s) { return String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
  function fmtP(p) { if (p == null || !Number.isFinite(p)) return '—'; const a = Math.abs(p); return p.toFixed(a >= 1000 ? 1 : a >= 1 ? 4 : a >= 0.01 ? 5 : 7); }
  function fmtPct(v) { return v == null || !Number.isFinite(Number(v)) ? '—' : `${Number(v) >= 0 ? '+' : ''}${Number(v).toFixed(2)} %`; }
  function hhmm(t) { return new Date(t * 1000).toISOString().slice(11, 16); }

  function ensureModal() {
    let m = document.getElementById('tc-modal');
    if (m) return m;
    const st = el('style', null, `
      #tc-modal{position:fixed;inset:0;z-index:60;display:none;align-items:center;justify-content:center;background:rgba(0,0,0,.65);padding:16px}
      #tc-modal.open{display:flex}
      #tc-box{background:#0f1720;border:1px solid rgba(255,255,255,.12);border-radius:12px;width:min(1100px,100%);max-height:94vh;overflow:auto;padding:14px 16px;color:#e5e7eb;font:13px/1.45 system-ui,-apple-system,sans-serif}
      #tc-box .tc-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 14px}
      #tc-box h3{margin:0;font-size:16px;font-weight:600}
      #tc-box .tc-res{font-weight:600;font-family:ui-monospace,Menlo,monospace;font-size:15px}
      #tc-box .tc-sub{color:#9ca3af;font-size:12px;width:100%}
      #tc-box .tc-x{margin-left:auto;background:none;border:0;color:#9ca3af;font-size:18px;cursor:pointer;padding:2px 8px;border-radius:6px}
      #tc-box .tc-x:hover,#tc-box .tc-x:focus-visible{color:#fff;background:rgba(255,255,255,.08);outline:none}
      #tc-box .tc-facts{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:6px;margin:10px 0}
      #tc-box .tc-f{background:rgba(255,255,255,.04);border-radius:6px;padding:4px 8px;min-width:0}
      #tc-box .tc-f .k{font-size:10.5px;color:#9ca3af;text-transform:uppercase;letter-spacing:.04em}
      #tc-box .tc-f .v{font-family:ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums}
      #tc-box .tc-bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:6px}
      #tc-box .tc-iv{border:1px solid rgba(255,255,255,.15);background:none;color:#d1d5db;border-radius:999px;padding:2px 10px;font-size:12px;cursor:pointer}
      #tc-box .tc-iv[aria-pressed="true"]{background:#2563eb;border-color:#2563eb;color:#fff}
      #tc-box .tc-iv:disabled{opacity:.35;cursor:not-allowed}
      #tc-box .tc-read{margin-left:auto;font-family:ui-monospace,Menlo,monospace;font-size:11.5px;color:#9ca3af}
      #tc-box canvas{width:100%;height:420px;display:block}
      #tc-box .tc-msg{height:420px;display:flex;align-items:center;justify-content:center;color:#9ca3af}
      #tc-box .tc-leg{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:11.5px;color:#9ca3af;margin-top:6px}
      #tc-box .tc-sw{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:5px}
      @media (max-width:600px){#tc-box canvas,#tc-box .tc-msg{height:340px}}`);
    document.head.appendChild(st);
    m = el('div'); m.id = 'tc-modal'; m.setAttribute('role', 'dialog'); m.setAttribute('aria-modal', 'true'); m.setAttribute('aria-label', 'Trade chart');
    m.innerHTML = '<div id="tc-box"></div>';
    m.addEventListener('click', e => { if (e.target === m) close(); });
    document.addEventListener('keydown', e => {   // review 536: the page's Escape convention — one layer per press
      if (e.key === 'Escape' && !e.defaultPrevented && m.classList.contains('open')) { e.preventDefault(); close(); }
    });
    document.body.appendChild(m);
    return m;
  }
  function close() {
    const m = document.getElementById('tc-modal');
    if (m) m.classList.remove('open');
    state = null; seq++;
    document.body.style.overflow = '';
    if (lastFocus && document.contains(lastFocus) && lastFocus.focus) { try { lastFocus.focus({ preventScroll: true }); } catch (_) { /* ignore */ } }
    lastFocus = null;
  }

  function render() {
    if (!state) return;
    const { o, iv } = state;
    const box = document.getElementById('tc-box');
    const pnl = Number(o.pnl_percentage), good = pnl >= 0;
    const lev = Number(o.leverage), notional = Number(o.notional_value) > 0 ? Number(o.notional_value) : Number(o.investment) * lev;
    box.innerHTML = `
      <div class="tc-head">
        <h3>${esc(o.pair)} · ${esc(o.direction)} · ${esc((o.entry_strategy || 'MOMENTUM').replace('FLIP:', 'FLIP '))}</h3>
        <span class="tc-res" style="color:${good ? '#34d399' : '#f87171'}">${fmtPct(pnl)} · ${Number(o.pnl) >= 0 ? '+' : '−'}$${Math.abs(Number(o.pnl) || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
        <button class="tc-x" type="button" aria-label="Close" title="Close (Esc)">✕</button>
        <span class="tc-sub">Opened ${esc(String(o.opened_at).replace('T', ' ').slice(0, 19))} UTC · closed ${esc(String(o.closed_at).replace('T', ' ').slice(0, 19))} UTC · ${esc(o.duration || '')} · ${esc(o.close_reason || '')}</span>
      </div>
      <div class="tc-facts">
        <div class="tc-f"><div class="k">Entry</div><div class="v">${fmtP(Number(o.entry_price))}</div></div>
        <div class="tc-f"><div class="k">Exit</div><div class="v">${fmtP(Number(o.exit_price))}</div></div>
        <div class="tc-f"><div class="k">Best / worst in trade</div><div class="v">${fmtPct(o.peak_pnl)} / ${fmtPct(o.trough_pnl)}</div></div>
        <div class="tc-f"><div class="k">Position</div><div class="v">$${Math.round(notional || 0).toLocaleString()} at ${lev || '—'}×</div></div>
        <div class="tc-f"><div class="k">Money in</div><div class="v">$${Math.round(Number(o.investment) || 0).toLocaleString()}</div></div>
        <div class="tc-f"><div class="k">After exit (45 min)</div><div class="v">${o.post_exit_final_pnl != null ? fmtPct(o.post_exit_final_pnl) + ' at the end' : '—'}</div></div>
      </div>
      <div class="tc-bar" role="group" aria-label="Candle size">
        ${Object.keys(IV).map(k => { const too = tooMany(k); return `<button class="tc-iv" type="button" data-iv="${k}" aria-pressed="${k === iv}"${too ? ' disabled title="Too many candles for one Binance request — pick a larger size"' : ''}>${k}</button>`; }).join('')}
        <span class="tc-read" id="tc-read"></span>
      </div>
      <div id="tc-area"><div class="tc-msg">Loading candles from Binance…</div></div>
      <div class="tc-leg">
        <span><span class="tc-sw" style="background:#60a5fa"></span>Entry</span>
        <span><span class="tc-sw" style="background:#fbbf24"></span>Exit</span>
        <span><span class="tc-sw" style="background:#34d399"></span>Best in trade</span>
        <span><span class="tc-sw" style="background:#f87171"></span>Worst in trade</span>
        <span><span class="tc-sw" style="background:rgba(96,165,250,.35)"></span>45 min after the exit</span>
        <span><span class="tc-sw" style="background:#9ca3af"></span>BTC, % from the entry</span>
        <span>Best / worst lines = the recorded net % with this trade's fees added back (its entry fee + the exit at the taker rate, as the engine estimates it).</span>
      </div>`;
    box.querySelector('.tc-x').onclick = close;
    box.querySelectorAll('.tc-iv').forEach(b => b.onclick = () => { if (b.disabled || !state) return; state.iv = b.dataset.iv; load(); });
    box.querySelector('.tc-x').focus();
  }

  function spanOf(o) { const a = parseUtc(o.opened_at); return a == null ? null : [a, parseUtc(o.closed_at) || a]; }
  function tooMany(iv) { const sp = state && spanOf(state.o); if (!sp) return false; const [t0, t1] = windowFor(sp[0], sp[1], iv); return barsFor(t0, t1, iv) > MAX_BARS; }

  async function load() {
    if (!state) return;
    render();
    const tok = ++seq, my = state, o = my.o, iv = my.iv;
    my.rows = null;
    const openS = parseUtc(o.opened_at), closeS = parseUtc(o.closed_at) || openS;
    const area = () => document.getElementById('tc-area');
    if (openS == null) { area().innerHTML = '<div class="tc-msg">This trade has no open time — nothing to chart.</div>'; return; }
    const [t0, t1] = windowFor(openS, closeS, iv);
    try {
      const [pair, btc] = await Promise.all([klines(o.pair, iv, t0, t1), klines('BTCUSDT', iv, t0, t1).catch(() => [])]);
      if (tok !== seq) return;                               // closed or switched while loading
      if (!pair.length) { area().innerHTML = `<div class="tc-msg">Binance has no ${esc(iv)} candles for ${esc(o.pair)} in this window.</div>`; return; }
      my.rows = { pair, btc, openS, closeS, t0, t1, iv };
      redraw();
    } catch (e) {
      if (tok !== seq) return;
      area().innerHTML = `<div class="tc-msg">Could not load the candles: ${esc(e.message || e)}.</div>`;
    }
  }

  function redraw() {                                       // review 536: resize = redraw the cached rows, never a new request / re-render
    if (!state || !state.rows) return;
    const a = document.getElementById('tc-area'); if (!a) return;
    a.innerHTML = '<canvas id="tc-cv" aria-label="Trade chart"></canvas>';
    const x = state.rows;
    draw(document.getElementById('tc-cv'), state.o, x.iv, x.pair, x.btc, x.openS, x.closeS, x.t0, x.t1);
  }

  function draw(cv, o, iv, K, B, openS, closeS, t0, t1) {
    const step = IV[iv], dpr = window.devicePixelRatio || 1, r = cv.getBoundingClientRect();
    cv.width = r.width * dpr; cv.height = r.height * dpr;
    const g = cv.getContext('2d'); g.scale(dpr, dpr);
    const W = r.width, H = r.height, L = 6, R = 78, T = 10, BH = 70, ph = H - T - BH - 34;
    const entry = Number(o.entry_price), exit = Number(o.exit_price);
    let best = priceAtPct(o, o.peak_pnl);
    const worst = priceAtPct(o, o.trough_pnl);
    if (String(o.exit_order_type || '').toUpperCase() === 'MAKER' && Number(o.exit_price) > 0 && o.peak_pnl != null
        && Math.abs(Number(o.peak_pnl) - Number(o.pnl_percentage)) < 1e-6) best = Number(o.exit_price);   // review 536: best = the resting-TP fill itself
    const lv = [entry, exit, best, worst].filter(v => v != null && Number.isFinite(v) && v > 0);
    let lo = Math.min(...K.map(k => k[3]), ...lv), hi = Math.max(...K.map(k => k[2]), ...lv);
    const pad = (hi - lo) * 0.06 || hi * 0.001; lo -= pad; hi += pad;
    const X = t => L + (t - t0) / (t1 - t0 + step) * (W - L - R);
    const Y = p => T + (hi - p) / (hi - lo) * ph;
    const bw = Math.max(1.2, (W - L - R) / ((t1 - t0) / step + 1) * 0.66);
    g.font = '11px ui-monospace,Menlo,monospace';
    // grid + price axis
    g.strokeStyle = 'rgba(255,255,255,.06)'; g.fillStyle = '#9ca3af'; g.lineWidth = 1;
    for (let i = 0; i <= 5; i++) { const p = lo + (hi - lo) * i / 5, y = Y(p); g.beginPath(); g.moveTo(L, y); g.lineTo(W - R, y); g.stroke(); g.fillText(fmtP(p), W - R + 6, y + 4); }
    const maxTicks = Math.max(3, Math.floor((W - L - R) / 64));   // never crowded at phone width
    const tick = Math.max(step, [5, 15, 30, 60, 120, 240, 480, 720, 1440, 2880].map(m => m * 60).find(s => (t1 - t0) / s <= maxTicks) || 86400 * 7);
    for (let t = Math.ceil(t0 / tick) * tick; t <= t1; t += tick) {
      const lab = (t % 86400 === 0 && tick >= 3600) ? new Date(t * 1000).toISOString().slice(5, 10) : hhmm(t);   // midnight → the date
      g.fillText(lab, X(t) - 14, T + ph + 15);
    }
    // after-exit band + trade span
    g.fillStyle = 'rgba(96,165,250,.10)'; g.fillRect(X(closeS), T, Math.max(0, X(Math.min(t1, closeS + AFTER_MIN * 60)) - X(closeS)), ph);
    g.fillStyle = 'rgba(255,255,255,.03)'; g.fillRect(X(openS), T, Math.max(1, X(closeS) - X(openS)), ph);
    // reference lines
    const line = (p, c, lab, right, below, dash) => { if (p == null || !(p > 0)) return; g.strokeStyle = c; g.setLineDash(dash || [5, 4]); g.beginPath(); g.moveTo(L, Y(p)); g.lineTo(W - R, Y(p)); g.stroke(); g.setLineDash([]);
      g.fillStyle = c; g.textAlign = right ? 'right' : 'left'; g.fillText(lab, right ? W - R - 4 : L + 4, Y(p) + (below ? 12 : -4)); g.textAlign = 'left'; };
    line(best, '#34d399', 'best in trade'); line(worst, '#f87171', 'worst in trade');
    // 📈 exit line (operator 11-13): dotted amber at the exit price; its label goes under the line when it would touch the entry label (both on the right)
    const _near = exit > 0 && Math.abs(Y(exit) - Y(entry)) < 14;   // both labels sit on the right: push them apart, whichever is higher
    if (exit > 0) line(exit, EXIT_C, 'exit', true, _near && Y(exit) >= Y(entry), [2, 3]);   // dotted (operator 11-13)
    line(entry, '#60a5fa', 'entry', true, _near && Y(entry) > Y(exit));   // entry label on the right: never on top of best / worst
    // candles
    for (const k of K) {
      const up = k[4] >= k[1], c = up ? '#34d399' : '#f87171', x = X(k[0]) + bw / 2;
      g.strokeStyle = c; g.beginPath(); g.moveTo(x, Y(k[2])); g.lineTo(x, Y(k[3])); g.stroke();
      const y1 = Y(Math.max(k[1], k[4])), y2 = Y(Math.min(k[1], k[4])); g.fillStyle = c; g.fillRect(x - bw / 2, y1, bw, Math.max(1, y2 - y1));
    }
    // entry / exit arrows
    const arrow = (t, p, up, lab, col) => { const x = X(t), y = Y(p); g.fillStyle = col || '#60a5fa'; g.beginPath();
      if (up) { g.moveTo(x, y + 3); g.lineTo(x - 6, y + 13); g.lineTo(x + 6, y + 13); } else { g.moveTo(x, y - 3); g.lineTo(x - 6, y - 13); g.lineTo(x + 6, y - 13); }
      g.fill(); g.font = '600 11px system-ui,sans-serif'; g.fillText(lab, x - lab.length * 3, up ? y + 26 : y - 17); g.font = '11px ui-monospace,Menlo,monospace'; };
    const long = o.direction !== 'SHORT';
    arrow(openS, entry, long, 'entry'); if (exit > 0) arrow(closeS, exit, !long, 'exit', EXIT_C);
    // BTC strip
    if (B && B.length) {
      const base = (B.find(k => k[0] <= openS && openS < k[0] + step) || B[0])[1], v = B.map(k => (k[4] / base - 1) * 100);
      const bl = Math.min(...v, -0.05), bh = Math.max(...v, 0.05), top = T + ph + 26, Y2 = x => top + (bh - x) / (bh - bl) * (BH - 6);
      g.strokeStyle = 'rgba(255,255,255,.08)'; g.beginPath(); g.moveTo(L, Y2(0)); g.lineTo(W - R, Y2(0)); g.stroke();
      g.fillStyle = '#9ca3af'; g.fillText(`BTC ${bh >= 0 ? '+' : ''}${bh.toFixed(2)}%`, W - R + 6, top + 8); g.fillText(`${bl.toFixed(2)}%`, W - R + 6, top + BH - 6);
      g.strokeStyle = '#9ca3af'; g.lineWidth = 1.4; g.beginPath(); B.forEach((k, i) => { const x = X(k[0]) + bw / 2, y = Y2(v[i]); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
    }
    // hover read-out
    const read = document.getElementById('tc-read');
    cv.onmousemove = ev => { const bx = cv.getBoundingClientRect(), t = t0 + (ev.clientX - bx.left - L) / (W - L - R) * (t1 - t0 + step);
      const ts = t0 + Math.floor((t - t0) / step) * step;            // review 536: the candle UNDER the cursor
      const k = K.find(b => b[0] === ts) || K.reduce((a, b) => Math.abs(b[0] - ts) < Math.abs(a[0] - ts) ? b : a, K[0]);
      const pct = entry > 0 ? ((k[4] / entry - 1) * 100 * (long ? 1 : -1)) : null;
      read.textContent = `${hhmm(k[0])}  O ${fmtP(k[1])}  H ${fmtP(k[2])}  L ${fmtP(k[3])}  C ${fmtP(k[4])}${pct != null ? `  (${pct >= 0 ? '+' : ''}${pct.toFixed(2)} % vs entry, gross)` : ''}`; };
    cv.onmouseleave = () => { read.textContent = ''; };
  }

  function open(o, from) {
    lastFocus = from || document.activeElement;
    ensureModal().classList.add('open');
    document.body.style.overflow = 'hidden';                 // review 536: the page behind does not scroll
    const openS = parseUtc(o.opened_at), closeS = parseUtc(o.closed_at) || openS;
    state = { o, iv: openS != null ? pickInterval(openS, closeS) : '1m', rows: null };
    load();
  }

  // ③ delegated click on the Closed Orders table (rows carry data-order-id; buttons / links inside a row keep their own action)
  document.addEventListener('click', e => {
    const tr = e.target.closest && e.target.closest('#closed-orders-table-body tr[data-order-id]');
    if (!tr || e.target.closest('button, a, input, select') || e.detail > 1) return;   // a double-click selects text, never opens
    if (window.getSelection && String(window.getSelection()).length) return;   // selecting text is not a click
    const o = (window._closedOrdersById || {})[tr.dataset.orderId];
    if (o) open(o, tr);
  });
  let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(redraw, 200); });

  window.TradeChart = { open, close, parseUtc, pickInterval, windowFor, priceAtPct, feePct, barsFor };
})();
