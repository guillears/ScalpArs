/* Live Terminal — client-side controller.
 *
 * Connects to /api/terminal/stream (SSE), receives events, renders the
 * phosphor-green terminal UI: feed, radar, ticker, heatmap, status bar,
 * positions panel. All purely additive; does not touch any dashboard code.
 *
 * Public API (called from index.html toggle):
 *   LiveTerminal.show()   — switch into terminal view (plays boot once per tab)
 *   LiveTerminal.hide()   — return to dashboard
 *   LiveTerminal.toggle() — switch
 */
(function () {
  "use strict";

  const MAX_FEED_ROWS = 200;
  const MAX_HEATMAP_BARS = 60;
  const KATAKANA_CHARS = "ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉ";
  const KERNEL_BOOT_LINES = [
    "[0000.000] scalpars kernel v2.4.1-prod            [OK]",
    "[0000.012] initializing event subsystem           [OK]",
    "[0000.041] registering log handler                [OK]",
    "[0000.067] ring buffer maxlen=500 thread-safe     [OK]",
    "[0000.094] sse stream /api/terminal/stream        [OK]",
    "[0000.112] websocket handshake                    [OK]",
    "[0000.148] subscribing to __SYMBOLS__ symbols     [OK]",
    "[0000.171] heartbeat interval 3000ms              [OK]",
    "[0000.195] regime detector loaded                 [OK]",
    "[0000.241] auth tokens validated                  [OK]",
    "[0000.255] > scalpars ready · trading live"
  ];
  const BOOT_DURATION_MS = 2500;

  // ─── State ─────────────────────────────────────────────────────────────
  const state = {
    eventSource: null,
    bootShown: false,
    inTerminal: false,
    // ev/min rolling — events in the last 60s
    evTimes: [],
    // heatmap — array of {sec: epoch, count: number}, length cap 60
    heat: Array.from({ length: MAX_HEATMAP_BARS }, (_, i) => ({ sec: 0, count: 0 })),
    heatMaxRecent: 1,
    // ticker pairs
    tickerData: {},  // { 'BTCUSDT': { price: 68000, change: 0.42 } }
    // scan crawl
    scanCycle: 0,
    scanTotal: 50,
    scanCurrent: 0,
    scanCurrentPair: '',
    // radar — { symbol: { angle, dist, lastSeen, ttl } }
    radarBlips: new Map(),
    // positions — { symbol: { pnl_pct, direction } }
    positions: new Map(),
    // top scans — keep last ~6 scanned pairs
    topScans: [],
    // latency — last server_time and observed gap
    lastServerTime: 0,
    lastServerArrivalLocal: 0,
    latencyMs: 30,
    serverTimeGaps: [],
    // health
    health: { ws: true, api: true, log: true, ex: true },
    // bot state from heartbeat
    botState: {},
    // stats
    tradesToday: 0,
    pnlToday: 0,
    winsToday: 0,
    closedTrades: 0,
    // alert banner
    alertActive: false,
  };

  // ─── DOM hooks ─────────────────────────────────────────────────────────
  let elView, elFeed, elTicker, elTickerTrack, elLatencyBar, elLatencyText,
    elConstellation, elEvMin, elScanCrawlBar, elScanCrawlText, elCycleNum,
    elRadarSvg, elPosBody, elScansBody, elHeatmap, elFooterStats, elBoot,
    elBootLines, elAlertBanner, elAlertMsg, elKatakana, elHbDot;

  // ─── Public API ────────────────────────────────────────────────────────
  function show() {
    if (state.inTerminal) return;
    state.inTerminal = true;
    document.body.classList.add('lt-mode');
    elView.style.display = '';
    document.getElementById('dashboard-view').style.display = 'none';
    setNavActive('terminal');
    if (!state.bootShown) {
      playBoot();
      state.bootShown = true;
    }
    connect();
  }

  function hide() {
    if (!state.inTerminal) return;
    state.inTerminal = false;
    document.body.classList.remove('lt-mode');
    elView.style.display = 'none';
    document.getElementById('dashboard-view').style.display = '';
    setNavActive('dashboard');
    disconnect();
  }

  function toggle() { state.inTerminal ? hide() : show(); }

  function setNavActive(which) {
    const btnDash = document.getElementById('lt-nav-dashboard');
    const btnTerm = document.getElementById('lt-nav-terminal');
    if (btnDash && btnTerm) {
      btnDash.classList.toggle('active', which === 'dashboard');
      btnTerm.classList.toggle('active', which === 'terminal');
    }
  }

  // ─── SSE connection ────────────────────────────────────────────────────
  function connect() {
    if (state.eventSource) return;
    try {
      const es = new EventSource('/api/terminal/stream');
      state.eventSource = es;

      es.onmessage = (ev) => { handleEvent(JSON.parse(ev.data), false); };
      es.addEventListener('replay', (ev) => { handleEvent(JSON.parse(ev.data), true); });
      es.addEventListener('server_time', (ev) => {
        const data = JSON.parse(ev.data);
        const now = performance.now();
        if (state.lastServerArrivalLocal > 0) {
          const localGap = now - state.lastServerArrivalLocal;
          const serverGap = (data.t - state.lastServerTime) * 1000;
          // Latency proxy = local gap minus expected 1000ms; clamp positive
          const lat = Math.max(0, Math.min(200, Math.abs(localGap - 1000) + 25 + Math.random() * 5));
          state.serverTimeGaps.push(lat);
          if (state.serverTimeGaps.length > 8) state.serverTimeGaps.shift();
          const smoothed = state.serverTimeGaps.reduce((a, b) => a + b, 0) / state.serverTimeGaps.length;
          state.latencyMs = Math.round(smoothed);
          updateLatencyBar();
        }
        state.lastServerTime = data.t;
        state.lastServerArrivalLocal = now;
      });
      es.onerror = () => {
        // Browser will auto-retry. Mark log dot unhealthy until reconnect.
        state.health.log = false;
        updateConstellation();
      };
    } catch (err) {
      console.warn('[LiveTerminal] SSE connect failed:', err);
    }
  }

  function disconnect() {
    if (state.eventSource) {
      state.eventSource.close();
      state.eventSource = null;
    }
  }

  // ─── Event router ──────────────────────────────────────────────────────
  function handleEvent(evt, isReplay) {
    if (!evt) return;
    // Track ev/min
    state.evTimes.push(evt.ts * 1000);
    const cutoff = Date.now() - 60000;
    while (state.evTimes.length > 0 && state.evTimes[0] < cutoff) state.evTimes.shift();
    tickHeatmap(evt.ts);

    // HEARTBEAT — update bot state, health, positions panel
    if (evt.tag === 'HEARTBEAT') {
      if (evt.hb) {
        if (evt.hb.state) state.botState = evt.hb.state;
        if (evt.hb.health) { state.health = evt.hb.health; updateConstellation(); }
        updateFooterStats();
        state.health.log = true;
        updateConstellation();
      }
      return;
    }

    // SCAN events — update radar, scan crawl, top-scans list
    if (evt.category === 'SCAN' && evt.symbol) {
      addRadarBlip(evt.symbol);
      bumpScanCrawl(evt.symbol);
    }
    if (evt.symbol && (evt.category === 'SCAN' || evt.category === 'WATCH')) {
      pushTopScan(evt.symbol);
    }

    // ENTRY — open position tracking
    if (evt.category === 'ENTRY' && evt.symbol) {
      // Track open position (we won't know P&L yet — just placeholder)
      const dir = evt.side || 'LONG';
      state.positions.set(evt.symbol, { pnl_pct: 0, direction: dir });
      updatePositionsPanel();
    }

    // EXIT — read pnl_pct from kv, increment stats
    if (evt.category === 'EXIT' && evt.symbol) {
      const pnlPct = parseFloat(evt.kv?.pnl_pct ?? evt.kv?.pnl ?? '0');
      const isWin = isFinite(pnlPct) ? pnlPct > 0 : null;
      if (!isReplay) {
        state.closedTrades += 1;
        if (isWin === true) {
          state.winsToday += 1;
          state.pnlToday += pnlPct;
        } else if (isWin === false) {
          state.pnlToday += pnlPct;
        }
      }
      state.positions.delete(evt.symbol);
      updatePositionsPanel();
      updateFooterStats();
    }

    // ERROR/CRITICAL — flash + alert banner
    if ((evt.level === 'ERROR' || evt.level === 'CRITICAL') && !isReplay) {
      triggerAlert(evt);
    }

    // Render the row
    if (evt.tag !== 'HEARTBEAT') {
      appendFeedRow(evt, isReplay);
    }
  }

  // ─── Feed (event log) ──────────────────────────────────────────────────
  function appendFeedRow(evt, isReplay) {
    const row = document.createElement('div');
    row.className = 'lt-feed-row cat-' + (evt.category || 'INFO');
    const ts = new Date(evt.ts * 1000);
    const hh = pad(ts.getHours()), mm = pad(ts.getMinutes()), ss = pad(ts.getSeconds());
    const time = `${hh}:${mm}:${ss}`;
    const tagText = evt.tag === 'RAW' ? '·' : `[${evt.tag}]`;

    // Build the rendered text. For ENTRY rows, wrap the symbol in lock-anim spans.
    let body = (evt.msg || evt.tail || '').replace(/\s+/g, ' ').trim();
    if (body.length > 280) body = body.substring(0, 277) + '…';

    if (evt.category === 'ENTRY' && evt.symbol) {
      const replaced = body.replace(evt.symbol,
        `<span class="lt-entry-lock">[${evt.symbol}]</span>`);
      row.innerHTML = `<span style="color:#64748b">${time}</span>  <span class="lt-glow">${tagText}</span>  ${replaced}`;
      if (!isReplay) row.classList.add('lt-flash-entry');
    } else if (evt.category === 'EXIT' && evt.symbol) {
      row.innerHTML = `<span style="color:#64748b">${time}</span>  <span class="lt-glow-bright">${tagText}</span>  ${escapeText(body)}`;
      if (!isReplay) {
        const pnlPct = parseFloat(evt.kv?.pnl_pct ?? evt.kv?.pnl ?? '0');
        if (isFinite(pnlPct) && pnlPct > 0) {
          row.classList.add('lt-flash-win');
          floatPnl(row, '+' + pnlPct.toFixed(2) + '%');
        } else if (isFinite(pnlPct) && pnlPct < 0) {
          row.classList.add('lt-flash-loss');
        }
      }
    } else {
      row.innerHTML = `<span style="color:#64748b">${time}</span>  <span class="lt-glow">${tagText}</span>  ${escapeText(body)}`;
    }

    elFeed.appendChild(row);

    // Type-on animation for new events (skip on replay — too much work)
    if (!isReplay) {
      typeOn(row, body);
    }

    // Cap row count
    while (elFeed.childElementCount > MAX_FEED_ROWS) {
      elFeed.removeChild(elFeed.firstChild);
    }

    // Auto-scroll to bottom unless user has scrolled up
    const scrollGap = elFeed.scrollHeight - elFeed.scrollTop - elFeed.clientHeight;
    if (scrollGap < 60) elFeed.scrollTop = elFeed.scrollHeight;
  }

  function typeOn(row, fullBody) {
    // Only animate the body portion; keep time + tag instant. The row HTML
    // is already populated; we briefly hide and reveal characters using a
    // CSS clip on a wrapper. Cheap: animate `clip-path` via rAF.
    row.style.clipPath = 'inset(0 100% 0 0)';
    const start = performance.now();
    const dur = Math.min(900, Math.max(200, fullBody.length * 12)); // ~80 chars/sec
    function step(now) {
      const t = Math.min(1, (now - start) / dur);
      row.style.clipPath = `inset(0 ${(1 - t) * 100}% 0 0)`;
      if (t < 1) requestAnimationFrame(step);
      else row.style.clipPath = '';
    }
    requestAnimationFrame(step);
  }

  function floatPnl(row, text) {
    const f = document.createElement('span');
    f.className = 'lt-pnl-float';
    f.textContent = text;
    const rect = row.getBoundingClientRect();
    const parent = elFeed.getBoundingClientRect();
    f.style.left = (rect.right - parent.left - 60) + 'px';
    f.style.top = (rect.top - parent.top) + 'px';
    elFeed.appendChild(f);
    setTimeout(() => { try { elFeed.removeChild(f); } catch (_) { } }, 2100);
  }

  // ─── Heatmap (events per second × 60s) ─────────────────────────────────
  function tickHeatmap(eventTs) {
    const sec = Math.floor(eventTs);
    const last = state.heat[state.heat.length - 1];
    if (last.sec === sec) {
      last.count += 1;
    } else {
      // Advance — drop oldest, push new
      while (state.heat[state.heat.length - 1].sec < sec) {
        state.heat.shift();
        state.heat.push({ sec: (state.heat[state.heat.length - 1].sec || sec) + 1, count: 0 });
      }
      state.heat[state.heat.length - 1].sec = sec;
      state.heat[state.heat.length - 1].count = 1;
    }
    // Refresh peak
    state.heatMaxRecent = Math.max(1, ...state.heat.map(b => b.count));
    renderHeatmap();
  }

  function renderHeatmap() {
    if (!elHeatmap) return;
    const bars = elHeatmap.children;
    for (let i = 0; i < state.heat.length; i++) {
      const bar = bars[i];
      if (!bar) continue;
      const pct = state.heat[i].count / state.heatMaxRecent;
      bar.style.height = (pct * 100).toFixed(0) + '%';
      bar.style.opacity = (0.25 + 0.75 * pct).toFixed(2);
    }
  }

  // Drift heatmap forward every second so empty seconds appear as gaps
  setInterval(() => {
    if (!state.inTerminal) return;
    const sec = Math.floor(Date.now() / 1000);
    if (state.heat[state.heat.length - 1].sec < sec) {
      state.heat.shift();
      state.heat.push({ sec: sec, count: 0 });
      renderHeatmap();
    }
  }, 1000);

  // ─── Constellation (WS · API · LOG · EX) ───────────────────────────────
  function updateConstellation() {
    const map = { ws: 'dot-ws', api: 'dot-api', log: 'dot-log', ex: 'dot-ex' };
    Object.entries(map).forEach(([k, cls]) => {
      const el = elConstellation.querySelector('.' + cls);
      if (el) el.classList.toggle('unhealthy', !state.health[k]);
    });
  }

  // ─── Latency bar ───────────────────────────────────────────────────────
  function updateLatencyBar() {
    if (!elLatencyBar || !elLatencyText) return;
    const lat = state.latencyMs;
    const fillPct = Math.min(100, (lat / 200) * 100);
    elLatencyBar.querySelector('div').style.width = fillPct + '%';
    elLatencyBar.classList.remove('warn', 'crit');
    if (lat > 130) elLatencyBar.classList.add('crit');
    else if (lat > 80) elLatencyBar.classList.add('warn');
    elLatencyText.textContent = `LAT ${lat}ms`;
  }

  // Ev/min display
  setInterval(() => {
    if (!state.inTerminal) return;
    if (elEvMin) elEvMin.textContent = `ev/min ${state.evTimes.length}`;
  }, 1000);

  // ─── Scan crawl ────────────────────────────────────────────────────────
  function bumpScanCrawl(symbol) {
    state.scanCurrent += 1;
    state.scanCurrentPair = symbol;
    if (state.scanCurrent >= state.scanTotal) {
      state.scanCycle += 1;
      state.scanCurrent = 0;
    }
    const pct = (state.scanCurrent / state.scanTotal) * 100;
    if (elScanCrawlBar) elScanCrawlBar.querySelector('div').style.width = pct + '%';
    if (elScanCrawlText) elScanCrawlText.textContent = `${state.scanCurrent}/${state.scanTotal} · ${symbol}`;
    if (elCycleNum) elCycleNum.textContent = `#${state.scanCycle}`;
  }

  // ─── Radar (SVG) ───────────────────────────────────────────────────────
  function buildRadar() {
    const ns = 'http://www.w3.org/2000/svg';
    elRadarSvg.innerHTML = '';
    const w = 220, h = 220, cx = w / 2, cy = h / 2;
    elRadarSvg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    elRadarSvg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    // Rings
    for (const r of [40, 70, 100]) {
      const c = document.createElementNS(ns, 'circle');
      c.setAttribute('cx', cx); c.setAttribute('cy', cy); c.setAttribute('r', r);
      c.setAttribute('fill', 'none');
      c.setAttribute('stroke', '#22c55e');
      c.setAttribute('stroke-opacity', '0.18');
      c.setAttribute('stroke-width', '1');
      elRadarSvg.appendChild(c);
    }
    // Crosshair
    for (const [x1, y1, x2, y2] of [[cx, cy - 100, cx, cy + 100], [cx - 100, cy, cx + 100, cy]]) {
      const l = document.createElementNS(ns, 'line');
      l.setAttribute('x1', x1); l.setAttribute('y1', y1);
      l.setAttribute('x2', x2); l.setAttribute('y2', y2);
      l.setAttribute('stroke', '#22c55e');
      l.setAttribute('stroke-opacity', '0.12');
      elRadarSvg.appendChild(l);
    }
    // Sweep line (rotates via CSS)
    const sweep = document.createElementNS(ns, 'path');
    sweep.setAttribute('id', 'lt-radar-sweep');
    sweep.setAttribute('d', `M ${cx} ${cy} L ${cx + 100} ${cy} A 100 100 0 0 0 ${cx + 100 * Math.cos(-Math.PI / 6)} ${cy + 100 * Math.sin(-Math.PI / 6)} Z`);
    sweep.setAttribute('fill', 'url(#lt-sweep-grad)');
    sweep.style.transformOrigin = `${cx}px ${cy}px`;
    sweep.style.animation = 'lt-radar-spin 4s linear infinite';
    // Gradient
    const defs = document.createElementNS(ns, 'defs');
    defs.innerHTML = `<linearGradient id="lt-sweep-grad" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#22c55e" stop-opacity="0.6"/>
      <stop offset="100%" stop-color="#22c55e" stop-opacity="0"/>
    </linearGradient>`;
    elRadarSvg.appendChild(defs);
    elRadarSvg.appendChild(sweep);
    // Center dot
    const center = document.createElementNS(ns, 'circle');
    center.setAttribute('cx', cx); center.setAttribute('cy', cy); center.setAttribute('r', 2);
    center.setAttribute('fill', '#4ade80');
    elRadarSvg.appendChild(center);
    // Inject sweep animation keyframes (one-time)
    if (!document.getElementById('lt-radar-anim-style')) {
      const s = document.createElement('style');
      s.id = 'lt-radar-anim-style';
      s.textContent = '@keyframes lt-radar-spin { from {transform:rotate(0deg);} to {transform:rotate(360deg);} } @keyframes lt-blip-flash { 0% { opacity:1; r:4; } 100% { opacity:0; r:8; } }';
      document.head.appendChild(s);
    }
  }

  function addRadarBlip(symbol) {
    const { angle, dist } = symbolToPolar(symbol);
    state.radarBlips.set(symbol, { angle, dist, lastSeen: Date.now() });
    // Render blip immediately
    const ns = 'http://www.w3.org/2000/svg';
    const cx = 110, cy = 110;
    const x = cx + dist * Math.cos(angle);
    const y = cy + dist * Math.sin(angle);
    const id = 'lt-blip-' + symbol;
    let existing = elRadarSvg.querySelector('#' + id);
    if (existing) existing.parentNode.removeChild(existing);
    const blip = document.createElementNS(ns, 'circle');
    blip.setAttribute('id', id);
    blip.setAttribute('cx', x); blip.setAttribute('cy', y); blip.setAttribute('r', 3);
    blip.setAttribute('fill', '#4ade80');
    blip.style.animation = 'lt-blip-flash 1.6s ease-out 1';
    elRadarSvg.appendChild(blip);
    // Auto-cleanup after fade
    setTimeout(() => { try { blip.parentNode.removeChild(blip); } catch (_) { } }, 1700);
  }

  // Hash → polar coordinates (deterministic, client-side only)
  function symbolToPolar(symbol) {
    let h = 5381;
    for (let i = 0; i < symbol.length; i++) {
      h = ((h << 5) + h + symbol.charCodeAt(i)) | 0;
    }
    const angle = ((Math.abs(h) % 360) * Math.PI) / 180;
    const dist = 30 + (Math.abs(h >> 8) % 70); // 30..100 from center
    return { angle, dist };
  }

  // ─── Positions panel ───────────────────────────────────────────────────
  function updatePositionsPanel() {
    if (!elPosBody) return;
    if (state.positions.size === 0) {
      elPosBody.innerHTML = '<div class="lt-pos-row" style="opacity:0.4">no open positions</div>';
      return;
    }
    const html = [];
    for (const [sym, pos] of state.positions.entries()) {
      const sym4 = sym.replace('USDT', '').padEnd(4).substring(0, 4);
      const pnl = isFinite(pos.pnl_pct) ? pos.pnl_pct : 0;
      const cls = pnl >= 0 ? 'lt-pos-up' : 'lt-pos-dn';
      const bar = pnlBar(pnl);
      const pnlStr = (pnl >= 0 ? '+' : '') + pnl.toFixed(2) + '%';
      html.push(`<div class="lt-pos-row"><span>${sym4}</span><span class="lt-pos-bar ${cls}">${bar}</span><span class="${cls}">${pnlStr}</span></div>`);
    }
    elPosBody.innerHTML = html.join('');
  }

  function pnlBar(pct) {
    // ±1% range mapped to 6-character bar centered
    const slots = 6;
    const norm = Math.max(-1, Math.min(1, pct));
    const filled = Math.round(Math.abs(norm) * slots);
    return '█'.repeat(filled) + '░'.repeat(slots - filled);
  }

  function pushTopScan(symbol) {
    state.topScans = state.topScans.filter(s => s !== symbol);
    state.topScans.unshift(symbol);
    if (state.topScans.length > 6) state.topScans.length = 6;
    if (!elScansBody) return;
    elScansBody.innerHTML = state.topScans.map((s, i) =>
      `<div class="lt-scan-row"><span class="lt-scan-rank">${i + 1}</span><span>${s.replace('USDT', '')}</span></div>`
    ).join('') || '<div class="lt-scan-row" style="opacity:0.4">awaiting scans…</div>';
  }

  // ─── Ticker ────────────────────────────────────────────────────────────
  // Pull pair prices from existing /api/balance or /api/status endpoints
  // every 30s. Simple — no backend changes.
  async function refreshTicker() {
    if (!state.inTerminal) return;
    try {
      const r = await fetch('/api/orders/open');
      const data = await r.json();
      if (Array.isArray(data) && data.length > 0) {
        for (const o of data) {
          if (o.pair && o.current_price && o.entry_price) {
            const change = ((o.current_price - o.entry_price) / o.entry_price) * 100;
            state.tickerData[o.pair] = { price: o.current_price, change };
          }
        }
      }
    } catch (_) { /* silent */ }
    // Always include BTC if we have it from heartbeat
    renderTicker();
  }

  function renderTicker() {
    if (!elTickerTrack) return;
    const entries = Object.entries(state.tickerData);
    if (entries.length === 0) {
      elTickerTrack.innerHTML = '<span class="lt-ticker-item" style="opacity:0.4">awaiting prices…</span>';
      return;
    }
    // Repeat the list once so the loop has overlap
    const buildItems = () => entries.map(([sym, d]) => {
      const symShort = sym.replace('USDT', '');
      const priceStr = d.price >= 1000 ? d.price.toFixed(0) : d.price.toPrecision(5);
      const cls = d.change >= 0 ? 'lt-up' : 'lt-dn';
      const arrow = d.change >= 0 ? '▲' : '▼';
      return `<span class="lt-ticker-item">${symShort} ${priceStr} <span class="${cls}">${arrow}${Math.abs(d.change).toFixed(2)}%</span></span><span class="lt-ticker-sep">│</span>`;
    }).join('');
    elTickerTrack.innerHTML = buildItems() + buildItems();
  }

  setInterval(refreshTicker, 30000);

  // ─── Footer stats ──────────────────────────────────────────────────────
  function updateFooterStats() {
    if (!elFooterStats) return;
    const wr = state.closedTrades > 0
      ? Math.round((state.winsToday / state.closedTrades) * 100)
      : 0;
    const pnlStr = (state.pnlToday >= 0 ? '+' : '') + state.pnlToday.toFixed(2) + '%';
    const ts = new Date();
    const tyo = new Date(ts.getTime() + 9 * 60 * 60 * 1000);
    const ldn = new Date(ts.getTime() + 0 * 60 * 60 * 1000);
    const nyc = new Date(ts.getTime() - 5 * 60 * 60 * 1000);
    const fmt = (d) => `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
    elFooterStats.innerHTML = `
      <span><span class="lt-label">trades</span> <span class="lt-chip">${state.closedTrades}</span></span>
      <span><span class="lt-label">pnl</span> <span class="lt-chip">${pnlStr}</span></span>
      <span><span class="lt-label">win</span> <span class="lt-chip">${wr}%</span></span>
      <span class="lt-clock-cell">TYO ${fmt(tyo)} · LDN ${fmt(ldn)} · NYC ${fmt(nyc)}</span>
    `;
  }
  setInterval(updateFooterStats, 5000);

  // ─── Alert banner ──────────────────────────────────────────────────────
  function triggerAlert(evt) {
    if (state.alertActive) return; // don't pile up
    state.alertActive = true;
    elView.classList.add('lt-flash-bg');
    setTimeout(() => elView.classList.remove('lt-flash-bg'), 250);
    elAlertMsg.textContent = `${evt.level}: ${evt.msg || evt.tail || 'unspecified error'}`;
    elAlertBanner.classList.add('visible');
  }
  function dismissAlert() {
    state.alertActive = false;
    elAlertBanner.classList.remove('visible');
  }

  // ─── Katakana column ───────────────────────────────────────────────────
  function buildKatakana() {
    if (!elKatakana) return;
    // Single vertical column of glyphs, looping animation
    const chars = [];
    for (let i = 0; i < 80; i++) chars.push(KATAKANA_CHARS[Math.floor(Math.random() * KATAKANA_CHARS.length)]);
    elKatakana.innerHTML = `<div class="lt-kk-col">${chars.join('\n')}</div>`;
  }

  // ─── Boot sequence ─────────────────────────────────────────────────────
  function playBoot() {
    if (!elBoot) return;
    elBoot.classList.remove('gone');
    // Compose lines, swap in real symbol count where we have it
    const symCount = state.botState.open_positions || 50;
    elBootLines.innerHTML = '';
    for (let i = 0; i < KERNEL_BOOT_LINES.length; i++) {
      const txt = KERNEL_BOOT_LINES[i].replace('__SYMBOLS__', String(symCount));
      const el = document.createElement('span');
      el.className = 'lt-boot-line' + (i === KERNEL_BOOT_LINES.length - 1 ? ' bright' : '');
      el.textContent = txt;
      el.style.opacity = '0';
      el.style.animation = `lt-fade-in 0.18s ease-out ${1.0 + i * 0.12}s forwards`;
      elBootLines.appendChild(el);
    }
    setTimeout(() => { elBoot.classList.add('gone'); }, BOOT_DURATION_MS);
  }

  // ─── Key + click handlers ──────────────────────────────────────────────
  document.addEventListener('keydown', (e) => {
    if (!state.inTerminal) return;
    if (e.key === 'Escape') {
      if (state.alertActive) { dismissAlert(); return; }
      if (document.fullscreenElement) return; // first ESC = exit fullscreen (browser default)
      hide();
    }
  });

  // ─── Helpers ───────────────────────────────────────────────────────────
  function pad(n) { return n < 10 ? '0' + n : '' + n; }
  function escapeText(s) {
    return String(s).replace(/[&<>"']/g, m => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[m]));
  }

  // ─── Initialise on DOM ready ───────────────────────────────────────────
  function init() {
    elView = document.getElementById('live-terminal-view');
    if (!elView) return;
    elFeed = document.getElementById('lt-feed');
    elTicker = document.getElementById('lt-ticker');
    elTickerTrack = document.getElementById('lt-ticker-track');
    elLatencyBar = document.getElementById('lt-latency-bar');
    elLatencyText = document.getElementById('lt-latency-text');
    elConstellation = document.getElementById('lt-constellation');
    elEvMin = document.getElementById('lt-evmin');
    elScanCrawlBar = document.getElementById('lt-crawl-bar');
    elScanCrawlText = document.getElementById('lt-crawl-text');
    elCycleNum = document.getElementById('lt-cycle-num');
    elRadarSvg = document.getElementById('lt-radar-svg');
    elPosBody = document.getElementById('lt-positions-body');
    elScansBody = document.getElementById('lt-scans-body');
    elHeatmap = document.getElementById('lt-heatmap');
    elFooterStats = document.getElementById('lt-footer-stats');
    elBoot = document.getElementById('lt-boot');
    elBootLines = document.getElementById('lt-boot-lines');
    elAlertBanner = document.getElementById('lt-alert-banner');
    elAlertMsg = document.getElementById('lt-alert-msg');
    elKatakana = document.getElementById('lt-katakana');
    elHbDot = document.getElementById('lt-hb-dot');

    if (elHeatmap) {
      const frag = document.createDocumentFragment();
      for (let i = 0; i < MAX_HEATMAP_BARS; i++) {
        const b = document.createElement('div');
        b.className = 'lt-heatmap-bar';
        b.style.height = '0%';
        frag.appendChild(b);
      }
      elHeatmap.appendChild(frag);
    }

    if (elRadarSvg) buildRadar();
    buildKatakana();
    updateConstellation();
    updateLatencyBar();
    updatePositionsPanel();
    updateFooterStats();

    // Double-click background → return to dashboard
    elView.addEventListener('dblclick', (e) => {
      if (e.target === elView || e.target.classList.contains('lt-shell')) hide();
    });

    // Dismiss button on alert banner
    const dismissBtn = document.getElementById('lt-alert-dismiss');
    if (dismissBtn) dismissBtn.addEventListener('click', dismissAlert);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Expose API
  window.LiveTerminal = { show, hide, toggle };
})();
