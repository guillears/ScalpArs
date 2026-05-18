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
  const KATAKANA_CHARS = "ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎ0123456789";
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
    // scan crawl
    scanCycle: 0,
    scanTotal: 50,
    scanCurrent: 0,
    scanCurrentPair: '',
    // radar — { symbol: { angle, brightness, state, lastSweepHit } }
    // Persistent across the session. State: 'dim' | 'bright' | 'amber' | 'reject'
    radarBlips: new Map(),
    radarSweepAngle: 0,             // current sweep angle (rad), driven by rAF
    radarSweepStarted: 0,           // performance.now() reference
    radarSymbolEvents: [],          // {ts, symbol} for last 60s — drives score
    radarRejectFlash: new Map(),    // symbol → expiry timestamp (ms)
    // positions — { symbol: { pnl_pct, direction } }
    positions: new Map(),
    // last closed trade (most recent EXIT seen) — drives positions waiting state
    lastClosed: null,               // { symbol, pnl_pct, at }
    // top scans — keep last ~6 scanned pairs
    topScans: [],
    // filter funnel — last parsed values, drives funnel panel
    funnel: {
      totalBinance: null,
      afterNewListing: null,
      afterAlpha: null,
      afterBlacklist: null,
      active: null,
      excludedNewListing: 0,
      excludedAlpha: 0,
      excludedBlacklist: 0,
    },
    // equity curve — cumulative pnl history from EXIT events
    equityHistory: [],   // {ts, cum_pnl}
    cumPnl: 0,
    // regime tracking for change banner
    lastSeenRegime: null,
    // activity pulse throttle
    lastPulseAt: 0,
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
  let elView, elFeed, elLatencyBar, elLatencyText,
    elConstellation, elEvMin, elScanCrawlBar, elScanCrawlText, elCycleNum,
    elRadarSvg, elPosBody, elScansBody, elHeatmap, elFooterStats, elBoot,
    elBootLines, elAlertBanner, elAlertMsg, elKatakana, elHbDot,
    elFeedPulse, elRegimeBanner, elFunnelBody, elEquityCurve,
    elHeatCompassSvg, elHcPanel, elHcFootLean, elHcFootRegime, elHcFootPressure;
  let _hcLastBeatAt = 0;

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
        if (evt.hb.state) {
          // Detect regime change for banner + glitch (May 18)
          const newRegime = evt.hb.state.regime;
          if (state.lastSeenRegime && newRegime && newRegime !== state.lastSeenRegime) {
            showRegimeBanner(state.lastSeenRegime, newRegime);
            triggerRegimeGlitch();
          }
          if (newRegime) state.lastSeenRegime = newRegime;
          state.botState = evt.hb.state;
        }
        if (evt.hb.health) { state.health = evt.hb.health; updateConstellation(); }
        updatePositionsPanel();
        updateFooterStats();
        if (evt.hb.state) updateHeatCompass(evt.hb.state);
        state.health.log = true;
        updateConstellation();
      }
      return;
    }

    // Heatmap and radar have DIFFERENT population rules (May 18, Option A):
    //   - RADAR = "currently being actively watched" — skips filter-out lines.
    //   - HEATMAP = "bot's full awareness universe" — parses filter-out lines
    //     too, adding cells in dim 'AWARENESS' state. Real scan events later
    //     upgrade those cells to score-based colors.
    if (!isReplay) {
      const msg = evt.msg || '';
      const isFilterOut = /\b(excluded|Blacklist active|new-listing filter|Alpha-subtype filter)\b/i.test(msg);
      if (isFilterOut) {
        // FILTER-OUT path: heatmap-only awareness. Do NOT touch radar/top-scans.
        // Discriminate the [SCAN] Blacklist active line — those symbols get
        // the sticky lt-pc-blacklist state. 20ms stagger creates a sweep
        // effect across simultaneously-revealed cells (May 18 polish).
        const filterSyms = msg.match(/\b(?:\d+)?[A-Z]{2,10}USDT\b/g);
        if (filterSyms) {
          const isBlacklist = /\bBlacklist active\b/.test(msg);
          const type = isBlacklist ? 'BLACKLIST' : 'AWARENESS';
          filterSyms.forEach((sym, i) => {
            updatePairHeatmap(sym, type, { pulseDelay: i * 20 });
          });
        }
      } else {
        // ACTIVE-SCAN path: radar + top-scans + heatmap.
        const allSyms = msg.match(/\b(?:\d+)?[A-Z]{2,10}USDT\b/g);
        if (allSyms && allSyms.length > 0) {
          const now = Date.now();
          for (const sym of allSyms) {
            state.radarSymbolEvents.push({ ts: now, symbol: sym });
            pushTopScan(sym);
            updatePairHeatmap(sym, 'SCAN');   // May 18: pair heatmap
          }
          const cutoff = now - 60000;
          while (state.radarSymbolEvents.length > 0 && state.radarSymbolEvents[0].ts < cutoff) {
            state.radarSymbolEvents.shift();
          }
        } else if (evt.symbol) {
          // Fallback: at least record the parsed primary symbol
          state.radarSymbolEvents.push({ ts: Date.now(), symbol: evt.symbol });
          updatePairHeatmap(evt.symbol, 'SCAN');
        }
      }
    }
    // Reject flash on radar — always honor parsed primary symbol (REJECT
    // category by definition mentions the rejected pair).
    if (evt.category === 'REJECT' && evt.symbol) {
      state.radarRejectFlash.set(evt.symbol, Date.now() + 1000);
      updatePairHeatmap(evt.symbol, 'REJECT');   // May 18: pair heatmap
    }
    // Heatmap state-overrides for WATCH/ENTRY on the primary symbol (May 18)
    if (evt.category === 'WATCH' && evt.symbol) {
      updatePairHeatmap(evt.symbol, 'WATCH');
    }
    if (evt.category === 'ENTRY' && evt.symbol) {
      updatePairHeatmap(evt.symbol, 'ENTRY');
    }
    // Filter funnel parsing — only [BINANCE] and [SCAN] tags carry the relevant lines
    if (evt.tag === 'BINANCE' || evt.tag === 'SCAN') {
      parseFilterFunnel(evt.msg || '');
    }

    // SCAN events — drive scan crawl (radar updates via the symbol-events
    // tracker added above, no per-event radar call needed)
    if (evt.category === 'SCAN' && evt.symbol) {
      bumpScanCrawl(evt.symbol);
    }
    // (TOP SCANS now fed from the all-symbol extraction above —
    // the old single-symbol SCAN/WATCH branch was getting stuck on
    // the first match of filter-out lists.)

    // ENTRY — open position tracking
    if (evt.category === 'ENTRY' && evt.symbol) {
      // Track open position (we won't know P&L yet — just placeholder)
      const dir = evt.side || 'LONG';
      state.positions.set(evt.symbol, { pnl_pct: 0, direction: dir });
      updatePositionsPanel();
    }

    // EXIT — read pnl_pct from kv, increment stats, append equity curve
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
        // Equity curve history (live trades only — replay would distort timestamps)
        if (isFinite(pnlPct)) {
          state.cumPnl += pnlPct;
          state.equityHistory.push({ ts: evt.ts * 1000, cum_pnl: state.cumPnl });
          if (state.equityHistory.length > 200) state.equityHistory.shift();
        }
      }
      // Record last-closed for positions waiting state
      if (isFinite(pnlPct)) {
        state.lastClosed = {
          symbol: evt.symbol,
          pnl_pct: pnlPct,
          at: evt.ts * 1000,
        };
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
      triggerActivityPulse(evt.level);
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
      // Logarithmic scaling: tiny bursts look substantial, heavy activity still differentiates
      // height_percent = min(100, ln(count + 1) * 30)
      const c = state.heat[i].count;
      const h = Math.min(100, Math.log(c + 1) * 30);
      bar.style.height = h.toFixed(0) + '%';
      bar.style.opacity = (0.25 + 0.75 * Math.min(1, c / Math.max(state.heatMaxRecent, 1))).toFixed(2);
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

  // ─── Radar (SVG) — persistent blips + sweep collision + frequency score ─
  // Layout: 220×220 viewBox, center (110,110), three rings at r=40/70/100.
  // Score proxy = symbol frequency in last 60s of events. Higher score →
  // closer to center. Angle is deterministic from symbol hash.
  // Sweep is rAF-driven so we can detect blip collisions and brighten them.
  function buildRadar() {
    const ns = 'http://www.w3.org/2000/svg';
    elRadarSvg.innerHTML = '';
    const w = 220, h = 220, cx = w / 2, cy = h / 2;
    elRadarSvg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    elRadarSvg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    // Rings + edge-right labels indicating what each ring means
    const ringSpec = [
      { r: 30, label: 'score>85' },
      { r: 60, label: 'score>70' },
      { r: 95, label: 'score>50' },
    ];
    for (const { r, label } of ringSpec) {
      const c = document.createElementNS(ns, 'circle');
      c.setAttribute('cx', cx); c.setAttribute('cy', cy); c.setAttribute('r', r);
      c.setAttribute('fill', 'none');
      c.setAttribute('stroke', '#22c55e');
      c.setAttribute('stroke-opacity', '0.18');
      c.setAttribute('stroke-width', '1');
      elRadarSvg.appendChild(c);
      // Ring label — placed at the TOP of each ring's right side so the
      // three labels are vertically staggered (one per ring) instead of
      // stacking on the same y and visually overlapping each other.
      const t = document.createElementNS(ns, 'text');
      t.setAttribute('class', 'lt-ring-label');
      t.setAttribute('x', cx + 4);
      t.setAttribute('y', cy - r - 2);
      t.textContent = label;
      elRadarSvg.appendChild(t);
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
    // Defs (gradient for sweep)
    const defs = document.createElementNS(ns, 'defs');
    defs.innerHTML = `<linearGradient id="lt-sweep-grad" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#22c55e" stop-opacity="0.6"/>
      <stop offset="100%" stop-color="#22c55e" stop-opacity="0"/>
    </linearGradient>`;
    elRadarSvg.appendChild(defs);
    // Sweep — rAF-driven so we can detect blip collisions. Radius 95 to
    // match outer ring exactly (was 100, overshooting). Pivot is specified
    // via SVG `transform="rotate(deg cx cy)"` attribute in radarTick — do
    // NOT set CSS transform-origin here (mixes coordinate systems on SVG
    // and visibly shifts the wedge off-center in some browsers).
    const R = 95;
    const sweep = document.createElementNS(ns, 'path');
    sweep.setAttribute('id', 'lt-radar-sweep');
    sweep.setAttribute('d', `M ${cx} ${cy} L ${cx + R} ${cy} A ${R} ${R} 0 0 0 ${cx + R * Math.cos(-Math.PI / 6)} ${cy + R * Math.sin(-Math.PI / 6)} Z`);
    sweep.setAttribute('fill', 'url(#lt-sweep-grad)');
    elRadarSvg.appendChild(sweep);
    // Container groups for blips and labels (rendered above sweep)
    const blipGroup = document.createElementNS(ns, 'g');
    blipGroup.setAttribute('id', 'lt-blip-group');
    elRadarSvg.appendChild(blipGroup);
    const labelGroup = document.createElementNS(ns, 'g');
    labelGroup.setAttribute('id', 'lt-blip-label-group');
    elRadarSvg.appendChild(labelGroup);
    // Center dot
    const center = document.createElementNS(ns, 'circle');
    center.setAttribute('cx', cx); center.setAttribute('cy', cy); center.setAttribute('r', 2);
    center.setAttribute('fill', '#4ade80');
    elRadarSvg.appendChild(center);
    // Kick off the rAF sweep loop
    state.radarSweepStarted = performance.now();
    requestAnimationFrame(radarTick);
  }

  // Per-frame: advance sweep, recompute blip positions/states, render.
  function radarTick(now) {
    if (!state.inTerminal || !elRadarSvg) {
      // Pause the loop when terminal hidden, but reschedule for when we come back
      setTimeout(() => requestAnimationFrame(radarTick), 200);
      return;
    }
    // Sweep angle — full rotation every 4s, clockwise.  Convert to radians.
    const elapsed = (now - state.radarSweepStarted) / 1000;
    const sweepDeg = (elapsed * 90) % 360;          // 90 deg/s = 4s/rev
    const sweepAngle = (sweepDeg * Math.PI) / 180;
    state.radarSweepAngle = sweepAngle;
    const sweepEl = elRadarSvg.querySelector('#lt-radar-sweep');
    if (sweepEl) sweepEl.setAttribute('transform', `rotate(${sweepDeg} 110 110)`);
    renderRadarBlips(sweepAngle);
    requestAnimationFrame(radarTick);
  }

  // Frequency score per symbol from last 60s. Returns Map<symbol, score 0..100>.
  function computeRadarScores() {
    const counts = new Map();
    for (const e of state.radarSymbolEvents) {
      counts.set(e.symbol, (counts.get(e.symbol) || 0) + 1);
    }
    if (counts.size === 0) return counts;
    // Normalize: max-frequency symbol = 100
    let maxC = 1;
    counts.forEach(v => { if (v > maxC) maxC = v; });
    const out = new Map();
    counts.forEach((v, k) => out.set(k, (v / maxC) * 100));
    return out;
  }

  // Hash → 0..2π (deterministic per symbol; preserves prior placement so blips
  // don't jump as their score changes).
  function symbolToAngle(symbol) {
    let h = 5381;
    for (let i = 0; i < symbol.length; i++) {
      h = ((h << 5) + h + symbol.charCodeAt(i)) | 0;
    }
    return ((Math.abs(h) % 360) * Math.PI) / 180;
  }

  // Score → distance from center (px). Higher score = closer to center.
  function scoreToDist(score) {
    // score 100 → r=20 (inside inner ring), score 0 → r=100 (outer edge)
    return 100 - Math.max(0, Math.min(100, score)) * 0.8;
  }

  function renderRadarBlips(sweepAngle) {
    const blipGroup = elRadarSvg.querySelector('#lt-blip-group');
    const labelGroup = elRadarSvg.querySelector('#lt-blip-label-group');
    if (!blipGroup || !labelGroup) return;
    const ns = 'http://www.w3.org/2000/svg';
    const cx = 110, cy = 110;
    const scores = computeRadarScores();
    if (scores.size === 0) {
      blipGroup.innerHTML = '';
      labelGroup.innerHTML = '';
      return;
    }
    // Determine top-3 for amber state
    const sortedSymbols = [...scores.entries()].sort((a, b) => b[1] - a[1]);
    const top3 = new Set(sortedSymbols.slice(0, 3).map(e => e[0]));
    const topClosest = sortedSymbols.slice(0, 5).map(e => e[0]);
    const now = Date.now();
    // Build set of existing blip elements for diff
    const seen = new Set();
    scores.forEach((score, symbol) => {
      seen.add(symbol);
      const angle = symbolToAngle(symbol);
      const dist = scoreToDist(score);
      const x = cx + dist * Math.cos(angle);
      const y = cy + dist * Math.sin(angle);
      // Determine state
      let stateClass = 'lt-blip-dim';
      // Reject flash (highest priority)
      const rejectUntil = state.radarRejectFlash.get(symbol);
      if (rejectUntil && rejectUntil > now) {
        stateClass = 'lt-blip-reject';
      } else if (rejectUntil) {
        state.radarRejectFlash.delete(symbol);
      }
      // Top-3 amber
      if (stateClass === 'lt-blip-dim' && top3.has(symbol)) {
        stateClass = 'lt-blip-amber';
      }
      // Sweep collision — bright for ~0.8s after pass
      const diff = Math.abs(((angle - sweepAngle + Math.PI * 3) % (Math.PI * 2)) - Math.PI);
      const onSweep = diff > (Math.PI - 0.18);  // within ~10° of sweep
      if (onSweep) {
        const blip = state.radarBlips.get(symbol) || {};
        blip.lastSweepHit = now;
        state.radarBlips.set(symbol, blip);
      }
      const hit = state.radarBlips.get(symbol);
      if (hit && hit.lastSweepHit && (now - hit.lastSweepHit) < 800) {
        if (stateClass === 'lt-blip-dim' || stateClass === 'lt-blip-amber') {
          stateClass = 'lt-blip-bright';
        }
      }
      const id = 'lt-blip-' + cssId(symbol);
      let circle = blipGroup.querySelector('#' + id);
      if (!circle) {
        circle = document.createElementNS(ns, 'circle');
        circle.setAttribute('id', id);
        circle.setAttribute('r', '2.5');
        circle.setAttribute('class', 'lt-blip ' + stateClass);
        blipGroup.appendChild(circle);
      }
      circle.setAttribute('cx', x.toFixed(1));
      circle.setAttribute('cy', y.toFixed(1));
      circle.setAttribute('class', 'lt-blip ' + stateClass);
    });
    // Cleanup blips for symbols that aged out
    Array.from(blipGroup.children).forEach(c => {
      const sym = c.id.replace('lt-blip-', '');
      const realSym = sym; // simplified — cssId is reversible since it just maps
      if (!seen.has(realSym)) c.remove();
    });
    // Render labels for top-5 closest only
    labelGroup.innerHTML = '';
    topClosest.forEach(symbol => {
      const angle = symbolToAngle(symbol);
      const dist = scoreToDist(scores.get(symbol));
      const x = cx + dist * Math.cos(angle);
      const y = cy + dist * Math.sin(angle);
      const t = document.createElementNS(ns, 'text');
      t.setAttribute('class', 'lt-blip-label');
      t.setAttribute('x', (x + 4).toFixed(1));
      t.setAttribute('y', (y + 2).toFixed(1));
      t.textContent = symbol.replace('USDT', '');
      labelGroup.appendChild(t);
    });
  }

  // Sanitize a symbol for use as an SVG id (USDT pairs are already safe but be defensive)
  function cssId(s) { return String(s).replace(/[^A-Za-z0-9_-]/g, '_'); }

  // ─── HEAT COMPASS (May 16) ─────────────────────────────────────────────
  // Cardinal-compass visualization: LONGS (N) / SHORTS (S) / VOL (E) / VLT (W).
  // Center drift dot summarizes overall market lean. Data from HEARTBEAT.
  const HC = {
    cx: 130, cy: 95,              // viewBox center (0 0 260 190)
    rings: [22, 38, 54],          // 3 concentric ring radii
    cardLabelOffset: 70,          // cardinal label distance from center
    barMaxN: 50,                  // max bar extent in viewBox units
    barThickness: 6,              // bar thickness
    bullScale: 30,                // bull/bear count → bar length (count of 30 → max bar)
    volScale: 2.0,                // global vol 0..2.0 maps to 0..1
  };
  function buildHeatCompassSvg() {
    if (!elHeatCompassSvg) return;
    const { cx, cy, rings, cardLabelOffset } = HC;
    elHeatCompassSvg.innerHTML = '';

    // Concentric rings
    rings.forEach((r, i) => {
      const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      c.setAttribute('cx', cx); c.setAttribute('cy', cy); c.setAttribute('r', r);
      c.setAttribute('class', 'lt-hc-ring' + (i === rings.length - 1 ? ' lt-hc-ring-outer' : ''));
      elHeatCompassSvg.appendChild(c);
    });

    // Cross lines (faint cardinal guides)
    const crossSpan = rings[rings.length - 1] + 4;
    [
      [cx, cy - crossSpan, cx, cy + crossSpan],
      [cx - crossSpan, cy, cx + crossSpan, cy],
    ].forEach(([x1, y1, x2, y2]) => {
      const l = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      l.setAttribute('x1', x1); l.setAttribute('y1', y1);
      l.setAttribute('x2', x2); l.setAttribute('y2', y2);
      l.setAttribute('class', 'lt-hc-cross');
      elHeatCompassSvg.appendChild(l);
    });

    // Cardinal labels (LONGS / SHORTS / VOL / VLT)
    const cards = [
      { id: 'lt-hc-card-n', x: cx,                       y: cy - cardLabelOffset,           t: 'LONGS' },
      { id: 'lt-hc-card-s', x: cx,                       y: cy + cardLabelOffset + 6,       t: 'SHORTS' },
      { id: 'lt-hc-card-e', x: cx + cardLabelOffset + 8, y: cy + 3,                         t: 'VOL' },
      { id: 'lt-hc-card-w', x: cx - cardLabelOffset - 8, y: cy + 3,                         t: 'VLT' },
    ];
    cards.forEach(({ id, x, y, t }) => {
      const tx = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      tx.setAttribute('id', id); tx.setAttribute('x', x); tx.setAttribute('y', y);
      tx.setAttribute('class', 'lt-hc-card-label'); tx.textContent = t;
      elHeatCompassSvg.appendChild(tx);
    });

    // Bars (rect elements, sized at update time) and numeric labels
    const t = HC.barThickness;
    const bars = [
      { id: 'lt-hc-bar-n', cls: 'lt-hc-bar lt-hc-bar-n', x: cx - t/2, y: cy,         w: t, h: 0 },
      { id: 'lt-hc-bar-s', cls: 'lt-hc-bar lt-hc-bar-s', x: cx - t/2, y: cy,         w: t, h: 0 },
      { id: 'lt-hc-bar-e', cls: 'lt-hc-bar lt-hc-bar-e', x: cx,       y: cy - t/2,   w: 0, h: t },
      { id: 'lt-hc-bar-w', cls: 'lt-hc-bar lt-hc-bar-w', x: cx,       y: cy - t/2,   w: 0, h: t },
    ];
    bars.forEach(b => {
      const r = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      r.setAttribute('id', b.id); r.setAttribute('class', b.cls);
      r.setAttribute('x', b.x); r.setAttribute('y', b.y);
      r.setAttribute('width', b.w); r.setAttribute('height', b.h);
      r.setAttribute('rx', 1); r.setAttribute('ry', 1);
      elHeatCompassSvg.appendChild(r);
    });

    // Numeric labels at bar tips (positioned at update time)
    const nums = [
      { id: 'lt-hc-num-n', cls: 'lt-hc-num lt-hc-num-green', x: cx, y: cy - 14 },
      { id: 'lt-hc-num-s', cls: 'lt-hc-num lt-hc-num-red',   x: cx, y: cy + 18 },
      { id: 'lt-hc-num-e', cls: 'lt-hc-num lt-hc-num-green', x: cx + 14, y: cy + 3 },
      { id: 'lt-hc-num-w', cls: 'lt-hc-num',                 x: cx - 14, y: cy + 3 },
    ];
    nums.forEach(n => {
      const tx = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      tx.setAttribute('id', n.id); tx.setAttribute('class', n.cls);
      tx.setAttribute('x', n.x); tx.setAttribute('y', n.y);
      tx.textContent = '—';
      elHeatCompassSvg.appendChild(tx);
    });

    // Center drift dot — outer <g> handles data-driven translate (CSS transition),
    // inner <g> runs the continuous wander animation (CSS keyframe), innermost is the circle itself
    const drift = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    drift.setAttribute('id', 'lt-hc-drift');
    drift.setAttribute('class', 'lt-hc-center-drift');
    drift.style.transform = 'translate(0px, 0px)';
    drift.style.transformOrigin = `${cx}px ${cy}px`;
    drift.style.transformBox = 'fill-box';

    const wander = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    wander.setAttribute('class', 'lt-hc-center-wander');

    const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    dot.setAttribute('class', 'lt-hc-center-dot');
    dot.setAttribute('cx', cx); dot.setAttribute('cy', cy); dot.setAttribute('r', 3.5);
    wander.appendChild(dot);
    drift.appendChild(wander);
    elHeatCompassSvg.appendChild(drift);
  }

  function _hcClassifyVlt(adx) {
    if (adx == null || isNaN(adx)) return { label: '—', frac: 0 };
    if (adx < 20) return { label: 'LOW',  frac: 0.30 };
    if (adx <= 25) return { label: 'MID', frac: 0.65 };
    return { label: 'HIGH', frac: 1.0 };
  }
  function _hcClassifyPressure(gv, adx) {
    // Simple heuristic: high pressure = high vol AND high ADX. Low = both low.
    if (gv == null && adx == null) return { label: '—', cls: 'lt-hc-foot-grey' };
    const v = gv == null ? 0 : Math.min(gv / 1.5, 1);
    const a = adx == null ? 0 : Math.min(adx / 35, 1);
    const score = (v + a) / 2;
    if (score < 0.40) return { label: 'LOW',  cls: 'lt-hc-foot-grey' };
    if (score < 0.70) return { label: 'MID',  cls: '' };           // default green
    return { label: 'HIGH', cls: 'lt-hc-foot-amber' };
  }

  function updateHeatCompass(s) {
    if (!elHeatCompassSvg || !s) return;
    // TEMP DIAGNOSTIC (May 16) — remove once payload confirmed
    console.log('[HC]', 'bull=', s.breadth_n_bull, 'bear=', s.breadth_n_bear,
                'gv=', s.global_volume_ratio, 'adx=', s.btc_adx, 'regime=', s.regime);
    _hcLastBeatAt = Date.now();
    if (elHcPanel) elHcPanel.classList.remove('lt-hc-stale');

    const { cx, cy, barMaxN, barThickness: bt, bullScale, volScale } = HC;
    const bull = s.breadth_n_bull;
    const bear = s.breadth_n_bear;
    const gv   = s.global_volume_ratio;
    const adx  = s.btc_adx;

    // N bar — LONGS count → height upward
    const barN = document.getElementById('lt-hc-bar-n');
    const numN = document.getElementById('lt-hc-num-n');
    if (barN && numN) {
      const h = bull == null ? 0 : Math.min(bull / bullScale, 1) * barMaxN;
      barN.setAttribute('y', cy - h);
      barN.setAttribute('height', h);
      const tipY = Math.max(cy - h - 4, cy - barMaxN - 4);
      numN.setAttribute('x', cx);
      numN.setAttribute('y', tipY);
      numN.textContent = bull == null ? '—' : String(bull);
    }
    // S bar — SHORTS count → height downward
    const barS = document.getElementById('lt-hc-bar-s');
    const numS = document.getElementById('lt-hc-num-s');
    if (barS && numS) {
      const h = bear == null ? 0 : Math.min(bear / bullScale, 1) * barMaxN;
      barS.setAttribute('y', cy);
      barS.setAttribute('height', h);
      const tipY = Math.min(cy + h + 10, cy + barMaxN + 10);
      numS.setAttribute('x', cx);
      numS.setAttribute('y', tipY);
      numS.textContent = bear == null ? '—' : String(bear);
    }
    // E bar — VOL → width rightward
    const barE = document.getElementById('lt-hc-bar-e');
    const numE = document.getElementById('lt-hc-num-e');
    if (barE && numE) {
      const w = gv == null ? 0 : Math.min(gv / volScale, 1) * barMaxN;
      barE.setAttribute('x', cx);
      barE.setAttribute('width', w);
      const tipX = Math.min(cx + w + 12, cx + barMaxN + 12);
      numE.setAttribute('x', tipX);
      numE.setAttribute('y', cy + 3);
      numE.textContent = gv == null ? '—' : gv.toFixed(2);
    }
    // W bar — VLT (BTC ADX classified) → width leftward
    const vlt = _hcClassifyVlt(adx);
    const barW = document.getElementById('lt-hc-bar-w');
    const numW = document.getElementById('lt-hc-num-w');
    if (barW && numW) {
      const w = vlt.frac * barMaxN;
      barW.setAttribute('x', cx - w);
      barW.setAttribute('width', w);
      const tipX = Math.max(cx - w - 12, cx - barMaxN - 12);
      numW.setAttribute('x', tipX);
      numW.setAttribute('y', cy + 3);
      numW.textContent = vlt.label;
    }

    // Center drift dot — data-driven offset on the outer <g>
    const drift = document.getElementById('lt-hc-drift');
    if (drift) {
      const totBB = (bull || 0) + (bear || 0);
      const xLean = totBB > 0 ? (((bull || 0) - (bear || 0)) / totBB) * 6 : 0;
      const adxNorm = adx == null ? 0 : Math.min(adx / 30, 1);
      const volNorm = gv == null ? 0 : Math.min(gv / 1.5, 1);
      const yLean = -((volNorm + adxNorm) / 2 - 0.5) * 6;
      const xOff = Math.max(-8, Math.min(8, xLean));
      const yOff = Math.max(-8, Math.min(8, yLean));
      drift.style.transform = `translate(${xOff.toFixed(2)}px, ${yOff.toFixed(2)}px)`;
    }

    // Footer meta line
    if (elHcFootLean) {
      if (bull == null || bear == null) {
        elHcFootLean.textContent = '—';
        elHcFootLean.className = 'lt-hc-foot-val lt-hc-foot-grey';
      } else {
        const diff = bull - bear;
        if (diff === 0) {
          elHcFootLean.textContent = 'BALANCED';
          elHcFootLean.className = 'lt-hc-foot-val lt-hc-foot-grey';
        } else if (diff > 0) {
          elHcFootLean.textContent = `LONG +${diff}`;
          elHcFootLean.className = 'lt-hc-foot-val';
        } else {
          elHcFootLean.textContent = `SHORT +${Math.abs(diff)}`;
          elHcFootLean.className = 'lt-hc-foot-val lt-hc-foot-red';
        }
      }
    }
    if (elHcFootRegime) {
      const r = s.regime || '—';
      elHcFootRegime.textContent = r;
      const isBull = /BULL/i.test(r);
      const isBear = /BEAR/i.test(r);
      elHcFootRegime.className = 'lt-hc-foot-val' +
        (isBull ? '' : isBear ? ' lt-hc-foot-red' : ' lt-hc-foot-grey');
    }
    if (elHcFootPressure) {
      const p = _hcClassifyPressure(gv, adx);
      elHcFootPressure.textContent = p.label;
      elHcFootPressure.className = 'lt-hc-foot-val' + (p.cls ? ' ' + p.cls : '');
    }
  }

  // Mark compass stale if no heartbeat for >6s (2 missed beats)
  setInterval(() => {
    if (!elHcPanel) return;
    if (_hcLastBeatAt && (Date.now() - _hcLastBeatAt) > 6000) {
      elHcPanel.classList.add('lt-hc-stale');
    }
  }, 2000);

  // ─── Positions panel ───────────────────────────────────────────────────
  function updatePositionsPanel() {
    if (!elPosBody) return;
    if (state.positions.size > 0) {
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
      return;
    }
    // Waiting state — always has motion + status
    const scanned = state.botState && state.botState.scanned_pairs_count
      ? state.botState.scanned_pairs_count
      : (state.funnel.active || '—');
    const cycle = state.scanCycle;
    let lastClosedLine = '';
    if (state.lastClosed) {
      const ago = Math.max(0, Math.floor((Date.now() - state.lastClosed.at) / 60000));
      const sym = state.lastClosed.symbol.replace('USDT', '');
      const pnl = state.lastClosed.pnl_pct;
      const pnlStr = (pnl >= 0 ? '+' : '') + pnl.toFixed(2) + '%';
      const cls = pnl >= 0 ? 'lt-pos-up' : 'lt-pos-dn';
      lastClosedLine = `<div class="lt-pos-line"><span class="lt-label-mini">last closed:</span><span>${sym}</span><span class="${cls}">${pnlStr}</span><span class="lt-label-mini">· ${ago}m ago</span></div>`;
    }
    elPosBody.innerHTML = `<div class="lt-pos-waiting">
      <div class="lt-pos-line" style="opacity:0.5">no open positions</div>
      <div class="lt-pos-line"><span class="lt-spin">◌</span><span>waiting for entry signal</span></div>
      <div class="lt-pos-line"><span class="lt-label-mini">scanning ${scanned} pairs · cycle</span><span>#${cycle}</span></div>
      ${lastClosedLine}
    </div>`;
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

  // Ticker removed (May 15 PM) — the empty "awaiting prices…" bar
  // was always-empty when no positions were open, so the whole strip
  // was retired. Scan crawl + radar + heatmap + feed type-on already
  // give the "alive" feel.

  // ─── Footer stats — bigger, glowing values + inline equity sparkline ───
  function updateFooterStats() {
    if (!elFooterStats) return;
    const wr = state.closedTrades > 0
      ? Math.round((state.winsToday / state.closedTrades) * 100)
      : 0;
    const pnlStr = (state.pnlToday >= 0 ? '+' : '') + state.pnlToday.toFixed(2) + '%';
    const pnlCls = state.pnlToday >= 0 ? '' : ' lt-stat-neg';
    const ts = new Date();
    const tyo = new Date(ts.getTime() + 9 * 60 * 60 * 1000);
    const ldn = new Date(ts.getTime() + 0 * 60 * 60 * 1000);
    const nyc = new Date(ts.getTime() - 5 * 60 * 60 * 1000);
    const fmt = (d) => `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
    elFooterStats.innerHTML = `
      <span><span class="lt-stat-label">trades</span><span class="lt-stat-value">${state.closedTrades}</span></span>
      <span><span class="lt-stat-label">pnl</span><span class="lt-stat-value${pnlCls}">${pnlStr}</span><svg class="lt-equity-curve" viewBox="0 0 120 24" preserveAspectRatio="none"></svg></span>
      <span><span class="lt-stat-label">win</span><span class="lt-stat-value">${wr}%</span></span>
      <span class="lt-clock-cell">TYO ${fmt(tyo)} · LDN ${fmt(ldn)} · NYC ${fmt(nyc)}</span>
    `;
    renderEquityCurve();
  }
  setInterval(updateFooterStats, 5000);

  // ─── Equity curve ──────────────────────────────────────────────────────
  function renderEquityCurve() {
    const svg = elFooterStats && elFooterStats.querySelector('.lt-equity-curve');
    if (!svg) return;
    const hist = state.equityHistory;
    const W = 120, H = 24;
    if (hist.length < 2) {
      // <2 trades — pulsing dot at right edge only (per spec)
      svg.innerHTML = `<circle class="lt-equity-dot" cx="${W - 4}" cy="${H / 2}" r="2"></circle>`;
      return;
    }
    // Auto-scale Y to session range; X spreads evenly across history
    const min = Math.min(...hist.map(p => p.cum_pnl));
    const max = Math.max(...hist.map(p => p.cum_pnl));
    const yPad = 2;
    const range = Math.max(0.01, max - min);
    const yFor = v => yPad + ((max - v) / range) * (H - yPad * 2);
    const xFor = i => (i / (hist.length - 1)) * W;
    const points = hist.map((p, i) => `${xFor(i).toFixed(1)},${yFor(p.cum_pnl).toFixed(1)}`);
    const d = 'M ' + points.join(' L ');
    const lastX = xFor(hist.length - 1).toFixed(1);
    const lastY = yFor(hist[hist.length - 1].cum_pnl).toFixed(1);
    svg.innerHTML = `<path d="${d}"></path><circle class="lt-equity-dot" cx="${lastX}" cy="${lastY}" r="2"></circle>`;
  }

  // ─── Filter funnel ─────────────────────────────────────────────────────
  function parseFilterFunnel(msg) {
    if (!msg) return;
    let changed = false;
    // [BINANCE] Fetched N pairs from Binance (limit=M)
    let m = msg.match(/Fetched (\d+) pairs from Binance/);
    if (m) {
      const n = parseInt(m[1], 10);
      if (state.funnel.active !== n) { state.funnel.active = n; changed = true; }
    }
    // [BINANCE] New-listing filter (180d): excluded N/M pairs
    m = msg.match(/New-listing filter[^:]*:\s*excluded (\d+)\/(\d+)\s*pairs/);
    if (m) {
      const excl = parseInt(m[1], 10);
      const total = parseInt(m[2], 10);
      const after = total - excl;
      if (state.funnel.totalBinance !== total) { state.funnel.totalBinance = total; changed = true; }
      if (state.funnel.afterNewListing !== after) { state.funnel.afterNewListing = after; changed = true; }
      if (state.funnel.excludedNewListing !== excl) { state.funnel.excludedNewListing = excl; changed = true; }
    }
    // [BINANCE] Alpha-subtype filter: excluded N/M pairs
    m = msg.match(/Alpha-subtype filter:\s*excluded (\d+)\/(\d+)\s*pairs/);
    if (m) {
      const excl = parseInt(m[1], 10);
      const inputTotal = parseInt(m[2], 10);
      const after = inputTotal - excl;
      if (state.funnel.afterAlpha !== after) { state.funnel.afterAlpha = after; changed = true; }
      if (state.funnel.excludedAlpha !== excl) { state.funnel.excludedAlpha = excl; changed = true; }
    }
    // [SCAN] Blacklist active: excluded N pairs (...)
    m = msg.match(/Blacklist active:\s*excluded (\d+)\s*pairs/);
    if (m) {
      const excl = parseInt(m[1], 10);
      if (state.funnel.excludedBlacklist !== excl) {
        state.funnel.excludedBlacklist = excl;
        if (state.funnel.afterAlpha != null) {
          state.funnel.afterBlacklist = state.funnel.afterAlpha - excl;
        }
        changed = true;
      }
    }
    if (changed) renderFunnel(true);
  }

  function renderFunnel(flash) {
    if (!elFunnelBody) return;
    const f = state.funnel;
    const fmt = v => v == null ? '—' : String(v);
    const fmtDelta = v => (v == null || v === 0) ? '' : `-${v}`;
    const flashCls = flash ? ' lt-funnel-flash' : '';
    // Labels shortened (drop "filter" suffix) so they fit the panel width.
    elFunnelBody.innerHTML = `
      <div class="lt-funnel-row${flashCls}"><span class="lt-funnel-num">${fmt(f.totalBinance)}</span><span class="lt-funnel-label">total</span></div>
      <div class="lt-funnel-arrow">▼ new-listing</div>
      <div class="lt-funnel-row${flashCls}"><span class="lt-funnel-num">${fmt(f.afterNewListing)}</span><span class="lt-funnel-delta">${fmtDelta(f.excludedNewListing)}</span></div>
      <div class="lt-funnel-arrow">▼ alpha-subtype</div>
      <div class="lt-funnel-row${flashCls}"><span class="lt-funnel-num">${fmt(f.afterAlpha)}</span><span class="lt-funnel-delta">${fmtDelta(f.excludedAlpha)}</span></div>
      <div class="lt-funnel-arrow">▼ blacklist</div>
      <div class="lt-funnel-row${flashCls}"><span class="lt-funnel-num">${fmt(f.afterBlacklist)}</span><span class="lt-funnel-delta">${fmtDelta(f.excludedBlacklist)}</span></div>
      <div class="lt-funnel-arrow">▼ scan limit</div>
      <div class="lt-funnel-row${flashCls}"><span class="lt-funnel-num">${fmt(f.active)}</span><span class="lt-funnel-label">active</span></div>
    `;
    if (flash) {
      setTimeout(() => {
        elFunnelBody.querySelectorAll('.lt-funnel-flash').forEach(el => el.classList.remove('lt-funnel-flash'));
      }, 220);
    }
  }

  // ─── Regime change banner ──────────────────────────────────────────────
  function showRegimeBanner(prev, curr) {
    if (!elRegimeBanner) return;
    // Strip subtype suffixes for the banner — show family only
    const fam = (r) => (r || '').replace(/_.*$/, '').replace('HEALTHY', '').trim() || (r || '—');
    elRegimeBanner.textContent = `▸ REGIME CHANGE   ${prev} → ${curr}`;
    // Re-trigger CSS animation by removing/adding the class
    elRegimeBanner.classList.remove('lt-regime-banner-active');
    void elRegimeBanner.offsetWidth;  // force reflow so animation replays
    elRegimeBanner.classList.add('lt-regime-banner-active');
  }

  // ─── Activity pulse (throttled to 100ms minimum gap) ──────────────────
  function triggerActivityPulse(level) {
    if (!elFeedPulse) return;
    const now = performance.now();
    if (now - state.lastPulseAt < 100) return;
    state.lastPulseAt = now;
    let cls = 'lt-pulse-info';
    if (level === 'ERROR' || level === 'CRITICAL') cls = 'lt-pulse-error';
    else if (level === 'WARNING') cls = 'lt-pulse-warn';
    elFeedPulse.className = 'lt-feed-pulse ' + cls;
    // Fade back to 0 after 150ms
    setTimeout(() => { if (elFeedPulse) elFeedPulse.className = 'lt-feed-pulse'; }, 150);
  }

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

  // ─── Katakana rain (Theater Pass v2, May 18) ──────────────────────────
  // Four edge containers (left/left2/right/right2), each with 2-3 streams.
  // CSS animates fall; JS mutates one glyph per stream every 200ms.
  const _rainStreams = [];  // refs to all stream <div>s for mutation tick
  function _randChar() {
    return KATAKANA_CHARS[Math.floor(Math.random() * KATAKANA_CHARS.length)];
  }
  function buildRainColumn(containerId, streamCount) {
    const el = document.getElementById(containerId);
    if (!el) return;
    el.innerHTML = '';
    const frag = document.createDocumentFragment();
    for (let s = 0; s < streamCount; s++) {
      const len = 16 + Math.floor(Math.random() * 9);    // 16-24 glyphs
      const dur = 3.5 + Math.random() * 4.5;             // 3.5-8s
      const delay = Math.random() * 5;                   // 0-5s stagger
      const stream = document.createElement('div');
      stream.className = 'lt-rain-stream';
      stream.style.animationDuration = dur.toFixed(2) + 's';
      stream.style.animationDelay = '-' + delay.toFixed(2) + 's';
      // Build glyphs — head bright, rest normal
      const glyphs = [];
      glyphs.push(`<span class="lt-rain-head">${_randChar()}</span>`);
      for (let i = 1; i < len; i++) glyphs.push(`<span>${_randChar()}</span>`);
      stream.innerHTML = glyphs.join('<br>');
      frag.appendChild(stream);
      _rainStreams.push(stream);
    }
    el.appendChild(frag);
  }
  function buildKatakana() {
    // Build 4 edge columns. ~2-3 streams per column → 10-12 total streams.
    buildRainColumn('lt-rain-l1', 3);
    buildRainColumn('lt-rain-l2', 2);
    buildRainColumn('lt-rain-r1', 3);
    buildRainColumn('lt-rain-r2', 2);
    // Mutation tick: replace one random glyph in each stream every 200ms.
    // Pure textContent swap on a <span> — no layout, no reflow.
    if (window._ltRainTick) return;  // guard against double-init
    window._ltRainTick = setInterval(() => {
      for (const stream of _rainStreams) {
        const spans = stream.children;  // HTMLCollection of <span>s
        if (!spans || spans.length === 0) continue;
        const idx = Math.floor(Math.random() * spans.length);
        if (spans[idx]) spans[idx].textContent = _randChar();
      }
    }, 200);
  }

  // ─── Pair heatmap (Theater Pass v2, May 18) ───────────────────────────
  // Populates organically from event stream — no pre-population.
  // Score proxy: count of recent appearances (60s sliding window from
  // state.radarSymbolEvents). Cells transition between states on each event.
  const _pairCells = new Map();  // symbol -> { el, state, rejectUntil }
  function _scoreForSymbol(sym) {
    // Reuse radar's appearance buffer. Score = recent appearances / max.
    const now = Date.now();
    const cutoff = now - 60000;
    let myCount = 0;
    const counts = new Map();
    for (const e of state.radarSymbolEvents) {
      if (e.ts < cutoff) continue;
      counts.set(e.symbol, (counts.get(e.symbol) || 0) + 1);
      if (e.symbol === sym) myCount++;
    }
    let maxN = 0;
    for (const v of counts.values()) if (v > maxN) maxN = v;
    if (maxN === 0) return 0;
    return Math.round((myCount / maxN) * 100);
  }
  // Short symbol for cell label: strip USDT suffix + leading digits, truncate >4.
  function _shortenSym(sym) {
    let s = sym.replace(/USDT$/, '').replace(/^\d+/, '');
    if (s.length > 4) s = s.slice(0, 4);
    return s || sym.slice(0, 4);
  }
  function _ensurePairCell(sym) {
    if (_pairCells.has(sym)) return _pairCells.get(sym);
    const grid = document.getElementById('lt-pair-heatmap-grid');
    if (!grid) return null;
    const el = document.createElement('div');
    el.className = 'lt-pair-cell';
    el.dataset.symbol = sym;
    // Label always present — visible via CSS opacity on hi/top tiers only.
    const label = document.createElement('span');
    label.className = 'lt-pc-label';
    label.textContent = _shortenSym(sym);
    el.appendChild(label);
    // Tooltip handlers
    el.addEventListener('mouseenter', _showPairTooltip);
    el.addEventListener('mouseleave', _hidePairTooltip);
    grid.appendChild(el);
    // aware       = mentioned in any event (filter-out OR real)
    // blacklisted = sticky terminal state (from [SCAN] Blacklist active line)
    const rec = { el, state: 'idle', rejectUntil: 0, aware: false, blacklisted: false };
    _pairCells.set(sym, rec);
    return rec;
  }
  function _classForScore(sc) {
    if (sc > 85) return 'lt-pc-top';
    if (sc >= 70) return 'lt-pc-hi';
    if (sc >= 50) return 'lt-pc-mid';
    if (sc > 0)   return 'lt-pc-low';
    return '';
  }
  function _applyCellClass(rec, extra) {
    if (!rec || !rec.el) return;
    const sym = rec.el.dataset.symbol;
    const score = _scoreForSymbol(sym);
    rec.el.dataset.score = String(score);
    let base = 'lt-pair-cell';
    const now = Date.now();
    if (rec.blacklisted) {
      // Sticky terminal — never upgrades regardless of score.
      base += ' lt-pc-blacklist';
    } else if (rec.rejectUntil && now < rec.rejectUntil) {
      base += ' lt-pc-reject';
    } else {
      const sc = _classForScore(score);
      if (sc) {
        base += ' ' + sc;
      } else if (rec.aware) {
        // Score=0 but symbol is in awareness universe → idle baseline.
        base += ' lt-pc-idle';
      }
    }
    if (extra) base += ' ' + extra;
    rec.el.className = base;
  }
  // Shimmer pulse — fires on every event touch (May 18 polish).
  // Reset cleanly via remove + reflow + setTimeout cleanup (NOT animationend).
  function _shimmer(rec) {
    if (!rec || !rec.el) return;
    rec.el.classList.remove('lt-shimmer');
    void rec.el.offsetWidth;          // force reflow so animation restarts
    rec.el.classList.add('lt-shimmer');
    setTimeout(() => {
      if (rec.el) rec.el.classList.remove('lt-shimmer');
    }, 620);
  }
  function updatePairHeatmap(sym, evtType, opts) {
    if (!sym) return;
    const rec = _ensurePairCell(sym);
    if (!rec) return;
    // Shimmer fires for EVERY touch — even on no-op state changes.
    // Optional pulseDelay (ms) lets callers stagger waves of simultaneous events.
    const delay = (opts && typeof opts.pulseDelay === 'number') ? opts.pulseDelay : 0;
    if (delay > 0) setTimeout(() => _shimmer(rec), delay);
    else _shimmer(rec);

    if (evtType === 'BLACKLIST') {
      // Blacklist is sticky — once set, never unset. Still gets shimmer above.
      if (!rec.blacklisted) {
        rec.blacklisted = true;
        rec.aware = true;
        _applyCellClass(rec);
      }
      return;
    }
    if (rec.blacklisted) {
      // Blacklisted cells don't react to other event types (still shimmer though).
      return;
    }
    if (evtType === 'AWARENESS') {
      if (!rec.aware) {
        rec.aware = true;
        _applyCellClass(rec);
      }
      return;
    }
    // Real event paths — mark aware so score=0 doesn't drop back to bare cell.
    rec.aware = true;
    if (evtType === 'REJECT') {
      rec.rejectUntil = Date.now() + 3000;
      _applyCellClass(rec);
      setTimeout(() => _applyCellClass(rec), 3100);
    } else if (evtType === 'WATCH') {
      _applyCellClass(rec, 'lt-pc-watch');
    } else if (evtType === 'ENTRY') {
      _applyCellClass(rec, 'lt-pc-flash');
      setTimeout(() => _applyCellClass(rec), 1300);
    } else {
      // SCAN / generic — just refresh based on score
      _applyCellClass(rec);
    }
  }
  function _showPairTooltip(e) {
    const tip = document.getElementById('lt-pair-heatmap-tooltip');
    if (!tip) return;
    const sym = e.target.dataset.symbol || '?';
    const sc = e.target.dataset.score || '0';
    tip.textContent = `${sym} · score ${sc}`;
    const rect = e.target.getBoundingClientRect();
    const panel = e.target.closest('.lt-pair-heatmap-panel').getBoundingClientRect();
    tip.style.left = (rect.left - panel.left) + 'px';
    tip.style.top  = (rect.top - panel.top - 18) + 'px';
    tip.classList.add('lt-visible');
  }
  function _hidePairTooltip() {
    const tip = document.getElementById('lt-pair-heatmap-tooltip');
    if (tip) tip.classList.remove('lt-visible');
  }

  // ─── Ambient flickers (Theater Pass v2, May 18) ───────────────────────
  // PURE THEATER. NOT real metrics. Random-walk values for visual life.
  const _flickerState = { net: 99.9, cpu: 15, mem: 42, rtt: 28, pkt: 0, err: 0.01, que: 2, fps: 60 };
  function _walk(v, range, min, max) {
    v += (Math.random() - 0.5) * range;
    if (v < min) v = min;
    if (v > max) v = max;
    return v;
  }
  function updateFlickers() {
    _flickerState.net = _walk(_flickerState.net, 0.15, 99.5, 100.0);
    _flickerState.cpu = _walk(_flickerState.cpu, 1.2, 12, 18);
    _flickerState.mem = _walk(_flickerState.mem, 0.3, 41, 44);
    _flickerState.rtt = _walk(_flickerState.rtt, 4, 22, 40);
    _flickerState.pkt += 5 + Math.floor(Math.random() * 20);
    _flickerState.err = Math.random() < 0.05 ? Math.random() * 0.05 : _walk(_flickerState.err, 0.005, 0.0, 0.02);
    _flickerState.que = Math.max(0, Math.min(6, Math.round(_walk(_flickerState.que, 2.5, 0, 6))));
    _flickerState.fps = Math.random() < 0.05 ? 58 + Math.floor(Math.random() * 2) : 60;
    const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
    set('lt-fl-net', _flickerState.net.toFixed(1));
    set('lt-fl-cpu', String(Math.round(_flickerState.cpu)));
    set('lt-fl-mem', String(Math.round(_flickerState.mem)));
    set('lt-fl-rtt', String(Math.round(_flickerState.rtt)));
    set('lt-fl-pkt', String(_flickerState.pkt));
    set('lt-fl-err', _flickerState.err.toFixed(2));
    set('lt-fl-que', String(_flickerState.que));
    set('lt-fl-fps', String(_flickerState.fps));
  }

  // ─── Regime glitch (Theater Pass v2, May 18) ──────────────────────────
  // Triggered alongside the regime banner. 5s throttle.
  let _lastGlitchAt = 0;
  function triggerRegimeGlitch() {
    const now = Date.now();
    if (now - _lastGlitchAt < 5000) return;
    _lastGlitchAt = now;
    if (!elView) return;
    elView.classList.remove('lt-glitching');
    void elView.offsetWidth;  // force reflow so animation replays
    elView.classList.add('lt-glitching');
    const sc = document.getElementById('lt-glitch-scanline');
    if (sc) {
      sc.classList.remove('lt-glitch-active');
      void sc.offsetWidth;
      sc.classList.add('lt-glitch-active');
    }
    setTimeout(() => {
      if (elView) elView.classList.remove('lt-glitching');
      if (sc) sc.classList.remove('lt-glitch-active');
    }, 150);
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
    elFeedPulse = document.getElementById('lt-feed-pulse');
    elRegimeBanner = document.getElementById('lt-regime-banner');
    elFunnelBody = document.getElementById('lt-funnel-body');
    elEquityCurve = null;  // resolved per render inside footer
    // HEAT COMPASS (May 16)
    elHcPanel = document.querySelector('#live-terminal-view .lt-heat-compass');
    elHeatCompassSvg = document.getElementById('lt-heat-compass-svg');
    elHcFootLean = document.getElementById('lt-hc-foot-lean');
    elHcFootRegime = document.getElementById('lt-hc-foot-regime');
    elHcFootPressure = document.getElementById('lt-hc-foot-pressure');
    buildHeatCompassSvg();

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
    // May 18: ambient flickers — pure-theater random-walk metrics, every 1200ms
    if (!window._ltFlickerTick) {
      updateFlickers();  // initial paint
      window._ltFlickerTick = setInterval(updateFlickers, 1200);
    }
    updateConstellation();
    updateLatencyBar();
    updatePositionsPanel();
    updateFooterStats();
    renderFunnel(false);

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
