#!/usr/bin/env python3
"""📊 Oct-8 research (vol24h / mcap) — the bot's OWN market-cap source for every study pair: the Binance Info-panel endpoint
services/mcap_service.py uses (data.mc market cap USD · data.cs circulating supply · data.rk CMC rank, CoinMarketCap data),
queried exactly like the engine (lookup_candidates: full base first, then the multiplier-stripped base; parse_detail with
expect=symbol → a different coin is never accepted). READ-ONLY research: never touches the bot, its API or its DB.

Budget: ONE request per candidate symbol (stop at the first that answers), ≥ 1.5 s between requests, every response cached to
reports/cache_vol_mcap/bapi/<symbol>.json and reused; stops on HTTP 418 / 429 (re-run resumes from cache).
Contract multiplier: mult = the power of ten nearest to (Binance futures price) / (mc / cs), so historical mcap at entry =
cs × entry price / mult (ASSUMPTION: circulating supply ≈ constant between the entry and today — checked in the report).
Writes reports/cache_vol_mcap/bapi_supply.json. Usage: venv/bin/python scripts/study_vol_mcap_bapi.py <pairs.txt>"""
import json, math, os, sys, time
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from services.mcap_service import URL, lookup_candidates, parse_detail  # noqa: E402

CD = os.path.join(ROOT, "reports", "cache_vol_mcap"); BD = os.path.join(CD, "bapi"); os.makedirs(BD, exist_ok=True)
_last = [0.0]


def fetch(sym):
    f = os.path.join(BD, f"{sym}.json")
    if os.path.exists(f):
        return json.load(open(f))
    w = 1.5 - (time.time() - _last[0])
    if w > 0:
        time.sleep(w)
    _last[0] = time.time()
    r = requests.get(URL, params={"symbol": sym}, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code in (418, 429):
        raise SystemExit(f"bapi {r.status_code} on {sym} — stopping (re-run resumes from cache)")
    j = r.json() if r.status_code == 200 else {"_status": r.status_code}
    json.dump(j, open(f, "w")); return j


def main(pairs_file):
    pairs = [p.strip() for p in open(pairs_file) if p.strip()]
    bp = json.load(open(os.path.join(CD, "binance_ticker_price.json")))
    out = {}
    for n, p in enumerate(pairs):
        res = None
        for sym in lookup_candidates(p):
            j = fetch(sym); mc, rk = parse_detail(j, expect=sym)
            if mc is not None:
                d = j.get("data") or {}
                try:
                    cs = float(d.get("cs"))
                except (TypeError, ValueError):
                    cs = None
                res = dict(symbol=sym, mc=mc, rk=rk, cs=cs if cs and cs > 0 else None)
                break
        if res and res["cs"]:
            unit = res["mc"] / res["cs"]                       # CMC coin price implied by the endpoint
            px = bp.get(p)
            if px:
                res["mult"] = 10 ** round(math.log10(px / unit)); res["px_ratio"] = px / (unit * res["mult"])
            else:
                res["mult"] = None; res["why"] = "no current Binance futures price (delisted) — multiplier from k5m at analysis time"
        out[p] = res or dict(why="endpoint has no mc for any candidate symbol")
        if n % 50 == 0:
            print(f"  {n}/{len(pairs)} {p} → {out[p]}", flush=True)
    json.dump(out, open(os.path.join(CD, "bapi_supply.json"), "w"), indent=1)
    ok = sum(1 for v in out.values() if v.get("cs"))
    print(f"supply readable for {ok}/{len(pairs)} pairs")
    for k, v in sorted(out.items()):
        if not v.get("cs"):
            print("  UNMAPPED", k, v.get("why"))


if __name__ == "__main__":
    main(sys.argv[1])
