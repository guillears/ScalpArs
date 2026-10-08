#!/usr/bin/env python3
"""📊 Oct-8 research (FRENZY vol24h / market-cap study) — map Binance futures pairs → CoinGecko ids and cache each id's
365-day daily market_caps / total_volumes / prices. READ-ONLY research tool: never touches the bot, its API or its DB.

Budget: CoinGecko free tier ≤ 1 request / 3 s (sleep 3.2 s between calls), back off 60 s → 120 s → 240 s on 429, every
response cached to reports/cache_vol_mcap/ and reused (a re-run makes no request for a cached id). Binance: ONE call to
fapi/v1/ticker/price (weight 2) for the current prices used to disambiguate tickers; X-MBX-USED-WEIGHT-1M is read and the
script stops on 418/429.

Mapping rules (per pair): candidates = every /coins/list id whose symbol equals the FULL base (1000cat) first, else the
multiplier-stripped base (pepe for 1000PEPE, mult 1000). Among candidates with a /coins/markets row, keep those whose
current CG price × mult is within ×/÷ 1.5 of Binance's current futures price; pick the highest market cap. No candidate
passing the price check → UNMAPPED (listed in the report; never a guess).
Usage: venv/bin/python scripts/study_vol_mcap_fetch.py <pairs.txt>"""
import json, os, re, sys, time
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CD = os.path.join(ROOT, "reports", "cache_vol_mcap")
os.makedirs(os.path.join(CD, "chart"), exist_ok=True)
CG = "https://api.coingecko.com/api/v3"
MULT = re.compile(r"^(1000000|100000|10000|1000|100)(?=[A-Z])")
_last = [0.0]


def cg_get(path, params=None, cache=None):
    if cache and os.path.exists(cache):
        return json.load(open(cache))
    back = 60
    for _ in range(6):
        w = 3.2 - (time.time() - _last[0])
        if w > 0:
            time.sleep(w)
        _last[0] = time.time()
        r = requests.get(CG + path, params=params, timeout=40, headers={"accept": "application/json"})
        if r.status_code == 429:
            print(f"  429 on {path} → sleep {back}s", flush=True); time.sleep(back); back = min(back * 2, 600); continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
        j = r.json()
        if cache:
            json.dump(j, open(cache, "w"))
        return j
    raise SystemExit(f"CoinGecko kept rate-limiting on {path}; stopping (re-run resumes from cache)")


def binance_prices():
    f = os.path.join(CD, "binance_ticker_price.json")
    if os.path.exists(f):
        return json.load(open(f))
    r = requests.get("https://fapi.binance.com/fapi/v1/ticker/price", timeout=30)
    print("binance used weight 1m:", r.headers.get("X-MBX-USED-WEIGHT-1M"), r.status_code)
    if r.status_code in (418, 429):
        raise SystemExit("Binance 418/429 — stop")
    r.raise_for_status()
    j = {x["symbol"]: float(x["price"]) for x in r.json()}
    json.dump(j, open(f, "w")); return j


def main(pairs_file):
    pairs = [p.strip() for p in open(pairs_file) if p.strip()]
    lst = cg_get("/coins/list", cache=os.path.join(CD, "coins_list.json"))
    bysym = {}
    for c in lst:
        bysym.setdefault(c["symbol"].lower(), []).append(c["id"])
    bp = binance_prices()
    cand = {}
    for p in pairs:
        full = p[:-4]; m = MULT.match(full); opts = [(full.lower(), 1)]
        if m:
            opts.append((full[m.end():].lower(), int(m.group(1))))
        cand[p] = [(i, mult) for s, mult in opts for i in bysym.get(s, [])]
    ids = sorted({i for v in cand.values() for i, _ in v})
    mk = {}
    for k in range(0, len(ids), 200):
        part = ids[k:k + 200]
        j = cg_get("/coins/markets", dict(vs_currency="usd", ids=",".join(part), per_page=250, page=1),
                   cache=os.path.join(CD, f"markets_{k:05d}_{len(part)}.json"))
        for x in j or []:
            mk[x["id"]] = x
    mapping = {}
    for p in pairs:
        px = bp.get(p); best = None; why = "no CG symbol match"
        rows = [(i, mult, mk.get(i)) for i, mult in cand[p]]
        rows = [r for r in rows if r[2] and r[2].get("current_price")]
        if cand[p] and not rows:
            why = "CG candidates have no market row"
        ok = []
        for i, mult, x in rows:
            if px is None:
                why = "no Binance price (delisted?)"; continue
            ratio = x["current_price"] * mult / px
            if 1 / 1.5 <= ratio <= 1.5:
                ok.append((x.get("market_cap") or 0, i, mult, ratio))
            else:
                why = f"price mismatch (best ratio {ratio:.3g})"
        if ok:
            ok.sort(reverse=True); best = ok[0]
            mapping[p] = dict(id=best[1], mult=best[2], price_ratio=round(best[3], 4), n_candidates=len(cand[p]), ok_candidates=len(ok))
        else:
            mapping[p] = dict(id=None, why=why, n_candidates=len(cand[p]))
    json.dump(mapping, open(os.path.join(CD, "mapping.json"), "w"), indent=1)
    todo = sorted({v["id"] for v in mapping.values() if v.get("id")})
    print(f"mapped {sum(1 for v in mapping.values() if v.get('id'))}/{len(pairs)} pairs → {len(todo)} ids", flush=True)
    for n, i in enumerate(todo):
        f = os.path.join(CD, "chart", f"{i}.json")
        if os.path.exists(f):
            continue
        cg_get(f"/coins/{i}/market_chart", dict(vs_currency="usd", days=365, interval="daily"), cache=f)
        if n % 25 == 0:
            print(f"  chart {n}/{len(todo)} {i}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
