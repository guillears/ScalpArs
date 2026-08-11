#!/usr/bin/env python3
"""
diagnose_algo_stop_testnet.py — TESTNET diagnosis of the Apr-17 -4120 stop rejection (Aug-11 2026).

FINDING THAT MOTIVATED THIS: -4120 = STOP_ORDER_SWITCH_ALGO. Binance MANDATORILY migrated all
conditional orders (STOP_MARKET / TAKE_PROFIT_MARKET / TRAILING_STOP_MARKET) from /fapi/v1/order
to the Algo Order API (POST /fapi/v1/algoOrder, algoType=CONDITIONAL), account waves through
2025-12-09. The four April hotfixes all retried params on the DEPRECATED endpoint. Never an
account defect. ccxt 4.5.34 has the raw endpoint (fapiPrivatePostAlgoOrder) but does not route
create_order stops to it — so the production backstop will call it directly, as this script does.

WHAT THIS SCRIPT DOES (Binance FUTURES TESTNET only — fake funds, zero real-money risk):
  1. connects with testnet keys, prints balance
  2. opens a MINIMAL market position on BTCUSDT
  3. attempts the OLD path (create_order STOP_MARKET on /fapi/v1/order) -> expects -4120
  4. attempts the NEW path (POST /fapi/v1/algoOrder, CONDITIONAL STOP_MARKET,
     closePosition=true, workingType=MARK_PRICE, priceProtect=TRUE) -> expects SUCCESS + algoId
  5. queries the algo order, CANCELS it, closes the position, prints a verdict table

HOW TO RUN (operator):
  1. Create a free testnet account at https://testnet.binancefuture.com (login button, instant)
  2. On that page: API Key tab -> generate key+secret
  3. TESTNET_API_KEY=xxx TESTNET_API_SECRET=yyy ./venv/bin/python scripts/diagnose_algo_stop_testnet.py

The script touches ONLY the testnet host. It refuses to run if the env vars are missing.
"""
import os, sys, time, json

API_KEY = os.environ.get("TESTNET_API_KEY")
API_SECRET = os.environ.get("TESTNET_API_SECRET")
if not API_KEY or not API_SECRET:
    sys.exit("ABORT: set TESTNET_API_KEY / TESTNET_API_SECRET env vars (create at https://testnet.binancefuture.com — fake funds). This script never touches the live account.")

import ccxt

ex = ccxt.binanceusdm({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True,
    "options": {"defaultType": "future"},
})
ex.set_sandbox_mode(True)  # routes to testnet.binancefuture.com

SYMBOL = "BTC/USDT:USDT"
RESULTS = []

def step(name, fn):
    print(f"\n=== {name} ===")
    try:
        out = fn()
        print("OK:", json.dumps(out, default=str)[:400] if out is not None else "(none)")
        RESULTS.append((name, "OK", out))
        return out
    except Exception as e:
        print("ERR:", type(e).__name__, str(e)[:400])
        RESULTS.append((name, "ERR", str(e)[:200]))
        return None

# 1. connectivity + balance
bal = step("testnet balance", lambda: ex.fetch_balance()["USDT"]["free"])
if bal is None:
    sys.exit("cannot reach testnet — check keys")

ex.load_markets()
px = ex.fetch_ticker(SYMBOL)["last"]
qty = max(ex.amount_to_precision(SYMBOL, 120.0 / px), ex.markets[SYMBOL]["limits"]["amount"]["min"])
print(f"\nBTC last={px}  test qty={qty} (~$120 notional, testnet funds)")

# 2. open minimal LONG
step("open minimal LONG (market)", lambda: ex.create_market_order(SYMBOL, "buy", qty))
time.sleep(1.5)

trigger = ex.price_to_precision(SYMBOL, px * 0.90)  # stop 10% below — never triggers during the test

# 3. OLD path — expected to fail -4120 (proves testnet reproduces the migration)
step("OLD path: STOP_MARKET via /fapi/v1/order (expect -4120)",
     lambda: ex.create_order(SYMBOL, "STOP_MARKET", "sell", qty, None,
                             {"stopPrice": trigger, "reduceOnly": True, "workingType": "MARK_PRICE"}))

# 4. NEW path — the Algo Order API
def new_path():
    return ex.fapiPrivatePostAlgoOrder({
        "algoType": "CONDITIONAL",
        "symbol": "BTCUSDT",
        "side": "SELL",
        "type": "STOP_MARKET",
        "triggerPrice": trigger,
        "closePosition": "true",
        "workingType": "MARK_PRICE",
        "priceProtect": "TRUE",
    })
algo = step("NEW path: CONDITIONAL STOP_MARKET via /fapi/v1/algoOrder (expect SUCCESS)", new_path)

# 5. query + cancel + cleanup
if algo and algo.get("algoId"):
    aid = str(algo["algoId"])
    step("query algo order", lambda: ex.fapiPrivateGetAlgoOrder({"algoId": aid}))
    step("cancel algo order", lambda: ex.fapiPrivateDeleteAlgoOrder({"algoId": aid}))
step("close test position (market)", lambda: ex.create_market_order(SYMBOL, "sell", qty, params={"reduceOnly": True}))

print("\n" + "=" * 60 + "\nVERDICT\n" + "=" * 60)
for name, st, out in RESULTS:
    print(f"  [{st:3s}] {name}")
old_err = next((o for n, s, o in RESULTS if n.startswith("OLD") and s == "ERR"), "")
new_ok  = any(n.startswith("NEW") and s == "OK" for n, s, o in RESULTS)
if new_ok:
    print("\n>>> ALGO ENDPOINT WORKS. The -4120 mystery is solved: production backstop = fapiPrivatePostAlgoOrder")
    print(">>> (CONDITIONAL STOP_MARKET, closePosition=true, MARK_PRICE, priceProtect). Ready to build the real thing.")
elif "-4120" in str(old_err):
    print("\n>>> testnet reproduces -4120 on the old path but the new path failed too — paste the NEW-path error to Claude.")
else:
    print("\n>>> unexpected pattern — paste the full output to Claude.")
