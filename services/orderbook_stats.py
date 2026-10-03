"""📖 Oct-3 (DECISION_LOG 192): order-book readings for research — what the book looked like when the operator clicked, and the same
readings once a minute on the FRENZY-flagged pairs (the control set: Binance keeps no historical order books). OBSERVE-ONLY: nothing reads
them for a decision. Pure functions: bids / asks are [[price, qty], …] best first (ccxt / Binance depth layout)."""
import math
from typing import Dict, List, Optional

BANDS = (0.25, 0.5, 1.0, 2.0)      # % from the mid price
WALL_BAND = 2.0                    # the largest single level within this % of the mid, each side


def _usd(levels, lo, hi):
    return sum(float(p) * float(q) for p, q in levels if lo <= float(p) <= hi)


def orderbook_metrics(bids: List, asks: List) -> Optional[Dict]:
    """→ {mid, spread_pct, top_bid_usd, top_ask_usd, bid_usd_<b>, ask_usd_<b>, imb_<b> (bid − ask) ÷ (bid + ask) for b in BANDS,
    wall_bid_dist_pct / wall_bid_usd / wall_bid_share, wall_ask_dist_pct / wall_ask_usd / wall_ask_share, depth_levels}. Band keys use
    '025' / '05' / '1' / '2'. None when the book is empty / unreadable. Never raises."""
    try:
        b = [(float(e[0]), float(e[1])) for e in (bids or [])[:1000] if float(e[0]) > 0 and float(e[1]) > 0]   # [p, q] or [p, q, count]
        a = [(float(e[0]), float(e[1])) for e in (asks or [])[:1000] if float(e[0]) > 0 and float(e[1]) > 0]
        if not b or not a:
            return None
        bb, ba = b[0][0], a[0][0]
        if not (ba > bb > 0):
            return None
        mid = (bb + ba) / 2
        reach_b = (1 - b[-1][0] / mid) * 100; reach_a = (a[-1][0] / mid - 1) * 100   # how far the returned levels reach (deep review)
        out = {"mid": mid, "spread_pct": (ba - bb) / mid * 100, "top_bid_usd": bb * b[0][1], "top_ask_usd": ba * a[0][1],
               "depth_levels": min(len(b), len(a)), "reach_bid_pct": reach_b, "reach_ask_pct": reach_a}
        for band in BANDS:
            k = str(band).replace("0.", "0").replace(".0", "").replace(".", "")
            if reach_b < band or reach_a < band:   # the book was cut short of this band on a side → no reading (never a biased one)
                out[f"bid_usd_{k}"] = out[f"ask_usd_{k}"] = out[f"imb_{k}"] = None
                continue
            bu = _usd(b, mid * (1 - band / 100), mid); au = _usd(a, mid, mid * (1 + band / 100))
            out[f"bid_usd_{k}"], out[f"ask_usd_{k}"] = bu, au
            out[f"imb_{k}"] = (bu - au) / (bu + au) if (bu + au) > 0 else None
        for side, lv, sign in (("bid", b, -1), ("ask", a, 1)):
            inb = [(p, p * q) for p, q in lv if abs(p / mid - 1) * 100 <= WALL_BAND]
            tot = sum(u for _, u in inb)
            if inb:
                p, u = max(inb, key=lambda x: x[1])
                out[f"wall_{side}_dist_pct"] = abs(p / mid - 1) * 100
                out[f"wall_{side}_usd"] = u
                out[f"wall_{side}_share"] = u / tot if tot > 0 else None
            else:
                out[f"wall_{side}_dist_pct"] = out[f"wall_{side}_usd"] = out[f"wall_{side}_share"] = None
        for k, v in list(out.items()):
            if isinstance(v, float):
                out[k] = float(f"{v:.6g}") if math.isfinite(v) else None   # 6 significant figures: smaller exports, no lost meaning
        return out
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return None


# the columns stamped on a MANUAL order (prefix manual_ob_) and stored per snapshot row — one list, used by the model, the migration and tests
OB_FIELDS = ["spread_pct", "top_bid_usd", "top_ask_usd", "bid_usd_025", "ask_usd_025", "imb_025", "bid_usd_05", "ask_usd_05", "imb_05",
             "bid_usd_1", "ask_usd_1", "imb_1", "bid_usd_2", "ask_usd_2", "imb_2", "wall_bid_dist_pct", "wall_bid_usd", "wall_bid_share",
             "wall_ask_dist_pct", "wall_ask_usd", "wall_ask_share", "depth_levels", "reach_bid_pct", "reach_ask_pct"]


def snapshot_pairs(flagged, manual_open, cap=12):
    """Pairs to snapshot this minute: every pair with an open MANUAL position first, then the FRENZY-flagged ones; ≤ cap, no duplicates."""
    out = []
    for p in list(manual_open or []) + list(flagged or []):
        if p and p not in out:
            out.append(p)
    return out[:cap]
