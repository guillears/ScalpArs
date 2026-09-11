"""Fee-provisioning cycle sim: OLD burn leg (12h x burn) vs NEW (12h x burn - BNB held).
Mirrors the engine rules: burn_24h/12h trailing windows, mature >=2h, target = max(24h x burn24, $50),
threshold = max(12h x burn12, $25), swap when BNB < threshold and (6h since last swap or BNB < 25% thr),
buy = min(target - BNB, free_usdt - $100) if >= $10; fee events trigger the emergency check too.
Book: 4 slots, equal-split on tradeable, 2x cell on half the fills, 20x lev, fees 0.063% RT of notional,
hold U(15,180) min. Trades arrive Poisson at `tpd` per day, only while a slot AND tradeable>=100."""
import random, statistics, sys
def run(rule, tpd, seed, days=30, equity0=3000.0, bnb0=100.0):
    rnd=random.Random(seed)
    dt=5/60.0; steps=int(days*24/dt)
    usdt=equity0-bnb0; bnb=bnb0; margin=0.0
    fee_events=[]   # (t, fee$)
    open_pos=[]     # (close_t, investment, fee_out)
    last_swap=-1e9; swaps=[]; starve=0; blocked=0; trades=0; first_fee=None
    idle_frac=[]; tradeable_series=[]; min_bnb=bnb
    lam=tpd/(24/dt)
    for i in range(steps):
        t=i*dt
        # closes
        for p in [p for p in open_pos if p[0]<=t]:
            open_pos.remove(p); margin-=p[1]; usdt+=p[1]   # margin returns to USDT (P&L-neutral sim)
            fee=p[2]
            if bnb>=fee: bnb-=fee
            else: starve+=1; usdt-=fee*1.1; bnb=0.0   # paid in USDT at undiscounted rate
            fee_events.append((t,fee))
        fee_events=[e for e in fee_events if e[0]>t-24.5]
        # burn estimates (trailing)
        f24=sum(f for tt,f in fee_events if tt>t-24); f12=sum(f for tt,f in fee_events if tt>t-12)
        if first_fee is None and fee_events: first_fee=fee_events[0][0]
        first=first_fee if first_fee is not None else t
        span=min(24.0, max(t-first, 1e-9)); mature=(first_fee is not None) and (t-first)>=2
        burn24=f24/max(span,1e-9) if mature else 0.0; burn12=f12/max(min(12.0,span),1e-9) if mature else 0.0
        target=max(24*burn24, 50.0); thr=max(12*burn12, 25.0)
        equity=usdt+margin+bnb
        free=usdt
        # reserve
        leg=15.0; leg=max(leg, 0.025*equity)
        if mature and burn12>0:
            leg=max(leg, 12*burn12 if rule=='old' else max(0.0, 12*burn12-bnb))
        reserve=min(leg, free); tradeable=max(0.0, free-reserve)
        # swap check (every 15 min)
        if i%3==0 and mature and bnb<thr and ((t-last_swap)>=6 or bnb<0.25*thr):
            short=target-bnb; cap=free-100.0; buy=min(short,cap)
            if buy>=10: usdt-=buy; bnb+=buy; last_swap=t; swaps.append(buy)
        # entries
        if rnd.random()<lam:
            if len(open_pos)<4 and tradeable>=100:
                base=tradeable/max(1,(4-len(open_pos))); inv=min(tradeable, base*(2.0 if rnd.random()<0.5 else 1.0))
                notional=inv*20; fee_in=notional*0.00018 if rnd.random()<0.6 else notional*0.00045
                if bnb>=fee_in: bnb-=fee_in
                else: starve+=1; usdt-=fee_in*1.1; bnb=0.0
                fee_events.append((t,fee_in))
                margin+=inv; usdt-=inv
                open_pos.append((t+rnd.uniform(15,180)/60.0, inv, notional*0.00045))
                trades+=1
                # emergency check on fee event
                if mature and bnb<thr:
                    short=target-bnb; cap=usdt-100.0; buy=min(short,cap)
                    if buy>=10: usdt-=buy; bnb+=buy; last_swap=t; swaps.append(buy)
            else: blocked+=1
        min_bnb=min(min_bnb,bnb)
        idle_frac.append((bnb+reserve)/max(equity,1)); tradeable_series.append(tradeable)
    return dict(trades=trades, blocked=blocked, starve=starve, swaps=len(swaps), swap_avg=(statistics.mean(swaps) if swaps else 0),
                min_bnb=min_bnb, idle=statistics.mean(idle_frac)*100, tradeable_avg=statistics.mean(tradeable_series))
print(f"{'tpd':>4} {'rule':4} | {'trades':>6} {'blocked':>7} {'STARVE':>6} {'swaps':>5} {'avg$':>6} {'minBNB':>7} {'idle%':>6} {'tradeable$':>10}")
for tpd in (8,20,40):
    for rule in ('old','new'):
        agg=[run(rule,tpd,seed) for seed in range(12)]
        m=lambda k: statistics.mean(a[k] for a in agg)
        print(f"{tpd:>4} {rule:4} | {m('trades'):>6.0f} {m('blocked'):>7.0f} {sum(a['starve'] for a in agg):>6} {m('swaps'):>5.1f} {m('swap_avg'):>6.0f} {min(a['min_bnb'] for a in agg):>7.1f} {m('idle'):>6.1f} {m('tradeable_avg'):>10.0f}")
