"""WILLY late take-profit grid (operator 2026-10-13): after X min (15/20/25/30) the TP drops from +1.25 to +Z (0.2/0.3/0.4); stop −4, cap 90.
Same fills / paths as scripts/study_willy_late_sl.py (reads reports/WILLY_LATE_SL_rows.csv). Output: stdout → reports/WILLY_LATE_TP_GRID.txt"""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT+"/scripts")
import study_willy_late_sl as S, scout_willy_timecap as WT, scout_frenzy_willy_exit as FX
V=[(x,z) for x in (15,20,25,30) for z in (0.2,0.3,0.4)]
def job(rows):
    WT._NET_BLOCKED=True; tc={}; out=[]
    for r in rows:
        P=FX.load_path(r["pair"],int(r["te"]),float(r["E"]),90,tc); b=S.ev(P); rec=dict(r,base=b["pct"])
        for x,z in V:
            v=S.ev(P,x,None,z); rec[f"TP{x}_{z:g}"]=v["pct"]; rec[f"TP{x}_{z:g}_why"]=v["why"]
        out.append(rec)
    return out
if __name__=="__main__":
    R0=pd.read_csv(ROOT+"/reports/WILLY_LATE_SL_rows.csv",low_memory=False); R0=R0[R0.base_why!="UNRES"]
    M=pd.read_csv(ROOT+"/reports/MASTER_POOL_stacked.csv",low_memory=False); M=M[(M.entry_strategy=="FRENZY_WILLY")&(M.stack_keep==True)]
    u={(str(a)[:19].replace("T"," "),p):s/c for a,p,s,c in zip(M.opened_at,M.pair,M.stack_pnl,M.stack_pct) if c}
    lv=R0.cohort!="BACKTEST"; R0.loc[lv,"usd_per_pt"]=[u.get((str(t)[:19],p),np.nan) for t,p in zip(R0.ts[lv],R0.pair[lv])]
    it=R0[["cohort","pair","te","E","ts","day","usd_per_pt"]].to_dict("records"); ps=sorted({r["pair"] for r in it}); g={p:k%6 for k,p in enumerate(ps)}
    with Pool(6) as P: R=pd.DataFrame(sum(P.map(job,[[r for r in it if g[r["pair"]]==k] for k in range(6)]),[]))
    rng=np.random.default_rng(7); sp=pd.Timestamp("2026-07-01")
    for coh in ("BACKTEST","B19","B20"):
        D=R[R.cohort==coh]; print(f"\n{coh}: {len(D)} fills · base WR {(D.base>0).mean()*100:.0f}% avg {D.base.mean():+.3f} ${(D.base*D.usd_per_pt).sum():+,.0f}")
        for x,z in V:
            k=f"TP{x}_{z:g}"; v=D[k]; d=v-D.base; sv=(d>1e-9).sum(); kl=(d<-1e-9).sum()
            gg=pd.DataFrame(dict(d=d,day=D.day)).groupby("day").d.agg(["sum","size"]); s_,n_=gg["sum"].values,gg["size"].values
            idx=rng.integers(0,len(gg),size=(4000,len(gg))); p=((s_[idx].sum(1)/n_[idx].sum(1))>0).mean() if len(gg)>2 else np.nan
            t=pd.to_datetime(D.ts,format="mixed"); h1=d[t<sp].mean(); h2=d[t>=sp].mean()
            print(f"  {k:9} WR {(v>0).mean()*100:3.0f}% avg {v.mean():+.3f} Δ {d.mean():+.3f} saved {sv:3} ({d[d>0].sum():+6.1f}) cut {kl:3} ({d[d<0].sum():+6.1f}) P {p:.2f} H1Δ {h1:+.3f} H2Δ {h2:+.3f} Δ$ {(d*D.usd_per_pt).sum():+,.0f}")
