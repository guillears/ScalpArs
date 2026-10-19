import pandas as pd, numpy as np, re
p="/Users/guillearslanian/Downloads/NOFA AI/reports/FADE_SIZE_MCAP_2026-10-19.md"; s=open(p).read()
r=pd.read_csv('rules_all.csv')
LAB={'a today 2x':'**(a) today 2×, liq 0.5 %**','b 1x all':'(b) 1× all','b2 0.5x all':'(b2) 0.5× all','c1 2x if mcap<T1 else 1x':"(c1) 2× if mcap < $40M else 1× (operator's idea)",
'c2 2x if mcap>T2 else 1x':'(c2) 2× if mcap > $102M else 1×','d liq 0.05%':'(d) 2×, liq 0.05 %','d liq 0.1%':'(d) 2×, liq 0.1 %','d liq 0.2%':'(d) 2×, liq 0.2 %',
'e acct 3xE':'(e) 2×, ≤ 3× account','e acct 4xE':'(e) 2×, ≤ 4× account','e acct 5xE':'(e) 2×, ≤ 5× account','e acct 6xE':'(e) 2×, ≤ 6× account',
'f 2x, <=5xE, <=0.1%V':'(f) 2×, ≤ 5× acct, ≤ 0.1 % vol','f 1x, <=0.1%V':'(f) 1×, ≤ 0.1 % vol','f 1x, <=0.05%V':'**(f) 1×, ≤ 0.05 % vol — first recommendation (§6)**',
'f 0.5x, <=0.1%V':'(f) 0.5×, ≤ 0.1 % vol','f 2x, <=3xE, <=0.1%V':'(f) 2×, ≤ 3× acct, ≤ 0.1 % vol','f 2x if mcap>T2 else 1x, <=5xE':'(f) 2× if mcap > $102M else 1×, ≤ 5× acct'}
def d(x): return f"{x:+,.0f}".replace('+-','−').replace('-','−')
def pc(x): return f"{x:+.2f}".replace('-','−')
def pr(x): return f"{x*100:.0f} %"
rows=[]
for k,lab in LAB.items():
    M=r[(r.rule==k)&(r.src=='MASTER')].iloc[0]; B=r[(r.rule==k)&(r.src=='BT')].iloc[0]; L=r[(r.rule==k)&(r.src=='LIVE')].iloc[0]
    xe=f"{M.med_notional_xE:.2f}" if abs(M.med_notional_xE-B.med_notional_xE)<0.05 else f"{M.med_notional_xE:.2f} / {B.med_notional_xE:.2f}"
    rows.append(f"| {lab} | {int(M.shrunk)} / {int(B.shrunk)} | {xe} | {pc(M.avg_pct)} | {d(M.usd)} | {M.worst_acct:.1f} | {pr(M.P30)} | {pr(M.P50)} | {d(M.usd_half)} | {d(M.usd_noimp)} | {M.usd_paper_noimp:,.0f} | {pc(B.avg_pct)} | {d(B.usd)} | {B.worst_acct:.1f} | {pr(B.P30)} | {pr(B.P50)} | {d(B.usd_half)} | {d(B.usd_noimp)} | {pr(B.P50_noimp)} | {d(L.usd)} |".replace('-','−').replace('|−−|','|---|'))
hdr="| rule | shrunk | x acct | MASTER avg | MASTER $ | worst | P30 | P50 | ½imp $ | paper $ | paper-era $ | BT avg | BT $ | worst | P30 | P50 | ½imp $ | paper $ | P50 paper | KMNO $ |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
tab=hdr+"\n".join(rows)+"\n"
a=s.index("| rule | shrunk | x acct |"); b=s.index("\nThe mcap cut points")
s=s[:a]+tab+s[b:]
# per batch
t=pd.read_csv('perbatch.csv',index_col=[0,1,2]).loc['imp']
cols=['a today 2x','b 1x all','b2 0.5x all','c1 2x if mcap<T1 else 1x','c2 2x if mcap>T2 else 1x','d liq 0.05%','e acct 5xE','f 1x, <=0.05%V','f 1x, <=0.1%V']
m=t.loc['MASTER']; G=[('B1',['B1']),('B2',['B2']),('B3',['B3']),('B5–B9',['B5','B6','B7','B8','B9']),('B10',['B10']),('B12–B13',['B12','B13']),('B15–B17',['B15','B16','B17']),('B18',['B18']),('B19–B21',['B19','B20','B21'])]
out=[]
for lab,l in G:
    x=m.loc[l].sum(); out.append(f"| {lab} | {int(x.N)} | "+" | ".join(d(x[c]) for c in cols)+" |")
x=m.drop('TOTAL',errors='ignore').sum() if 'TOTAL' in m.index else m.sum()
mt=t.loc['MASTER'].loc['TOTAL'] if 'TOTAL' in t.loc['MASTER'].index else None
mt=t.loc[('MASTER','TOTAL')]
out.append(f"| **MASTER TOTAL** | **{int(mt.N)}** | "+" | ".join(f"**{d(mt[c])}**" for c in cols)+" |")
for reg in ('2025 BULL','2026-H1 BEAR','2026-H2 BULL'):
    x=t.loc[('BT',reg)]; out.append(f"| BT {reg} | {int(x.N)} | "+" | ".join(d(x[c]) for c in cols)+" |")
x=t.loc[('BT','TOTAL')]; out.append(f"| **BT TOTAL** | **{int(x.N)}** | "+" | ".join(f"**{d(x[c])}**" for c in cols)+" |")
x=t.loc[('LIVE','TOTAL')]; out.append(f"| **LIVE (KMNO)** | **1** | "+" | ".join(d(x[c]) for c in cols)+" |")
a=s.index("| B1 | 6 |"); b=s.index("\nThe recommended rule beats today in")
s=s[:a]+"\n".join(out)+"\n"+s[b:]
open(p,'w').write(s)
print("\n".join(out)); print(tab)
