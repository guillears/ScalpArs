# STATE_EXIT_AUDIT_2026-11-21 — audit helper (nullaudit). Research only, read-only on study data. Run from the repo root: venv/bin/python scripts/STATE_EXIT_AUDIT_2026-11-21_nullaudit.py
import importlib.util, numpy as np, pandas as pd, os, sys
H='scripts/'
sp=importlib.util.spec_from_file_location("EV",H+"STATE_EXIT_2026-11-21_eval.py"); EV=importlib.util.module_from_spec(sp); sp.loader.exec_module(EV)
C=EV.C
for s in sys.argv[1:]:
    O=pd.read_csv(C.D+'/fills_out.csv'); O=O[(O.sleeve==s)&O.base_net.notna()].copy()
    dec=pd.read_pickle(C.D+f'/dec_{s}.pkl').sort_values(['fid','k']).reset_index(drop=True).replace([np.inf,-np.inf],np.nan)
    TH=EV.thesis(s)
    for per in ['MASTER','MB','HOLDOUT','DESIGN']:
        fids=O[O.period.isin(['MASTER','BATCH'] if per=='MB' else [per])].fid.values
        if len(fids)<8: continue
        dp=dec[dec.fid.isin(fids)].reset_index(drop=True); fv,lab=dp.fid.values,dp.label.values
        NS=300
        res=[]; mxm=np.zeros(NS); mxt=np.zeros(NS)
        for gi,(gn,gf) in enumerate(EV.GATES.items()):
            gm=gf(dp).values; idx,perms=EV.shift_perms(dp,gm,NS,11+gi)
            for nm,(_,fn) in TH.items():
                try: cm=fn(dp).fillna(False).values
                except AttributeError: continue
                def dvec(sel):
                    d=np.zeros(len(fids))
                    if len(sel):
                        f=fv[sel]; u,first=np.unique(f,return_index=True)
                        d[:len(u)]=lab[sel[first]]
                    return d
                ob=dvec(idx[cm[idx]]); om=ob.mean(); ot=om/(ob.std(ddof=1)/np.sqrt(len(ob))) if ob.std()>0 else 0
                nm_=np.zeros(NS); nt=np.zeros(NS)
                for b in range(NS):
                    d=dvec(np.sort(idx[cm[perms[b]]])); nm_[b]=d.mean(); nt[b]=d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if d.std()>0 else 0
                mxm=np.maximum(mxm,nm_); mxt=np.maximum(mxt,nt)
                res.append(dict(rule=f'{nm}·{gn}',fired=int((ob!=0).sum()),d=om,t=ot,p_rule_mean=(nm_>=om).mean(),null_rule_mean_med=np.median(nm_),null_q95=np.percentile(nm_,95)))
        R=pd.DataFrame(res)
        R['p_fw_mean']=[(mxm>=v).mean() for v in R.d]; R['p_fw_t']=[(mxt>=v).mean() for v in R.t]
        print(f'== {s} {per} N={len(fids)}  FW q95 mean={np.percentile(mxm,95):.3f}  FW q95 t={np.percentile(mxt,95):.2f}')
        print(R[R.rule.str.endswith('G3')].sort_values('d',ascending=False).round(3).to_string(index=False))
