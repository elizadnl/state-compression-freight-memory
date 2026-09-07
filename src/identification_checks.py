#!/usr/bin/env python3
"""Identification and stability checks for the public BDI analysis.

Requires data/public_bdi_2009_2025.csv produced by prepare_public_bdi.py.
Writes results/identification_checks.json.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analyze_public_bdi as core
DATA=ROOT/'data'/'public_bdi_2009_2025.csv'; OUT=ROOT/'results'/'identification_checks.json'

def add_path(d):
    d=d.copy(); d['run_abs_cum']=d.groupby('spell')['r'].transform(lambda x:x.abs().cumsum()); d['prior_run_abs']=d.run_abs_cum-d.r.abs(); return d

def fit(d, age=False, path=False, age4=False, hac=False):
    X=pd.DataFrame({'state_up':(d.state>0).astype(float),'r1':d.r,'r2':d.r**2},index=d.index)
    if path:X['prior_run_abs']=d.prior_run_abs
    if age:X['age']=d.age
    if age4:X['age4plus']=(d.age>=4).astype(float)
    z=pd.concat([d[['exit','spell']],X],axis=1).dropna(); y=z.pop('exit'); g=z.pop('spell'); X=sm.add_constant(z,has_constant='add')
    if hac: f=sm.GLM(y,X,family=sm.families.Binomial()).fit(cov_type='HAC',cov_kwds={'maxlags':4})
    else: f=sm.GLM(y,X,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':g})
    return f,len(y),int(g.nunique())

def design(d,age=False,path=False):
    X=pd.DataFrame({'state_up':(d.state>0).astype(float),'r1':d.r,'r2':d.r**2},index=d.index)
    if path:X['prior_run_abs']=d.prior_run_abs
    if age:X['age']=d.age
    return X

def pred(train,row,age=False,path=False):
    Xt=design(train,age,path); z=pd.concat([train[['exit']],Xt],axis=1).dropna(); y=z.pop('exit').astype(int).values; X=z.values
    xr=design(row,age,path).reindex(columns=z.columns).values; sc=StandardScaler().fit(X)
    m=LogisticRegression(penalty=None,solver='lbfgs',max_iter=2000).fit(sc.transform(X),y)
    return float(m.predict_proba(sc.transform(xr))[0,1])

def expanding(d,age=False,path=False,start='2020-01-01'):
    rows=[]
    for i in d.index[(d.date>=pd.Timestamp(start)) & d.exit.notna()]:
        tr=d.loc[(d.index<i)&d.exit.notna()]
        try:p=pred(tr,d.loc[[i]],age,path)
        except Exception:continue
        rows.append((d.loc[i,'date'],int(d.loc[i,'exit']),np.clip(p,1e-6,1-1e-6)))
    return pd.DataFrame(rows,columns=['date','y','p'])

def compare(ref,alt):
    o=ref.merge(alt,on=['date','y'],suffixes=('_ref','_alt')); y=o.y.values; pr=o.p_ref.values; pa=o.p_alt.values
    br=(y-pr)**2; ba=(y-pa)**2; diff=br-ba
    lr=-(y*np.log(pr)+(1-y)*np.log(1-pr)); la=-(y*np.log(pa)+(1-y)*np.log(1-pa))
    return {'n':len(o),'brier_ref':float(br.mean()),'brier_alt':float(ba.mean()),'relative_improvement':float(diff.mean()/br.mean()),
            'brier_hac4':core.hac_test(diff,4),'logloss_ref':float(lr.mean()),'logloss_alt':float(la.mean()),'logloss_hac4':core.hac_test(lr-la,4),
            'auc_ref':float(roc_auc_score(y,pr)),'auc_alt':float(roc_auc_score(y,pa))}

def main():
    d=add_path(core.build_panel(DATA)); out={'age_path_corr':float(d[['age','prior_run_abs']].dropna().corr().iloc[0,1])}
    fa,n,sp=fit(d,age=True); fp,_,_=fit(d,path=True); fb,_,_=fit(d,age=True,path=True); f4,_,_=fit(d,path=True,age4=True); fh,_,_=fit(d,age=True,hac=True)
    out['identification']={'age_only':{'coef':float(fa.params.age),'p':float(fa.pvalues.age)},
      'path_only':{'coef':float(fp.params.prior_run_abs),'p':float(fp.pvalues.prior_run_abs)},
      'path_plus_age':{'coef_age':float(fb.params.age),'p_age':float(fb.pvalues.age),'coef_path':float(fb.params.prior_run_abs),'p_path':float(fb.pvalues.prior_run_abs)},
      'path_plus_age4':{'coef_age4':float(f4.params.age4plus),'p_age4':float(f4.pvalues.age4plus),'or_age4':float(np.exp(f4.params.age4plus))},
      'age_hac4':{'coef':float(fh.params.age),'p':float(fh.pvalues.age)},'n':n,'spells':sp}
    base=expanding(d); age=expanding(d,age=True); path=expanding(d,path=True); both=expanding(d,age=True,path=True)
    out['oos_2020_2025']={'age_vs_base':compare(base,age),'path_vs_base':compare(base,path),'age_plus_path_vs_path':compare(path,both)}
    # Start-date sensitivity can be obtained by generating the earliest forecast stream once, then subsetting.
    base17=expanding(d,start='2017-01-01'); age17=expanding(d,age=True,start='2017-01-01')
    starts={}
    for st in ['2017-01-01','2018-03-01','2019-01-01','2020-01-01','2021-01-01','2022-01-01']:
        cut=pd.Timestamp(st); starts[st]=compare(base17[base17.date>=cut],age17[age17.date>=cut])
    out['oos_start_sensitivity']=starts
    br={}
    for name,mask in {'pre_2018_03':d.date<pd.Timestamp('2018-03-01'),'post_2018_03':d.date>=pd.Timestamp('2018-03-01'),'post_2018_exclude_2020_2022':(d.date>=pd.Timestamp('2018-03-01')) & (~d.date.dt.year.isin([2020,2021,2022]))}.items():
        f,nn,ss=fit(d.loc[mask & d.exit.notna()],age=True); br[name]={'n':nn,'spells':ss,'coef_age':float(f.params.age),'p':float(f.pvalues.age)}
    out['bdi_methodology_break']=br; OUT.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
