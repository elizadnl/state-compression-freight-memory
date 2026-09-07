#!/usr/bin/env python3
import argparse, json, warnings, math
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

SEED=20260904
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'public_bdi_2009_2025.csv'
OUT=ROOT/'results'; OUT.mkdir(parents=True,exist_ok=True)


def build_panel(path=DATA, zero_rule='carry'):
    if not Path(path).exists():
        raise FileNotFoundError(
            f'{path} is not bundled. Run `python src/prepare_public_bdi.py` first to fetch and verify the public source.'
        )
    d=pd.read_csv(path,parse_dates=['date']).sort_values('date').reset_index(drop=True)
    assert len(d)==808, f'Expected 808 rows, got {len(d)}'
    assert d.date.min()==pd.Timestamp('2009-11-06') and d.date.max()==pd.Timestamp('2025-04-25')
    assert d.date.duplicated().sum()==0 and d.bdi.isna().sum()==0
    assert set(d.date.diff().dropna().dt.days.unique())=={7}
    d['log_bdi']=np.log(d.bdi); d['r']=d.log_bdi.diff()
    s=np.sign(d.r)
    if zero_rule=='carry': s=s.replace(0,np.nan).ffill()
    elif zero_rule=='drop': s=s.replace(0,np.nan)
    d['state']=s
    spell=[]; age=[]; current=np.nan; sid=-1; a=0
    for x in s:
        if pd.isna(x): spell.append(np.nan); age.append(np.nan); continue
        if pd.isna(current) or x!=current: sid+=1; a=1; current=x
        else: a+=1
        spell.append(sid); age.append(a)
    d['spell']=spell; d['age']=age
    d['next_state']=d.state.shift(-1)
    d['exit']=(d.next_state!=d.state).astype(float)
    d.loc[d.state.isna()|d.next_state.isna(),'exit']=np.nan
    d['woy']=d.date.dt.isocalendar().week.astype(float)
    d['vol4']=d.r.rolling(4).std(); d['vol8']=d.r.rolling(8).std()
    for L in [1,2,4,8]: d[f'r_lag{L}']=d.r.shift(L)
    return d


def design(d, degree=2, age=False, season=False, vols=(), lags=()):
    X=pd.DataFrame(index=d.index)
    X['state_up']=(d.state>0).astype(float)
    for k in range(1,degree+1): X[f'r{k}']=d.r**k
    if age: X['age']=d.age
    if season:
        ang=2*np.pi*d.woy/52.1775; X['sin_woy']=np.sin(ang); X['cos_woy']=np.cos(ang)
    for v in vols: X[v]=d[v]
    for L in lags: X[f'r_lag{L}']=d[f'r_lag{L}']
    return X


def cluster_glm(d, degree=2, age=True, season=False, vols=(), lags=()):
    X=design(d,degree,age,season,vols,lags)
    z=pd.concat([d[['exit','spell']],X],axis=1).dropna()
    y=z.pop('exit'); g=z.pop('spell'); X=sm.add_constant(z,has_constant='add')
    fit=sm.GLM(y,X,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':g})
    return fit, z.index


def sklearn_fit_predict(train,row,degree=2,age=False,season=False,vols=(),lags=()):
    Xt=design(train,degree,age,season,vols,lags)
    z=pd.concat([train[['exit']],Xt],axis=1).dropna(); y=z.pop('exit').astype(int).values; X=z.values
    xr=design(row,degree,age,season,vols,lags).reindex(columns=z.columns).values
    scaler=StandardScaler().fit(X); Xs=scaler.transform(X); xrs=scaler.transform(xr)
    m=LogisticRegression(penalty=None,solver='lbfgs',max_iter=2000).fit(Xs,y)
    return float(m.predict_proba(xrs)[0,1])


def expanding_predictions(d,start,degree,age=False,season=False,vols=(),lags=()):
    rows=[]
    for i in d.index[(d.date>=pd.Timestamp(start)) & d.exit.notna()]:
        train=d.loc[(d.index<i)&d.exit.notna()]
        if len(train)<100: continue
        try: p=sklearn_fit_predict(train,d.loc[[i]],degree,age,season,vols,lags)
        except Exception: continue
        rows.append((d.loc[i,'date'],int(d.loc[i,'exit']),np.clip(p,1e-6,1-1e-6)))
    return pd.DataFrame(rows,columns=['date','y','p'])


def degree_selection(d):
    ans={}
    for deg in range(1,6):
        o=expanding_predictions(d,'2015-01-01',deg,False)
        o=o[o.date<pd.Timestamp('2020-01-01')]
        y=o.y.values;p=o.p.values
        ans[deg]={'n':len(o),'brier':float(np.mean((y-p)**2)),'logloss':float(log_loss(y,p,labels=[0,1]))}
    chosen=min(ans,key=lambda x:ans[x]['brier'])
    return chosen,ans


def hac_test(x,lags=4):
    r=sm.OLS(np.asarray(x),np.ones((len(x),1))).fit(cov_type='HAC',cov_kwds={'maxlags':lags})
    return {'mean':float(r.params[0]),'se':float(r.bse[0]),'t':float(r.tvalues[0]),'p':float(r.pvalues[0]),'lags':lags}


def moving_block_bootstrap_mean_test(x, block=4, B=5000, seed=SEED+77):
    x=np.asarray(x,float); n=len(x); obs=float(x.mean()); centered=x-obs
    rng=np.random.default_rng(seed); starts=np.arange(n-block+1); k=math.ceil(n/block)
    sims=np.empty(B)
    for b in range(B):
        idx=np.concatenate([np.arange(s,s+block) for s in rng.choice(starts,k,replace=True)])[:n]
        sims[b]=centered[idx].mean()
    p_two=float((1+(np.abs(sims)>=abs(obs)).sum())/(B+1))
    return {'mean':obs,'block':block,'B':B,'se_boot':float(sims.std(ddof=1)),'p_two_sided':p_two}


def oos_compare(d,degree=2,start='2020-01-01'):
    b=expanding_predictions(d,start,degree,False).rename(columns={'p':'p_base'})
    a=expanding_predictions(d,start,degree,True).rename(columns={'p':'p_age'})
    o=b.merge(a,on=['date','y'])
    y=o.y.values; pb=o.p_base.values; pa=o.p_age.values
    o['brier_base']=(y-pb)**2; o['brier_age']=(y-pa)**2
    o['ll_base']=-(y*np.log(pb)+(1-y)*np.log(1-pb));o['ll_age']=-(y*np.log(pa)+(1-y)*np.log(1-pa))
    def summarize(q):
        yy=q.y.values; xb=q.p_base.values; xa=q.p_age.values
        return {'n':len(q),'brier_base':float(q.brier_base.mean()),'brier_age':float(q.brier_age.mean()),
                'brier_rel_improvement':float((q.brier_base.mean()-q.brier_age.mean())/q.brier_base.mean()),
                'brier_hac4':hac_test(q.brier_base-q.brier_age,4),
                'logloss_base':float(q.ll_base.mean()),'logloss_age':float(q.ll_age.mean()),
                'logloss_hac4':hac_test(q.ll_base-q.ll_age,4),
                'auc_base':float(roc_auc_score(yy,xb)),'auc_age':float(roc_auc_score(yy,xa))}
    out={'all':summarize(o),'2020_2022':summarize(o[o.date<'2023-01-01']),'2023_2025':summarize(o[o.date>='2023-01-01'])}
    diff=(o.brier_base-o.brier_age).values
    out['inference_sensitivity']={'hac':{str(L):hac_test(diff,L) for L in [0,1,2,4,8]},
                                  'moving_block_4':moving_block_bootstrap_mean_test(diff,4,5000)}
    o.to_csv(OUT/'oos_predictions.csv',index=False)
    return out


def rolling_fit(d,h):
    q=d.copy(); q['rollret']=q.log_bdi-q.log_bdi.shift(h); s=np.sign(q.rollret).replace(0,np.nan).ffill();q['rr_state']=s
    spell=[];age=[];cur=np.nan;sid=-1;a=0
    for x in s:
        if pd.isna(x):spell.append(np.nan);age.append(np.nan);continue
        if pd.isna(cur) or x!=cur:sid+=1;a=1;cur=x
        else:a+=1
        spell.append(sid);age.append(a)
    q['rr_spell']=spell;q['rr_age']=age;q['rr_exit']=(q.rr_state.shift(-1)!=q.rr_state).astype(float)
    q.loc[q.rr_state.isna()|q.rr_state.shift(-1).isna(),'rr_exit']=np.nan
    X=pd.DataFrame({'state_up':(q.rr_state>0).astype(float),'dist':q.rollret.abs(),'dist2':q.rollret.abs()**2,'age':q.rr_age})
    z=pd.concat([q[['rr_exit','rr_spell']],X],axis=1).dropna(); y=z.pop('rr_exit');g=z.pop('rr_spell');X=sm.add_constant(z,has_constant='add')
    fit=sm.GLM(y,X,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':g})
    return {'coef_age':float(fit.params.age),'se':float(fit.bse.age),'p':float(fit.pvalues.age),'n':int(fit.nobs),'spells':int(g.nunique())}


def ar_fit(r,p):
    x=pd.Series(r).dropna().values; Y=x[p:];X=np.ones((len(Y),p+1))
    for j in range(1,p+1):X[:,j]=x[p-j:len(x)-j]
    beta=np.linalg.lstsq(X,Y,rcond=None)[0];res=Y-X@beta
    return beta,res


def sim_ar(beta,resid,n,rng,burn=200):
    p=len(beta)-1;x=np.zeros(n+burn+p);x[:p]=rng.choice(resid,p,replace=True)
    for t in range(p,len(x)):
        x[t]=beta[0]+sum(beta[j]*x[t-j] for j in range(1,p+1))+rng.choice(resid)
    return x[burn+p:]


def sim_weekly_test(rr):
    s=np.sign(rr);s[s==0]=1;spell=np.zeros(len(rr),int);age=np.ones(len(rr),int);sid=0;a=1
    for i in range(1,len(rr)):
        if s[i]!=s[i-1]:sid+=1;a=1
        else:a+=1
        spell[i]=sid;age[i]=a
    y=(s[1:]!=s[:-1]).astype(int);r=rr[:-1];st=s[:-1];ag=age[:-1];g=spell[:-1]
    X=np.column_stack([np.ones(len(y)),(st>0).astype(float),r,r*r,ag])
    try:
        fit=sm.GLM(y,X,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':g})
        return float(fit.params[-1]),float(fit.pvalues[-1])
    except Exception:return np.nan,np.nan


def sim_rolling_test(rr,h):
    lp=np.r_[0,np.cumsum(rr)];R=lp[h:]-lp[:-h];s=np.sign(R);s[s==0]=1
    spell=np.zeros(len(R),int);age=np.ones(len(R),int);sid=0;a=1
    for i in range(1,len(R)):
        if s[i]!=s[i-1]:sid+=1;a=1
        else:a+=1
        spell[i]=sid;age[i]=a
    y=(s[1:]!=s[:-1]).astype(int);z=R[:-1];st=s[:-1];ag=age[:-1];g=spell[:-1]
    X=np.column_stack([np.ones(len(y)),(st>0).astype(float),np.abs(z),np.abs(z)**2,ag])
    try:
        fit=sm.GLM(y,X,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':g})
        return float(fit.params[-1]),float(fit.pvalues[-1])
    except Exception:return np.nan,np.nan


def simulations(d,obs,rolling_obs,B_week=1000,B_roll=500):
    r=d.r.dropna().values; rng=np.random.default_rng(SEED);out={}
    for p in [1,4]:
        beta,res=ar_fit(r,p); co=[];pv=[]
        for _ in range(B_week):
            c,v=sim_weekly_test(sim_ar(beta,res,len(r),rng),);co.append(c);pv.append(v)
        co=np.array(co,float);pv=np.array(pv,float);ok=np.isfinite(co)&np.isfinite(pv);co=co[ok];pv=pv[ok]
        out[f'ar{p}']={'B':int(len(co)),'beta':beta.tolist(),'coef_mean':float(co.mean()),'coef_sd':float(co.std(ddof=1)),
                       'reject5':float((pv<.05).mean()),'upper_tail':float((1+(co>=obs).sum())/(1+len(co)))}
        pd.DataFrame({'coef':co,'p':pv}).to_csv(OUT/f'sim_weekly_ar{p}.csv',index=False)
    beta,res=ar_fit(r,1);roll={}
    for h in [4,8,13,26]:
        co=[];pv=[]
        for _ in range(B_roll):
            c,v=sim_rolling_test(sim_ar(beta,res,len(r),rng),h);co.append(c);pv.append(v)
        co=np.array(co,float);pv=np.array(pv,float);ok=np.isfinite(co)&np.isfinite(pv);co=co[ok];pv=pv[ok]
        roll[str(h)]={'B':int(len(co)),'coef_mean':float(co.mean()),'coef_sd':float(co.std(ddof=1)),
                      'reject5':float((pv<.05).mean()),'upper_tail':float((1+(co>=rolling_obs[h]).sum())/(1+len(co)))}
        pd.DataFrame({'coef':co,'p':pv}).to_csv(OUT/f'sim_rolling_ar1_h{h}.csv',index=False)
    out['rolling_ar1']=roll
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--simulations',action='store_true');ap.add_argument('--b-week',type=int,default=1000);ap.add_argument('--b-roll',type=int,default=500);args=ap.parse_args()
    d=build_panel();degree,cv=degree_selection(d);assert degree==2
    full=d[d.exit.notna()].copy();fit,idx=cluster_glm(full,degree,True)
    primary={'coef_age':float(fit.params.age),'se_cluster':float(fit.bse.age),'p_cluster':float(fit.pvalues.age),
             'odds_ratio_per_week':float(np.exp(fit.params.age)),'n':int(len(idx)),'spells':int(full.spell.nunique())}
    # 4+ indicator
    X=design(full,degree,False);X['age4plus']=(full.age>=4).astype(float);z=pd.concat([full[['exit','spell']],X],axis=1).dropna();y=z.pop('exit');g=z.pop('spell');X=sm.add_constant(z,has_constant='add')
    f4=sm.GLM(y,X,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':g})
    age4={'coef':float(f4.params.age4plus),'p':float(f4.pvalues.age4plus),'odds_ratio':float(np.exp(f4.params.age4plus))}
    # hazard
    hz=full.assign(age_bin=pd.cut(full.age,[0,1,2,3,np.inf],labels=['1','2','3','4+'])).groupby('age_bin',observed=True).exit.agg(['mean','count']).reset_index();hz.to_csv(OUT/'hazard_by_age.csv',index=False)
    # subperiods
    periods={'pre2020':full.date<'2020-01-01','2020_2022':(full.date>='2020-01-01')&(full.date<'2023-01-01'),'2023_2025':full.date>='2023-01-01','exclude_2020_2022':~((full.date>='2020-01-01')&(full.date<'2023-01-01'))}
    sub={}
    for name,m in periods.items():
        f,ii=cluster_glm(full[m],degree,True);sub[name]={'n':int(len(ii)),'spells':int(full[m].spell.nunique()),'coef_age':float(f.params.age),'p':float(f.pvalues.age)}
    # robustness
    specs={'season':dict(season=True),'vol4':dict(vols=('vol4',)),'vol8':dict(vols=('vol8',)),'vol4_vol8':dict(vols=('vol4','vol8')),
           'lag1':dict(lags=(1,)),'lags12':dict(lags=(1,2)),'lags124':dict(lags=(1,2,4)),'lags1248':dict(lags=(1,2,4,8))}
    robust={}
    for k,kw in specs.items():
        f,ii=cluster_glm(full,degree,True,**kw);robust[k]={'n':int(len(ii)),'coef_age':float(f.params.age),'p':float(f.pvalues.age)}
    ddrop=build_panel(zero_rule='drop');fdrop,ii=cluster_glm(ddrop[ddrop.exit.notna()],degree,True);robust['drop_zero_returns']={'n':int(len(ii)),'coef_age':float(fdrop.params.age),'p':float(fdrop.pvalues.age)}
    rolling={h:rolling_fit(d,h) for h in [4,8,13,26]}
    oos=oos_compare(d,degree)
    result={'data':{'n':len(d),'start':str(d.date.min().date()),'end':str(d.date.max().date()),'zero_returns':int((d.r==0).sum())},
            'degree_selection':{'chosen':degree,'validation':cv},'primary':primary,'age4plus':age4,
            'hazard':hz.assign(age_bin=hz.age_bin.astype(str)).to_dict('records'),'subperiods':sub,'robustness':robust,
            'rolling_regimes':{str(k):v for k,v in rolling.items()},'oos':oos}
    if args.simulations:
        result['simulations']=simulations(d,primary['coef_age'],{h:rolling[h]['coef_age'] for h in rolling},args.b_week,args.b_roll)
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
