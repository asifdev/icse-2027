from __future__ import annotations
import math
import numpy as np
import pandas as pd

COLUMNS=["dataset","project","feature","window_prev","window_curr","detector","score","threshold","drift_detected","p_value","notes"]
def _hist(a,b,bins=10):
    lo=min(np.min(a),np.min(b)); hi=max(np.max(a),np.max(b)); hi=hi if hi>lo else lo+1
    p,_=np.histogram(a,bins=bins,range=(lo,hi)); q,_=np.histogram(b,bins=bins,range=(lo,hi)); p=p/p.sum(); q=q/q.sum(); return p,q
def _hellinger(p,q): return float(np.sqrt(np.sum((np.sqrt(p)-np.sqrt(q))**2))/np.sqrt(2))
def _js(p,q):
    m=(p+q)/2
    def kl(x,y):
        mask=x>0
        return np.sum(x[mask]*np.log2(x[mask]/np.maximum(y[mask],1e-12)))
    return float(np.sqrt((kl(p,m)+kl(q,m))/2))
def _ks(a,b):
    try:
        from scipy.stats import ks_2samp
        r=ks_2samp(a,b); return float(r.statistic),float(r.pvalue),"scipy"
    except ImportError:
        values=np.sort(np.unique(np.r_[a,b])); d=max(abs(np.searchsorted(np.sort(a),values,side="right")/len(a)-np.searchsorted(np.sort(b),values,side="right")/len(b)))
        ne=len(a)*len(b)/(len(a)+len(b)); p=min(1.0,2*math.exp(-2*ne*d*d)); return float(d),float(p),"approximate KS fallback"
def run_generic_detectors(test_features,window_features,thresholds,detectors):
    rows=[]; feature_cols=[c for c in ["failure_rate","recent_failure_rate","last_failure_age","duration_mean","duration_std","test_age"] if c in test_features]
    meta=window_features[["dataset","project","window_id","granularity","strategy"]]
    for (_,project,gran,strategy),g in meta.groupby(["dataset","project","granularity","strategy"]):
      g=g.sort_values("window_id")
      for prev,curr in zip(g.itertuples(index=False),list(g.itertuples(index=False))[1:]):
        for feature in feature_cols:
          a=pd.to_numeric(test_features[(test_features.window_id==prev.window_id)][feature],errors="coerce").dropna().to_numpy(); b=pd.to_numeric(test_features[(test_features.window_id==curr.window_id)][feature],errors="coerce").dropna().to_numpy()
          if not len(a) or not len(b): continue
          p,q=_hist(a,b)
          for detector in detectors:
            pval=np.nan; notes=""
            if detector=="ks_test": score,pval,notes=_ks(a,b); threshold=thresholds["p_value"]; hit=pval<threshold
            elif detector=="hellinger": score=_hellinger(p,q); threshold=thresholds["distance"]; hit=score>=threshold
            elif detector=="jensen_shannon": score=_js(p,q); threshold=thresholds["distance"]; hit=score>=threshold
            elif detector=="cusum": score=abs(np.mean(b)-np.mean(a))/(np.std(a)+1e-9); threshold=1.0; hit=score>=threshold; notes="standardized mean-shift CUSUM proxy"
            elif detector=="adwin_light": score=abs(np.mean(b)-np.mean(a))/(abs(np.mean(a))+1e-9); threshold=thresholds["relative_change"]; hit=score>=threshold; notes="adaptive two-window approximation; not full ADWIN"
            else: continue
            rows.append({"dataset":curr.dataset,"project":project,"feature":feature,"window_prev":prev.window_id,"window_curr":curr.window_id,"detector":detector,"score":score,"threshold":threshold,"drift_detected":bool(hit),"p_value":pval,"notes":notes})
    return pd.DataFrame(rows,columns=COLUMNS)
