from __future__ import annotations
import math
import numpy as np
import pandas as pd

COLUMNS=["dataset","project","window_id","budget_percent","delay","sampling_policy","available_labels","detected_degradation","missed_degradation","detection_delay","label_cost_proxy"]
ALIASES={
    "failure_rate":["failure_rate","test_failure_rate","train_failure_rate"],
    "recent_failure_rate":["recent_failure_rate","test_failure_rate","failure_rate","train_failure_rate"],
    "duration_mean":["duration_mean","mean_duration","median_duration"],
    "failure_count":["failure_count","test_failures","train_failures"],
}
def _series(g,logical,default=0.0):
    for col in ALIASES.get(logical,[logical]):
        if col in g.columns:
            return pd.to_numeric(g[col],errors="coerce").fillna(default)
    return pd.Series(default,index=g.index,dtype="float64")
def _order(g,policy,rng):
    if policy=="random": return rng.permutation(len(g))
    col={"failure_risk_based":"failure_rate","duration_risk_based":"duration_mean","change_aware":"recent_failure_rate","prioritization_cutoff":"failure_count"}.get(policy,"failure_rate")
    return np.argsort(-_series(g,col).to_numpy())
def simulate_labels(test_features,truth,budgets,delays,policies,seed):
    truth_map=truth.set_index(["dataset","project","window_curr"]).drift_detected.astype(bool).to_dict() if not truth.empty else {}
    rng=np.random.default_rng(seed); rows=[]
    for (dataset,project,window),g in test_features.groupby(["dataset","project","window_id"]):
      degraded=truth_map.get((dataset,project,window),False)
      failure_rate=_series(g,"failure_rate")
      median_failure=float(failure_rate.median()) if len(failure_rate) else 0.0
      for budget in budgets:
        n=min(len(g),max(1,math.ceil(len(g)*budget/100))) if len(g) else 0
        for policy in policies:
          sample_idx=_order(g,policy,rng)[:n]
          sample_failure_rate=failure_rate.iloc[sample_idx] if len(sample_idx) else pd.Series(dtype="float64")
          evidence=bool((sample_failure_rate>median_failure).any()) if len(sample_failure_rate) else False
          for delay in delays:
            detected=bool(degraded and evidence); rows.append({"dataset":dataset,"project":project,"window_id":window,"budget_percent":budget,"delay":delay,"sampling_policy":policy,"available_labels":n,"detected_degradation":detected,"missed_degradation":bool(degraded and not detected),"detection_delay":delay if detected else np.nan,"label_cost_proxy":n})
    frame=pd.DataFrame(rows,columns=COLUMNS)
    return frame[frame.delay==0].copy(),frame
