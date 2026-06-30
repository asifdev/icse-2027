from __future__ import annotations
import numpy as np
import pandas as pd

COLUMNS=["dataset","project","window_id","policy","APFD","APFDc","NAPFD","recovery_time","retraining_count","unnecessary_retraining_rate","missed_recovery_rate","label_cost","audit_cost","total_cost","cost_benefit_score","decision_basis"]
def simulate_retraining(baseline,software,audit,policies,cost_cfg):
    if baseline.empty: return pd.DataFrame(columns=COLUMNS)
    perf=baseline.groupby(["dataset","project","window_id","granularity","strategy"],as_index=False).agg(APFD=("APFD","mean"),APFDc=("APFDc","mean"),NAPFD=("NAPFD","mean"))
    drift=software.groupby(["dataset","project","window_curr"],as_index=False).drift_detected.max().rename(columns={"window_curr":"window_id"})
    aud=audit.groupby(["dataset","project","window_id"],as_index=False).audit_signal.max() if not audit.empty else pd.DataFrame(columns=["dataset","project","window_id","audit_signal"])
    perf=perf.merge(drift,on=["dataset","project","window_id"],how="left").merge(aud,on=["dataset","project","window_id"],how="left").fillna({"drift_detected":False,"audit_signal":False})
    rows=[]
    for (_,project,gran,strategy),g in perf.groupby(["dataset","project","granularity","strategy"]):
      seq=list(g.sort_values("window_id").itertuples(index=False))
      for i,r in enumerate(seq):
        for policy in policies:
          trigger={"none":False,"periodic_monthly":True,"periodic_quarterly":i%3==0,"rolling_window":True,"drift_triggered":r.drift_detected,"audit_triggered":r.audit_signal,"ensemble_weighting":r.drift_detected,"heuristic_fallback":r.drift_detected or r.audit_signal}[policy]
          unnecessary=bool(trigger and not r.drift_detected); missed=bool(r.drift_detected and not trigger); retrain=int(trigger)
          # Decision layer reports observed performance; it does not claim a counterfactual gain.
          total=retrain*cost_cfg["retraining_cost"]+unnecessary*cost_cfg["false_alarm_cost"]+missed*cost_cfg["missed_drift_cost"]
          rows.append({"dataset":r.dataset,"project":project,"window_id":r.window_id,"policy":policy,"APFD":r.APFD,"APFDc":r.APFDc,"NAPFD":r.NAPFD,"recovery_time":0 if trigger and r.drift_detected else np.nan,"retraining_count":retrain,"unnecessary_retraining_rate":float(unnecessary),"missed_recovery_rate":float(missed),"label_cost":0.0,"audit_cost":float(policy=="audit_triggered"),"total_cost":total+float(policy=="audit_triggered"),"cost_benefit_score":-total,"decision_basis":"policy simulation; metrics are observed pre-counterfactual values"})
    return pd.DataFrame(rows,columns=COLUMNS)
