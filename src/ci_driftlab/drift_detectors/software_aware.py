from __future__ import annotations
import numpy as np
import pandas as pd

COLUMNS=["dataset","project","feature","window_prev","window_curr","detector","score","threshold","drift_detected","p_value","notes"]
SIGNALS={"failure_rate_shift":"failure_rate","duration_shift":"duration_mean","suite_size_shift":"suite_size","test_churn_shift":"test_churn","build_cadence_shift":"build_cadence"}
def run_software_aware_detectors(features,baseline,thresholds,signals):
    rows=[]; rel=float(thresholds["relative_change"])
    for (dataset,project,gran,strategy),g in features.groupby(["dataset","project","granularity","strategy"]):
      g=g.sort_values("window_id"); seq=list(g.itertuples(index=False))
      for prev,curr in zip(seq,seq[1:]):
        for detector,feature in SIGNALS.items():
          if detector not in signals: continue
          a=getattr(prev,feature); b=getattr(curr,feature)
          score=abs(b-a)/(abs(a)+1e-9) if pd.notna(a) and pd.notna(b) else np.nan
          rows.append({"dataset":dataset,"project":project,"feature":feature,"window_prev":prev.window_id,"window_curr":curr.window_id,"detector":detector,"score":score,"threshold":rel,"drift_detected":bool(pd.notna(score) and score>=rel),"p_value":np.nan,"notes":"relative change"})
    if "performance_degradation" in signals and not baseline.empty:
      best=baseline.groupby(["dataset","project","granularity","strategy","window_id"],as_index=False).APFDc.mean()
      for (dataset,project,gran,strategy),g in best.groupby(["dataset","project","granularity","strategy"]):
        seq=list(g.sort_values("window_id").itertuples(index=False))
        for prev,curr in zip(seq,seq[1:]):
          score=prev.APFDc-curr.APFDc; threshold=float(thresholds["performance_drop"])
          rows.append({"dataset":dataset,"project":project,"feature":"APFDc","window_prev":prev.window_id,"window_curr":curr.window_id,"detector":"performance_degradation","score":score,"threshold":threshold,"drift_detected":bool(pd.notna(score) and score>=threshold),"p_value":np.nan,"notes":"mean model performance drop"})
    return pd.DataFrame(rows,columns=COLUMNS)
