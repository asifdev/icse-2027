import numpy as np
import pandas as pd

COLUMNS=["detector_family","detector","precision","recall","F1","false_alarm_rate","detection_delay","missed_degradation_rate","true_positive","false_positive","false_negative","true_negative"]
def evaluate_detector_frame(frame, truth, family):
    if frame.empty or truth.empty: return pd.DataFrame(columns=COLUMNS)
    t=truth[["dataset","project","window_curr","drift_detected"]].rename(columns={"drift_detected":"truth"}).drop_duplicates(["dataset","project","window_curr"])
    rows=[]
    for detector,g in frame.groupby("detector"):
      pred=g.groupby(["dataset","project","window_curr"],as_index=False).drift_detected.max().merge(t,on=["dataset","project","window_curr"],how="inner")
      p=pred.drift_detected.astype(bool); y=pred.truth.astype(bool); tp=int((p&y).sum()); fp=int((p&~y).sum()); fn=int((~p&y).sum()); tn=int((~p&~y).sum())
      precision=tp/(tp+fp) if tp+fp else np.nan; recall=tp/(tp+fn) if tp+fn else np.nan
      rows.append({"detector_family":family,"detector":detector,"precision":precision,"recall":recall,"F1":2*precision*recall/(precision+recall) if precision+recall else np.nan,"false_alarm_rate":fp/(fp+tn) if fp+tn else np.nan,"detection_delay":0.0 if tp else np.nan,"missed_degradation_rate":fn/(fn+tp) if fn+tp else np.nan,"true_positive":tp,"false_positive":fp,"false_negative":fn,"true_negative":tn})
    return pd.DataFrame(rows,columns=COLUMNS)
