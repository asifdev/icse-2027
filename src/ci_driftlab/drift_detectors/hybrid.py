import pandas as pd

COLUMNS=["dataset","project","window_curr","rule","generic_vote","software_vote","score","threshold","drift_detected"]
def combine_detectors(generic,software):
    def votes(df,name):
        if df.empty: return pd.DataFrame(columns=["dataset","project","window_curr",name])
        return df.groupby(["dataset","project","window_curr"],as_index=False).drift_detected.mean().rename(columns={"drift_detected":name})
    merged=votes(generic,"generic_vote").merge(votes(software,"software_vote"),on=["dataset","project","window_curr"],how="outer").fillna(0)
    rows=[]
    for r in merged.itertuples(index=False):
      for rule,score,threshold in [("OR",max(r.generic_vote,r.software_vote),1e-12),("AND",min(r.generic_vote,r.software_vote),1e-12),("weighted_score",.5*r.generic_vote+.5*r.software_vote,.5)]:
        rows.append({"dataset":r.dataset,"project":r.project,"window_curr":r.window_curr,"rule":rule,"generic_vote":r.generic_vote,"software_vote":r.software_vote,"score":score,"threshold":threshold,"drift_detected":score>=threshold})
    return pd.DataFrame(rows,columns=COLUMNS)
