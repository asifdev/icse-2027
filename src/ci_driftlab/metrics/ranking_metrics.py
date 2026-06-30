from __future__ import annotations
import numpy as np

def _ordered(ranked_tests, values):
    if isinstance(values,dict): return np.asarray([values.get(x,False) for x in ranked_tests])
    a=np.asarray(values); return a[np.asarray(ranked_tests,dtype=int)] if len(a) else a
def apfd(ranked_tests, failed_flags):
    f=_ordered(ranked_tests,failed_flags).astype(bool); n=len(f); m=int(f.sum())
    return float(1-(np.flatnonzero(f)+1).sum()/(n*m)+1/(2*n)) if n and m else np.nan
def napfd(ranked_tests, failed_flags):
    # With all failures observable, NAPFD equals APFD.
    return apfd(ranked_tests,failed_flags)
def apfdc(ranked_tests, failed_flags, durations):
    f=_ordered(ranked_tests,failed_flags).astype(bool); c=_ordered(ranked_tests,durations).astype(float)
    c=np.nan_to_num(c,nan=0.0); total=c.sum(); m=int(f.sum())
    if not len(f) or not m or total<=0: return np.nan
    suffix=np.cumsum(c[::-1])[::-1]
    return float(sum((suffix[i]-c[i]/2) for i in np.flatnonzero(f))/(total*m))
def first_failure_rank(ranked_tests, failed_flags):
    f=_ordered(ranked_tests,failed_flags).astype(bool); idx=np.flatnonzero(f); return int(idx[0]+1) if len(idx) else np.nan
def time_to_first_failure(ranked_tests, failed_flags, durations):
    rank=first_failure_rank(ranked_tests,failed_flags); c=_ordered(ranked_tests,durations).astype(float)
    return float(np.nansum(c[:int(rank)])) if not np.isnan(rank) else np.nan
def time_saved(ranked_tests, failed_flags, durations):
    t=time_to_first_failure(ranked_tests,failed_flags,durations); total=float(np.nansum(_ordered(ranked_tests,durations)))
    return total-t if not np.isnan(t) else 0.0
