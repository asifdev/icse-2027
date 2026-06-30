from __future__ import annotations
from pathlib import Path
import pandas as pd

from .base import DatasetLoader
from ci_driftlab.preprocessing.normalize import normalize_outcome

BUILD_COLS=["build_id","project","commit_id","build_started_at","build_status","raw_source_file"]
TEST_COLS=["build_id","project","test_id","test_name","test_status","is_failed","duration_sec","started_at","raw_source_file"]


class LRTSLoader(DatasetLoader):
    """Native loader for LRTS build metadata plus per-build zipped test classes."""

    def _metadata_path(self):
        path=self.root/"dataset.csv"
        if not path.exists(): raise FileNotFoundError(f"LRTS metadata file does not exist: {path}")
        return path

    def _select_smoke_project(self, metadata):
        configured=self.options.get("smoke_project")
        if configured:
            if not metadata.project.astype(str).eq(str(configured)).any(): raise ValueError(f"Configured LRTS smoke_project not found: {configured}")
            self.logger.info("LRTS smoke selected configured project %s",configured); return str(configured)
        dated=metadata.assign(_month=pd.to_datetime(metadata["build_date"],errors="coerce").dt.to_period("M"))
        monthly=dated.groupby(["project","_month"]).size(); eligible=monthly[monthly>=int(self.options.get("smoke_min_builds_per_month",5))].groupby(level=0).size()
        counts=dated.groupby("project").size(); candidates=counts.to_frame("rows").join(eligible.rename("eligible_months")).fillna(0)
        candidates=candidates[candidates.eligible_months>=2].sort_values(["rows"])
        if candidates.empty: raise ValueError("No LRTS project has two smoke-eligible calendar months")
        selected=str(candidates.index[0]); self.logger.info("LRTS smoke selected smallest eligible project %s (%d builds)",selected,int(candidates.iloc[0].rows)); return selected

    @staticmethod
    def _canonical_build_id(row):
        return f"{row.project}|{row.pr_name}|build{row.build_id}|stage_{row.stage_id}"

    def _test_path(self,row):
        return self.root/"processed_test_result"/str(row.project)/f"{row.pr_name}_build{row.build_id}"/f"stage_{row.stage_id}"/"test_class.csv.zip"

    def inspect(self):
        metadata_path=self._metadata_path(); columns=list(pd.read_csv(metadata_path,nrows=0).columns)
        samples=sorted((self.root/"processed_test_result").rglob("test_class.csv.zip"))[:1]
        rows=[{"file":str(metadata_path),"kind":"lrts_build_metadata","columns":columns,
               "mapping":{"build_id":"project+pr_name+build_id+stage_id","project":"project","started_at":"build_timestamp","commit_id":"build_head_sha","build_status":"build_result"}}]
        for sample in samples:
            rows.append({"file":str(sample),"kind":"lrts_test_execution_zip","columns":list(pd.read_csv(sample,nrows=0).columns),
                         "mapping":{"test_name":"testclass","duration_sec":"duration","test_status":"outcome"}})
        return rows

    def load(self):
        metadata_path=self._metadata_path(); metadata=pd.read_csv(metadata_path,low_memory=False)
        if self.options.get("mode")=="smoke": metadata=metadata[metadata.project.astype(str).eq(self._select_smoke_project(metadata))].copy()
        metadata["canonical_build_id"]=[self._canonical_build_id(row) for row in metadata.itertuples(index=False)]
        metadata["started_at"]=pd.to_datetime(pd.to_numeric(metadata["build_timestamp"],errors="coerce"),unit="s",errors="coerce",utc=True)
        builds=pd.DataFrame({"build_id":metadata["canonical_build_id"],"project":metadata["project"].astype(str),"commit_id":metadata["build_head_sha"],"build_started_at":metadata["started_at"],"build_status":metadata["build_result"],"raw_source_file":str(metadata_path)})
        tests=[]; missing=0
        for index,row in enumerate(metadata.itertuples(index=False),start=1):
            path=self._test_path(row)
            if not path.exists(): missing+=1; continue
            try: frame=pd.read_csv(path,low_memory=False)
            except Exception as exc: self.logger.warning("Skipping unreadable LRTS test archive %s: %s",path,exc); continue
            outcome=frame["outcome"].astype(str)
            normalized_outcome=outcome.map(normalize_outcome)
            test_name=frame["testclass"].astype(str)
            tests.append(pd.DataFrame({"build_id":row.canonical_build_id,"project":str(row.project),"test_id":test_name,"test_name":test_name,"test_status":outcome,"is_failed":normalized_outcome.eq("fail"),"duration_sec":pd.to_numeric(frame["duration"],errors="coerce"),"started_at":row.started_at,"raw_source_file":str(path)}))
            if index%500==0: self.logger.info("LRTS test archive progress: %d/%d",index,len(metadata))
        if missing: self.logger.warning("LRTS has %d build rows without a matching test_class archive",missing)
        if not tests: raise ValueError("No LRTS test_class archives matched the selected build metadata")
        return builds[BUILD_COLS],pd.concat(tests,ignore_index=True)[TEST_COLS],self.inspect()
