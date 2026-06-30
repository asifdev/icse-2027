from __future__ import annotations
from datetime import datetime,timezone
from pathlib import Path
from time import monotonic,sleep
from typing import Callable,Any
import hashlib,json,traceback
import pandas as pd

def _now(): return datetime.now(timezone.utc).isoformat()
def _hash(value): return hashlib.sha256(json.dumps(value,sort_keys=True,default=str,separators=(",",":")).encode("utf-8")).hexdigest()

class CheckpointManager:
    """Durable substep state with stale-input and corrupt-output detection."""
    def __init__(self,out:Path,logger,resume=False,force=False):
        self.out=Path(out); self.logger=logger; self.resume=resume; self.force=force; self.dir=self.out/"checkpoints"; self.path=self.dir/"pipeline_state.json"; self.state=self._load()
    def _load(self):
        if not self.path.exists(): return {"version":1,"created_at":_now(),"updated_at":_now(),"steps":{}}
        try:
            value=json.loads(self.path.read_text(encoding="utf-8")); value.setdefault("steps",{}); return value
        except Exception as exc: raise RuntimeError(f"Checkpoint state is unreadable: {self.path}: {exc}") from exc
    def _save(self):
        self.dir.mkdir(parents=True,exist_ok=True); self.state["updated_at"]=_now(); temp=self.path.with_name(f"{self.path.name}.{id(self)}.tmp"); temp.write_text(json.dumps(self.state,indent=2,default=str),encoding="utf-8")
        for attempt in range(6):
            try: temp.replace(self.path); return
            except PermissionError:
                if attempt==5: raise
                sleep(.05*(attempt+1))
    def _sample_hash(self,path:Path):
        digest=hashlib.sha256(); size=path.stat().st_size
        with path.open("rb") as handle:
            digest.update(handle.read(65536))
            if size>65536:
                handle.seek(max(0,size-65536)); digest.update(handle.read(65536))
        return digest.hexdigest()
    def _signature(self,path:Path):
        path=Path(path)
        if not path.exists(): return {"path":str(path.resolve()),"exists":False}
        if path.is_dir():
            files=[]
            for p in sorted(x for x in path.rglob("*") if x.is_file()):
                stat=p.stat(); files.append([str(p.relative_to(path)),stat.st_size,stat.st_mtime_ns])
            return {"path":str(path.resolve()),"exists":True,"kind":"directory","files":files}
        stat=path.stat(); return {"path":str(path.resolve()),"exists":True,"kind":"file","size":stat.st_size,"mtime_ns":stat.st_mtime_ns,"sample_sha256":self._sample_hash(path)}
    def input_digest(self,inputs,parameters): return _hash({"inputs":[self._signature(Path(x)) for x in inputs],"parameters":parameters})
    def validate_outputs(self,outputs):
        errors=[]
        for raw in outputs:
            path=Path(raw)
            if not path.exists(): errors.append(f"missing: {path}"); continue
            if path.is_file() and path.stat().st_size==0: errors.append(f"empty: {path}"); continue
            try:
                if path.suffix.lower()==".csv": pd.read_csv(path,nrows=1)
                elif path.suffix.lower()==".json": json.loads(path.read_text(encoding="utf-8"))
                elif path.suffix.lower() in {".md",".txt",".yaml",".yml"} and not path.read_text(encoding="utf-8",errors="ignore").strip(): errors.append(f"blank: {path}")
            except Exception as exc: errors.append(f"invalid {path}: {exc}")
        return errors
    def run_step(self,name:str,outputs:list[Path],fn:Callable[[],Any],inputs:list[Path]|None=None,parameters:dict|None=None):
        inputs=inputs or []; parameters=parameters or {}; digest=self.input_digest(inputs,parameters); prior=self.state["steps"].get(name,{})
        errors=self.validate_outputs(outputs); current_signatures=[self._signature(Path(x)) for x in outputs]; signatures_match=current_signatures==prior.get("output_signatures")
        if self.resume and not self.force and prior.get("status")=="completed" and prior.get("input_digest")==digest and not errors and signatures_match:
            self.logger.info("CHECKPOINT HIT  | %s | outputs=%d | completed=%s",name,len(outputs),prior.get("completed_at")); return "skipped"
        reason="forced" if self.force else "no completed checkpoint" if prior.get("status")!="completed" else "inputs changed" if prior.get("input_digest")!=digest else "outputs invalid: "+"; ".join(errors) if errors else "outputs changed since checkpoint"
        attempt=int(prior.get("attempt",0))+1; started=monotonic(); self.state["steps"][name]={"status":"running","attempt":attempt,"started_at":_now(),"input_digest":digest,"inputs":[str(Path(x)) for x in inputs],"outputs":[str(Path(x)) for x in outputs],"restart_reason":reason}; self._save()
        self.logger.info("CHECKPOINT START| %s | attempt=%d | reason=%s",name,attempt,reason)
        try: result=fn()
        except KeyboardInterrupt:
            self._mark_stopped(name,"interrupted",started,"KeyboardInterrupt"); self.logger.warning("CHECKPOINT INTERRUPTED | %s",name); raise
        except BaseException as exc:
            self._mark_stopped(name,"failed",started,f"{type(exc).__name__}: {exc}",traceback.format_exc()); self.logger.exception("CHECKPOINT FAILED | %s",name); raise
        errors=self.validate_outputs(outputs)
        if errors:
            exc=RuntimeError(f"Step {name} finished but outputs failed validation: {'; '.join(errors)}"); self._mark_stopped(name,"failed",started,str(exc)); raise exc
        elapsed=monotonic()-started; self.state["steps"][name].update({"status":"completed","completed_at":_now(),"elapsed_seconds":round(elapsed,3),"output_signatures":[self._signature(Path(x)) for x in outputs]}); self._save(); self.logger.info("CHECKPOINT DONE | %s | %.2fs | outputs=%d",name,elapsed,len(outputs)); return result
    def _mark_stopped(self,name,status,started,error,details=None):
        self.state["steps"][name].update({"status":status,"stopped_at":_now(),"elapsed_seconds":round(monotonic()-started,3),"error":error,"traceback":details}); self._save()
    def invalidate_from(self,marker,ordered_steps):
        matches=[i for i,x in enumerate(ordered_steps) if x==marker or x.startswith(marker+".")]
        if not matches: raise ValueError(f"Unknown restart marker {marker!r}. Valid steps: {ordered_steps}")
        affected=ordered_steps[min(matches):]
        for name in affected:
            if name in self.state["steps"]: self.state["steps"][name].update({"status":"invalidated","invalidated_at":_now(),"invalidated_from":marker})
        self._save(); self.logger.warning("Invalidated checkpoints from %s (%d steps). Existing output files remain until atomically replaced.",marker,len(affected))
    def status(self,ordered_steps):
        rows=[]
        for name in ordered_steps:
            value=self.state["steps"].get(name,{}); errors=self.validate_outputs([Path(x) for x in value.get("outputs",[])]) if value else ["not run"]
            rows.append({"step":name,"status":value.get("status","pending"),"attempt":value.get("attempt",0),"completed_at":value.get("completed_at"),"elapsed_seconds":value.get("elapsed_seconds"),"outputs_valid":not errors,"validation_errors":errors})
        return rows
