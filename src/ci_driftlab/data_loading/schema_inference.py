from __future__ import annotations
import re

ALIASES = {
 "build_id": ["build_id", "build", "build_number", "job_id", "buildid"],
 "commit_id": ["commit_id", "commit", "sha", "commit_sha"],
 "build_status": ["build_status", "build_result", "state"],
}

def _norm(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())
def find_column(columns, candidates):
    exact = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in exact: return exact[candidate.lower()]
    normalized = {_norm(c): c for c in columns}
    for candidate in candidates:
        if _norm(candidate) in normalized: return normalized[_norm(candidate)]
    return None

def infer_mapping(columns, candidate_config):
    mapping = {}
    config_map = {
      "project": candidate_config.get("project_column_candidates", []),
      "started_at": candidate_config.get("date_column_candidates", []),
      "test_name": candidate_config.get("test_column_candidates", []),
      "test_status": candidate_config.get("outcome_column_candidates", []),
      "duration_sec": candidate_config.get("duration_column_candidates", []),
    }
    for target, candidates in config_map.items(): mapping[target] = find_column(columns, candidates)
    for target, candidates in ALIASES.items(): mapping[target] = find_column(columns, candidates)
    return mapping
