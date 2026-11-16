"""Frozen cross-market factor test; structural data audit is not PIT certification."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

from csi300_data import build_panel
from miner_lite import BASELINE_FEATURES, candidate_from_dict, load_candidates
from run_candidate_batch import canonical, failure_reasons, remove_tradable_artifacts
from trader_lite import file_fingerprint
from temporal import is_feature_column

HERE = Path(__file__).resolve().parent
ML = HERE.parent / "machine-learn"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        parser.error("Output must be a new directory")
    out.mkdir(parents=True)

    def write(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    write("status.json", {"status": "preparing"})
    source = ML / "data/raw/qlib_csi300_quantaalpha"
    prior = ML / "data/qlib/alphacrafter_feedback_search_v1/miner/candidates.json"
    schema = ML / "data/qlib/alphacrafter_feedback_search_v1/features.csv.gz"
    columns = list(pd.read_csv(schema, nrows=0).columns)
    columns = [c for c in columns if c != "split"]
    available = [c for c in columns if is_feature_column(c) and c not in BASELINE_FEATURES]
    candidates = [{"candidate_id": "csi_raw_" + feature, "features": [feature],
                   "expression": {"op": "feature", "feature": feature},
                   "hypothesis": "Frozen single-feature cross-sectional test: " + feature}
                  for feature in available]
    for candidate in load_candidates(prior, available):
        row = asdict(candidate)
        row["expression"] = candidate.expression_spec
        row["features"] = list(candidate.features)
        candidates.append(row)
    unique, seen = [], set()
    for row in candidates:
        key = canonical(row["expression"])
        if key not in seen:
            unique.append(row)
            seen.add(key)
    write("candidates_frozen.json", unique)
    calendar = pd.DatetimeIndex(pd.to_datetime((source / "qlib_day.txt").read_text().splitlines()))
    pd.DataFrame({"date": calendar}).to_csv(out / "calendar.csv", index=False)
    inputs = [source / "daily_pv.h5", source / "qlib_csi300.txt", source / "qlib_day.txt", prior,
              out / "candidates_frozen.json", HERE / "csi300_data.py", Path(__file__), HERE / "miner_lite.py",
              ML / "build_features.py", ML / "data_contract.py", HERE / "run_candidate_batch.py", HERE / "temporal.py"]
    manifest = {"scope": "retrospective_CSI300_factor_quality_not_executable_returns_or_certified_PIT",
                "started_at": datetime.now(timezone.utc).isoformat(), "candidate_count": len(unique),
                "train": ["2010-01-01", "2018-12-31"], "validation": ["2019-01-01", "2021-12-31"],
                "holdout": ["2022-01-01", "2025-12-31"], "label_mode": "paper_close",
                "holdout_policy": "evaluate frozen admitted candidates only; skip if none admitted",
                "candidate_sources": "52 raw safe fields plus 28 pre-existing US-generated LLM proposals; no CSI-adaptive generation",
                "limitations": ["Supplied membership/calendar/adjustment provenance not independently certified",
                                "Retrospectively generated hypotheses; no historical LLM availability claim",
                                "Close-to-close research labels; no A-share execution or profit claim"],
                "files": {str(p): file_fingerprint(p) for p in inputs}}
    write("manifest.json", manifest)
    snapshots = out / "source_snapshot"
    snapshots.mkdir()
    for path in inputs:
        if path.suffix == ".py":
            shutil.copyfile(path, snapshots / path.name)

    def check_sources():
        for path, expected in manifest["files"].items():
            if file_fingerprint(Path(path)) != expected:
                raise ValueError("Source changed during research: " + path)

    try:
        stats = build_panel(source / "daily_pv.h5", source / "qlib_csi300.txt", calendar,
                            out / "features.csv.gz", columns, "2010-01-01", "2025-12-31")
        write("feature_audit.json", stats)
        write("status.json", {"status": "evaluating_train_validation"})
        first = True
        for chunk in pd.read_csv(out / "features.csv.gz", chunksize=100000):
            subset = chunk[chunk.date <= "2021-12-31"]
            subset.to_csv(out / "admission_features.csv.gz", index=False, mode="w" if first else "a", header=first,
                          compression={"method": "gzip", "compresslevel": 1})
            first = False

        def run_miner(folder, input_path, candidate_path, cutoff):
            cmd = [sys.executable, str(HERE / "miner_lite.py"), "--input", str(input_path),
                   "--calendar", str(out / "calendar.csv"), "--require-explicit-label-dates",
                   "--label-mode", "paper_close", "--candidate-file", str(candidate_path),
                   "--research-market", "cn",
                   "--output-dir", str(out / folder), "--as-of", cutoff,
                   "--train-start", "2010-01-01", "--train-end", "2018-12-31",
                   "--validation-start", "2019-01-01", "--validation-end", "2021-12-31",
                   "--backtest-start", "2022-01-01", "--backtest-end", "2025-12-31"]
            write(folder + "_command.json", cmd)
            try:
                with (out / (folder + ".log")).open("x") as log:
                    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
                return json.loads((out / folder / "candidate_metrics.json").read_text())
            finally:
                remove_tradable_artifacts(out / folder)

        rows = run_miner("admission", out / "admission_features.csv.gz", out / "candidates_frozen.json", "2021-12-31")
        if {r["candidate_id"] for r in rows} != {r["candidate_id"] for r in unique}:
            raise ValueError("Evaluated candidate set differs from frozen set")
        accepted = [r["candidate_id"] for r in rows if r["accepted"]]
        selected = [r for r in unique if r["candidate_id"] in accepted]
        write("holdout_selection_frozen.json", {"candidate_ids": accepted,
                                               "directions": {r["candidate_id"]: r["direction_from_train"] for r in rows if r["accepted"]}})
        write("holdout_candidates.json", selected)
        findings = []
        for row in rows:
            direction = row["direction_from_train"]
            failures = {f"{split}_{h}d": failure_reasons(row["horizons"][str(h)][split], direction, h)
                        for split in ("train", "validation") for h in (1, 5)}
            if all(not v for v in failures.values()) != row["accepted"]:
                raise ValueError("Failure diagnosis disagrees with strict gate")
            findings.append({"candidate_id": row["candidate_id"], "expression": row["expression"],
                             "direction": direction, "strict_accepted": row["accepted"],
                             "horizons": row["horizons"], "failed_checks": failures})
        write("findings.json", findings)
        holdout = []
        selection_fingerprint = file_fingerprint(out / "holdout_selection_frozen.json")
        write("selection_freeze_fingerprint.json", selection_fingerprint)
        if accepted:
            write("status.json", {"status": "evaluating_frozen_holdout", "accepted": accepted})
            holdout = run_miner("holdout", out / "features.csv.gz", out / "holdout_candidates.json", "2025-12-31")
            if file_fingerprint(out / "holdout_selection_frozen.json") != selection_fingerprint:
                raise ValueError("Holdout selection changed")
            for row in holdout:
                expected = next(r for r in findings if r["candidate_id"] == row["candidate_id"])
                if row["direction_from_train"] != expected["direction"] or not row["accepted"]:
                    raise ValueError("Frozen admission/direction changed in holdout run")
        write("holdout_metrics.json", holdout)
        check_sources()
        summary = {"candidate_count": len(rows), "strict_accepted": accepted,
                   "holdout_evaluated": bool(accepted), "scope": manifest["scope"]}
        write("summary.json", summary)
        write("status.json", {"status": "completed_research"})
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        write("status.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
