"""Frozen cross-market factor test; structural data audit is not PIT certification."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

import pandas as pd

from alphax.experiment import CSI300_DATA_FILES, CSI300_RECORD, CSI300_SCOPE, artifact_inventory, build_manifest
from csi300_data import build_panel
from miner_lite import BASELINE_FEATURES, load_candidates
from run_candidate_batch import canonical, failure_reasons, remove_tradable_artifacts
from trader_lite import file_fingerprint
from temporal import is_feature_column

HERE = Path(__file__).resolve().parent
ML = HERE.parent / "machine-learn"
TRAIN = ("2010-01-01", "2018-12-31")
VALIDATION = ("2019-01-01", "2021-12-31")
HOLDOUT = ("2022-01-01", "2025-12-31")
LABEL_MODE = "paper_close"
HOLDOUT_POLICY = "evaluate frozen admitted candidates only; skip if none admitted"


def freeze_candidates(schema: Path, prior: Path) -> tuple[list[str], list[dict]]:
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
    return columns, unique


def run(out: Path, source: Path, prior: Path, schema: Path) -> dict:
    out = out.resolve()
    out.mkdir(parents=True)
    run_id = str(uuid4())

    def write(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def set_status(status, **details):
        write("status.json", {"status": status, "record_type": CSI300_RECORD, "run_id": run_id, **details})

    stage = "preparing"
    try:
        set_status(stage)
        started_at = datetime.now(timezone.utc).isoformat()
        input_paths = {
            **{role: (source / name).resolve() for role, name in CSI300_DATA_FILES.items()},
            "candidate_file": prior.resolve(),
            "feature_schema": schema.resolve(),
        }
        checkout = HERE.parents[1]
        code_paths = [HERE / name for name in (
            "run_csi300_research.py", "csi300_data.py", "miner_lite.py", "run_candidate_batch.py",
            "temporal.py", "trader_lite.py", "admission.py", "alphax/__init__.py",
            "alphax/factor_dsl.py", "alphax/experiment.py",
        )] + [ML / "build_features.py", ML / "data_contract.py"]
        input_fingerprints = {role: file_fingerprint(path) for role, path in input_paths.items()}
        code_fingerprints = {path.relative_to(checkout).as_posix(): file_fingerprint(path) for path in code_paths}
        columns, unique = freeze_candidates(input_paths["feature_schema"], input_paths["candidate_file"])
        write("candidates_frozen.json", unique)
        calendar = pd.DatetimeIndex(pd.to_datetime(input_paths["calendar"].read_text().splitlines()))
        pd.DataFrame({"date": calendar}).to_csv(out / "calendar.csv", index=False)
        pinned = {entry["path"]: entry for entry in (*input_fingerprints.values(), *code_fingerprints.values())}
        config = {
            "train": TRAIN, "validation": VALIDATION, "holdout": HOLDOUT, "label_mode": LABEL_MODE,
            "research_market": "cn", "holdout_policy": HOLDOUT_POLICY,
        }
        windows = {
            "runner_version": "csi300-factor-research-v1",
            "selection": dict(zip(("start", "end"), VALIDATION)),
            "holdout": dict(zip(("start", "end"), HOLDOUT)),
        }
        manifest = build_manifest(
            {"output_dir": str(out)}, config, input_fingerprints, code_fingerprints, pinned,
            windows, checkout, record_type=CSI300_RECORD, extra_packages=("h5py",), run_id=run_id,
        )
        frozen_candidates = file_fingerprint(out / "candidates_frozen.json")
        checked_files = {**pinned, frozen_candidates["path"]: frozen_candidates}
        manifest.update(
            scope=CSI300_SCOPE, started_at=started_at, candidate_count=len(unique),
            train=TRAIN, validation=VALIDATION, holdout=HOLDOUT, label_mode=LABEL_MODE,
            holdout_policy=HOLDOUT_POLICY,
            candidate_sources="Raw safe fields plus pre-existing US-generated LLM proposals; no CSI-adaptive generation",
            limitations=[
                "Supplied membership/calendar/adjustment provenance not independently certified",
                "Retrospectively generated hypotheses; no historical LLM availability claim",
                "Close-to-close research labels; no A-share execution or profit claim",
            ],
            files=checked_files,
        )
        write("manifest.json", manifest)
        snapshots = out / "source_snapshot"
        snapshots.mkdir()
        for path in code_paths:
            shutil.copyfile(path, snapshots / path.name)

        stage = "building_features"
        set_status(stage)
        stats = build_panel(input_paths["prices"], input_paths["membership"], calendar,
                            out / "features.csv.gz", columns, TRAIN[0], HOLDOUT[1])
        write("feature_audit.json", stats)
        stage = "evaluating_train_validation"
        set_status(stage)
        first = True
        for chunk in pd.read_csv(out / "features.csv.gz", chunksize=100000):
            subset = chunk[chunk.date <= VALIDATION[1]]
            subset.to_csv(out / "admission_features.csv.gz", index=False, mode="w" if first else "a", header=first,
                          compression={"method": "gzip", "compresslevel": 1})
            first = False

        def run_miner(folder, input_path, candidate_path, cutoff):
            cmd = [sys.executable, str(HERE / "miner_lite.py"), "--input", str(input_path),
                   "--calendar", str(out / "calendar.csv"), "--require-explicit-label-dates",
                   "--label-mode", LABEL_MODE, "--candidate-file", str(candidate_path),
                   "--research-market", "cn",
                   "--output-dir", str(out / folder), "--as-of", cutoff,
                   "--train-start", TRAIN[0], "--train-end", TRAIN[1],
                   "--validation-start", VALIDATION[0], "--validation-end", VALIDATION[1],
                   "--backtest-start", HOLDOUT[0], "--backtest-end", HOLDOUT[1]]
            write(folder + "_command.json", cmd)
            try:
                with (out / (folder + ".log")).open("x") as log:
                    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
                return json.loads((out / folder / "candidate_metrics.json").read_text())
            finally:
                remove_tradable_artifacts(out / folder)

        rows = run_miner("admission", out / "admission_features.csv.gz", out / "candidates_frozen.json", VALIDATION[1])
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
            stage = "evaluating_frozen_holdout"
            set_status(stage, accepted=accepted)
            holdout = run_miner("holdout", out / "features.csv.gz", out / "holdout_candidates.json", HOLDOUT[1])
            if file_fingerprint(out / "holdout_selection_frozen.json") != selection_fingerprint:
                raise ValueError("Holdout selection changed")
            for row in holdout:
                expected = next(r for r in findings if r["candidate_id"] == row["candidate_id"])
                if row["direction_from_train"] != expected["direction"] or not row["accepted"]:
                    raise ValueError("Frozen admission/direction changed in holdout run")
        stage = "finalizing"
        write("holdout_metrics.json", holdout)
        for path, expected in checked_files.items():
            if file_fingerprint(Path(path)) != expected:
                raise ValueError("Source changed during research: " + path)
        summary = {"candidate_count": len(rows), "strict_accepted": accepted,
                   "holdout_evaluated": bool(accepted), "scope": manifest["scope"]}
        write("summary.json", summary)
        write("artifacts.json", artifact_inventory(out))
        set_status("completed_research")
    except (Exception, KeyboardInterrupt) as exc:
        try:
            write("artifacts.json", artifact_inventory(out))
        except OSError:
            pass
        try:
            set_status("failed", stage=stage, error=str(exc))
        except OSError:
            pass
        raise
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output must be a new directory")
    prior_run = ML / "data/qlib/alphacrafter_feedback_search_v1"
    summary = run(
        args.output_dir, ML / "data/raw/qlib_csi300_quantaalpha",
        prior_run / "miner/candidates.json", prior_run / "features.csv.gz",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
