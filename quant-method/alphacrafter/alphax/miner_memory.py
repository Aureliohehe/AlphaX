"""Append-only research evidence for AlphaCrafter Miner."""

import argparse
from contextlib import closing
from datetime import date, datetime, timezone
import hashlib
import json
import re
import sqlite3
from pathlib import Path

from .factor_dsl import expression_features, parse_expression, render_expression


DSL_VERSION = "factor-dsl-v1"
SCHEMA = """
CREATE TABLE IF NOT EXISTS factors (
    id TEXT PRIMARY KEY, dsl_version TEXT NOT NULL, ast TEXT NOT NULL,
    expression TEXT NOT NULL, features TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY, factor_id TEXT NOT NULL REFERENCES factors(id),
    text TEXT NOT NULL, control_id TEXT, details TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, market TEXT NOT NULL, universe TEXT NOT NULL, label_mode TEXT NOT NULL,
    available_at TEXT NOT NULL, metadata TEXT NOT NULL, source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluations (
    id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
    hypothesis_id TEXT NOT NULL REFERENCES hypotheses(id), candidate_id TEXT NOT NULL,
    horizon INTEGER NOT NULL, target TEXT NOT NULL, kind TEXT NOT NULL,
    evaluation_end TEXT NOT NULL, evidence TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memory_run_scope ON runs(market, universe, label_mode, available_at);
CREATE INDEX IF NOT EXISTS memory_evaluation_scope ON evaluations(run_id, horizon, target, evaluation_end);
"""


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(encode(value).encode("utf-8")).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def research_metadata(audit, market, universe, data_version):
    if market not in {"cn", "us"} or not universe.strip() or not data_version.strip():
        raise ValueError("Miner memory requires a known market, universe and data version")
    return {
        "market": market, "universe": universe.strip(), "data_version": data_version.strip(),
        "as_of": audit["as_of"], "label_mode": audit["label_mode"],
        "evaluation_policy_version": audit["policy_version"],
        "target_sources": audit["target_sources"], "label_end_source": audit["label_end_source"],
        "windows": {name: list(audit["windows"][name]) for name in ("train", "validation")},
        "cost_model": "not_evaluated",
    }


def search_evaluations(report, definitions, mechanisms):
    """Read current reports and older reports whose ASTs live in candidates.json."""
    if report["scope"] != "training_only_research_not_admission":
        raise ValueError("Expected a training-only Miner search report")
    for row in report["candidates"]:
        cid = row["candidate_id"]
        definition = definitions[cid]
        ast = parse_expression(row.get("expression_ast", definition["expression"]))
        if render_expression(ast) != row["expression"]:
            raise ValueError(f"Search evidence and candidate expression disagree for {cid}")
        if definition["hypothesis"] != row["hypothesis"]:
            raise ValueError(f"Search evidence and candidate hypothesis disagree for {cid}")
        for horizon in sorted({fold["horizon"] for fold in row["folds"]}):
            folds = [fold for fold in row["folds"] if fold["horizon"] == horizon]
            tasks = [task for task in row.get("next_validation_tasks", []) if task["horizon"] == horizon]
            yield {
                "candidate_id": cid, "ast": ast, "hypothesis": row["hypothesis"],
                "control_id": row.get("control_id", definition.get("control_id")),
                "details": hypothesis_details(definition.get("iteration_note"), mechanisms.get(cid)),
                "horizon": horizon, "kind": "search_fold",
                "evaluation_end": max(fold["end"] for fold in folds),
                "evidence": {"folds": folds, "next_validation_tasks": tasks},
            }


def hypothesis_details(iteration_note, mechanism):
    return {key: value for key, value in (("iteration_note", iteration_note), ("mechanism", mechanism)) if value}


class MinerMemory:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.executescript(SCHEMA)

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.row_factory = sqlite3.Row
        return connection

    def record(self, metadata, source, evaluations):
        """Publish a complete evidence batch and keep IDs and availability stable on retry."""
        items = list(evaluations)
        run_id = digest([metadata, items])
        as_of = date.fromisoformat(metadata["as_of"])
        with closing(self._connect()) as connection, connection:
            connection.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO NOTHING",
                               (run_id, metadata["market"], metadata["universe"], metadata["label_mode"],
                                utc_now(), encode(metadata), source))
            for row in items:
                ast = parse_expression(row["ast"])
                factor_id = digest([DSL_VERSION, ast])
                connection.execute("INSERT INTO factors VALUES (?, ?, ?, ?, ?) ON CONFLICT(id) DO NOTHING",
                                   (factor_id, DSL_VERSION, encode(ast), render_expression(ast),
                                    encode(sorted(set(expression_features(ast))))))
                hypothesis = (factor_id, row["hypothesis"], row["control_id"], encode(row["details"]))
                hypothesis_id = digest(hypothesis)
                connection.execute("INSERT INTO hypotheses VALUES (?, ?, ?, ?, ?) ON CONFLICT(id) DO NOTHING",
                                   (hypothesis_id, *hypothesis))
                end = date.fromisoformat(row["evaluation_end"])
                window = "train" if row["kind"] == "search_fold" else row["kind"]
                if window not in {"train", "validation"}:
                    raise ValueError(f"Unsupported Miner evidence kind {row['kind']}")
                start, window_end = map(date.fromisoformat, metadata["windows"][window])
                if not start <= end <= min(as_of, window_end):
                    raise ValueError("Evidence ends outside its research window or after the source cutoff")
                connection.execute(
                    "INSERT INTO evaluations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO NOTHING",
                    (digest([run_id, row]), run_id, hypothesis_id, row["candidate_id"], row["horizon"],
                     metadata["target_sources"][f"{row['horizon']}d"], row["kind"], end.isoformat(),
                     encode(row["evidence"])),
                )
        return run_id

    def recall(self, metadata, available, controls, limit):
        """Use only compatible evidence available before this session and inside its training cutoff."""
        as_of = metadata["as_of"]
        train_end = min(as_of, metadata["windows"]["train"][1])
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT e.*, r.available_at, r.metadata, h.text, h.control_id, f.expression, f.features,
                          f.id AS factor_id
                   FROM evaluations e JOIN runs r ON r.id = e.run_id
                   JOIN hypotheses h ON h.id = e.hypothesis_id JOIN factors f ON f.id = h.factor_id
                   WHERE r.market = ? AND r.universe = ? AND r.label_mode = ?
                     AND r.available_at < ? AND e.evaluation_end <= ? AND f.dsl_version = ?
                   ORDER BY e.evaluation_end DESC, r.available_at DESC, e.id""",
                (metadata["market"], metadata["universe"], metadata["label_mode"], as_of, train_end, DSL_VERSION),
            ).fetchall()
        available = set(available)
        control_families = {cid: {re.sub(r"\d+", "", f) for f in expression_features(ast)}
                            for cid, ast in controls.items()}
        ranked = []
        for row in rows:
            horizon = str(row["horizon"])
            if metadata["target_sources"].get(horizon + "d") != row["target"]:
                continue
            previous = json.loads(row["metadata"])
            if (previous["as_of"] > as_of
                    or previous.get("evaluation_policy_version") != metadata["evaluation_policy_version"]):
                continue
            if previous["label_end_source"].get(horizon) != metadata["label_end_source"].get(horizon):
                continue
            features = set(json.loads(row["features"]))
            if not features <= available:
                continue
            families = {re.sub(r"\d+", "", feature) for feature in features}
            control_ids = [
                cid for cid, inputs in control_families.items()
                if cid == row["control_id"] or families.intersection(inputs)
            ]
            if controls and not control_ids:
                continue
            priority = (row["control_id"] in control_ids, previous["data_version"] == metadata["data_version"])
            ranked.append((priority, row, previous, control_ids))
        selected = []
        seen = set()
        for _, row, previous, control_ids in sorted(ranked, key=lambda match: match[0], reverse=True):
            key = (row["factor_id"], row["horizon"], row["kind"])
            if key in seen:
                continue
            seen.add(key)
            selected.append({
                "evidence_id": row["id"], "run_id": row["run_id"], "expression": row["expression"],
                "hypothesis": row["text"][:600], "control_id": row["control_id"], "for_controls": control_ids,
                "horizon": row["horizon"], "target": row["target"], "kind": row["kind"],
                "window": previous["windows"]["train" if row["kind"] == "search_fold" else row["kind"]],
                "evaluation_end": row["evaluation_end"], "available_at": row["available_at"],
                "data_version": previous["data_version"], **summarize(row),
            })
            if len(selected) >= limit:
                break
        return selected


def summarize(row):
    evidence = json.loads(row["evidence"])
    if row["kind"] == "search_fold":
        folds = evidence["folds"]
        return {
            "failure_reasons": sorted({reason for fold in folds for reason in fold["failure_reasons"]}),
            "fold_count": len(folds),
            "recent_folds": [
                {"fold": fold["fold"], "direction": fold["direction"],
                 "rank_ic": fold["metrics"].get("mean_daily_rank_ic"),
                 "rank_icir": fold["metrics"].get("rank_icir"),
                 "failure_reasons": fold["failure_reasons"]}
                for fold in folds[-3:]
            ],
            "next_validation_tasks": evidence["next_validation_tasks"][:1],
        }
    return {"failure_reasons": evidence["failure_reasons"], "direction": evidence["direction"],
            "rank_ic": evidence["metrics"]["mean_daily_rank_ic"],
            "rank_icir": evidence["metrics"]["rank_icir"]}


def main():
    parser = argparse.ArgumentParser(description="Import an existing Miner search report into SQLite")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--search-feedback", type=Path, required=True)
    parser.add_argument("--market", choices=("us", "cn"), required=True)
    parser.add_argument("--universe", required=True)
    parser.add_argument("--data-version", required=True, help="Version or SHA-256 of the source feature panel")
    args = parser.parse_args()
    source = args.search_feedback.resolve()
    try:
        report = json.loads(source.read_text(encoding="utf-8"))
        definitions = {row["candidate_id"]: row for row in
                       json.loads(source.with_name("candidates.json").read_text(encoding="utf-8"))}
        audit = json.loads(source.with_name("temporal_audit.json").read_text(encoding="utf-8"))
        metadata = research_metadata(audit, args.market, args.universe, args.data_version)
        mechanism_path = source.with_name("economic_mechanisms.json")
        mechanisms = json.loads(mechanism_path.read_text(encoding="utf-8")) if mechanism_path.is_file() else {}
        rows = list(search_evaluations(report, definitions, mechanisms))
        run_id = MinerMemory(args.db).record(metadata, str(source), rows)
    except (KeyError, TypeError, ValueError, OSError, sqlite3.Error) as exc:
        parser.error(str(exc))
    print(f"Imported {len(rows)} evaluation records as {run_id}")


if __name__ == "__main__":
    main()
