"""Local settings, factor artifacts, and research command preparation."""

import csv
from datetime import date, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from threading import Lock
from urllib.parse import urlsplit


WEB = Path(__file__).resolve().parent
ROOT = WEB.parents[1]
AC = ROOT / "quant-method/alphacrafter"
DATA = ROOT / "quant-method/machine-learn/data"
OUTPUTS = WEB / ".runs"
DEFAULTS = {
    "model": "deepseek-chat",
    "api_base": "",
    "panel": str((DATA / "processed/sp500_features_label_dates.csv.gz").relative_to(ROOT)),
    "calendar": str((DATA / "processed/sp500_xnys_xcal_4_13_3.csv").relative_to(ROOT)),
    "library_root": str((DATA / "qlib").relative_to(ROOT)),
    "train_start": "2016-01-04",
    "initial_as_of": "2022-12-30",
    "selection_start": "2023-01-03",
    "selection_end": "2023-12-29",
    "holdout_start": "2024-01-02",
    "holdout_end": "2026-02-27",
    "admission_sessions": 252,
    "min_train_sessions": 252,
    "maintenance_days": 90,
    "warmup_sessions": 120,
    "cost_rate": 0.0001,
    "configs": "0.8,0.5,50,25;0.8,0.0,50,50;0.8,1.0,50,0",
    "agents": 3,
    "candidates": 12,
    "rounds": 1,
}
DATE_KEYS = ("train_start", "initial_as_of", "selection_start", "selection_end", "holdout_start", "holdout_end")
GRADES = ("strict", "research", "unadmitted", "unknown")


def read_json(path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def valid_settings(body):
    if not isinstance(body, dict) or set(body) - DEFAULTS.keys():
        raise ValueError("设置字段无效。")
    settings = DEFAULTS | body
    for name in ("panel", "calendar", "library_root", "configs"):
        value = settings[name]
        if not isinstance(value, str) or not value.strip() or len(value) > 4096 or "\0" in value:
            raise ValueError(f"请填写有效的 {name}。")
        settings[name] = value.strip()
    model = settings["model"]
    if not isinstance(model, str) or not model.strip() or len(model) > 160 or "\0" in model:
        raise ValueError("请输入有效的模型名称。")
    settings["model"] = model.strip()
    base = settings["api_base"]
    if not isinstance(base, str) or len(base) > 2048:
        raise ValueError("API 地址无效。")
    base = base.strip().rstrip("/")
    if base:
        parts = urlsplit(base)
        if (parts.scheme not in ("http", "https") or not parts.hostname or parts.username
                or parts.password or parts.query or parts.fragment
                or any(character.isspace() for character in base)):
            raise ValueError("请输入 HTTP(S) API 基础地址，地址应省略凭据和查询参数。")
        _ = parts.port
    settings["api_base"] = base
    if not base and settings["model"] not in ("deepseek-chat", "deepseek-reasoner"):
        raise ValueError("自定义模型需要配置兼容 Chat Completions 的 API 基础地址。")
    try:
        dates = [date.fromisoformat(settings[name]) for name in DATE_KEYS]
    except (ValueError, TypeError):
        raise ValueError("日期格式应为 YYYY-MM-DD。") from None
    settings.update(zip(DATE_KEYS, map(str, dates)))
    train, cutoff, select_start, select_end, holdout_start, holdout_end = dates
    if not train < cutoff < select_start <= select_end < holdout_start <= holdout_end:
        raise ValueError("日期必须依次覆盖训练、准入截点、策略验证和留出期。")
    bounds = {
        "admission_sessions": (6, 5000), "min_train_sessions": (6, 5000),
        "maintenance_days": (1, 3650), "warmup_sessions": (0, 5000),
        "agents": (1, 12), "candidates": (1, 100), "rounds": (1, 10),
    }
    for name, (low, high) in bounds.items():
        value = settings[name]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} 应为 {low}–{high} 的整数。")
    cost = settings["cost_rate"]
    if type(cost) not in (int, float) or not 0 <= cost <= 1:
        raise ValueError("成本率应位于 0 到 1 之间。")
    return settings


class Settings:
    def __init__(self, path):
        self.path = path
        self.lock = Lock()
        self.values = valid_settings(read_json(path, {}))
        self.api_key = None

    def public(self):
        with self.lock:
            return {"values": dict(self.values), "has_key": bool(self.api_key or self.environment_key())}

    def environment_key(self):
        return os.getenv("ALPHAX_LLM_API_KEY" if self.values["api_base"] else "DEEPSEEK_API_KEY")

    def save(self, payload):
        values = valid_settings(payload.get("values"))
        key = payload.get("api_key")
        if key is not None and (not isinstance(key, str) or len(key) > 4096 or any(c.isspace() for c in key)):
            raise ValueError("API 密钥格式无效。")
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, delete=False) as file:
                    temporary = Path(file.name)
                    json.dump(values, file, ensure_ascii=False, indent=2)
                    file.write("\n")
                temporary.replace(self.path)
            finally:
                if temporary:
                    temporary.unlink(missing_ok=True)
            if key is None and self.values["api_base"] != values["api_base"]:
                self.api_key = None
            self.values = values
            if key is not None:
                self.api_key = key or None
        return self.public()

    def snapshot(self):
        with self.lock:
            return dict(self.values), self.api_key or self.environment_key()


def local_path(value):
    return (ROOT / Path(value).expanduser()).resolve()


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def rows(path):
    payload = read_json(path, [])
    if isinstance(payload, dict):
        payload = payload.get("candidates", [])
    return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []


def factor_rows(directory):
    candidates = rows(directory / "candidates.json")
    accepted_rows = rows(directory / "accepted_factors.json")
    library = read_json(directory / "factor_library.json", {})
    library_rows = library.get("factors", []) if isinstance(library, dict) else library
    if isinstance(library_rows, list):
        accepted_rows.extend(row for row in library_rows if isinstance(row, dict))
    exploratory = {row.get("candidate_id") for row in accepted_rows
                   if row.get("strict_paper_accepted") is False}
    accepted = {row.get("candidate_id") for row in accepted_rows} - exploratory
    research = {row.get("candidate_id") for row in rows(directory / "research_assessments.json")
                if row.get("research_eligible") is True} | exploratory
    pool = read_json(directory / "research_pool.json", {})
    by_horizon = pool.get("candidates_by_horizon", {}) if isinstance(pool, dict) else {}
    pool_definitions = []
    metrics = {}
    for horizon, entries in by_horizon.items():
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("definition"), dict):
                continue
            definition = entry["definition"]
            candidate_id = definition.get("candidate_id")
            if not isinstance(candidate_id, str):
                continue
            pool_definitions.append(definition)
            research.add(candidate_id)
            if horizon == "1":
                metrics[candidate_id] = entry.get("metrics", {}).get("validation", {})
    definitions = {row["candidate_id"]: row for row in pool_definitions + accepted_rows + candidates
                   if isinstance(row.get("candidate_id"), str) and row["candidate_id"]}
    evaluated = set()
    try:
        with (directory / "candidate_metrics.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                candidate_id = row.get("candidate_id")
                evaluated.add(candidate_id)
                if str(row.get("accepted")).lower() == "true" and candidate_id not in exploratory:
                    accepted.add(candidate_id)
                if row.get("split") == "validation" and row.get("horizon", "1") == "1":
                    metrics[candidate_id] = row
    except (OSError, UnicodeError, csv.Error):
        pass
    for candidate_id, definition in definitions.items():
        grade = ("strict" if candidate_id in accepted else "research" if candidate_id in research
                 else "unadmitted" if candidate_id in evaluated else "unknown")
        expression = definition.get("expression_spec") or definition.get("expression")
        features = definition.get("features")
        if not isinstance(features, list) or not all(isinstance(feature, str) for feature in features):
            continue
        candidate = {**definition, "expression": expression}
        display = definition.get("expression_text") or expression
        if not isinstance(display, (str, dict)):
            operation = definition.get("op")
            if not isinstance(operation, str) or not operation:
                continue
            display = f"{operation}({', '.join(features)})"
        yield candidate, {
            "id": candidate_id, "grade": grade,
            "expression": display if isinstance(display, str) else json.dumps(display, ensure_ascii=False),
            "hypothesis": str(definition.get("hypothesis", "")),
            "rank_ic": number(metrics.get(candidate_id, {}).get("mean_daily_rank_ic")),
        }


def library_id(directory):
    return hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:24]


def libraries(settings):
    visited = set()
    for root in (local_path(settings["library_root"]), OUTPUTS):
        for directory, children, filenames in os.walk(root):
            children[:] = sorted(name for name in children if not name.startswith(".")
                                 and name not in ("snapshots", "logs", "checkpoints"))
            definitions = {"candidates.json", "accepted_factors.json", "factor_library.json", "research_pool.json"}
            if definitions.isdisjoint(filenames):
                continue
            path = Path(directory).resolve()
            if path in visited:
                continue
            visited.add(path)
            if path.is_relative_to(OUTPUTS) and path.name == "miner" and path.parent.parent == OUTPUTS:
                if read_json(path.parent / "miner-status.json", {}).get("status") != "passed":
                    continue
            factors = list(factor_rows(path))
            if not factors:
                continue
            name = str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
            yield library_id(path), name, factors


def catalog(settings):
    return [{"id": uid, "name": name, "factors": [display for _, display in factors]}
            for uid, name, factors in libraries(settings)]


def option_args(values, names):
    return [part for name in names for part in ("--" + name.replace("_", "-"), str(values[name]))]


def prepare_job(kind, payload, output, settings, api_key):
    panel = local_path(settings["panel"])
    calendar = local_path(settings["calendar"])
    if not panel.is_file() or not calendar.is_file():
        raise ValueError("未找到特征面板或交易日历，请检查系统设置。")
    shared = ["--input", str(panel), "--calendar", str(calendar), "--require-explicit-label-dates"]
    if kind == "miner":
        brief = payload.get("brief")
        if not isinstance(brief, str) or not 1 <= len(brief.strip()) <= 4000:
            raise ValueError("请输入 1–4000 字的研究假设。")
        if not api_key:
            raise ValueError("请先在系统设置中配置 LLM API 密钥。")
        try:
            with calendar.open(encoding="utf-8-sig", newline="") as stream:
                sessions = sorted({date.fromisoformat(row["date"]) for row in csv.DictReader(stream)
                                   if settings["train_start"] <= row["date"] <= settings["initial_as_of"]})
        except (KeyError, ValueError, OSError) as exc:
            raise ValueError("交易日历必须包含 YYYY-MM-DD 格式的 date 列。") from exc
        admission = settings["admission_sessions"]
        if len(sessions) < admission + settings["min_train_sessions"]:
            raise ValueError("交易日历在截点前的交易日不足以覆盖训练和准入窗口。")
        output.mkdir(parents=True)
        brief_file = output / "research-brief.txt"
        brief_file.write_text(brief.strip() + "\n", encoding="utf-8")
        cutoff = sessions[-1]
        future = cutoff + timedelta(days=1)
        command = [sys.executable, str(AC / "miner_lite.py"), *shared,
                   "--output-dir", str(output / "miner"), "--as-of", str(cutoff),
                   "--train-start", settings["train_start"], "--train-end", str(sessions[-admission-1]),
                   "--validation-start", str(sessions[-admission]), "--validation-end", str(cutoff),
                   "--backtest-start", str(future), "--backtest-end", str(future),
                   "--llm-model", settings["model"], "--llm-agents", str(settings["agents"]),
                   "--llm-count", str(settings["candidates"]), "--search-rounds", str(settings["rounds"]),
                   "--search-budget", str(settings["candidates"] * settings["rounds"]),
                   "--search-batch-size", str(settings["candidates"]),
                   "--research-brief-file", str(brief_file)]
        if settings["api_base"]:
            env = {"ALPHAX_LLM_API_BASE": settings["api_base"], "ALPHAX_LLM_API_KEY": api_key}
        else:
            env = {"ALPHAX_LLM_API_BASE": "", "DEEPSEEK_API_KEY": api_key}
        return command, env, [api_key]
    selected_library = payload.get("library")
    grade = payload.get("grade", "")
    if not isinstance(selected_library, str) or not isinstance(grade, str) or grade not in ("", *GRADES):
        raise ValueError("请选择有效的因子来源和质量范围。")
    name, factors = next(((name, factors) for uid, name, factors in libraries(settings)
                          if uid == selected_library), ("", []))
    candidates = [candidate for candidate, display in factors if not grade or display["grade"] == grade]
    if not candidates:
        raise ValueError("此来源和质量范围下没有可提交的因子。")
    output.mkdir(parents=True)
    candidate_file = output / "fixed-candidates.json"
    candidate_file.write_text(json.dumps(candidates, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    source = {"name": name, "grade": grade, "count": len(candidates)}
    (output / "factor-source.json").write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
    command = [sys.executable, str(AC / "run_pipeline.py"), *shared,
               "--candidate-file", str(candidate_file), "--output-dir", str(output / "backtest"),
               "--allow-feature-warmup-sessions", str(settings["warmup_sessions"]),
               *option_args(settings, (*DATE_KEYS, "admission_sessions", "min_train_sessions",
                                        "maintenance_days", "cost_rate", "configs"))]
    return command, {}, []


def backtest_result(path):
    if read_json(path / "status.json", {}).get("status") != "completed":
        return None
    summary = read_json(path / "summary.json", {})
    if not isinstance(summary.get("holdout"), dict):
        return None
    points = []
    nav = benchmark = 1.0
    with (path / "trader/holdout_returns.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            nav *= 1 + float(row["net_return"])
            benchmark *= 1 + float(row["benchmark_return"])
            points.append({"date": row["date"], "nav": nav, "benchmark": benchmark})
    return {"metrics": summary["holdout"], "selected_config": summary.get("selected_config"),
            "invested_days": summary.get("holdout_invested_days"), "series": points,
            "source": read_json(path.parent / "factor-source.json", {})}


def completed_backtests():
    paths = [path for path in OUTPUTS.glob("*/backtest")
             if read_json(path / "status.json", {}).get("status") == "completed"]
    return [path.parent.name for path in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)]
