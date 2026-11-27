"""Local AlphaX dashboard and launcher for repository validation tasks."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import research


WEB = Path(__file__).resolve().parent
ROOT = WEB.parents[1]
AC = Path("quant-method/alphacrafter")
ML = Path("quant-method/machine-learn")
RUNS = ROOT / ML / "data/qlib"
STATIC = WEB / "static"
OUTPUTS = WEB / ".runs"
SOFTWARE = sys.executable

SOURCES = {
    "holdout": "alphacrafter_positive_holdout_20261117",
    "paired": "llm_paired_feedback_mSKURa",
    "followup": "llm_frozen_2026_0UhKTx",
    "search": "alphacrafter_feedback_search_v1",
    "mechanism": "llm_three_round_mechanisms_DcgAyJ",
}


def read_json(key, name):
    try:
        return json.loads((RUNS / SOURCES[key] / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_returns(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_series():
    try:
        directory = RUNS / SOURCES["holdout"]
        files = [read_returns(directory / f"cost{cost}_returns.csv") for cost in (1, 2)]
        if [row["date"] for row in files[0]] != [row["date"] for row in files[1]]:
            return None
        matched = 1.0
        points = []
        for standard, stress in zip(*files):
            matched *= 1 + float(standard["matched_index_net_return"])
            points.append({
                "date": standard["date"], "standard": float(standard["nav"]),
                "stress": float(stress["nav"]), "benchmark": matched,
            })
        return points
    except (OSError, ValueError, KeyError):
        return None


def dashboard():
    summary = read_json("holdout", "summary.json")
    status = read_json("holdout", "status.json")
    holds = None
    if summary and status and status.get("status") == "completed_research_holdout":
        holds = {"trials": summary.get("trials", []), "series": read_series(),
                 "admitted": summary.get("formal_strategy_admitted"), "source": SOURCES["holdout"]}
    paired = read_json("paired", "replay_summary.json")
    paired_result = None
    if paired:
        rows = paired.get("rounds", {}).get("feedback", {}).get("variants", [])
        best = next((row for row in rows if row.get("id") == "low_vol60_tsmean_120"), None)
        if best:
            paired_result = {"window": paired["window"], "training": paired["training_window"],
                             "variant": best, "followup": read_json("followup", "decision.json"),
                             "source": SOURCES["paired"]}
    search = read_json("search", "analysis.json")
    mechanism = read_json("mechanism", "frequency_counts.json")
    if mechanism:
        mechanism = mechanism["all"]["eligible_counts"]
    return {"holdout": holds, "paired": paired_result, "search": search, "mechanism": mechanism}


def commands(output):
    panel = ML / "data/processed/sp500_features_label_dates.csv.gz"
    calendar = ML / "data/processed/sp500_xnys_xcal_4_13_3.csv"
    return {
        "demo": [SOFTWARE, str(AC / "run_offline_example.py"), "--output-dir", str(output)],
        "smoke": ["make", "test-smoke", f"PYTHON={SOFTWARE}"],
        "preflight": [
            SOFTWARE, str(AC / "run_pipeline.py"), "--input", str(panel), "--calendar", str(calendar),
            "--require-explicit-label-dates", "--allow-feature-warmup-sessions", "120",
            "--output-dir", str(output), "--dry-run",
        ],
    }


class Jobs:
    def __init__(self):
        self.lock = threading.Lock()
        self.entries = {}

    def list(self):
        with self.lock:
            entries = [{**entry, "lines": list(entry["lines"])}
                       for entry in reversed(self.entries.values())]
        for entry in entries:
            if entry["kind"] == "backtest":
                entry["progress"] = self.pipeline_progress(entry["id"])
            if entry["kind"] == "miner" and entry["status"] == "passed":
                entry["library"] = research.library_id(OUTPUTS / entry["id"] / "miner")
        return entries

    @staticmethod
    def pipeline_progress(uid):
        try:
            path = OUTPUTS / uid / "backtest/events.jsonl"
            with path.open(encoding="utf-8") as stream:
                lines = deque(stream, maxlen=200)
            events = []
            for line in lines:
                if line.endswith("\n"):
                    event = json.loads(line)
                    if event.get("event") in {"step_started", "step_finished"}:
                        label = "开始" if event["event"] == "step_started" else "结束"
                        events.append(f"{label} {event.get('name', '')}")
                    elif event.get("event") == "trader_frozen":
                        events.append("策略配置已冻结")
            return events
        except (OSError, ValueError):
            return []

    def start(self, kind, payload=None, settings=None):
        with self.lock:
            if any(entry["status"] == "running" for entry in self.entries.values()):
                raise RuntimeError("已有任务在运行，请等待完成。")
            uid = secrets.token_hex(6)
            output = OUTPUTS / uid
            if kind in ("miner", "backtest"):
                values, api_key = settings.snapshot()
                command, env, redactions = research.prepare_job(kind, payload, output, values, api_key)
            else:
                command, env, redactions = commands(output)[kind], {}, []
            if len(self.entries) == 20:
                del self.entries[next(iter(self.entries))]
            entry = {
                "id": uid, "kind": kind, "status": "running", "started": time.time(),
                "finished": None, "exit_code": None, "lines": deque(maxlen=300),
                "command": shlex.join(command),
                "output": str(output.relative_to(ROOT)) if kind in ("demo", "miner", "backtest") else None,
            }
            self.entries[uid] = entry
        threading.Thread(target=self.run, args=(entry, command, env, redactions), daemon=True).start()
        return uid

    def run(self, entry, command, env, redactions):
        def redact(line):
            for secret in redactions:
                line = line.replace(secret, "[已隐藏]")
            return line

        code = 1
        try:
            with subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  env={**os.environ, **env, "PYTHONUNBUFFERED": "1",
                                       "PYTHONDONTWRITEBYTECODE": "1"},
                                  encoding="utf-8", errors="replace") as process:
                for line in process.stdout:
                    with self.lock:
                        entry["lines"].append(redact(line).rstrip()[:2000])
                code = process.wait()
        except OSError as error:
            with self.lock:
                entry["lines"].append(redact(str(error)))
        finally:
            if entry["kind"] == "miner":
                try:
                    status = {"status": "passed" if code == 0 else "failed", "exit_code": code}
                    (OUTPUTS / entry["id"] / "miner-status.json").write_text(
                        json.dumps(status), encoding="utf-8")
                except OSError as error:
                    code = 1
                    with self.lock:
                        entry["lines"].append(str(error))
            with self.lock:
                entry.update(status="passed" if code == 0 else "failed",
                             exit_code=code, finished=time.time())


class RequestHandler(BaseHTTPRequestHandler):
    jobs = Jobs()
    settings = None
    assets = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/workbench.js": ("workbench.js", "text/javascript; charset=utf-8"),
        "/style.css": ("style.css", "text/css; charset=utf-8"),
    }

    def send_bytes(self, payload, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, payload, status=200):
        self.send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                        "application/json; charset=utf-8", status)

    def local_request(self):
        hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        return self.headers.get("Host") in hosts

    def do_GET(self):
        if not self.local_request():
            return self.send_json({"error": "请通过本机地址访问。"}, 403)
        if self.path in self.assets:
            name, content_type = self.assets[self.path]
            try:
                payload = (STATIC / name).read_bytes()
            except FileNotFoundError:
                return self.send_json({"error": "请先在 report/web 运行 npm ci 和 npm run build。"}, 503)
            return self.send_bytes(payload, content_type)
        if self.path == "/favicon.ico":
            return self.send_bytes(b"", "image/x-icon", 204)
        if self.path == "/api/dashboard":
            return self.send_json(dashboard())
        if self.path == "/api/jobs":
            return self.send_json(self.jobs.list())
        if self.path == "/api/settings":
            return self.send_json(self.settings.public())
        if self.path == "/api/factors":
            return self.send_json(research.catalog(self.settings.public()["values"]))
        if self.path == "/api/backtests":
            return self.send_json(research.completed_backtests())
        path = urlsplit(self.path)
        if path.path == "/api/backtest":
            uid = parse_qs(path.query).get("id", [""])[0]
            if re.fullmatch(r"[0-9a-f]{12}", uid):
                try:
                    result = research.backtest_result(OUTPUTS / uid / "backtest")
                except (OSError, ValueError, KeyError, TypeError):
                    return self.send_json({"error": "回测产物缺失或格式无效。"}, 422)
                if result is not None:
                    return self.send_json(result)
        self.send_json({"error": "页面或接口不存在。"}, 404)

    def do_POST(self):
        if not self.local_request() or self.headers.get("Origin") != f"http://{self.headers.get('Host')}":
            return self.send_json({"error": "仅允许本地页面发起任务。"}, 403)
        if self.path not in ("/api/jobs", "/api/settings"):
            return self.send_json({"error": "接口不存在。"}, 404)
        if self.headers.get_content_type() != "application/json":
            return self.send_json({"error": "请使用 JSON 请求。"}, 415)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 32768:
                raise ValueError
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError
        except (ValueError, TypeError):
            return self.send_json({"error": "JSON 请求无效。"}, 400)
        if self.path == "/api/settings":
            try:
                return self.send_json(self.settings.save(payload))
            except ValueError as error:
                return self.send_json({"error": str(error)}, 400)
            except OSError:
                return self.send_json({"error": "无法保存本地设置。"}, 500)
        kind = payload.get("kind")
        if not isinstance(kind, str) or kind not in (*commands(OUTPUTS), "miner", "backtest"):
            return self.send_json({"error": "任务类型无效。"}, 400)
        try:
            uid = self.jobs.start(kind, payload, self.settings)
        except RuntimeError as error:
            return self.send_json({"error": str(error)}, 409)
        except ValueError as error:
            return self.send_json({"error": str(error)}, 400)
        except OSError:
            return self.send_json({"error": "无法准备任务目录。"}, 500)
        self.send_json({"id": uid}, 201)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not (STATIC / "workbench.js").is_file():
        parser.error("Build the React app first: cd report/web && npm ci && npm run build")
    RequestHandler.settings = research.Settings(OUTPUTS / "settings.json")
    with ThreadingHTTPServer(("127.0.0.1", args.port), RequestHandler) as server:
        print(f"AlphaX http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
