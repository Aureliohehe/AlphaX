"""Command-line arguments for Miner-lite."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser(base: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or evaluate AlphaCrafter factor candidates.",
        epilog="Default: generate template candidates. Use --llm-model for LangGraph proposals "
               "or --candidate-file for fixed candidates. See list.md for command examples.",
    )

    files_args = parser.add_argument_group("Input and output")
    files_args.add_argument("--input", type=Path, default=base / "data" / "processed" / "sp500_features.csv.gz")
    files_args.add_argument(
        "--output-dir",
        type=Path,
        default=base / "data" / "qlib" / "alphacrafter_miner_lite",
    )
    files_args.add_argument("--calendar", type=Path, help="Reference session CSV; validate label horizons against market sessions")

    candidates_args = parser.add_argument_group("Candidate source and return ranking")
    candidates_args.add_argument("--candidate-file", type=Path, default=None, help="Evaluate candidates from this JSON file")
    candidates_args.add_argument("--factor-return-model", type=Path,
                                 help="Rank by predicted 24-month net return; requires --candidate-file and --as-of")
    candidates_args.add_argument("--max-pairs", type=int, default=60)
    candidates_args.add_argument(
        "--max-temporal",
        type=int,
        default=20,
        help="Maximum deterministic candidates using the safe time-series DSL",
    )

    llm_args = parser.add_argument_group("LLM proposal generation")
    llm_args.add_argument(
        "--llm-model",
        default=None,
        help=(
            "Optional model; deepseek-chat/deepseek-reasoner use DEEPSEEK_API_KEY "
            "directly, other names use LiteLLM"
        ),
    )
    llm_args.add_argument("--llm-count", type=int, default=40,
                          help="Single-round candidate budget (default: %(default)s)")
    llm_args.add_argument(
        "--llm-agents",
        type=int,
        default=3,
        help="Number of specialist Miner agents used with --llm-model (default: %(default)s)",
    )
    llm_args.add_argument("--research-brief-file", type=Path,
                        help="UTF-8 research brief for LLM proposals (up to 4000 characters)")
    llm_args.add_argument("--llm-mechanism-first", action="store_true",
                        help="Freeze falsifiable economic mechanisms before generating candidate expressions")
    llm_args.add_argument(
        "--llm-orchestrator",
        choices=("langgraph",),
        default="langgraph",
        help="Optional explicit selection of the LangGraph backend",
    )

    search_args = parser.add_argument_group("Multi-round training search")
    search_args.add_argument("--search-rounds", type=int, default=1,
                             help="Search rounds (default: %(default)s); more than one requires --llm-model")
    search_args.add_argument("--search-budget", type=int, default=40, help="Total proposal slots across feedback rounds")
    search_args.add_argument("--search-batch-size", type=int, default=10,
                             help="Proposal slots per round (default: %(default)s)")
    search_args.add_argument("--search-folds", type=int, default=3,
                             help="Rolling training folds (default: %(default)s; minimum: 2)")
    search_args.add_argument("--search-reserve", type=int, default=0,
                        help="Proposal slots within search-budget to replace missing candidates with new hypotheses")

    feedback_args = parser.add_argument_group("Targeted research and feedback")
    feedback_args.add_argument("--llm-seed-file", type=Path,
                        help="JSON controls for single-round LLM substitution research")
    feedback_args.add_argument("--llm-factor-mine-skill-dir", type=Path,
                        help="Local upstream factor-mine skill checkout, for one LLM proposal round")
    feedback_args.add_argument("--llm-xgboost-complement", action="store_true",
                        help="Ask an unseeded Miner for complementary inputs to the four-factor XGBoost model")
    feedback_args.add_argument("--llm-weekly-feedback-dir", type=Path,
                        help="Completed earlier Trader research to inform seeded LangGraph proposals")
    feedback_args.add_argument("--llm-weekly-xgboost-feedback", type=Path,
                        help="Completed paired XGBoost replay summary for seeded or complementary proposals")

    memory_args = parser.add_argument_group("Research memory")
    memory_args.add_argument("--research-market", choices=("us", "cn", "unknown"), default="unknown",
                        help="Market metadata for the separate horizon-specific research pool")
    memory_args.add_argument("--memory-db", type=Path, help="SQLite research evidence to recall and update")
    memory_args.add_argument("--memory-universe", help="Stable identifier for the stock universe and membership policy")
    memory_args.add_argument("--memory-limit", type=int, default=6, help="Maximum historical evidence records per prompt (1–20)")

    windows_args = parser.add_argument_group("Dates and label policy")
    windows_args.add_argument(
        "--as-of",
        default=None,
        help="Information cutoff at session close; future rows and unsettled labels are invisible",
    )
    windows_args.add_argument("--train-start", default="2016-01-04")
    windows_args.add_argument("--train-end", default="2022-12-30")
    windows_args.add_argument("--validation-start", default="2023-01-03")
    windows_args.add_argument("--validation-end", default="2023-12-29")
    windows_args.add_argument("--backtest-start", default="2024-01-02")
    windows_args.add_argument("--backtest-end", default="2026-02-27")
    windows_args.add_argument("--label-mode", choices=("execution", "paper_close"), default="execution",
                        help="Execution-aligned T+1 open to T+h close (default), or paper-style T close to T+h close for Miner diagnostics")
    windows_args.add_argument("--require-explicit-label-dates", action="store_true", help="Reject legacy feature-row label-date inference")

    admission_args = parser.add_argument_group("Admission thresholds")
    admission_args.add_argument(
        "--min-train-ic",
        type=float,
        default=None,
        help="Optional override for paper IC and Rank IC thresholds on training data",
    )
    admission_args.add_argument(
        "--min-valid-ic",
        type=float,
        default=None,
        help="Optional override for paper IC and Rank IC thresholds on validation data",
    )
    admission_args.add_argument(
        "--min-train-icir",
        type=float,
        default=None,
        help="Optional override for paper ICIR and Rank ICIR thresholds on training data",
    )
    admission_args.add_argument(
        "--min-valid-icir",
        type=float,
        default=None,
        help="Optional override for paper ICIR and Rank ICIR thresholds on validation data",
    )
    admission_args.add_argument(
        "--min-train-hit",
        type=float,
        default=None,
        help="Optional directional hit-ratio floor for training data (paper default: 0.60)",
    )
    admission_args.add_argument(
        "--min-valid-hit",
        type=float,
        default=None,
        help="Optional directional hit-ratio floor for validation data (paper default: 0.60)",
    )
    admission_args.add_argument(
        "--min-coverage",
        type=float,
        default=0.90,
        help="Minimum train/validation signal coverage (paper threshold: 0.90)",
    )
    admission_args.add_argument(
        "--max-turnover",
        type=float,
        default=0.40,
        help="Maximum train/validation cross-sectional rank turnover (paper threshold: 0.40)",
    )
    admission_args.add_argument(
        "--novelty-baseline",
        type=Path,
        default=None,
        help="Optional candidate JSON/AST library used as the external novelty baseline",
    )

    library_args = parser.add_argument_group("Factor library and maintenance")
    library_args.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Persistent factor-library registry; defaults to <output-dir>/registry.json",
    )
    library_args.add_argument("--snapshot-dir", type=Path, default=None, help="Write-once archive; defaults to <output-dir>/snapshots")
    library_args.add_argument(
        "--maintenance-interval-days",
        type=int,
        default=90,
        help="Revalidate active factors after this many days (paper default: 90)",
    )
    return parser
