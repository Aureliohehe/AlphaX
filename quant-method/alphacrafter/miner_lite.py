#!/usr/bin/env python3
"""AlphaCrafter Miner-lite.

This module implements the first, deterministic version of the Miner Harness:

1. Generate safe candidate factor expressions from the existing feature schema.
2. Evaluate them with cross-sectional Rank IC and portfolio diagnostics.
3. Accept candidates using a train/validation gate and keep 2024+ as holdout.

An LLM is intentionally not required.  Later, an LLM can provide a JSON file
with the same candidate schema used by ``--candidate-file``; the validator and
registry remain unchanged.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


BASELINE_FEATURES = {"return_1d", "one_day_reversal"}
META_COLUMNS = {"date", "ticker", "target_1d", "split"}
ALLOWED_OPS = {"feature", "difference", "sum"}

# Related features are paired only within a family.  This keeps the first
# search interpretable and prevents a combinatorial explosion.
FEATURE_FAMILIES = {
    "momentum": [
        "return_5d",
        "return_10d",
        "return_20d",
        "return_60d",
        "sma_5_ratio",
        "sma_10_ratio",
        "sma_20_ratio",
        "sma_60_ratio",
        "ema_12_ratio",
        "ema_26_ratio",
    ],
    "range_reversal": [
        "open_close_return",
        "close_location",
        "stochastic_14",
        "bollinger_z",
        "drawdown_from_high_5d",
        "drawdown_from_high_20d",
        "distance_from_low_5d",
        "distance_from_low_20d",
    ],
    "volatility": [
        "volatility_5d",
        "volatility_20d",
        "volatility_60d",
        "atr_14_ratio",
        "bollinger_width",
    ],
    "volume": [
        "volume_ratio_20d",
        "volume_change_1d",
        "volume_volatility_20d",
    ],
}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    op: str
    features: Tuple[str, ...]
    hypothesis: str

    @property
    def expression(self) -> str:
        if self.op == "feature":
            return f"rank({self.features[0]})"
        if self.op == "difference":
            return f"rank({self.features[0]}) - rank({self.features[1]})"
        if self.op == "sum":
            return f"rank({self.features[0]}) + rank({self.features[1]})"
        raise ValueError(f"Unsupported operation: {self.op}")


@dataclass
class Metric:
    rows: int
    mean_daily_rank_ic: float
    daily_rank_ic_std: float
    rank_icir: float
    rank_ic_hit_ratio: float
    mean_top_bottom_decile_spread: float


def candidate_from_dict(value: Dict[str, object]) -> Candidate:
    """Validate the JSON boundary used by future LLM candidate generators."""
    op = str(value.get("op", ""))
    features_value = value.get("features", [])
    if not isinstance(features_value, list) or not all(isinstance(x, str) for x in features_value):
        raise ValueError("candidate.features must be a list of strings")
    candidate = Candidate(
        candidate_id=str(value.get("candidate_id", value.get("id", ""))),
        op=op,
        features=tuple(features_value),
        hypothesis=str(value.get("hypothesis", "")),
    )
    validate_candidate(candidate)
    return candidate


def validate_candidate(candidate: Candidate, available: Optional[Iterable[str]] = None) -> None:
    if not candidate.candidate_id:
        raise ValueError("candidate_id cannot be empty")
    if candidate.op not in ALLOWED_OPS:
        raise ValueError(f"Unsupported op: {candidate.op}")
    expected = 1 if candidate.op == "feature" else 2
    if len(candidate.features) != expected:
        raise ValueError(f"{candidate.op} expects {expected} features")
    if available is not None:
        missing = sorted(set(candidate.features) - set(available))
        if missing:
            raise ValueError(f"Candidate uses unavailable features: {missing}")


def template_candidates(available: Sequence[str], max_pairs: int) -> List[Candidate]:
    """Generate an interpretable, deterministic candidate set."""
    available_set = set(available)
    candidates: List[Candidate] = []
    for feature in available:
        candidates.append(
            Candidate(
                candidate_id=f"single__{feature}",
                op="feature",
                features=(feature,),
                hypothesis=f"Cross-sectional rank of {feature} has incremental predictive information.",
            )
        )

    pairs: List[Candidate] = []
    for family, family_features in FEATURE_FAMILIES.items():
        usable = [feature for feature in family_features if feature in available_set]
        for left_index, left in enumerate(usable):
            for right in usable[left_index + 1 :]:
                pairs.append(
                    Candidate(
                        candidate_id=f"spread__{left}__{right}",
                        op="difference",
                        features=(left, right),
                        hypothesis=(
                            f"The relative {family} signal {left} versus {right} "
                            "contains more information than either level alone."
                        ),
                    )
                )

    # Stable ordering makes experiments reproducible.  The limit is a safety
    # valve for future schemas with many more fields.
    candidates.extend(sorted(pairs, key=lambda item: item.candidate_id)[:max_pairs])
    return candidates


def load_candidates(path: Path, available: Sequence[str]) -> List[Candidate]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("candidates", [])
    if not isinstance(payload, list):
        raise ValueError("Candidate file must contain a list or {'candidates': [...]}")
    candidates = [candidate_from_dict(value) for value in payload]
    for candidate in candidates:
        validate_candidate(candidate, available)
    return candidates


def build_market_context(frame: pd.DataFrame, train_mask: pd.Series) -> str:
    """Create a compact, point-in-time context for an optional LLM proposer."""
    history = frame.loc[train_mask].sort_values("date")
    if history.empty:
        return "No historical context available."
    latest_date = history["date"].max()
    recent = history[history["date"] > latest_date - pd.Timedelta(days=60)]
    fields = [
        "return_1d",
        "return_5d",
        "return_20d",
        "volatility_20d",
        "volume_ratio_20d",
        "intraday_range",
    ]
    summary = {field: float(recent[field].median()) for field in fields if field in recent}
    return json.dumps(
        {
            "as_of": str(latest_date.date()),
            "window": "last 60 calendar days available in training data",
            "cross_sectional_medians": summary,
        },
        ensure_ascii=False,
    )


def generate_llm_candidates(
    model: str, available: Sequence[str], market_context: str, count: int
) -> List[Candidate]:
    """Use LiteLLM only when explicitly requested; templates remain the default."""
    try:
        from litellm import completion
    except ImportError as exc:
        raise SystemExit(
            "--llm-model requires LiteLLM. Install it with: python3 -m pip install litellm"
        ) from exc

    prompt = f"""You are the Miner component of a cross-sectional quantitative research system.
Generate exactly {count} candidate factors using only the approved feature names and operations.
Do not use future data, prices after the as_of date, arbitrary Python, or unavailable fields.
A candidate must be a JSON object with:
  candidate_id: short unique snake_case string
  op: one of feature, difference, sum
  features: one feature for feature, exactly two for difference/sum
  hypothesis: one sentence
The validator will calculate all metrics. Return only a JSON array.
Approved features: {json.dumps(list(available), ensure_ascii=False)}
Approved operations: feature, difference, sum
Point-in-time market context: {market_context}
"""
    response = completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Return machine-readable JSON and respect the supplied schema.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=4000,
    )
    content = response.choices[0].message.content
    if not isinstance(content, str):
        raise ValueError("LLM response content is not text")
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    payload = json.loads(content)
    if not isinstance(payload, list):
        payload = payload.get("candidates", []) if isinstance(payload, dict) else []
    candidates = [candidate_from_dict(value) for value in payload]
    for candidate in candidates:
        validate_candidate(candidate, available)
    return candidates


def evaluate_score(
    dates: pd.Series, score: pd.Series, target: pd.Series, mask: pd.Series
) -> Metric:
    values = pd.DataFrame(
        {"date": dates, "score": score, "target": target}, index=dates.index
    ).loc[mask].dropna()
    daily_ic: List[float] = []
    spreads: List[float] = []
    for _, group in values.groupby("date", sort=False):
        if len(group) < 3 or group["score"].nunique() < 2 or group["target"].nunique() < 2:
            continue
        # Rank the score and target inside each trading day.  This is
        # equivalent to Spearman correlation and avoids scipy as a dependency.
        score_rank = group["score"].rank(method="average").to_numpy(dtype=float)
        target_rank = group["target"].rank(method="average").to_numpy(dtype=float)
        score_rank -= score_rank.mean()
        target_rank -= target_rank.mean()
        denominator = np.sqrt(np.dot(score_rank, score_rank) * np.dot(target_rank, target_rank))
        if denominator == 0:
            continue
        daily_ic.append(float(np.dot(score_rank, target_rank) / denominator))
        count = max(1, len(group) // 10)
        ordered = group.sort_values("score")
        spreads.append(
            float(ordered["target"].iloc[-count:].mean() - ordered["target"].iloc[:count].mean())
        )

    ic = pd.Series(daily_ic, dtype=float).dropna()
    spread = pd.Series(spreads, dtype=float).dropna()
    if ic.empty:
        return Metric(0, np.nan, np.nan, np.nan, np.nan, np.nan)
    std = float(ic.std())
    return Metric(
        rows=int(len(values)),
        mean_daily_rank_ic=float(ic.mean()),
        daily_rank_ic_std=std,
        rank_icir=float(ic.mean() / std) if std else np.nan,
        rank_ic_hit_ratio=float((ic > 0).mean()),
        mean_top_bottom_decile_spread=float(spread.mean()),
    )


def evaluate_candidate_score(candidate: Candidate, ranked: pd.DataFrame) -> pd.Series:
    validate_candidate(candidate, ranked.columns)
    if candidate.op == "feature":
        return ranked[candidate.features[0]]
    left = ranked[candidate.features[0]]
    right = ranked[candidate.features[1]]
    if candidate.op == "difference":
        return left - right
    if candidate.op == "sum":
        return left + right
    raise ValueError(f"Unsupported operation: {candidate.op}")


def metric_dict(metric: Metric) -> Dict[str, float]:
    return asdict(metric)


def main() -> None:
    base = Path(__file__).resolve().parents[1] / "machine-learn"
    parser = argparse.ArgumentParser(description="Run the deterministic Miner-lite harness")
    parser.add_argument("--input", type=Path, default=base / "data" / "processed" / "sp500_features.csv.gz")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "data" / "qlib" / "alphacrafter_miner_lite",
    )
    parser.add_argument("--candidate-file", type=Path, default=None)
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Optional LiteLLM model, e.g. anthropic/claude-sonnet-4-5",
    )
    parser.add_argument("--llm-count", type=int, default=40)
    parser.add_argument("--max-pairs", type=int, default=60)
    parser.add_argument("--min-train-ic", type=float, default=0.010)
    parser.add_argument("--min-valid-ic", type=float, default=0.005)
    parser.add_argument("--min-valid-hit", type=float, default=0.51)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, compression="infer", parse_dates=["date"])
    frame["one_day_reversal"] = -frame["return_1d"]
    available = [column for column in frame.columns if column not in META_COLUMNS | BASELINE_FEATURES]
    frame = frame.dropna(subset=available + ["target_1d"]).reset_index(drop=True)
    dates = frame["date"]
    target = frame["target_1d"]

    train_mask = dates.dt.year <= 2022
    valid_mask = dates.dt.year == 2023
    backtest_mask = dates.dt.year >= 2024

    if args.candidate_file:
        candidates = load_candidates(args.candidate_file, available)
        generator = "external_json"
    elif args.llm_model:
        candidates = generate_llm_candidates(
            args.llm_model,
            available,
            build_market_context(frame, train_mask),
            args.llm_count,
        )
        generator = f"llm:{args.llm_model}"
    else:
        candidates = template_candidates(available, args.max_pairs)
        generator = "template_v1"

    # Cross-sectional rank normalization is computed once for all source fields.
    ranked = frame[available].groupby(dates, sort=False).rank(pct=True).astype("float32")
    baseline = frame["one_day_reversal"].groupby(dates, sort=False).rank(pct=True)
    results: List[Dict[str, object]] = []
    accepted: List[Dict[str, object]] = []
    for candidate in candidates:
        score = evaluate_candidate_score(candidate, ranked)
        train = evaluate_score(dates, score, target, train_mask)
        valid = evaluate_score(dates, score, target, valid_mask)
        backtest = evaluate_score(dates, score, target, backtest_mask)
        direction = 1 if train.mean_daily_rank_ic >= 0 else -1
        directional_valid_ic = direction * valid.mean_daily_rank_ic
        directional_valid_hit = (
            valid.rank_ic_hit_ratio if direction > 0 else 1.0 - valid.rank_ic_hit_ratio
        )
        accepted_flag = (
            abs(train.mean_daily_rank_ic) >= args.min_train_ic
            and directional_valid_ic >= args.min_valid_ic
            and directional_valid_hit >= args.min_valid_hit
        )
        row = {
            "candidate_id": candidate.candidate_id,
            "expression": candidate.expression,
            "op": candidate.op,
            "features": ",".join(candidate.features),
            "hypothesis": candidate.hypothesis,
            "direction_from_train": direction,
            "accepted": accepted_flag,
            "train": metric_dict(train),
            "validation": metric_dict(valid),
            "backtest": metric_dict(backtest),
        }
        results.append(row)
        if accepted_flag:
            accepted.append(
                {
                    **asdict(candidate),
                    "features": list(candidate.features),
                    "expression": candidate.expression,
                    "direction_expression": (
                        candidate.expression if direction > 0 else f"-({candidate.expression})"
                    ),
                    "direction": direction,
                    "train_rank_ic": train.mean_daily_rank_ic,
                    "validation_rank_ic": valid.mean_daily_rank_ic,
                    "validation_directional_rank_ic": directional_valid_ic,
                    "validation_directional_hit_ratio": directional_valid_hit,
                    "backtest_directional_rank_ic": direction * backtest.mean_daily_rank_ic,
                    "backtest_rank_ic": backtest.mean_daily_rank_ic,
                }
            )

    accepted.sort(key=lambda item: item["validation_directional_rank_ic"], reverse=True)
    baseline_metrics = {
        "train": metric_dict(evaluate_score(dates, baseline, target, train_mask)),
        "validation": metric_dict(evaluate_score(dates, baseline, target, valid_mask)),
        "backtest": metric_dict(evaluate_score(dates, baseline, target, backtest_mask)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_payload = []
    for candidate in candidates:
        payload = asdict(candidate)
        payload["features"] = list(candidate.features)
        payload["expression"] = candidate.expression
        candidate_payload.append(payload)
    (args.output_dir / "candidates.json").write_text(
        json.dumps(candidate_payload, indent=2, ensure_ascii=False) + "\n"
    )
    (args.output_dir / "candidate_metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=True) + "\n"
    )
    (args.output_dir / "accepted_factors.json").write_text(
        json.dumps(accepted, indent=2, ensure_ascii=False, allow_nan=True) + "\n"
    )
    (args.output_dir / "registry.json").write_text(
        json.dumps(
            {
                "registry_version": "miner-lite-v1",
                "generator": generator,
                "input": str(args.input),
                "candidate_count": len(candidates),
                "accepted_count": len(accepted),
                "selection": {
                    "train_end": "2022-12-30",
                    "validation": "2023",
                    "holdout": "2024-2026",
                    "min_train_abs_rank_ic": args.min_train_ic,
                    "min_validation_directional_rank_ic": args.min_valid_ic,
                    "min_validation_hit_ratio": args.min_valid_hit,
                },
                "baseline": baseline_metrics,
                "accepted_factors": accepted,
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )
        + "\n"
    )

    metrics_rows: List[Dict[str, object]] = []
    for result in results:
        for split in ("train", "validation", "backtest"):
            metrics_rows.append(
                {
                    "candidate_id": result["candidate_id"],
                    "expression": result["expression"],
                    "accepted": result["accepted"],
                    "split": split,
                    **result[split],
                }
            )
    pd.DataFrame(metrics_rows).to_csv(args.output_dir / "candidate_metrics.csv", index=False)

    report_lines = [
        "# AlphaCrafter Miner-lite 报告",
        "",
        f"- 候选生成器：`{generator}`",
        f"- 候选数：`{len(candidates)}`",
        f"- 通过数：`{len(accepted)}`",
        "- 训练集：2016–2022；验证集：2023；留出回测：2024–2026",
        "- `return_1d` 与 `one_day_reversal` 不进入候选池，避免重复计算简单反转因子。",
        "",
        "## 基线",
        "",
        "| split | Rank IC | Rank ICIR | Hit Ratio | Top/Bottom spread |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in ("train", "validation", "backtest"):
        metric = baseline_metrics[split]
        report_lines.append(
            f"| {split} | {metric['mean_daily_rank_ic']:.6f} | {metric['rank_icir']:.4f} | "
            f"{metric['rank_ic_hit_ratio']:.3f} | {metric['mean_top_bottom_decile_spread']:.6f} |"
        )
    report_lines.extend(["", "## 通过验证的因子（按验证集方向调整后的 Rank IC 排序）", ""])
    if accepted:
        report_lines.extend([
            "| 因子 | 表达式 | 训练 Rank IC | 验证 Rank IC | 回测 Rank IC |",
            "| --- | --- | ---: | ---: | ---: |",
        ])
        for factor in accepted[:20]:
            report_lines.append(
                f"| `{factor['candidate_id']}` | `{factor['direction_expression']}` | "
                f"{factor['train_rank_ic']:.6f} | {factor['validation_directional_rank_ic']:.6f} | "
                f"{factor['backtest_directional_rank_ic']:.6f} |"
            )
    else:
        report_lines.append("没有候选因子通过当前训练集和验证集门槛。")
    report_lines.extend(
        [
            "",
            "> 验证集用于因子准入；2024–2026 只作为最终留出报告，不能用于调阈值。",
            "> 当前版本是 Miner-lite：候选由安全模板生成，也支持通过 `--candidate-file` 接收未来 LLM 生成的 JSON。",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(report_lines) + "\n")

    print(f"Generated candidates: {len(candidates)}")
    print(f"Accepted factors: {len(accepted)}")
    print(f"Results written to {args.output_dir}")
    for factor in accepted[:10]:
        print(
            f"{factor['candidate_id']}: valid={factor['validation_directional_rank_ic']:.6f}, "
            f"backtest={factor['backtest_directional_rank_ic']:.6f}"
        )


if __name__ == "__main__":
    main()
