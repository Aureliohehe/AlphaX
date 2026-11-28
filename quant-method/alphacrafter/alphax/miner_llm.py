"""Model requests and validated LLM proposals for Miner and Screener."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from .factor_dsl import Candidate
from .miner_candidates import parse_llm_candidates

MINER_AGENT_ROLES = (
    "momentum_and_reversal_researcher",
    "volatility_and_volume_researcher",
    "cross_signal_composition_researcher",
)


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
    model: str,
    available: Sequence[str],
    market_context: str,
    count: int,
    role: str = "general_factor_researcher",
    audit: Optional[List[Dict[str, object]]] = None,
) -> List[Candidate]:
    """Generate candidates with DeepSeek directly or another LiteLLM provider.

    DeepSeek is implemented with the standard library because its API is
    OpenAI-compatible and the Miner should work without installing a second
    LLM framework.  Other model names retain the existing LiteLLM path.
    """
    prompt = f"""You are the {role}, one specialist in a Miner Cluster for a cross-sectional quantitative research system.
Generate exactly {count} candidate factors using only the approved feature names and operations.
Do not use future data, prices after the as_of date, arbitrary Python, or unavailable fields.
A candidate must be a JSON object with:
  candidate_id: short unique snake_case string
  hypothesis: one or two short sentences naming the expected direction, holding horizon,
    and one observable result that would count against the idea
  features: the unique leaf features used by the expression
  expression: a safe expression object, for example:
    {{"op":"ts_zscore","feature":"return_5d","window":20}}
  Allowed expression ops: feature, rank, ts_mean, ts_std, ts_sum, ts_min,
    ts_max, ts_quantile, ts_delta, ts_slope, ts_rsquare, ts_resi, ts_zscore,
    ts_rank, ts_corr, sum, difference, product, divide, negate, abs, log.
  AST shapes are strict: rank/negate use `input`; sum/difference/product/divide
    use `inputs` with exactly two expression objects; ts_corr uses `features`
    with exactly two feature names; time-series unary ops use `feature` and `window`.
  Time-series windows must be integers from 2 through 252.
Feature leaves read raw values. Arithmetic uses raw inputs; rank is never implicit.
Use an explicit rank node for cross-sectional percentile ranking. ts_rank ranks within each stock.
The validator calculates Pearson IC on the expression output and Rank IC by ranking that output.
The validator will calculate all metrics. Return only a JSON array.
Approved features: {json.dumps(list(available), ensure_ascii=False)}
Do not emit Python, arbitrary function names, or operators outside this grammar.
Point-in-time market context: {market_context}
"""
    messages = [
        {
            "role": "system",
            "content": "Return machine-readable JSON and respect the supplied schema.",
        },
        {"role": "user", "content": prompt},
    ]
    content = request_llm_json(model, messages)
    payload = json.loads(content)
    if not isinstance(payload, list):
        payload = payload.get("candidates", []) if isinstance(payload, dict) else []
    if audit is not None:
        audit.append({"status": "llm_response", "role": role,
                      "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                      "proposed_count": len(payload) if isinstance(payload, list) else None})
    return parse_llm_candidates(payload, available, audit if audit is not None else [])


def request_llm_json(model, messages):
    """Request a JSON response through the configured Miner provider."""
    deepseek_model = model.split("/", 1)[1] if model.startswith("deepseek/") else model
    if (compatible_base := os.environ.get("ALPHAX_LLM_API_BASE")):
        if not (api_key := os.environ.get("ALPHAX_LLM_API_KEY")):
            raise ValueError("ALPHAX_LLM_API_BASE requires ALPHAX_LLM_API_KEY")
        content = _compatible_completion(messages, model, compatible_base, api_key)
    elif deepseek_model in {"deepseek-chat", "deepseek-reasoner"}:
        content = _deepseek_completion(messages, deepseek_model)
    else:
        try:
            from litellm import completion
        except ImportError as exc:
            raise SystemExit(
                "This model requires LiteLLM. Install it with: python3 -m pip install litellm"
            ) from exc
        response = completion(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=4000,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise ValueError("LLM response content is not text")

    content = content.strip()
    if content.startswith("```"):
        content = content.partition("\n")[2].rsplit("```", 1)[0].strip()
    return content


def _deepseek_tool_call(
    messages: List[Dict[str, object]],
    model: str,
    tool: Dict[str, object],
) -> Dict[str, object]:
    """Call DeepSeek and return validated JSON arguments from one tool call."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DeepSeek tool calling requires DEEPSEEK_API_KEY")
    base_url = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com").rstrip("/")
    request_payload = {
        "model": model,
        "messages": messages,
        "tools": [tool],
        "tool_choice": {
            "type": "function",
            "function": {"name": tool["function"]["name"]},
        },
        "max_tokens": 1000,
    }
    request = Request(
        f"{base_url}/chat/completions",
        data=json.dumps(request_payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"DeepSeek tool API returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach DeepSeek tool API: {exc.reason}") from exc

    try:
        tool_calls = payload["choices"][0]["message"]["tool_calls"]
        arguments = tool_calls[0]["function"]["arguments"]
        result = json.loads(arguments) if isinstance(arguments, str) else arguments
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("DeepSeek response did not contain valid tool arguments") from exc
    if not isinstance(result, dict):
        raise ValueError("DeepSeek tool arguments must be a JSON object")
    return result


def _deepseek_completion(messages: List[Dict[str, str]], model: str) -> str:
    """Call DeepSeek's OpenAI-compatible endpoint without exposing the API key."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("--llm-model deepseek-chat requires DEEPSEEK_API_KEY")
    base_url = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com").rstrip("/")
    return _compatible_completion(messages, model, base_url, api_key, provider="DeepSeek")


def _compatible_completion(messages, model, base_url, api_key, provider="LLM"):
    request_payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 4000,
    }
    # DeepSeek Reasoner does not accept sampling controls such as temperature.
    if model == "deepseek-chat":
        request_payload["temperature"] = 0.7
    request = Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(request_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    request.add_unredirected_header("Authorization", f"Bearer {api_key}")
    try:
        with urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"{provider} API returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach {provider} API: {exc.reason}") from exc

    if "error" in payload:
        raise RuntimeError(f"{provider} API error: {payload['error']}")
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"{provider} response did not contain message content") from exc
    if not isinstance(content, str):
        raise ValueError(f"{provider} response content is not text")
    return content
