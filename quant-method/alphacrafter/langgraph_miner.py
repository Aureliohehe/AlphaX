#!/usr/bin/env python3
"""Optional LangGraph orchestration for the AlphaCrafter Miner cluster.

LangGraph is deliberately kept at the candidate-generation boundary.  The
existing safe AST parser, factor evaluator, admission gates, and persistent
library remain the source of truth after this graph returns.
"""

from __future__ import annotations

import json
import operator
from math import ceil
from typing import Dict, List, Sequence

from typing_extensions import Annotated, TypedDict

from miner_lite import (
    Candidate,
    MINER_AGENT_ROLES,
    candidate_from_dict,
    generate_llm_candidates,
    validate_candidate,
)


# Filled with langgraph.types.Send when the optional backend is invoked.
Send = None


class MinerGraphState(TypedDict, total=False):
    available: List[str]
    market_context: str
    model: str
    requested_count: int
    agent_count: int
    per_agent_count: int
    role: str
    candidate_payloads: Annotated[List[Dict[str, object]], operator.add]
    errors: Annotated[List[Dict[str, str]], operator.add]


def _candidate_payload(candidate: Candidate) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        "op": candidate.op,
        "features": list(candidate.features),
        "hypothesis": candidate.hypothesis,
    }
    if candidate.expression_spec is not None:
        payload["expression"] = candidate.expression_spec
    return payload


def _fan_out(state: MinerGraphState):
    """Create one Send task per specialist; Send tasks run in one graph step."""
    roles = list(MINER_AGENT_ROLES)
    agent_count = max(1, int(state.get("agent_count", len(roles))))
    per_agent_count = int(state.get("per_agent_count", 1))
    return [
        Send(
            "generate_agent_candidates",
            {
                "available": state["available"],
                "market_context": state["market_context"],
                "model": state["model"],
                "per_agent_count": per_agent_count,
                "role": roles[index % len(roles)],
            },
        )
        for index in range(agent_count)
    ]


def _generate_agent_candidates(state: MinerGraphState) -> Dict[str, object]:
    role = state["role"]
    try:
        candidates = generate_llm_candidates(
            state["model"],
            state["available"],
            state["market_context"],
            int(state["per_agent_count"]),
            role=role,
        )
        return {"candidate_payloads": [_candidate_payload(candidate) for candidate in candidates]}
    except Exception as exc:  # preserve partial results from other agents
        return {
            "candidate_payloads": [],
            "errors": [{"role": role, "error": f"{type(exc).__name__}: {exc}"}],
        }


def _deduplicate(
    payloads: Sequence[Dict[str, object]], available: Sequence[str], count: int
) -> List[Candidate]:
    candidates: List[Candidate] = []
    seen_ids = set()
    seen_expressions = set()
    for payload in payloads:
        candidate = candidate_from_dict(dict(payload))
        validate_candidate(candidate, available)
        if candidate.expression in seen_expressions:
            continue
        if candidate.candidate_id in seen_ids:
            continue
        seen_ids.add(candidate.candidate_id)
        seen_expressions.add(candidate.expression)
        candidates.append(candidate)
        if len(candidates) >= count:
            break
    return candidates


def generate_langgraph_candidates(
    model: str,
    available: Sequence[str],
    market_context: str,
    count: int,
    agents: int,
) -> List[Candidate]:
    """Generate candidates through a parallel LangGraph fan-out/fan-in graph."""
    if count < 1:
        raise ValueError("count must be positive")
    if agents < 1:
        raise ValueError("agents must be positive")
    try:
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import Send
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph backend requires langgraph. Install it with: "
            "python3 -m pip install langgraph"
        ) from exc

    # _fan_out closes over Send only through the module global.  Assigning it
    # here keeps importing this optional backend harmless when LangGraph is not
    # installed in the default deterministic Miner environment.
    globals()["Send"] = Send
    graph = StateGraph(MinerGraphState)
    graph.add_node("generate_agent_candidates", _generate_agent_candidates)
    graph.add_conditional_edges(START, _fan_out)
    graph.add_edge("generate_agent_candidates", END)
    compiled = graph.compile()
    per_agent_count = max(1, int(ceil(count / agents)))
    result = compiled.invoke(
        {
            "available": list(available),
            "market_context": market_context,
            "model": model,
            "requested_count": count,
            "agent_count": agents,
            "per_agent_count": per_agent_count,
            "candidate_payloads": [],
            "errors": [],
        },
        config={"max_concurrency": agents},
    )
    candidates = _deduplicate(result.get("candidate_payloads", []), available, count)
    if not candidates:
        errors = result.get("errors", [])
        detail = json.dumps(errors, ensure_ascii=False) if errors else "no candidates returned"
        raise RuntimeError(f"LangGraph Miner produced no valid candidates: {detail}")
    return candidates
