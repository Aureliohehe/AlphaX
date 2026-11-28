"""Real LangGraph and LiteLLM execution with local model responses only."""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quant-method/alphacrafter"))
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["LANGSMITH_TRACING"] = "false"

import litellm
from langgraph_miner import generate_langgraph_candidates
from miner_lite import MINER_AGENT_ROLES


class LLMIntegrationTests(unittest.TestCase):
    def setUp(self):
        # An accidental provider request must fail, even on a machine with API keys.
        for target in ("socket.socket.connect", "socket.socket.connect_ex"):
            guard = patch(target, side_effect=AssertionError("Network forbidden in offline integration"))
            guard.start()
            self.addCleanup(guard.stop)

    def test_real_graph_distributes_candidates_and_preserves_other_agent_on_failure(self):
        def response(model, messages):
            prompt = messages[-1]["content"]
            index = next(i for i, role in enumerate(MINER_AGENT_ROLES) if role in prompt)
            if index == 1:
                raise RuntimeError("Injected model failure")
            return json.dumps([{"candidate_id": f"candidate_{index}", "op": "feature",
                                "features": [f"feature_{index}"], "hypothesis": "Synthetic mechanism"}])

        audit = []
        with patch("alphax.miner_llm.request_llm_json", side_effect=response):
            result = generate_langgraph_candidates(
                "offline-model", ["feature_0", "feature_1", "feature_2"],
                "Offline integration", count=3, agents=3, audit=audit,
            )
        self.assertEqual({c.candidate_id for c in result}, {"candidate_0", "candidate_2"})
        self.assertTrue(any(row.get("status") == "agent_error" for row in audit))

    def test_litellm_response_contract_without_provider_request(self):
        response = litellm.completion(
            model="openai/gpt-4o-mini", messages=[{"role": "user", "content": "offline"}],
            mock_response='{"offline": true}',
        )
        self.assertEqual(json.loads(response.choices[0].message.content), {"offline": True})


if __name__ == "__main__":
    unittest.main()
