from pathlib import Path
import tempfile
import unittest

from factor_mine_skill import SKILL_FILES, load_factor_mine_skill, note_failure
from langgraph_miner import _candidate_payload
from miner_lite import candidate_from_dict


class FactorMineSkillTests(unittest.TestCase):
    def test_loads_documents_and_excludes_ui_form(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "references").mkdir()
            (root / "SKILL.md").write_text(
                "---\nname: factor-mine\n---\n```json qsh-form\n{}\n```\n# Experiment rules\n")
            for name in SKILL_FILES[1:]:
                (root / name).write_text(f"Reference for {name}")
            context, metadata = load_factor_mine_skill(root)
            self.assertIn("Experiment rules", context)
            self.assertNotIn("qsh-form", context)
            self.assertEqual([item["file"] for item in metadata["files"]], list(SKILL_FILES))
            self.assertTrue(all(len(item["sha256"]) == 64 for item in metadata["files"]))

    def test_note_survives_graph_serialization_and_has_expected_operation(self):
        note = {"op_type": "add_factor", "hypothesis": "liquidity shock",
                "change": "add a turnover term", "expected": "Rank IC +.01 to +.02"}
        candidate = candidate_from_dict({"id": "test", "op": "feature", "features": ["volume"],
                                         "hypothesis": "liquidity shock", "iteration_note": note})
        recovered = candidate_from_dict(_candidate_payload(candidate))
        self.assertIsNone(note_failure(recovered.iteration_note, "add_factor"))
        self.assertEqual(note_failure(recovered.iteration_note, "modify_factor"),
                         "iteration_note_operation_mismatch")
        self.assertEqual(note_failure({"op_type": "add_factor"}, "add_factor"),
                         "missing_or_invalid_iteration_note")
