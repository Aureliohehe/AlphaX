import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from miner_lite import Candidate
from miner_search import rolling_feedback, run_search


class SearchTests(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2020-01-01", periods=40)
        self.frame = pd.DataFrame({"date": dates.repeat(4), "ticker": list("ABCD") * 40,
                                   "x": [1., 2., 3., 4.] * 40})
        self.targets = {h: pd.Series(np.tile([-.02, -.01, .01, .02], 40)) for h in (1, 5)}
        self.ends = {h: self.frame.date.map(dict(zip(dates[:-h], dates[h:]))) for h in (1, 5)}
        self.candidate = Candidate("x", "feature", ("x",), "test")

    def test_fold_labels_are_purged_and_direction_uses_prefix(self):
        first = rolling_feedback(self.frame, self.candidate, ["x"], self.targets, self.ends)
        changed = {h: values.copy() for h, values in self.targets.items()}
        # Corrupt only labels maturing after the first evaluation block.
        end = pd.Timestamp(first["folds"][0]["end"])
        for h in changed:
            changed[h].loc[self.ends[h] > end] *= -100
        second = rolling_feedback(self.frame, self.candidate, ["x"], changed, self.ends)
        self.assertEqual(first["folds"][0]["direction"], second["folds"][0]["direction"])
        for index in (0, 1):
            self.assertEqual(first["folds"][index]["metrics"]["rows"], second["folds"][index]["metrics"]["rows"])
            self.assertEqual(first["folds"][index]["metrics"]["mean_daily_rank_ic"], second["folds"][index]["metrics"]["mean_daily_rank_ic"])
        five = first["folds"][1]
        self.assertEqual(five["metrics"]["rows"], 20)

    def test_budget_duplicates_and_feedback(self):
        calls = []
        def propose(count, feedback):
            calls.append((count, json.loads(feedback)))
            return [self.candidate] * count
        candidates, audit = run_search(propose, self.frame, ["x"], self.targets, self.ends,
                                       rounds=5, budget=3, batch_size=2)
        self.assertEqual([c[0] for c in calls], [2, 1])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(audit["budget_spent"], 3)
        self.assertEqual(calls[0][1], [])
        self.assertEqual(calls[1][1][0]["expression"], self.candidate.expression)
        json.dumps(audit, allow_nan=False)

    def test_insufficient_sessions_fail(self):
        with self.assertRaises(ValueError):
            rolling_feedback(self.frame.iloc[:8], self.candidate, ["x"], self.targets, self.ends)

    def test_specialists_share_requested_budget_even_with_duplicates(self):
        from miner_lite import generate_llm_candidate_cluster
        with patch("miner_lite.generate_llm_candidates", return_value=[self.candidate]) as provider:
            generate_llm_candidate_cluster("mock", ["x"], "context", 10, 3, strict_budget=True)
        self.assertEqual([call.args[3] for call in provider.call_args_list], [4, 4, 2])

    def test_cli_passes_only_training_rows_to_search(self):
        from miner_lite import main
        from test_temporal import fixture_panel
        from miner_search import run_search as real_search
        panel = fixture_panel(100)
        sessions = sorted(panel.date.unique())
        def day(index):
            return str(pd.Timestamp(sessions[index]).date())
        observed = []
        def inspect(propose, frame, available, targets, ends, **kwargs):
            self.assertLessEqual(frame.date.max(), pd.Timestamp(day(59)))
            for h, target in targets.items():
                self.assertTrue(target.loc[ends[h] > pd.Timestamp(day(59))].isna().all())
            result = real_search(propose, frame, available, targets, ends, **kwargs)
            observed.append(result[1])
            return result
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "input.csv", Path(temp) / "out"
            panel.to_csv(source, index=False)
            args = ["miner", "--input", str(source), "--output-dir", str(output),
                    "--train-start", day(0), "--train-end", day(59),
                    "--validation-start", day(60), "--validation-end", day(79),
                    "--backtest-start", day(80), "--backtest-end", day(99),
                    "--llm-model", "mock", "--search-rounds", "2", "--search-budget", "2",
                    "--search-batch-size", "1"]
            candidate = Candidate("x", "feature", ("feature_x",), "fixture")
            with patch("sys.argv", args), patch("miner_search.run_search", side_effect=inspect), patch(
                "miner_lite.generate_llm_candidate_cluster", return_value=[candidate]
            ):
                main()
            report = json.loads((output / "search_feedback.json").read_text())
            self.assertEqual(report, observed[0])
            registry = json.loads((output / "registry.json").read_text())
            self.assertEqual(registry["selection"]["paper_min_hit_deviation"], .1)


if __name__ == "__main__":
    unittest.main()
