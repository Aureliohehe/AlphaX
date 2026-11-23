"""Regression tests for the standalone factor DSL API."""

import unittest

import pandas as pd

from alphax.factor_dsl import evaluate_expression, parse_expression, validate_expression


class FactorDSLTests(unittest.TestCase):
    def test_parse_and_evaluate_nested_json_shorthand(self):
        frame = pd.DataFrame(
            {"date": [1, 1, 2, 2], "x": [2.0, 1.0, 1.0, 1.0], "y": [3.0, 2.0, 0.0, 0.0]},
            index=[10, 20, 30, 40],
        )
        raw = {
            "op": "divide",
            "left": {"op": "rank", "feature": "x"},
            "right": {"op": "feature", "feature": "y"},
        }
        expression = parse_expression(raw, iter(frame.columns))
        self.assertEqual(expression["inputs"][0], {"op": "rank", "input": {"op": "feature", "feature": "x"}})
        self.assertIn("left", raw)
        expected = pd.Series([1 / 3, 0.25, float("nan"), float("nan")], index=frame.index)
        pd.testing.assert_series_equal(evaluate_expression(expression, frame), expected)

    def test_parse_rejects_invalid_expressions(self):
        deep = {"op": "feature", "feature": "x"}
        for _ in range(5):
            deep = {"op": "rank", "input": deep}
        cases = (
            (["feature", "x"], "JSON objects"),
            ({"op": ["feature"], "feature": "x"}, "Unsupported expression op"),
            ({"op": "ts_corr", "features": [{"feature": "x"}, {}], "window": 2}, "two feature names"),
            ({"op": "feature", "feature": "missing"}, "unavailable features"),
            (deep, "expression depth"),
        )
        for expression, message in cases:
            with self.subTest(expression=expression), self.assertRaisesRegex(ValueError, message):
                parse_expression(expression, ["x"])

    def test_validate_checks_canonical_ast(self):
        with self.assertRaisesRegex(ValueError, "input expression"):
            validate_expression({"op": "rank", "feature": "x"})
