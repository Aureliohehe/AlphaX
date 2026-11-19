import io
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from download_historical_member_prices import get_quotes, historical_symbols
from audit_yahoo_supplement import audit


class HistoricalMemberPriceTests(unittest.TestCase):
    def test_membership_includes_start_snapshot_and_only_changes_through_end(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "membership.csv"
            path.write_text('date,tickers\n2021-12-31,"A,B"\n2022-01-04,"A,C"\n2022-02-01,"D,E"\n')
            self.assertEqual(historical_symbols(path, date(2022, 1, 1), date(2022, 1, 31)),
                             {"A", "B", "C"})
            path.write_text('date,tickers\n2022-01-01,"A,B"\n2022-02-01,"D,E"\n')
            self.assertEqual(historical_symbols(path, date(2022, 1, 1), date(2022, 1, 31)),
                             {"A", "B"})

    def test_missing_start_snapshot_and_duplicate_ticker_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "membership.csv"
            path.write_text('date,tickers\n2022-01-04,"A,B"\n')
            with self.assertRaisesRegex(ValueError, "covering start"):
                historical_symbols(path, date(2022, 1, 1), date(2022, 1, 31))
            path.write_text('date,tickers\n2021-12-31,"A,A"\n')
            with self.assertRaisesRegex(ValueError, "duplicate ticker"):
                historical_symbols(path, date(2022, 1, 1), date(2022, 1, 31))

    def test_audit_flags_reused_symbol_without_historical_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            membership, calendar, quotes = root / "members.csv", root / "calendar.csv", root / "quotes"
            quotes.mkdir()
            membership.write_text('date,tickers\n2020-01-01,"OLD,GOOD"\n2020-01-03,"GOOD"\n')
            calendar.write_text('date\n2020-01-02\n2020-01-03\n')
            (quotes / "OLD.csv").write_text('date,ticker,close\n2020-01-03,OLD,10\n')
            (quotes / "GOOD.csv").write_text('date,ticker,close\n2020-01-02,GOOD,10\n')
            result = audit(membership, calendar, quotes, date(2020, 1, 2), date(2020, 1, 3))
            self.assertEqual(result["zero_overlap_files"], ["OLD"])
            self.assertEqual(result["files_checked"], 2)

    @patch("download_historical_member_prices.urlopen")
    def test_yahoo_404_does_not_claim_success(self, fetch):
        fetch.side_effect = HTTPError("https://example.test", 404, "Not Found", {}, None)
        status, rows, reason = get_quotes("ABC", date(2016, 1, 1), date(2016, 1, 2))
        self.assertEqual((status, rows, reason), ("unavailable", [], "HTTP 404"))
        self.assertEqual(fetch.call_count, 1)

    @patch("download_historical_member_prices.urlopen")
    def test_yahoo_quote_field_mapping(self, fetch):
        response = {"chart": {"result": [{"timestamp": [1451692800],
            "indicators": {"quote": [{"open": [1], "high": [2], "low": [0.5],
                                       "close": [1.5], "volume": [123]}],
                           "adjclose": [{"adjclose": [1.4]}]}}]}}
        fetch.return_value.__enter__.return_value = io.BytesIO(json.dumps(response).encode())
        status, rows, reason = get_quotes("TEST", date(2016, 1, 1), date(2016, 1, 3))
        self.assertEqual((status, reason), ("downloaded", None))
        self.assertEqual(rows, [{"date": "2016-01-02", "ticker": "TEST", "open": 1,
                                 "high": 2, "low": 0.5, "close": 1.5, "volume": 123, "adj_close": 1.4}])


if __name__ == "__main__":
    unittest.main()
