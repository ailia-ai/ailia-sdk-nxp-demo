"""Current-frame timings and table reset do not reuse historical measurements."""

import unittest
from unittest.mock import Mock

from app import current_timing, reset_measurement_table
from settings import BACKENDS


class CurrentTimingTests(unittest.TestCase):
    def test_prepare_and_live_have_current_values(self):
        for action in ("prepare", "live", "detect", "classify", "estimate"):
            self.assertEqual(
                current_timing(dict(encoder_ms=12.3, click_ms=0, mode="latest", action=action))[:4],
                ["12.3", "—", "0.0", "12.3"],
            )

    def test_cached_encoder_is_not_counted_again(self):
        result = dict(encoder_ms=900, click_ms=120, cached=True, mode="latest", action="segment")
        self.assertEqual(current_timing(result)[:4], ["0.0", "—", "120.0", "120.0"])
        result["action"] = "benchmark"
        self.assertEqual(current_timing(result)[3], "1020.0")

    def test_encoder_only_prepare_has_total(self):
        self.assertEqual(current_timing(dict(encoder_ms=900, mode="latest"))[3], "900.0")

    def test_reset_clears_all_values_but_not_backend_names(self):
        cells = {backend: [Mock() for _ in range(6)] for backend in BACKENDS}
        reset_measurement_table(cells)
        for row in cells.values():
            row[0].set_text.assert_not_called()
            for cell in row[1:]:
                cell.set_text.assert_called_once_with("—")
