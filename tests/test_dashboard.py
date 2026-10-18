"""Unit tests for FinSight dashboard formatting and helper functions."""

import unittest
from app import (
    format_currency_value,
    format_pct_value,
    format_ratio_value,
    safe_metric_value,
    safe_metric_source_page,
    safe_metric_obj,
)
from extraction.financial_schema import (
    FinancialValue,
    IncomeStatement,
    BalanceSheet,
)


class TestDashboardHelpers(unittest.TestCase):
    """Test dashboard data formatting and unavailable value display."""

    def test_format_currency_value(self) -> None:
        """Verify currency and unit formatting with large and standard numbers."""
        # Value with currency and unit
        formatted = format_currency_value(12500.5, currency="USD", unit="millions")
        self.assertEqual(formatted, "USD 12,500.50 millions")

        # Standard number without currency or unit
        formatted_simple = format_currency_value(450.25)
        self.assertEqual(formatted_simple, "450.25")

        # Missing / None value must display 'Unavailable', never invented or 0
        self.assertEqual(format_currency_value(None), "Unavailable")
        self.assertEqual(format_currency_value(None, currency="USD"), "Unavailable")

    def test_format_pct_value(self) -> None:
        """Verify percentage formatting for positive, negative, zero, and missing."""
        self.assertEqual(format_pct_value(15.42), "+15.42%")
        self.assertEqual(format_pct_value(-8.2), "-8.20%")
        self.assertEqual(format_pct_value(0.0), "0.00%")
        self.assertEqual(format_pct_value(None), "Unavailable")

    def test_format_ratio_value(self) -> None:
        """Verify decimal ratio formatting."""
        self.assertEqual(format_ratio_value(1.75), "1.75x")
        self.assertEqual(format_ratio_value(0.4), "0.40x")
        self.assertEqual(format_ratio_value(None), "Unavailable")

    def test_safe_metric_helpers_with_standard_schema(self) -> None:
        """Verify safe metric extraction when fields are populated or default."""
        inc = IncomeStatement(
            revenue=FinancialValue(value=1000.0, source_page=2),
            cost_of_goods_sold=FinancialValue(value=600.0, source_page=2),
        )
        self.assertEqual(safe_metric_value(inc, "revenue"), 1000.0)
        self.assertEqual(safe_metric_source_page(inc, "revenue"), 2)
        self.assertEqual(safe_metric_value(inc, "cost_of_goods_sold"), 600.0)
        self.assertEqual(safe_metric_source_page(inc, "cost_of_goods_sold"), 2)

        # Omitted field has is_unavailable=True and value=None
        self.assertIsNone(safe_metric_value(inc, "operating_profit"))
        self.assertIsNone(safe_metric_source_page(inc, "operating_profit"))

    def test_safe_metric_helpers_resilient_to_missing_attributes(self) -> None:
        """Verify no AttributeError is raised when accessing missing attributes on legacy objects."""
        # Simulate legacy or mocked object that has no cost_of_goods_sold attribute
        class LegacyIncomeStatement:
            revenue = FinancialValue(value=5000.0, source_page=1)
            # deliberately missing cost_of_goods_sold

        legacy_inc = LegacyIncomeStatement()
        self.assertFalse(hasattr(legacy_inc, "cost_of_goods_sold"))

        # Accessing cost_of_goods_sold must safely return None without raising AttributeError
        val = safe_metric_value(legacy_inc, "cost_of_goods_sold")
        self.assertIsNone(val)

        src_page = safe_metric_source_page(legacy_inc, "cost_of_goods_sold")
        self.assertIsNone(src_page)

        obj = safe_metric_obj(legacy_inc, "cost_of_goods_sold")
        self.assertIsInstance(obj, FinancialValue)
        self.assertIsNone(obj.value)
        self.assertTrue(obj.is_unavailable)

    def test_safe_metric_helpers_with_none_container(self) -> None:
        """Verify safe metric functions handle None statement containers safely."""
        self.assertIsNone(safe_metric_value(None, "cost_of_goods_sold"))
        self.assertIsNone(safe_metric_source_page(None, "cost_of_goods_sold"))
        obj = safe_metric_obj(None, "cost_of_goods_sold")
        self.assertIsInstance(obj, FinancialValue)
        self.assertIsNone(obj.value)
        self.assertTrue(obj.is_unavailable)

    def test_balance_sheet_total_debt_and_liabilities_separation(self) -> None:
        """Verify total_debt is an explicit field and distinct from total_liabilities."""
        bs = BalanceSheet(
            total_liabilities=FinancialValue(value=1000.0, source_page=4),
            total_debt=FinancialValue(value=400.0, source_page=4),
            equity=FinancialValue(value=600.0, source_page=4),
        )
        self.assertEqual(safe_metric_value(bs, "total_debt"), 400.0)
        self.assertEqual(safe_metric_value(bs, "total_liabilities"), 1000.0)
        self.assertNotEqual(safe_metric_value(bs, "total_debt"), safe_metric_value(bs, "total_liabilities"))


if __name__ == "__main__":
    unittest.main()
