"""Unit tests for FinSight Financial Ratio Engine."""

import unittest

from analytics.ratios import (
    FinancialRatios,
    calculate_annual_ratios,
    calculate_debt_to_equity,
    calculate_gross_margin_pct,
    calculate_net_profit_margin_pct,
    calculate_report_ratios,
    calculate_revenue_growth_yoy_pct,
    calculate_roe,
    calculate_ratios_series,
)
from extraction.financial_schema import (
    AnnualFinancialData,
    BalanceSheet,
    FinancialExtractionReport,
    FinancialValue,
    IncomeStatement,
)


class TestFinancialRatiosIndividualCalculations(unittest.TestCase):
    """Test individual ratio calculation functions with normal values, zero denominators, and edge cases."""

    def test_gross_margin_normal_and_edge_cases(self) -> None:
        """Verify Gross Margin % = (Revenue - COGS) / Revenue * 100."""
        # Normal calculation: (1000 - 600) / 1000 * 100 = 40.0%
        self.assertEqual(calculate_gross_margin_pct(1000.0, 600.0), 40.0)

        # Missing inputs -> None
        self.assertIsNone(calculate_gross_margin_pct(None, 600.0))
        self.assertIsNone(calculate_gross_margin_pct(1000.0, None))
        self.assertIsNone(calculate_gross_margin_pct(None, None))

        # Zero denominator -> None (no division by zero)
        self.assertIsNone(calculate_gross_margin_pct(0.0, 0.0))
        self.assertIsNone(calculate_gross_margin_pct(0.0, 50.0))

        # Decimal values
        self.assertAlmostEqual(calculate_gross_margin_pct(300.0, 100.0, round_digits=2), 66.67)

    def test_net_profit_margin_normal_and_edge_cases(self) -> None:
        """Verify Net Profit Margin % = Net Profit / Revenue * 100."""
        # Normal calculation: 150 / 1000 * 100 = 15.0%
        self.assertEqual(calculate_net_profit_margin_pct(150.0, 1000.0), 15.0)

        # Net loss (negative profit): -200 / 1000 * 100 = -20.0%
        self.assertEqual(calculate_net_profit_margin_pct(-200.0, 1000.0), -20.0)

        # Missing inputs -> None
        self.assertIsNone(calculate_net_profit_margin_pct(None, 1000.0))
        self.assertIsNone(calculate_net_profit_margin_pct(150.0, None))
        self.assertIsNone(calculate_net_profit_margin_pct(None, None))

        # Zero denominator -> None
        self.assertIsNone(calculate_net_profit_margin_pct(150.0, 0.0))

    def test_roe_normal_and_edge_cases(self) -> None:
        """Verify ROE % = Net Profit / Shareholders' Equity * 100."""
        # Normal calculation: 200 / 1000 * 100 = 20.0%
        self.assertEqual(calculate_roe(200.0, 1000.0), 20.0)

        # Negative equity
        self.assertEqual(calculate_roe(100.0, -500.0), -20.0)

        # Missing inputs -> None
        self.assertIsNone(calculate_roe(None, 1000.0))
        self.assertIsNone(calculate_roe(200.0, None))

        # Zero denominator -> None
        self.assertIsNone(calculate_roe(200.0, 0.0))

    def test_debt_to_equity_normal_and_liabilities_separation(self) -> None:
        """Verify Debt-to-Equity = Total Debt / Shareholders' Equity."""
        # Normal calculation: 500 / 1000 = 0.5
        self.assertEqual(calculate_debt_to_equity(500.0, 1000.0), 0.5)

        # Missing inputs -> None
        self.assertIsNone(calculate_debt_to_equity(None, 1000.0))
        self.assertIsNone(calculate_debt_to_equity(500.0, None))

        # Zero denominator -> None
        self.assertIsNone(calculate_debt_to_equity(500.0, 0.0))

    def test_revenue_growth_yoy_normal_and_edge_cases(self) -> None:
        """Verify YoY Revenue Growth % = (Current - Previous) / Previous * 100."""
        # Positive growth: (1200 - 1000) / 1000 * 100 = 20.0%
        self.assertEqual(calculate_revenue_growth_yoy_pct(1200.0, 1000.0), 20.0)

        # Contraction: (800 - 1000) / 1000 * 100 = -20.0%
        self.assertEqual(calculate_revenue_growth_yoy_pct(800.0, 1000.0), -20.0)

        # Missing inputs -> None
        self.assertIsNone(calculate_revenue_growth_yoy_pct(None, 1000.0))
        self.assertIsNone(calculate_revenue_growth_yoy_pct(1200.0, None))
        self.assertIsNone(calculate_revenue_growth_yoy_pct(None, None))

        # Zero denominator (previous revenue == 0) -> None
        self.assertIsNone(calculate_revenue_growth_yoy_pct(1200.0, 0.0))


class TestAnnualRatiosEngine(unittest.TestCase):
    """Test calculate_annual_ratios integration with AnnualFinancialData."""

    def test_calculate_annual_ratios_with_all_metrics(self) -> None:
        """Ensure full calculation works with all schema fields including COGS and Total Debt."""
        annual = AnnualFinancialData(
            year=2023,
            income_statement=IncomeStatement(
                revenue=FinancialValue(value=10000.0),
                cost_of_goods_sold=FinancialValue(value=4000.0),
                net_profit=FinancialValue(value=2500.0),
            ),
            balance_sheet=BalanceSheet(
                equity=FinancialValue(value=8000.0),
                total_debt=FinancialValue(value=3200.0),
                total_liabilities=FinancialValue(value=6000.0),  # Not debt!
            ),
        )
        prev_annual = AnnualFinancialData(
            year=2022,
            income_statement=IncomeStatement(
                revenue=FinancialValue(value=8000.0),
            ),
        )

        # No kwargs needed: engine reads cost_of_goods_sold and total_debt directly from schema
        ratios = calculate_annual_ratios(
            current=annual,
            previous=prev_annual,
        )

        self.assertEqual(ratios.year, 2023)
        self.assertEqual(ratios.gross_margin_pct, 60.0)  # (10000 - 4000)/10000 * 100
        self.assertEqual(ratios.net_profit_margin_pct, 25.0)  # 2500/10000 * 100
        self.assertEqual(ratios.roe, 31.25)  # 2500/8000 * 100
        self.assertEqual(ratios.roe_pct, 31.25)
        self.assertEqual(ratios.debt_to_equity, 0.4)  # 3200/8000
        self.assertEqual(ratios.revenue_growth_yoy_pct, 25.0)  # (10000 - 8000)/8000 * 100

    def test_missing_cogs_never_inferred(self) -> None:
        """Requirement: Never infer missing COGS; return None when not reported."""
        annual = AnnualFinancialData(
            year=2023,
            income_statement=IncomeStatement(
                revenue=FinancialValue(value=10000.0),
                operating_profit=FinancialValue(value=3000.0),
                operating_expenses=FinancialValue(value=4000.0),
                # cost_of_goods_sold is omitted / None
            ),
            balance_sheet=BalanceSheet(equity=FinancialValue(value=5000.0)),
        )

        ratios = calculate_annual_ratios(current=annual)
        self.assertIsNone(ratios.gross_margin_pct)

    def test_total_liabilities_is_never_treated_as_total_debt(self) -> None:
        """Requirement: Do not treat total liabilities as total debt unless explicitly defined."""
        annual = AnnualFinancialData(
            year=2023,
            balance_sheet=BalanceSheet(
                equity=FinancialValue(value=5000.0),
                total_liabilities=FinancialValue(value=3000.0),
                # total_debt is omitted / None
            ),
        )

        # total_debt is None -> debt_to_equity MUST be None even though total_liabilities=3000.0 exists
        ratios = calculate_annual_ratios(current=annual)
        self.assertIsNone(ratios.debt_to_equity)

    def test_missing_and_zero_values_return_none(self) -> None:
        """Test that missing inputs and zero denominators yield None without error."""
        annual = AnnualFinancialData(
            year=2023,
            income_statement=IncomeStatement(
                revenue=FinancialValue(value=0.0),
                net_profit=FinancialValue(value=None),
            ),
            balance_sheet=BalanceSheet(
                equity=FinancialValue(value=0.0),
            ),
        )

        ratios = calculate_annual_ratios(current=annual)
        self.assertEqual(ratios.year, 2023)
        self.assertIsNone(ratios.gross_margin_pct)
        self.assertIsNone(ratios.net_profit_margin_pct)
        self.assertIsNone(ratios.roe)
        self.assertIsNone(ratios.debt_to_equity)
        self.assertIsNone(ratios.revenue_growth_yoy_pct)


class TestMultiYearRatiosSeries(unittest.TestCase):
    """Test multi-year sequence and YoY consecutive year logic."""

    def test_consecutive_multi_year_series(self) -> None:
        """Verify 3 consecutive years calculate YoY growth correctly."""
        y2021 = AnnualFinancialData(
            year=2021,
            income_statement=IncomeStatement(
                revenue=FinancialValue(value=1000.0),
                net_profit=FinancialValue(value=100.0),
            ),
            balance_sheet=BalanceSheet(equity=FinancialValue(value=500.0)),
        )
        y2022 = AnnualFinancialData(
            year=2022,
            income_statement=IncomeStatement(
                revenue=FinancialValue(value=1200.0),
                net_profit=FinancialValue(value=150.0),
            ),
            balance_sheet=BalanceSheet(equity=FinancialValue(value=600.0)),
        )
        y2023 = AnnualFinancialData(
            year=2023,
            income_statement=IncomeStatement(
                revenue=FinancialValue(value=1500.0),
                net_profit=FinancialValue(value=225.0),
            ),
            balance_sheet=BalanceSheet(equity=FinancialValue(value=750.0)),
        )

        # Pass in scrambled order to verify chronological sorting
        series = calculate_ratios_series([y2023, y2021, y2022])

        self.assertEqual(len(series), 3)
        self.assertEqual(series[0].year, 2021)
        self.assertEqual(series[1].year, 2022)
        self.assertEqual(series[2].year, 2023)

        # 2021: first available year -> YoY is None
        self.assertIsNone(series[0].revenue_growth_yoy_pct)

        # 2022: (1200 - 1000) / 1000 * 100 = 20.0%
        self.assertEqual(series[1].revenue_growth_yoy_pct, 20.0)

        # 2023: (1500 - 1200) / 1200 * 100 = 25.0%
        self.assertEqual(series[2].revenue_growth_yoy_pct, 25.0)

    def test_non_consecutive_years_returns_none_for_yoy(self) -> None:
        """When consecutive prior year is missing (e.g. 2021 and 2023), YoY for 2023 is None."""
        y2021 = AnnualFinancialData(
            year=2021,
            income_statement=IncomeStatement(revenue=FinancialValue(value=1000.0)),
        )
        y2023 = AnnualFinancialData(
            year=2023,
            income_statement=IncomeStatement(revenue=FinancialValue(value=1500.0)),
        )

        series = calculate_ratios_series([y2021, y2023], strict_consecutive=True)

        self.assertEqual(len(series), 2)
        self.assertIsNone(series[0].revenue_growth_yoy_pct)
        # 2022 is missing, so 2023 does not have consecutive prior year
        self.assertIsNone(series[1].revenue_growth_yoy_pct)

    def test_middle_year_with_missing_revenue(self) -> None:
        """If 2022 revenue is missing, 2022 YoY is None and 2023 YoY is None (do not skip to 2021)."""
        y2021 = AnnualFinancialData(
            year=2021,
            income_statement=IncomeStatement(revenue=FinancialValue(value=1000.0)),
        )
        y2022 = AnnualFinancialData(
            year=2022,
            income_statement=IncomeStatement(revenue=FinancialValue(value=None)),  # missing
        )
        y2023 = AnnualFinancialData(
            year=2023,
            income_statement=IncomeStatement(revenue=FinancialValue(value=1500.0)),
        )

        series = calculate_ratios_series([y2021, y2022, y2023])

        self.assertIsNone(series[0].revenue_growth_yoy_pct)
        self.assertIsNone(series[1].revenue_growth_yoy_pct)
        self.assertIsNone(series[2].revenue_growth_yoy_pct)

    def test_calculate_report_ratios_wrapper(self) -> None:
        """Test calculate_report_ratios with FinancialExtractionReport."""
        report = FinancialExtractionReport(
            company_name="Alpha Corp",
            annual_reports=[
                AnnualFinancialData(
                    year=2022,
                    income_statement=IncomeStatement(
                        revenue=FinancialValue(value=500.0),
                        net_profit=FinancialValue(value=50.0),
                    ),
                    balance_sheet=BalanceSheet(equity=FinancialValue(value=250.0)),
                ),
                AnnualFinancialData(
                    year=2023,
                    income_statement=IncomeStatement(
                        revenue=FinancialValue(value=600.0),
                        net_profit=FinancialValue(value=75.0),
                    ),
                    balance_sheet=BalanceSheet(equity=FinancialValue(value=300.0)),
                ),
            ],
        )

        ratios = calculate_report_ratios(report)
        self.assertEqual(len(ratios), 2)
        self.assertEqual(ratios[0].year, 2022)
        self.assertEqual(ratios[0].net_profit_margin_pct, 10.0)
        self.assertEqual(ratios[0].roe, 20.0)
        self.assertIsNone(ratios[0].revenue_growth_yoy_pct)

        self.assertEqual(ratios[1].year, 2023)
        self.assertEqual(ratios[1].net_profit_margin_pct, 12.5)
        self.assertEqual(ratios[1].roe, 25.0)
        self.assertEqual(ratios[1].revenue_growth_yoy_pct, 20.0)  # (600 - 500)/500 * 100


if __name__ == "__main__":
    unittest.main()
