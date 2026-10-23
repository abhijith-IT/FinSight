"""Unit tests for FinSight AI Executive Summary generation module."""

import unittest
from unittest.mock import MagicMock

from analytics.ratios import FinancialRatios
from extraction.financial_schema import (
    AnnualFinancialData,
    BalanceSheet,
    CashFlowStatement,
    FinancialExtractionReport,
    FinancialValue,
    IncomeStatement,
)
from generation.summary import ExecutiveSummary, ExecutiveSummaryGenerator
from ingestion.pdf_parser import ParsedPage, ParsedPDF
from rag.pipeline import RAGPipeline


class TestExecutiveSummaryGenerator(unittest.TestCase):
    """Test suite for ExecutiveSummaryGenerator with mocked Gemini responses."""

    def setUp(self) -> None:
        """Create sample extracted financial report and calculated ratios."""
        self.report = FinancialExtractionReport(
            company_name="Apex Global Technologies Inc.",
            annual_reports=[
                AnnualFinancialData(
                    year=2022,
                    currency="USD",
                    unit="millions",
                    income_statement=IncomeStatement(
                        revenue=FinancialValue(value=1000.0, source_page=3),
                        cost_of_goods_sold=FinancialValue(value=600.0, source_page=3),
                        operating_profit=FinancialValue(value=200.0, source_page=3),
                        net_profit=FinancialValue(value=150.0, source_page=3),
                    ),
                    balance_sheet=BalanceSheet(
                        total_assets=FinancialValue(value=2500.0, source_page=4),
                        total_liabilities=FinancialValue(value=1200.0, source_page=4),
                        total_debt=FinancialValue(value=500.0, source_page=4),
                        equity=FinancialValue(value=1300.0, source_page=4),
                    ),
                    cash_flow_statement=CashFlowStatement(
                        operating_cash_flow=FinancialValue(value=280.0, source_page=5),
                        capital_expenditure=FinancialValue(value=90.0, source_page=5),
                    ),
                    source_pages=[3, 4, 5],
                ),
                AnnualFinancialData(
                    year=2023,
                    currency="USD",
                    unit="millions",
                    income_statement=IncomeStatement(
                        revenue=FinancialValue(value=1250.0, source_page=3),
                        cost_of_goods_sold=FinancialValue(value=725.0, source_page=3),
                        operating_profit=FinancialValue(value=260.0, source_page=3),
                        net_profit=FinancialValue(value=195.0, source_page=3),
                    ),
                    balance_sheet=BalanceSheet(
                        total_assets=FinancialValue(value=2900.0, source_page=4),
                        total_liabilities=FinancialValue(value=1350.0, source_page=4),
                        total_debt=FinancialValue(value=550.0, source_page=4),
                        equity=FinancialValue(value=1550.0, source_page=4),
                    ),
                    cash_flow_statement=CashFlowStatement(
                        operating_cash_flow=FinancialValue(value=350.0, source_page=5),
                        capital_expenditure=FinancialValue(value=110.0, source_page=5),
                    ),
                    source_pages=[3, 4, 5],
                ),
            ],
        )

        self.ratios = [
            FinancialRatios(
                year=2022,
                gross_margin_pct=40.0,
                net_profit_margin_pct=15.0,
                roe=11.54,
                debt_to_equity=0.38,
                revenue_growth_yoy_pct=None,
            ),
            FinancialRatios(
                year=2023,
                gross_margin_pct=42.0,
                net_profit_margin_pct=15.6,
                roe=12.58,
                debt_to_equity=0.35,
                revenue_growth_yoy_pct=25.0,
            ),
        ]

        # RAG Pipeline with qualitative risk and MD&A disclosure
        p_qual = ParsedPage(
            page_number=8,
            text="Item 1A. Risk Factors: Global component shortages and geopolitical tensions "
                 "could increase procurement lead times and constrain enterprise shipments.",
            has_text=True,
            char_count=165,
        )
        pdf = ParsedPDF(total_pages=8, pages=[p_qual], total_characters=165)
        self.pipeline = RAGPipeline()
        self.pipeline.ingest(pdf)

        self.mock_client = MagicMock()
        self.generator = ExecutiveSummaryGenerator(
            api_key="mock_key",
            model_name="gemini-3.1-flash-lite",
            client=self.mock_client,
        )

    def test_generate_summary_success_with_citations(self) -> None:
        """Verify end-to-end executive summary generation with citations and structured metrics."""
        mock_response = MagicMock()
        mock_response.text = (
            "## 🏢 Executive Overview\n"
            "Apex Global Technologies reported strong growth across FY 2022 – FY 2023 [Page 3].\n\n"
            "## 📈 Revenue & Profitability Trends\n"
            "Revenue expanded 25.0% YoY to $1,250M [Page 3]. Net profit rose from $150M to $195M.\n\n"
            "## ⚖️ Financial Health & Capital Structure\n"
            "Gross Margin improved to 42.0% while Debt-to-Equity remained conservative at 0.35x [Page 4].\n\n"
            "## ⚠️ Strategic Highlights & Key Risks\n"
            "Management highlighted component procurement delays as the primary operational risk [Page 8]."
        )
        self.mock_client.models.generate_content.return_value = mock_response

        summary = self.generator.generate_summary(self.report, self.ratios, self.pipeline)

        self.assertIsInstance(summary, ExecutiveSummary)
        self.assertEqual(summary.company_name, "Apex Global Technologies Inc.")
        self.assertEqual(summary.reporting_period, "FY 2022 – FY 2023")
        self.assertIn("Revenue expanded 25.0%", summary.summary_text)
        self.assertIn("[Page 3]", summary.summary_text)
        self.assertIn("[Page 8]", summary.summary_text)
        self.assertEqual(summary.cited_pages, [3, 4, 8])
        self.assertIsNone(summary.error)

        # Verify prompt passed to Gemini contains both financial statement data and calculated ratios
        call_kwargs = self.mock_client.models.generate_content.call_args[1]
        prompt = call_kwargs["contents"]
        self.assertIn("Apex Global Technologies Inc.", prompt)
        self.assertIn("Revenue: 1250.0", prompt)
        self.assertIn("Gross Margin: 42.00%", prompt)
        self.assertIn("YoY Revenue Growth: +25.00%", prompt)
        self.assertIn("[Page 8]", prompt)

    def test_citation_reconciliation_drops_invented_pages(self) -> None:
        """Verify hallucinated page citations not in the document are excluded from cited_pages."""
        mock_response = MagicMock()
        # Page 99 was never part of statement pages (3, 4, 5) or qualitative page (8)
        mock_response.text = "Revenue grew strongly [Page 3] and dividend payout was $50M [Page 99]."
        self.mock_client.models.generate_content.return_value = mock_response

        summary = self.generator.generate_summary(self.report, self.ratios, self.pipeline)

        self.assertIn(3, summary.cited_pages)
        self.assertNotIn(99, summary.cited_pages)

    def test_gemini_api_error_returns_fallback_without_raising(self) -> None:
        """Verify API error produces a structured fallback summary without raising an uncaught exception."""
        self.mock_client.models.generate_content.side_effect = Exception("Resource exhausted: 429")

        summary = self.generator.generate_summary(self.report, self.ratios, self.pipeline)

        self.assertIsNotNone(summary.error)
        self.assertIn("Resource exhausted", summary.error)
        self.assertIn("Apex Global Technologies Inc.", summary.summary_text)
        self.assertIn("Revenue =", summary.summary_text)
        self.assertIn("Net Margin:", summary.summary_text)

    def test_empty_report_handled_safely(self) -> None:
        """Verify empty report returns clean notice without invoking Gemini."""
        empty_report = FinancialExtractionReport(company_name=None, annual_reports=[])
        summary = self.generator.generate_summary(empty_report, [])

        self.assertIn("No extracted financial statement data", summary.summary_text)
        self.assertIsNotNone(summary.error)
        self.mock_client.models.generate_content.assert_not_called()

    def test_missing_metrics_handled_without_crashing(self) -> None:
        """Verify report with missing COGS and Total Debt generates safely without error."""
        single_year_report = FinancialExtractionReport(
            company_name="Single Year Corp",
            annual_reports=[
                AnnualFinancialData(
                    year=2023,
                    currency="EUR",
                    unit="thousands",
                    income_statement=IncomeStatement(
                        revenue=FinancialValue(value=5000.0, source_page=2),
                        cost_of_goods_sold=FinancialValue(value=None, is_unavailable=True),
                        net_profit=FinancialValue(value=800.0, source_page=2),
                    ),
                    balance_sheet=BalanceSheet(
                        total_assets=FinancialValue(value=12000.0, source_page=3),
                        total_debt=FinancialValue(value=None, is_unavailable=True),
                        equity=FinancialValue(value=6000.0, source_page=3),
                    ),
                    missing_fields=["cost_of_goods_sold", "total_debt"],
                    source_pages=[2, 3],
                )
            ],
        )
        single_ratios = [
            FinancialRatios(
                year=2023,
                gross_margin_pct=None,
                net_profit_margin_pct=16.0,
                roe=13.33,
                debt_to_equity=None,
                revenue_growth_yoy_pct=None,
            )
        ]
        mock_response = MagicMock()
        mock_response.text = (
            "## 🏢 Executive Overview\nSingle Year Corp reported FY 2023 performance [Page 2].\n"
            "Cost of goods sold and Total Debt were not reported."
        )
        self.mock_client.models.generate_content.return_value = mock_response

        summary = self.generator.generate_summary(single_year_report, single_ratios)

        self.assertEqual(summary.company_name, "Single Year Corp")
        self.assertEqual(summary.reporting_period, "FY 2023")
        self.assertIn("[Page 2]", summary.summary_text)
        self.assertIsNone(summary.error)


if __name__ == "__main__":
    unittest.main()
