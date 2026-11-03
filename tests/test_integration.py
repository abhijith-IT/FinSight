"""End-to-End Integration and Quality Assurance tests for FinSight.

Validates the complete workflow:
PDF Upload -> PDF Parsing -> Financial Extraction -> Validation -> Ratio Calculation
-> Dashboard Data Binding -> Executive Summary -> RAG Q&A.
"""

import unittest
from unittest.mock import MagicMock, patch

from analytics.ratios import FinancialRatios, calculate_report_ratios
from app import (
    compute_pdf_hash,
    format_currency_value,
    format_pct_value,
    format_ratio_value,
    safe_metric_source_page,
    safe_metric_value,
)
from extraction.extractor import FinancialExtractor
from extraction.financial_schema import (
    AnnualFinancialData,
    BalanceSheet,
    CashFlowStatement,
    FinancialExtractionReport,
    FinancialValue,
    IncomeStatement,
)
from extraction.validation import sanitize_and_flag_annual_data
from generation.summary import ExecutiveSummary, ExecutiveSummaryGenerator
from ingestion.pdf_parser import ParsedPage, ParsedPDF
from rag.chunker import DocumentChunker
from rag.pipeline import RAGPipeline, create_rag_pipeline
from rag.qa import QAResult, ReportQAEngine


class TestEndToEndWorkflow(unittest.TestCase):
    """Integration test suite covering the entire FinSight pipeline."""

    def setUp(self) -> None:
        """Create sample multi-page parsed PDF mimicking an audited corporate annual report."""
        p1 = ParsedPage(
            page_number=1,
            text="Corporate Profile: Zenith Technologies Limited Annual Report for Fiscal Years 2022-23 and 2023-24.",
            has_text=True,
            char_count=105,
        )
        p2 = ParsedPage(
            page_number=2,
            text=(
                "Statement of Profit and Loss (in USD millions):\n"
                "Revenue from operations: FY2023 = $1,000.0M, FY2024 = $1,250.0M.\n"
                "Cost of goods sold: FY2023 = $600.0M, FY2024 = $725.0M.\n"
                "Operating profit (EBIT): FY2023 = $200.0M, FY2024 = $260.0M.\n"
                "Net profit after tax: FY2023 = $150.0M, FY2024 = $195.0M."
            ),
            has_text=True,
            char_count=295,
        )
        p3 = ParsedPage(
            page_number=3,
            text=(
                "Balance Sheet (in USD millions):\n"
                "Total assets: FY2023 = $2,500.0M, FY2024 = $2,900.0M.\n"
                "Total liabilities: FY2023 = $1,200.0M, FY2024 = $1,350.0M.\n"
                "Total debt (borrowings): FY2023 = $400.0M, FY2024 = $450.0M.\n"
                "Shareholders' Equity: FY2023 = $1,300.0M, FY2024 = $1,550.0M."
            ),
            has_text=True,
            char_count=298,
        )
        p4 = ParsedPage(
            page_number=4,
            text=(
                "Statement of Cash Flows (in USD millions):\n"
                "Operating cash flow: FY2023 = $280.0M, FY2024 = $350.0M.\n"
                "Capital expenditures: FY2023 = $90.0M, FY2024 = $110.0M."
            ),
            has_text=True,
            char_count=180,
        )
        p5 = ParsedPage(
            page_number=5,
            text=(
                "Management Discussion & Risk Factors:\n"
                "The group faces exposure to semiconductor supply chain lead times and foreign currency fluctuations.\n"
                "Enterprise software cloud revenues grew rapidly across North America and Europe."
            ),
            has_text=True,
            char_count=235,
        )

        self.pages = [p1, p2, p3, p4, p5]
        self.parsed_pdf = ParsedPDF(
            file_name="zenith_annual_report.pdf",
            total_pages=5,
            pages=self.pages,
            total_characters=sum(p.char_count for p in self.pages),
        )

    def test_complete_pipeline_data_flow_and_consistency(self) -> None:
        """Verify seamless data flow from PDF -> Extraction -> Ratios -> Summary -> RAG."""
        # 1. Document hashing
        raw_pdf_bytes = b"%PDF-1.4 Zenith Technologies Sample Audited Financial Document Content"
        doc_hash = compute_pdf_hash(raw_pdf_bytes)
        self.assertEqual(len(doc_hash), 64)

        # 2. Simulated Structured Extraction
        extraction_report = FinancialExtractionReport(
            company_name="Zenith Technologies Limited",
            annual_reports=[
                AnnualFinancialData(
                    year=2023,
                    currency="USD",
                    unit="millions",
                    income_statement=IncomeStatement(
                        revenue=FinancialValue(value=1000.0, source_page=2),
                        cost_of_goods_sold=FinancialValue(value=600.0, source_page=2),
                        operating_expenses=FinancialValue(value=200.0, source_page=2),
                        operating_profit=FinancialValue(value=200.0, source_page=2),
                        net_profit=FinancialValue(value=150.0, source_page=2),
                    ),
                    balance_sheet=BalanceSheet(
                        total_assets=FinancialValue(value=2500.0, source_page=3),
                        total_liabilities=FinancialValue(value=1200.0, source_page=3),
                        total_debt=FinancialValue(value=400.0, source_page=3),
                        equity=FinancialValue(value=1300.0, source_page=3),
                    ),
                    cash_flow_statement=CashFlowStatement(
                        operating_cash_flow=FinancialValue(value=280.0, source_page=4),
                        capital_expenditure=FinancialValue(value=90.0, source_page=4),
                    ),
                    source_pages=[2, 3, 4],
                ),
                AnnualFinancialData(
                    year=2024,
                    currency="USD",
                    unit="millions",
                    income_statement=IncomeStatement(
                        revenue=FinancialValue(value=1250.0, source_page=2),
                        cost_of_goods_sold=FinancialValue(value=725.0, source_page=2),
                        operating_expenses=FinancialValue(value=265.0, source_page=2),
                        operating_profit=FinancialValue(value=260.0, source_page=2),
                        net_profit=FinancialValue(value=195.0, source_page=2),
                    ),
                    balance_sheet=BalanceSheet(
                        total_assets=FinancialValue(value=2900.0, source_page=3),
                        total_liabilities=FinancialValue(value=1350.0, source_page=3),
                        total_debt=FinancialValue(value=450.0, source_page=3),
                        equity=FinancialValue(value=1550.0, source_page=3),
                    ),
                    cash_flow_statement=CashFlowStatement(
                        operating_cash_flow=FinancialValue(value=350.0, source_page=4),
                        capital_expenditure=FinancialValue(value=110.0, source_page=4),
                    ),
                    source_pages=[2, 3, 4],
                ),
            ],
            is_incomplete=False,
        )

        # 3. Validation step
        for ann in extraction_report.annual_reports:
            sanitize_and_flag_annual_data(ann)
            self.assertEqual(len(ann.missing_fields), 0)

        # 4. Pure-Python Ratio Calculation
        ratios = calculate_report_ratios(extraction_report)
        self.assertEqual(len(ratios), 2)

        r23 = ratios[0]
        self.assertEqual(r23.year, 2023)
        self.assertAlmostEqual(r23.gross_margin_pct, 40.0)  # (1000 - 600) / 1000 * 100
        self.assertAlmostEqual(r23.net_profit_margin_pct, 15.0)  # 150 / 1000 * 100
        self.assertAlmostEqual(r23.debt_to_equity, 400.0 / 1300.0, places=2)
        self.assertIsNone(r23.revenue_growth_yoy_pct)  # No 2022 data

        r24 = ratios[1]
        self.assertEqual(r24.year, 2024)
        self.assertAlmostEqual(r24.gross_margin_pct, 42.0)  # (1250 - 725) / 1250 * 100
        self.assertAlmostEqual(r24.net_profit_margin_pct, 15.6)  # 195 / 1250 * 100
        self.assertAlmostEqual(r24.revenue_growth_yoy_pct, 25.0)  # (1250 - 1000) / 1000 * 100

        # 5. RAG Ingestion (reuses existing ParsedPDF without re-parsing)
        rag_pipeline = create_rag_pipeline(api_key=None, retriever_type="bm25")
        chunks = rag_pipeline.ingest(self.parsed_pdf, doc_id="zenith")
        self.assertGreater(len(chunks), 0)
        self.assertEqual(rag_pipeline.indexed_pages, [1, 2, 3, 4, 5])

        # Test RAG retrieval and Q&A grounding
        mock_qa_client = MagicMock()
        mock_qa_resp = MagicMock()
        mock_qa_resp.text = (
            "Zenith Technologies faces risks related to semiconductor supply chain lead times "
            "and foreign currency fluctuations [Page 5]."
        )
        mock_qa_client.models.generate_content.return_value = mock_qa_resp

        qa_engine = ReportQAEngine(
            api_key="mock_key",
            model_name="gemini-3.8-flash",
            client=mock_qa_client,
            top_k=2,
            base_delay_seconds=0.01,
        )
        qa_res = qa_engine.answer_question("What are the primary risk factors?", rag_pipeline)

        self.assertFalse(qa_res.has_insufficient_evidence)
        self.assertIn("semiconductor supply chain", qa_res.answer)
        self.assertEqual(qa_res.cited_pages, [5])
        self.assertGreater(len(qa_res.sources), 0)

        # 6. Executive Summary Generation
        mock_summary_client = MagicMock()
        mock_summary_resp = MagicMock()
        mock_summary_resp.text = (
            "## 🏢 Executive Overview\nZenith Technologies Limited reported FY 2023 – FY 2024 results [Page 2].\n\n"
            "## 📈 Revenue & Profitability Trends\nRevenue grew 25.0% YoY to $1,250.0M with gross margin of 42.0% [Page 2].\n\n"
            "## 💪 Financial Strengths & Core Drivers\nHealthy operating cash flow of $350.0M [Page 4].\n\n"
            "## ⚠️ Weaknesses, Strategic Risks & Constraints\nSemiconductor supply disruptions [Page 5]."
        )
        mock_summary_client.models.generate_content.return_value = mock_summary_resp

        summary_gen = ExecutiveSummaryGenerator(
            api_key="mock_key",
            client=mock_summary_client,
            base_delay_seconds=0.01,
        )
        summary = summary_gen.generate_summary(extraction_report, ratios, rag_pipeline)

        self.assertFalse(summary.is_incomplete)
        self.assertEqual(summary.company_name, "Zenith Technologies Limited")
        self.assertEqual(summary.reporting_period, "FY 2023 – FY 2024")
        self.assertEqual(summary.cited_pages, [2, 4, 5])
        self.assertIn("Revenue grew 25.0%", summary.summary_text)

        # 7. Dashboard Value Formatting Consistency
        rev_24 = safe_metric_value(extraction_report.annual_reports[1].income_statement, "revenue")
        pg_24 = safe_metric_source_page(extraction_report.annual_reports[1].income_statement, "revenue")
        self.assertEqual(rev_24, 1250.0)
        self.assertEqual(pg_24, 2)
        self.assertEqual(
            format_currency_value(rev_24, "USD", "millions"),
            "USD 1,250.00 millions",
        )
        self.assertEqual(format_pct_value(r24.revenue_growth_yoy_pct), "+25.00%")
        self.assertEqual(format_ratio_value(r24.debt_to_equity), f"{r24.debt_to_equity:.2f}x")

    def test_incomplete_extraction_gracefully_propagates(self) -> None:
        """Verify incomplete extraction status propagates through summary and dashboard."""
        incomplete_report = FinancialExtractionReport(
            company_name="Partial Data Corp",
            annual_reports=[
                AnnualFinancialData(
                    year=2024,
                    income_statement=IncomeStatement(
                        revenue=FinancialValue(value=500.0, source_page=2),
                    ),
                    source_pages=[2],
                )
            ],
            is_incomplete=True,
            failed_sections=[3, 4],
            unprocessed_sections=[5],
        )

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value.text = (
            "## 🏢 Executive Overview\nPartial Data Corp reported partial results [Page 2]."
        )

        summary_gen = ExecutiveSummaryGenerator(
            api_key="mock_key",
            client=mock_client,
            base_delay_seconds=0.01,
        )
        summary = summary_gen.generate_summary(incomplete_report, [])

        self.assertTrue(summary.is_incomplete)

        # Fallback summary also communicates incomplete status
        fallback_summary_text = summary_gen._build_fallback_summary(
            "Partial Data Corp", "FY 2024", incomplete_report, [], "API error"
        )
        self.assertIn("Incomplete Extraction Notice", fallback_summary_text)
        self.assertIn("Failed sections: [3, 4]", fallback_summary_text)

    def test_unsupported_out_of_scope_questions_not_fabricated(self) -> None:
        """Verify that asking out-of-scope questions does not hallucinate answers."""
        rag_pipeline = create_rag_pipeline(api_key=None, retriever_type="bm25")
        rag_pipeline.ingest(self.parsed_pdf)

        mock_client = MagicMock()
        qa_engine = ReportQAEngine(
            api_key="mock_key",
            client=mock_client,
            base_delay_seconds=0.01,
        )

        # 1. Query completely missing from document terms (retrieval returns 0 chunks)
        unsupported_query = "What is the quantum teleportation frequency and hyperdrive fuel capacity?"
        qa_result = qa_engine.answer_question(unsupported_query, rag_pipeline)

        self.assertTrue(qa_result.has_insufficient_evidence)
        self.assertEqual(qa_result.sources, [])
        self.assertEqual(qa_result.cited_pages, [])
        self.assertIn("not contain enough information", qa_result.answer)
        # Ensure Gemini model was never called
        mock_client.models.generate_content.assert_not_called()

        # 2. Query where passages are retrieved, but Gemini properly identifies insufficient evidence
        mock_resp = MagicMock()
        mock_resp.text = "The report does not contain enough information to answer what the aircraft hangar address is. [Page 1]"
        mock_client.models.generate_content.return_value = mock_resp

        partially_matched_query = "What is the corporate aircraft hangar address?"
        qa_result_partial = qa_engine.answer_question(partially_matched_query, rag_pipeline)
        self.assertTrue(qa_result_partial.has_insufficient_evidence)
        self.assertIn("not contain enough information", qa_result_partial.answer)

    def test_zero_values_clearly_distinguished_from_unavailable_metrics(self) -> None:
        """Verify that zero debt or zero profit is distinct from missing/unavailable metrics."""
        report = FinancialExtractionReport(
            company_name="Zero Debt Corp",
            annual_reports=[
                AnnualFinancialData(
                    year=2024,
                    income_statement=IncomeStatement(
                        revenue=FinancialValue(value=500.0, source_page=2),
                        cost_of_goods_sold=FinancialValue(value=None, is_unavailable=True),
                        operating_profit=FinancialValue(value=0.0, source_page=2),
                    ),
                    balance_sheet=BalanceSheet(
                        total_assets=FinancialValue(value=1000.0, source_page=3),
                        total_debt=FinancialValue(value=0.0, source_page=3),
                        equity=FinancialValue(value=1000.0, source_page=3),
                    ),
                    source_pages=[2, 3],
                )
            ],
        )

        ratios = calculate_report_ratios(report)
        self.assertEqual(len(ratios), 1)

        # Gross margin is None because COGS is unavailable
        self.assertIsNone(ratios[0].gross_margin_pct)

        # Debt to equity is 0.0 because debt is zero (not unavailable)
        self.assertEqual(ratios[0].debt_to_equity, 0.0)

        # safe_metric_value returns None for unavailable COGS, but returns 0.0 for zero debt
        cogs_val = safe_metric_value(report.annual_reports[0].income_statement, "cost_of_goods_sold")
        debt_val = safe_metric_value(report.annual_reports[0].balance_sheet, "total_debt")
        self.assertIsNone(cogs_val)
        self.assertEqual(debt_val, 0.0)


if __name__ == "__main__":
    unittest.main()
