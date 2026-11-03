"""Unit tests for FinSight export functionality and RAG embedding fallback.

Tests:
1. Consolidated CSV export with complete and unavailable metrics.
2. Executive report PDF generation using ReportLab.
3. Preservation of reporting years, currency, units, and source pages.
4. Embedding fallback: retaining indexed documents and switching to BM25 when embeddings fail.
5. Preservation of cached embeddings upon failure.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from analytics.ratios import FinancialRatios, calculate_report_ratios
from extraction.financial_schema import (
    AnnualFinancialData,
    BalanceSheet,
    CashFlowStatement,
    FinancialExtractionReport,
    FinancialValue,
    IncomeStatement,
)
from generation.export import generate_consolidated_csv, generate_executive_pdf
from generation.summary import ExecutiveSummary
from rag.pipeline import RAGPipeline, create_rag_pipeline
from rag.retriever import (
    BM25Retriever,
    EmbeddingCache,
    GeminiEmbeddingGenerator,
    HybridRetriever,
    VectorRetriever,
)
from rag.schema import DocumentChunk


class TestConsolidatedCSVExport(unittest.TestCase):
    """Test suite for consolidated CSV export functionality."""

    def setUp(self) -> None:
        self.sample_report = FinancialExtractionReport(
            company_name="Apex Global Technologies",
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
        )
        self.sample_ratios = calculate_report_ratios(self.sample_report)

    def test_consolidated_csv_includes_all_statements_and_ratios(self) -> None:
        """Verify consolidated CSV includes income, balance sheet, cash flow, and ratios."""
        csv_text = generate_consolidated_csv(self.sample_report, self.sample_ratios)
        lines = csv_text.strip().splitlines()

        # Check headers
        header = lines[0]
        self.assertIn("Statement", header)
        self.assertIn("Metric", header)
        self.assertIn("Currency", header)
        self.assertIn("Unit", header)
        self.assertIn("FY 2023 Value", header)
        self.assertIn("FY 2023 Source Page", header)
        self.assertIn("FY 2024 Value", header)
        self.assertIn("FY 2024 Source Page", header)

        # Check specific statements and metrics present
        self.assertIn("Income Statement,Revenue,USD,millions,1000.0,Page 2,1250.0,Page 2", csv_text)
        self.assertIn("Income Statement,Net Profit (Net Income),USD,millions,150.0,Page 2,195.0,Page 2", csv_text)
        self.assertIn("Balance Sheet,Total Debt (Borrowings),USD,millions,400.0,Page 3,450.0,Page 3", csv_text)
        self.assertIn("Balance Sheet,Shareholders' Equity,USD,millions,1300.0,Page 3,1550.0,Page 3", csv_text)
        self.assertIn("Cash Flow Statement,Operating Cash Flow,USD,millions,280.0,Page 4,350.0,Page 4", csv_text)

        # Check ratios
        self.assertIn("Financial Ratios,Gross Margin (%),Ratio,%,40.0,Calculated,42.0,Calculated", csv_text)
        self.assertIn("Financial Ratios,Net Profit Margin (%),Ratio,%,15.0,Calculated,15.6,Calculated", csv_text)
        self.assertIn("Financial Ratios,YoY Revenue Growth (%),Ratio,%,Unavailable,Unavailable,25.0,Calculated", csv_text)

    def test_consolidated_csv_preserves_unavailable_metrics_without_fabrication(self) -> None:
        """Verify unavailable metrics appear explicitly as 'Unavailable' in CSV."""
        partial_report = FinancialExtractionReport(
            company_name="Partial Metrics Corp",
            annual_reports=[
                AnnualFinancialData(
                    year=2024,
                    currency="EUR",
                    unit="thousands",
                    income_statement=IncomeStatement(
                        revenue=FinancialValue(value=800.0, source_page=5),
                        cost_of_goods_sold=FinancialValue(value=None, is_unavailable=True),
                        operating_profit=FinancialValue(value=100.0, source_page=5),
                        net_profit=FinancialValue(value=80.0, source_page=5),
                    ),
                    balance_sheet=BalanceSheet(
                        total_assets=FinancialValue(value=1500.0, source_page=6),
                        total_liabilities=FinancialValue(value=None, is_unavailable=True),
                        total_debt=FinancialValue(value=0.0, source_page=6),  # Zero debt, not unavailable
                        equity=FinancialValue(value=1500.0, source_page=6),
                    ),
                    cash_flow_statement=CashFlowStatement(
                        operating_cash_flow=FinancialValue(value=None, is_unavailable=True),
                        capital_expenditure=FinancialValue(value=50.0, source_page=7),
                    ),
                    source_pages=[5, 6, 7],
                )
            ],
        )
        partial_ratios = calculate_report_ratios(partial_report)
        csv_text = generate_consolidated_csv(partial_report, partial_ratios)

        # COGS should be Unavailable
        self.assertIn("Cost of Goods Sold (COGS),EUR,thousands,Unavailable,Unavailable", csv_text)
        # Gross Margin % must be Unavailable since COGS was missing
        self.assertIn("Gross Margin (%),Ratio,%,Unavailable,Unavailable", csv_text)
        # Total Debt is 0.0, which is distinct from Unavailable
        self.assertIn("Total Debt (Borrowings),EUR,thousands,0.0,Page 6", csv_text)
        # Operating Cash Flow is Unavailable
        self.assertIn("Operating Cash Flow,EUR,thousands,Unavailable,Unavailable", csv_text)

    def test_consolidated_csv_empty_report_handled_safely(self) -> None:
        """Verify empty report returns clean CSV notification without crashing."""
        empty_report = FinancialExtractionReport(company_name="Empty Corp", annual_reports=[])
        csv_text = generate_consolidated_csv(empty_report, [])
        self.assertIn("Notice,No annual financial data available,Unavailable", csv_text)


class TestExecutivePDFExport(unittest.TestCase):
    """Test suite for ReportLab PDF executive report generation."""

    def setUp(self) -> None:
        self.report = FinancialExtractionReport(
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
        )
        self.ratios = calculate_report_ratios(self.report)
        self.summary = ExecutiveSummary(
            company_name="Zenith Technologies Limited",
            reporting_period="FY 2023 - FY 2024",
            summary_text=(
                "## 🏢 Executive Overview\nZenith demonstrated strong top-line revenue expansion of 25.0% [Page 2].\n\n"
                "## 💪 Financial Strengths\n- Healthy net margin of 15.6% [Page 2].\n- Robust operating cash flows [Page 4].\n\n"
                "## ⚠️ Strategic Risks\n- Increasing borrowings require debt service monitoring [Page 3]."
            ),
            cited_pages=[2, 3, 4],
        )

    def test_pdf_export_generates_valid_pdf_bytes(self) -> None:
        """Verify generate_executive_pdf produces a valid, non-empty PDF binary."""
        pdf_bytes = generate_executive_pdf(self.report, self.ratios, self.summary)

        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 2000)
        # Check standard PDF magic header
        self.assertTrue(pdf_bytes.startswith(b"%PDF-1."))

    def test_pdf_export_with_incomplete_extraction_notice(self) -> None:
        """Verify PDF includes incomplete extraction notice without failing layout."""
        incomplete_report = FinancialExtractionReport(
            company_name="Incomplete Extraction Corp",
            is_incomplete=True,
            failed_sections=[3, 4],
            unprocessed_sections=[5],
            annual_reports=[
                AnnualFinancialData(
                    year=2024,
                    income_statement=IncomeStatement(
                        revenue=FinancialValue(value=500.0, source_page=2),
                        net_profit=FinancialValue(value=50.0, source_page=2),
                    ),
                    source_pages=[2],
                )
            ],
        )
        ratios = calculate_report_ratios(incomplete_report)
        incomplete_summary = ExecutiveSummary(
            company_name="Incomplete Extraction Corp",
            reporting_period="FY 2024",
            summary_text="Partial extraction summary.",
            is_incomplete=True,
        )

        pdf_bytes = generate_executive_pdf(incomplete_report, ratios, incomplete_summary)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-1."))
        self.assertGreater(len(pdf_bytes), 1500)

    def test_pdf_export_with_none_summary_handles_cleanly(self) -> None:
        """Verify PDF generates cleanly even if summary is None."""
        pdf_bytes = generate_executive_pdf(self.report, self.ratios, summary=None)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-1."))
        self.assertGreater(len(pdf_bytes), 2000)


class TestRAGEmbeddingFallback(unittest.TestCase):
    """Test suite for graceful fallback from Gemini embeddings to BM25."""

    def test_embedding_failure_retains_document_and_uses_bm25(self) -> None:
        """Verify that when Gemini embedding fails, document remains indexed and BM25 answers."""
        bm25 = BM25Retriever()

        # Mock vector retriever that raises quota / network error on index
        def failing_embed_fn(texts: list[str]) -> list[list[float]]:
            raise RuntimeError("503 UNAVAILABLE: Model is overloaded or quota exhausted")

        vec_retriever = VectorRetriever(embed_fn=failing_embed_fn)
        hybrid = HybridRetriever(bm25_retriever=bm25, vector_retriever=vec_retriever)

        chunks = [
            DocumentChunk(
                chunk_id="chunk_1",
                page_number=2,
                text="Revenue from cloud computing operations grew to $1,250 million in FY2024.",
                char_count=73,
            ),
            DocumentChunk(
                chunk_id="chunk_2",
                page_number=5,
                text="Risk factors include currency volatility and semiconductor component lead times.",
                char_count=80,
            ),
        ]

        # Ingest chunks: VectorRetriever fails, but Hybrid catches and retains BM25
        hybrid.index(chunks)

        # Chunks must NOT be discarded
        self.assertEqual(hybrid.chunk_count, 2)
        self.assertTrue(hybrid.semantic_unavailable)
        self.assertIn("503 UNAVAILABLE", hybrid.embedding_error or "")

        # Retrieval should still succeed via BM25
        results = hybrid.retrieve("cloud computing operations", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "chunk_1")
        self.assertEqual(results[0].page_number, 2)

    def test_embedding_cache_not_discarded_on_subsequent_failure(self) -> None:
        """Verify that successfully generated embeddings remain cached even when a later batch fails."""
        cache = EmbeddingCache()
        cache.set("First successfully embedded financial passage", [0.1, 0.2, 0.3])
        self.assertEqual(cache.size, 1)

        # Simulate subsequent call failing
        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = RuntimeError("Resource exhausted: 429")

        generator = GeminiEmbeddingGenerator(
            api_key="mock_key",
            client=mock_client,
            cache=cache,
            max_retries=1,
            base_delay_seconds=0.01,
        )

        with self.assertRaises(RuntimeError):
            generator.embed_texts(["New text that fails to embed"])

        # The initial cached embedding was NOT discarded
        self.assertTrue(cache.contains("First successfully embedded financial passage"))
        self.assertEqual(cache.get("First successfully embedded financial passage"), [0.1, 0.2, 0.3])

    def test_rag_pipeline_semantic_unavailable_property(self) -> None:
        """Verify RAGPipeline exposes is_semantic_unavailable and embedding_error properties."""
        bm25 = BM25Retriever()
        vec = VectorRetriever(embed_fn=lambda texts: [[1.0, 0.0] for _ in texts])
        hybrid = HybridRetriever(bm25_retriever=bm25, vector_retriever=vec)

        pipeline = RAGPipeline(retriever=hybrid)
        self.assertFalse(pipeline.is_semantic_unavailable)
        self.assertIsNone(pipeline.embedding_error)

        # Manually flag unavailable
        hybrid.semantic_unavailable = True
        hybrid.embedding_error = "Gemini embedding quota exceeded"

        self.assertTrue(pipeline.is_semantic_unavailable)
        self.assertEqual(pipeline.embedding_error, "Gemini embedding quota exceeded")


if __name__ == "__main__":
    unittest.main()

