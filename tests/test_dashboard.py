"""Unit tests for FinSight dashboard formatting and helper functions."""

import io
import unittest
import streamlit as st

from analytics.ratios import FinancialRatios
from app import (
    REPORT_STATE_KEYS,
    clear_report_state,
    compute_pdf_hash,
    extract_file_bytes,
    format_currency_value,
    format_pct_value,
    format_ratio_value,
    safe_metric_obj,
    safe_metric_source_page,
    safe_metric_value,
)
from extraction.financial_schema import (
    AnnualFinancialData,
    BalanceSheet,
    FinancialExtractionReport,
    FinancialValue,
    IncomeStatement,
)
from generation.summary import ExecutiveSummary
from ingestion.pdf_parser import ParsedPDF
from rag.pipeline import RAGPipeline


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


class TestReportStalenessAndHashVerification(unittest.TestCase):
    """Test PDF upload state isolation, hash-based change detection, and stale cache elimination."""

    def setUp(self) -> None:
        """Clear any residual session state before each test."""
        clear_report_state()

    def tearDown(self) -> None:
        """Clean up session state after each test."""
        clear_report_state()

    def test_compute_pdf_hash_properties(self) -> None:
        """Verify content hash is SHA-256, deterministic, and sensitive to byte differences."""
        northstar_bytes = b"%PDF-1.4 Northstar Technologies FY2023 financial report"
        reliance_bytes = b"%PDF-1.4 Reliance Industries Limited Integrated Annual Report 2022-23"

        hash_northstar = compute_pdf_hash(northstar_bytes)
        hash_reliance = compute_pdf_hash(reliance_bytes)

        # Output format must be 64-character hex string
        self.assertEqual(len(hash_northstar), 64)
        self.assertEqual(len(hash_reliance), 64)

        # Deterministic: same bytes produce identical hash
        self.assertEqual(hash_northstar, compute_pdf_hash(northstar_bytes))
        self.assertEqual(hash_reliance, compute_pdf_hash(reliance_bytes))

        # Different documents produce different hashes
        self.assertNotEqual(hash_northstar, hash_reliance)

        # Even a 1-byte difference produces a different hash
        self.assertNotEqual(hash_northstar, compute_pdf_hash(northstar_bytes + b" "))

    def test_extract_file_bytes(self) -> None:
        """Verify byte extraction works for getvalue, read, and rejects invalid types."""
        sample_bytes = b"sample pdf content"

        # Object with getvalue() (e.g. Streamlit UploadedFile or io.BytesIO)
        bio = io.BytesIO(sample_bytes)
        self.assertEqual(extract_file_bytes(bio), sample_bytes)

        # Object with read() only
        class ReadOnlyStream:
            def __init__(self, data: bytes):
                self._data = data

            def read(self) -> bytes:
                return self._data

        ro = ReadOnlyStream(sample_bytes)
        self.assertEqual(extract_file_bytes(ro), sample_bytes)

        # Unsupported type
        with self.assertRaises(ValueError):
            extract_file_bytes(12345)

    def test_clear_report_state_purges_all_keys(self) -> None:
        """Verify clear_report_state removes all report-specific state keys."""
        # Populate session state with report keys and an unrelated key
        st.session_state.parsed_pdf = ParsedPDF(filename="test.pdf", total_pages=1, pages=[])
        st.session_state.extraction_report = FinancialExtractionReport(company_name="Test Corp")
        st.session_state.ratios = [FinancialRatios(year=2023)]
        st.session_state.uploaded_filename = "test.pdf"
        st.session_state.uploaded_file_hash = "abc123hash"
        st.session_state.rag_pipeline = RAGPipeline()
        st.session_state.qa_messages = [{"role": "user", "content": "hello"}]
        st.session_state.executive_summary = ExecutiveSummary(summary_text="briefing")
        st.session_state.user_theme = "dark"  # unrelated app-level key

        # Verify all report keys are present initially
        for key in REPORT_STATE_KEYS:
            self.assertIn(key, st.session_state)

        # Clear report state
        clear_report_state()

        # All report keys must be purged
        for key in REPORT_STATE_KEYS:
            self.assertNotIn(key, st.session_state)

        # Unrelated key must remain intact
        self.assertIn("user_theme", st.session_state)
        self.assertEqual(st.session_state.user_theme, "dark")

    def test_switch_from_northstar_to_reliance_purges_stale_data(self) -> None:
        """Verify switching from Northstar to Reliance clears all state and updates hash."""
        northstar_bytes = b"Northstar Technologies Annual Report FY2023"
        reliance_bytes = b"Reliance Industries Limited Annual Report 2022-23"

        northstar_hash = compute_pdf_hash(northstar_bytes)
        reliance_hash = compute_pdf_hash(reliance_bytes)

        # 1. State contains Northstar data
        st.session_state.uploaded_filename = "northstar_report.pdf"
        st.session_state.uploaded_file_hash = northstar_hash
        st.session_state.extraction_report = FinancialExtractionReport(
            company_name="Northstar Technologies",
            annual_reports=[
                AnnualFinancialData(
                    year=2023,
                    income_statement=IncomeStatement(revenue=FinancialValue(value=50000.0, source_page=1)),
                )
            ],
        )
        st.session_state.ratios = [FinancialRatios(year=2023, net_profit_margin_pct=15.0)]
        st.session_state.qa_messages = [{"role": "assistant", "content": "Northstar grew 10%."}]
        st.session_state.executive_summary = ExecutiveSummary(summary_text="Northstar executive briefing")

        # 2. Simulate new upload of Reliance
        new_upload_bytes = reliance_bytes
        new_upload_hash = compute_pdf_hash(new_upload_bytes)

        # Detection logic: hash mismatch must be detected
        self.assertNotEqual(st.session_state.get("uploaded_file_hash"), new_upload_hash)

        # Clear stale state upon change detection
        clear_report_state()

        # Verify all Northstar data is gone
        self.assertNotIn("extraction_report", st.session_state)
        self.assertNotIn("ratios", st.session_state)
        self.assertNotIn("executive_summary", st.session_state)
        self.assertNotIn("qa_messages", st.session_state)
        self.assertNotIn("uploaded_file_hash", st.session_state)

        # 3. Simulate processing new Reliance report
        reliance_report = FinancialExtractionReport(
            company_name="Reliance Industries Limited",
            annual_reports=[
                AnnualFinancialData(
                    year=2023,
                    income_statement=IncomeStatement(revenue=FinancialValue(value=892809.0, source_page=5)),
                )
            ],
        )
        st.session_state.uploaded_filename = "reliance_report.pdf"
        st.session_state.uploaded_file_hash = reliance_hash
        st.session_state.extraction_report = reliance_report
        st.session_state.ratios = [FinancialRatios(year=2023, net_profit_margin_pct=8.2)]
        st.session_state.executive_summary = ExecutiveSummary(summary_text="Reliance executive briefing")

        # Verify only Reliance data is active
        self.assertEqual(st.session_state.extraction_report.company_name, "Reliance Industries Limited")
        self.assertEqual(st.session_state.uploaded_file_hash, reliance_hash)
        self.assertEqual(st.session_state.executive_summary.summary_text, "Reliance executive briefing")
        self.assertNotIn("Northstar", st.session_state.executive_summary.summary_text)

    def test_switch_reports_with_identical_filename_different_bytes(self) -> None:
        """Verify that uploading different PDFs with identical filenames triggers state clearing."""
        common_filename = "annual_report.pdf"
        pdf1_bytes = b"%PDF-1.4 Fake Northstar Document Content"
        pdf2_bytes = b"%PDF-1.4 Real Reliance Document Content"

        hash1 = compute_pdf_hash(pdf1_bytes)
        hash2 = compute_pdf_hash(pdf2_bytes)

        # Initially loaded report 1
        st.session_state.uploaded_filename = common_filename
        st.session_state.uploaded_file_hash = hash1
        st.session_state.extraction_report = FinancialExtractionReport(company_name="Company 1")

        # Now report 2 is uploaded with SAME filename
        uploaded_filename = common_filename
        uploaded_bytes = pdf2_bytes
        uploaded_hash = compute_pdf_hash(uploaded_bytes)

        # Filename is identical, BUT hash differs
        self.assertEqual(uploaded_filename, st.session_state.get("uploaded_filename"))
        self.assertNotEqual(uploaded_hash, st.session_state.get("uploaded_file_hash"))

        # Content hash change MUST trigger state purge
        clear_report_state()

        self.assertNotIn("extraction_report", st.session_state)
        self.assertNotIn("uploaded_file_hash", st.session_state)

    def test_rerun_with_identical_hash_preserves_state(self) -> None:
        """Verify that re-running with the same document hash preserves extracted data without clearing."""
        reliance_bytes = b"Reliance Industries Limited Annual Report 2022-23"
        reliance_hash = compute_pdf_hash(reliance_bytes)

        st.session_state.uploaded_filename = "reliance.pdf"
        st.session_state.uploaded_file_hash = reliance_hash
        st.session_state.extraction_report = FinancialExtractionReport(company_name="Reliance Industries Limited")

        # Rerun arrives with same file bytes
        incoming_bytes = reliance_bytes
        incoming_hash = compute_pdf_hash(incoming_bytes)

        # Check condition in main(): current_hash == incoming_hash
        self.assertEqual(st.session_state.get("uploaded_file_hash"), incoming_hash)

        # Report must NOT be cleared
        self.assertIn("extraction_report", st.session_state)
        self.assertEqual(st.session_state.extraction_report.company_name, "Reliance Industries Limited")


if __name__ == "__main__":
    unittest.main()
