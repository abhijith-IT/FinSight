"""Unit tests for FinSight financial extraction and validation modules."""

import json
import unittest
from unittest.mock import MagicMock, patch

import config
from extraction.financial_schema import (
    AnnualFinancialData,
    BalanceSheet,
    CashFlowStatement,
    FinancialExtractionReport,
    FinancialValue,
    IncomeStatement,
)
from extraction.extractor import (
    CORE_METRIC_KEYWORDS,
    ExtractionError,
    FinancialExtractor,
    clear_section_cache,
    extract_retry_delay,
    is_permanent_error,
    is_transient_error,
)
from extraction.validation import (
    clean_and_parse_llm_json,
    sanitize_and_flag_annual_data,
    validate_financial_report,
)
from ingestion.pdf_parser import ParsedPDF, ParsedPage


class TestFinancialSchemaAndValidation(unittest.TestCase):
    """Test schema instantiation, JSON parsing, and validation checks."""

    def test_schema_defaults_and_provenance(self) -> None:
        """Verify field defaults, provenance tracking, and missing flags."""
        val = FinancialValue(value=10500.5, source_page=12)
        self.assertEqual(val.value, 10500.5)
        self.assertEqual(val.source_page, 12)
        self.assertFalse(val.is_unavailable)

        # Null value flag
        empty_val = FinancialValue()
        self.assertIsNone(empty_val.value)
        self.assertFalse(empty_val.is_unavailable)

    def test_sanitize_and_flag_missing_values(self) -> None:
        """Verify unavailable metrics are set to None, flagged True, and collected."""
        annual = AnnualFinancialData(
            year=2023,
            currency="USD",
            unit="millions",
            income_statement=IncomeStatement(
                revenue=FinancialValue(value=50000.0, source_page=5),
            ),
        )

        sanitized = sanitize_and_flag_annual_data(annual)

        # Revenue was provided
        self.assertFalse(sanitized.income_statement.revenue.is_unavailable)
        self.assertEqual(sanitized.income_statement.revenue.value, 50000.0)

        # Net profit was omitted -> should be flagged as unavailable
        self.assertTrue(sanitized.income_statement.net_profit.is_unavailable)
        self.assertIsNone(sanitized.income_statement.net_profit.value)

        # Total assets, total debt, and COGS omitted -> flagged unavailable
        self.assertTrue(sanitized.balance_sheet.total_assets.is_unavailable)
        self.assertTrue(sanitized.balance_sheet.total_debt.is_unavailable)
        self.assertTrue(sanitized.income_statement.cost_of_goods_sold.is_unavailable)

        # Check missing_fields list
        self.assertIn("net_profit", sanitized.missing_fields)
        self.assertIn("total_assets", sanitized.missing_fields)
        self.assertIn("total_debt", sanitized.missing_fields)
        self.assertIn("cost_of_goods_sold", sanitized.missing_fields)
        self.assertNotIn("revenue", sanitized.missing_fields)

    def test_clean_and_parse_llm_json(self) -> None:
        """Test markdown code fence stripping and malformed json handling."""
        raw_markdown = """```json
        {
            "company_name": "Acme Corp",
            "annual_reports": []
        }
        ```"""
        parsed = clean_and_parse_llm_json(raw_markdown)
        self.assertEqual(parsed["company_name"], "Acme Corp")

        # Invalid JSON
        with self.assertRaises(ValueError):
            clean_and_parse_llm_json("This is not JSON at all")

        with self.assertRaises(ValueError):
            clean_and_parse_llm_json("")

    def test_balance_sheet_equation_validation(self) -> None:
        """Test that balanced and unbalanced balance sheets are flagged accurately."""
        # Unbalanced: Assets=1000, Liab=400, Equity=400 (diff=200)
        unbalanced_annual = AnnualFinancialData(
            year=2023,
            balance_sheet=BalanceSheet(
                total_assets=FinancialValue(value=1000.0, source_page=10),
                total_liabilities=FinancialValue(value=400.0, source_page=10),
                equity=FinancialValue(value=400.0, source_page=10),
            ),
        )
        report = FinancialExtractionReport(annual_reports=[unbalanced_annual])
        val_report = validate_financial_report(report)

        self.assertTrue(val_report.is_valid)  # Valid format, but with warning
        self.assertTrue(any("Balance Sheet equation discrepancy" in issue.message for issue in val_report.warnings))

        # Balanced: Assets=1000, Liab=600, Equity=400
        balanced_annual = AnnualFinancialData(
            year=2023,
            balance_sheet=BalanceSheet(
                total_assets=FinancialValue(value=1000.0, source_page=10),
                total_liabilities=FinancialValue(value=600.0, source_page=10),
                equity=FinancialValue(value=400.0, source_page=10),
            ),
        )
        report_balanced = FinancialExtractionReport(annual_reports=[balanced_annual])
        val_report_balanced = validate_financial_report(report_balanced)
        self.assertEqual(len(val_report_balanced.warnings), 0)


class TestFinancialExtractor(unittest.TestCase):
    """Test extractor partitioning, Gemini API integration, and section merging."""

    def setUp(self) -> None:
        clear_section_cache()

    def tearDown(self) -> None:
        clear_section_cache()

    def test_partition_pages_respects_limits_and_keywords(self) -> None:
        """Ensure long documents are partitioned into bounded sections."""
        pages = [
            ParsedPage(page_number=1, text="Cover page and overview", has_text=True),
            ParsedPage(page_number=2, text="Consolidated Balance Sheet Total Assets $1000", has_text=True),
            ParsedPage(page_number=3, text="Notes to financial statements", has_text=True),
            ParsedPage(page_number=4, text="Consolidated Statement of Operations Revenues $500", has_text=True),
            ParsedPage(page_number=5, text="Executive compensation and governance", has_text=True),
        ]
        parsed_pdf = ParsedPDF(total_pages=5, pages=pages)

        extractor = FinancialExtractor(api_key="test_key", max_pages_per_section=2)
        sections = extractor.partition_pages(parsed_pdf)

        # Verify pages with keywords (2, 4) plus adjacent pages are captured
        self.assertTrue(len(sections) >= 1)
        for sec in sections:
            self.assertLessEqual(len(sec), 2)
            total_chars = sum(len(p.text) for p in sec)
            self.assertLessEqual(total_chars, 15000)

    def test_extract_section_mock_success(self) -> None:
        """Verify extract_section correctly parses JSON response into FinancialExtractionReport."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "company_name": "TestCorp",
            "annual_reports": [{
                "year": 2024,
                "currency": "USD",
                "unit": "millions",
                "income_statement": {
                    "revenue": {"value": 5000.0, "source_page": 2, "is_unavailable": False},
                    "net_profit": {"value": 800.0, "source_page": 2, "is_unavailable": False},
                },
                "balance_sheet": {
                    "total_assets": {"value": 15000.0, "source_page": 3, "is_unavailable": False},
                },
                "cash_flow_statement": {},
                "source_pages": [2, 3],
            }],
            "extraction_notes": ["Parsed successfully"]
        })
        mock_client.models.generate_content.return_value = mock_response

        extractor = FinancialExtractor(client=mock_client)
        pages = [
            ParsedPage(page_number=2, text="Revenues 5000 Net income 800", has_text=True),
            ParsedPage(page_number=3, text="Total assets 15000", has_text=True),
        ]

        result = extractor.extract_section(pages)

        self.assertEqual(result.company_name, "TestCorp")
        self.assertEqual(len(result.annual_reports), 1)
        self.assertEqual(result.annual_reports[0].income_statement.revenue.value, 5000.0)
        self.assertEqual(result.annual_reports[0].income_statement.revenue.source_page, 2)
        self.assertEqual(result.annual_reports[0].balance_sheet.total_assets.value, 15000.0)
        self.assertEqual(result.annual_reports[0].source_pages, [2, 3])

    def test_extract_merges_across_multiple_sections(self) -> None:
        """Verify that multi-section extractions merge by year seamlessly."""
        mock_client = MagicMock()

        # Section 1 returns Income Statement with COGS for 2023
        resp_sec1 = MagicMock()
        resp_sec1.text = json.dumps({
            "company_name": "Consolidated Corp",
            "annual_reports": [{
                "year": 2023,
                "currency": "USD",
                "income_statement": {
                    "revenue": {"value": 1200.0, "source_page": 2, "is_unavailable": False},
                    "cost_of_goods_sold": {"value": 700.0, "source_page": 2, "is_unavailable": False},
                },
                "source_pages": [2],
            }],
        })

        # Section 2 returns Balance Sheet with Total Debt for 2023
        resp_sec2 = MagicMock()
        resp_sec2.text = json.dumps({
            "company_name": "Consolidated Corp",
            "annual_reports": [{
                "year": 2023,
                "currency": "USD",
                "balance_sheet": {
                    "total_assets": {"value": 3000.0, "source_page": 4, "is_unavailable": False},
                    "total_debt": {"value": 600.0, "source_page": 4, "is_unavailable": False},
                },
                "source_pages": [4],
            }],
        })

        mock_client.models.generate_content.side_effect = [resp_sec1, resp_sec2]

        pages = [
            ParsedPage(page_number=2, text="Income Statement Revenue $1200", has_text=True),
            ParsedPage(page_number=4, text="Balance Sheet Total Assets $3000", has_text=True),
        ]
        parsed_pdf = ParsedPDF(total_pages=4, pages=pages)

        # Force each page into a separate section
        extractor = FinancialExtractor(client=mock_client, max_pages_per_section=1)
        report = extractor.extract(parsed_pdf)

        self.assertEqual(len(report.annual_reports), 1)
        annual = report.annual_reports[0]
        self.assertEqual(annual.year, 2023)
        self.assertEqual(annual.income_statement.revenue.value, 1200.0)
        self.assertEqual(annual.income_statement.cost_of_goods_sold.value, 700.0)
        self.assertEqual(annual.balance_sheet.total_assets.value, 3000.0)
        self.assertEqual(annual.balance_sheet.total_debt.value, 600.0)
        self.assertEqual(annual.source_pages, [2, 4])

    @patch("time.sleep", return_value=None)
    def test_api_failure_handled_gracefully(self, mock_sleep: MagicMock) -> None:
        """Verify API error does not crash overall execution and is noted."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("Rate limit exceeded")

        pages = [ParsedPage(page_number=1, text="Income Statement", has_text=True)]
        parsed_pdf = ParsedPDF(total_pages=1, pages=pages)

        extractor = FinancialExtractor(client=mock_client)
        report = extractor.extract(parsed_pdf)

        self.assertEqual(len(report.annual_reports), 0)
        self.assertTrue(any("extraction note" in note.lower() or "paused" in note.lower() for note in report.extraction_notes))

    def test_missing_api_key_raises_extraction_error(self) -> None:
        """Verify missing API key raises an informative ExtractionError."""
        with patch.object(config, "GEMINI_API_KEY", None):
            extractor = FinancialExtractor(api_key=None, client=None)
            with self.assertRaises(ExtractionError):
                _ = extractor.client

    def test_early_stopping_avoids_unnecessary_gemini_calls(self) -> None:
        """Verify that when all financial metrics are complete, subsequent sections are skipped."""
        mock_client = MagicMock()
        resp = MagicMock()
        resp.text = json.dumps({
            "company_name": "Fast Corp",
            "annual_reports": [{
                "year": 2023,
                "currency": "USD",
                "income_statement": {
                    "revenue": {"value": 1000.0, "source_page": 2, "is_unavailable": False},
                    "cost_of_goods_sold": {"value": 600.0, "source_page": 2, "is_unavailable": False},
                    "operating_expenses": {"value": 200.0, "source_page": 2, "is_unavailable": False},
                    "operating_profit": {"value": 200.0, "source_page": 2, "is_unavailable": False},
                    "net_profit": {"value": 150.0, "source_page": 2, "is_unavailable": False},
                },
                "balance_sheet": {
                    "total_assets": {"value": 5000.0, "source_page": 2, "is_unavailable": False},
                    "total_liabilities": {"value": 2000.0, "source_page": 2, "is_unavailable": False},
                    "total_debt": {"value": 500.0, "source_page": 2, "is_unavailable": False},
                    "equity": {"value": 3000.0, "source_page": 2, "is_unavailable": False},
                },
                "cash_flow_statement": {
                    "operating_cash_flow": {"value": 400.0, "source_page": 2, "is_unavailable": False},
                    "capital_expenditure": {"value": 100.0, "source_page": 2, "is_unavailable": False},
                },
                "source_pages": [2],
                "missing_fields": [],
            }],
        })
        mock_client.models.generate_content.return_value = resp

        # 3 separate sections
        pages = [
            ParsedPage(page_number=2, text="Consolidated Income Statement and Balance Sheet", has_text=True),
            ParsedPage(page_number=4, text="Note 1 Summary of Significant Accounting Policies", has_text=True),
            ParsedPage(page_number=6, text="Note 2 Additional General Disclosures", has_text=True),
        ]
        parsed_pdf = ParsedPDF(total_pages=6, pages=pages)

        extractor = FinancialExtractor(client=mock_client, max_pages_per_section=1)
        report = extractor.extract(parsed_pdf)

        self.assertEqual(len(report.annual_reports), 1)
        # Verify early stopping: Gemini was called ONLY ONCE because section 1 was complete!
        self.assertEqual(mock_client.models.generate_content.call_count, 1)

    def test_extractor_caches_extraction_for_same_document(self) -> None:
        """Verify that extracting the same parsed PDF twice reuses cached results without calling Gemini."""
        mock_client = MagicMock()
        resp = MagicMock()
        resp.text = json.dumps({
            "company_name": "Cached Corp",
            "annual_reports": [{
                "year": 2023,
                "income_statement": {"revenue": {"value": 5000.0, "source_page": 1, "is_unavailable": False}},
            }],
        })
        mock_client.models.generate_content.return_value = resp

        pages = [ParsedPage(page_number=1, text="Income Statement Revenue $5000", has_text=True)]
        parsed_pdf = ParsedPDF(total_pages=1, pages=pages)

        extractor = FinancialExtractor(client=mock_client)
        # First extraction: invokes Gemini
        report1 = extractor.extract(parsed_pdf)
        self.assertEqual(mock_client.models.generate_content.call_count, 1)

        # Second extraction with same ParsedPDF: served from internal cache!
        report2 = extractor.extract(parsed_pdf)
        self.assertEqual(mock_client.models.generate_content.call_count, 1)
        self.assertEqual(report1.company_name, report2.company_name)

    def test_partition_pages_prioritizes_primary_statement_pages(self) -> None:
        """Verify that in large documents, primary statement pages and their adjacent context are prioritized."""
        pages = []
        for i in range(1, 21):
            if i == 10:
                pages.append(ParsedPage(page_number=i, text="Consolidated Balance Sheet Total Assets $1000", has_text=True))
            elif i == 11:
                pages.append(ParsedPage(page_number=i, text="Consolidated Statement of Operations Income Statement", has_text=True))
            elif i == 12:
                pages.append(ParsedPage(page_number=i, text="Consolidated Statement of Cash Flows", has_text=True))
            else:
                # Narrative pages that happen to mention revenues
                pages.append(ParsedPage(page_number=i, text=f"Page {i}: General corporate overview and revenues discussion.", has_text=True))

        parsed_pdf = ParsedPDF(total_pages=20, pages=pages)
        extractor = FinancialExtractor(api_key="test_key", max_pages_per_section=4)
        sections = extractor.partition_pages(parsed_pdf)

        # Candidate pages should be focused on statement pages and immediate context
        all_candidate_pages = [p.page_number for sec in sections for p in sec]
        self.assertIn(10, all_candidate_pages)
        self.assertIn(11, all_candidate_pages)
        self.assertIn(12, all_candidate_pages)
        # Total candidate pages must be a small fraction of the 20 pages, not all 20 pages
        self.assertLessEqual(len(all_candidate_pages), 8)

    @patch("time.sleep", return_value=None)
    def test_retry_transient_503_eventual_success(self, mock_sleep: MagicMock) -> None:
        """Verify that HTTP 503 transient errors are retried with backoff and succeed."""
        mock_client = MagicMock()
        resp = MagicMock()
        resp.text = json.dumps({
            "company_name": "Resilient Corp",
            "annual_reports": [{
                "year": 2023,
                "income_statement": {
                    "revenue": {"value": 1000.0, "source_page": 1, "is_unavailable": False},
                },
            }],
        })
        err_503 = RuntimeError("503 UNAVAILABLE: The model is experiencing high demand. Please try again later.")
        mock_client.models.generate_content.side_effect = [err_503, resp]

        page = ParsedPage(page_number=1, text="Income Statement Revenue $1000", has_text=True)
        extractor = FinancialExtractor(client=mock_client, max_retries=3, base_delay=0.1)

        result = extractor.extract_section([page])

        self.assertEqual(result.company_name, "Resilient Corp")
        self.assertEqual(mock_client.models.generate_content.call_count, 2)
        mock_sleep.assert_called_once()
        self.assertGreater(mock_sleep.call_args[0][0], 0)

    @patch("time.sleep", return_value=None)
    def test_retry_transient_429_eventual_success(self, mock_sleep: MagicMock) -> None:
        """Verify that HTTP 429 rate limit errors are retried with backoff."""
        mock_client = MagicMock()
        resp = MagicMock()
        resp.text = json.dumps({
            "company_name": "RateLimited Corp",
            "annual_reports": [{
                "year": 2023,
                "income_statement": {
                    "revenue": {"value": 2000.0, "source_page": 1, "is_unavailable": False},
                },
            }],
        })
        err_429 = RuntimeError("429 RESOURCE_EXHAUSTED: Rate limit exceeded.")
        mock_client.models.generate_content.side_effect = [err_429, resp]

        page = ParsedPage(page_number=1, text="Revenue $2000", has_text=True)
        extractor = FinancialExtractor(client=mock_client, max_retries=3, base_delay=0.1)

        result = extractor.extract_section([page])
        self.assertEqual(result.company_name, "RateLimited Corp")
        self.assertEqual(mock_client.models.generate_content.call_count, 2)
        mock_sleep.assert_called_once()

    @patch("time.sleep", return_value=None)
    def test_retry_respects_server_delay(self, mock_sleep: MagicMock) -> None:
        """Verify that server-provided retry delay is respected."""
        mock_client = MagicMock()
        resp = MagicMock()
        resp.text = json.dumps({
            "company_name": "Delayed Corp",
            "annual_reports": [{
                "year": 2023,
                "income_statement": {"revenue": {"value": 3000.0, "source_page": 1, "is_unavailable": False}},
            }],
        })
        err = RuntimeError("503 UNAVAILABLE: Model overloaded. retry after 3.5s")
        mock_client.models.generate_content.side_effect = [err, resp]

        page = ParsedPage(page_number=1, text="Revenue $3000", has_text=True)
        extractor = FinancialExtractor(client=mock_client, max_retries=3, base_delay=1.0)

        result = extractor.extract_section([page])
        self.assertEqual(result.company_name, "Delayed Corp")
        mock_sleep.assert_called_once_with(3.5)

    @patch("time.sleep", return_value=None)
    def test_permanent_error_no_retry(self, mock_sleep: MagicMock) -> None:
        """Verify that permanent errors (e.g. 401 invalid API key) are never retried."""
        mock_client = MagicMock()
        err_401 = RuntimeError("401 UNAUTHENTICATED: Invalid API key provided.")
        mock_client.models.generate_content.side_effect = err_401

        page = ParsedPage(page_number=1, text="Revenue $1000", has_text=True)
        extractor = FinancialExtractor(client=mock_client, max_retries=3)

        with self.assertRaises(ExtractionError) as ctx:
            extractor.extract_section([page])

        self.assertIn("permanent error", str(ctx.exception).lower())
        self.assertEqual(mock_client.models.generate_content.call_count, 1)
        mock_sleep.assert_not_called()

    @patch("time.sleep", return_value=None)
    def test_repeated_failures_pause_gracefully(self, mock_sleep: MagicMock) -> None:
        """Verify that consecutive transient failures trigger early pause without crashing."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("503 UNAVAILABLE: high demand")

        pages = [
            ParsedPage(page_number=1, text="Section 1 Balance Sheet", has_text=True),
            ParsedPage(page_number=2, text="Section 2 Income Statement", has_text=True),
            ParsedPage(page_number=3, text="Section 3 Statement of Cash Flows", has_text=True),
            ParsedPage(page_number=4, text="Section 4 Statement of Financial Position", has_text=True),
        ]
        parsed_pdf = ParsedPDF(total_pages=4, pages=pages)

        extractor = FinancialExtractor(
            client=mock_client,
            max_pages_per_section=1,
            max_retries=1,
            max_consecutive_failures=2,
            max_sections_to_extract=4,
            fallback_model_name=None,
        )

        report = extractor.extract(parsed_pdf)

        self.assertTrue(report.is_incomplete)
        self.assertEqual(report.failed_sections, [1, 2])
        self.assertEqual(report.unprocessed_sections, [3, 4])
        # Section 1 called 2 times (initial + 1 retry), Section 2 called 2 times (initial + 1 retry).
        # Sections 3 and 4 were NOT called because extraction paused!
        self.assertEqual(mock_client.models.generate_content.call_count, 4)
        self.assertTrue(any("paused" in note.lower() for note in report.extraction_notes))

    @patch("time.sleep", return_value=None)
    def test_preserve_partial_results_when_later_section_fails(self, mock_sleep: MagicMock) -> None:
        """Verify successfully extracted data is preserved when a later section fails."""
        mock_client = MagicMock()

        resp_sec1 = MagicMock()
        resp_sec1.text = json.dumps({
            "company_name": "Partial Corp",
            "annual_reports": [{
                "year": 2023,
                "currency": "USD",
                "income_statement": {
                    "revenue": {"value": 50000.0, "source_page": 1, "is_unavailable": False},
                },
                "source_pages": [1],
            }],
        })
        err_503 = RuntimeError("503 UNAVAILABLE: Model overloaded")
        mock_client.models.generate_content.side_effect = [resp_sec1, err_503, err_503]

        pages = [
            ParsedPage(page_number=1, text="Income Statement Revenue $50000", has_text=True),
            ParsedPage(page_number=2, text="Balance Sheet Total Assets $80000", has_text=True),
        ]
        parsed_pdf = ParsedPDF(total_pages=2, pages=pages)

        extractor = FinancialExtractor(
            client=mock_client,
            max_pages_per_section=1,
            max_retries=1,
            max_consecutive_failures=2,
            fallback_model_name=None,
        )

        report = extractor.extract(parsed_pdf)

        self.assertTrue(report.is_incomplete)
        self.assertEqual(report.company_name, "Partial Corp")
        self.assertEqual(len(report.annual_reports), 1)
        self.assertEqual(report.annual_reports[0].income_statement.revenue.value, 50000.0)
        self.assertIn(2, report.failed_sections)

    @patch("time.sleep", return_value=None)
    def test_partial_result_recovery_reuses_cached_sections(self, mock_sleep: MagicMock) -> None:
        """Verify that re-running extraction on an incomplete document reuses successful sections."""
        mock_client = MagicMock()

        resp_sec1 = MagicMock()
        resp_sec1.text = json.dumps({
            "company_name": "Recovery Corp",
            "annual_reports": [{
                "year": 2023,
                "currency": "USD",
                "income_statement": {
                    "revenue": {"value": 90000.0, "source_page": 1, "is_unavailable": False},
                },
                "source_pages": [1],
            }],
        })

        err_503 = RuntimeError("503 UNAVAILABLE: Model overloaded")

        # First run: Section 1 succeeds, Section 2 fails (1 initial + 1 retry = 2 calls)
        mock_client.models.generate_content.side_effect = [resp_sec1, err_503, err_503]

        pages = [
            ParsedPage(page_number=1, text="Income Statement Revenue $90000", has_text=True),
            ParsedPage(page_number=2, text="Balance Sheet Total Assets $120000", has_text=True),
        ]
        parsed_pdf = ParsedPDF(total_pages=2, pages=pages)

        extractor = FinancialExtractor(
            client=mock_client,
            max_pages_per_section=1,
            max_retries=1,
            max_consecutive_failures=2,
            fallback_model_name=None,
        )

        report1 = extractor.extract(parsed_pdf)
        self.assertTrue(report1.is_incomplete)
        self.assertEqual(mock_client.models.generate_content.call_count, 3)

        # Recovery run: Section 2 now succeeds!
        resp_sec2 = MagicMock()
        resp_sec2.text = json.dumps({
            "company_name": "Recovery Corp",
            "annual_reports": [{
                "year": 2023,
                "currency": "USD",
                "balance_sheet": {
                    "total_assets": {"value": 120000.0, "source_page": 2, "is_unavailable": False},
                },
                "source_pages": [2],
            }],
        })
        mock_client.models.generate_content.side_effect = [resp_sec2]
        mock_client.models.generate_content.reset_mock()

        report2 = extractor.extract(parsed_pdf)
        # Verify: Section 1 was retrieved from section cache (NOT called again!)
        # Only Section 2 was requested from Gemini!
        self.assertEqual(mock_client.models.generate_content.call_count, 1)
        self.assertFalse(report2.is_incomplete)
        self.assertEqual(len(report2.annual_reports), 1)
        ann = report2.annual_reports[0]
        self.assertEqual(ann.income_statement.revenue.value, 90000.0)
        self.assertEqual(ann.balance_sheet.total_assets.value, 120000.0)

    def test_merge_metric_does_not_overwrite_with_null(self) -> None:
        """Verify that merging values never replaces existing valid numbers with null."""
        extractor = FinancialExtractor(api_key="test_key")
        target = FinancialValue(value=500.0, source_page=2, is_unavailable=False)
        incoming_null = FinancialValue(value=None, source_page=4, is_unavailable=True)

        extractor._merge_metric(target, incoming_null)
        self.assertEqual(target.value, 500.0)
        self.assertEqual(target.source_page, 2)
        self.assertFalse(target.is_unavailable)

    def test_error_classification_helpers(self) -> None:
        """Verify transient and permanent error classification."""
        self.assertTrue(is_transient_error(RuntimeError("503 UNAVAILABLE")))
        self.assertTrue(is_transient_error(RuntimeError("429 RESOURCE_EXHAUSTED")))
        self.assertTrue(is_transient_error(RuntimeError("high demand, please try again")))
        self.assertFalse(is_transient_error(RuntimeError("401 Invalid API key")))
        self.assertFalse(is_transient_error(RuntimeError("400 Bad Request")))

        self.assertTrue(is_permanent_error(RuntimeError("401 UNAUTHENTICATED")))
        self.assertTrue(is_permanent_error(RuntimeError("invalid api key")))
        self.assertTrue(is_permanent_error(RuntimeError("400 Bad Request")))
        self.assertFalse(is_permanent_error(RuntimeError("503 UNAVAILABLE")))

        # Delay extraction
        self.assertEqual(extract_retry_delay(RuntimeError("retry after 5.0s")), 5.0)
        self.assertEqual(extract_retry_delay(RuntimeError("wait 12s")), 12.0)
        self.assertIsNone(extract_retry_delay(RuntimeError("unknown error")))


class TestExtractionMetricMappingRegressions(unittest.TestCase):
    """Regression tests for known incorrect metric mappings.

    These tests guard against systematic prompt-level extraction errors:
    - "Other Expenses" misused as operating_expenses
    - EBITDA substituted for operating_profit
    - Total liabilities used instead of borrowings for total_debt
    - NCI-inclusive total equity used instead of owners' equity
    - Prior-year comparative columns (e.g. FY2023 in a FY2024 report) being dropped
    """

    def setUp(self) -> None:
        clear_section_cache()

    def tearDown(self) -> None:
        clear_section_cache()

    def _make_extractor_with_response(self, response_json: dict) -> tuple:
        """Helper: create mock client returning a fixed JSON response."""
        mock_client = MagicMock()
        resp = MagicMock()
        resp.text = json.dumps(response_json)
        mock_client.models.generate_content.return_value = resp
        extractor = FinancialExtractor(client=mock_client)
        return extractor, mock_client

    # -------------------------------------------------------------------------
    # Regression 1: operating_expenses must not be populated from "Other Expenses"
    # -------------------------------------------------------------------------
    def test_other_expenses_must_not_populate_operating_expenses(self) -> None:
        """Verify that when the LLM correctly leaves operating_expenses null (no explicit OPEX line),
        the system preserves null and does not treat Other Expenses as OPEX."""
        response = {
            "company_name": "Indian Corp",
            "annual_reports": [{
                "year": 2024,
                "currency": "INR",
                "unit": "crores",
                "income_statement": {
                    "revenue": {"value": 900000.0, "source_page": 5, "is_unavailable": False},
                    "cost_of_goods_sold": {"value": None, "source_page": None, "is_unavailable": True},
                    "operating_expenses": {"value": None, "source_page": None, "is_unavailable": True},
                    "operating_profit": {"value": 75000.0, "source_page": 5, "is_unavailable": False},
                    "net_profit": {"value": 60000.0, "source_page": 5, "is_unavailable": False},
                },
                "balance_sheet": {},
                "cash_flow_statement": {},
                "source_pages": [5],
            }],
        }
        extractor, _ = self._make_extractor_with_response(response)
        page = ParsedPage(page_number=5, text="Statement of Profit & Loss Revenue 900000 Other Expenses 50000", has_text=True)
        result = extractor.extract_section([page])

        ann = result.annual_reports[0]
        # operating_expenses should remain null — "Other Expenses" is NOT a valid OPEX aggregate
        self.assertIsNone(ann.income_statement.operating_expenses.value)
        self.assertTrue(ann.income_statement.operating_expenses.is_unavailable)
        # operating_profit sourced from explicit line should be present
        self.assertEqual(ann.income_statement.operating_profit.value, 75000.0)

    # -------------------------------------------------------------------------
    # Regression 2: EBITDA must not be substituted for operating_profit
    # -------------------------------------------------------------------------
    def test_ebitda_must_not_substitute_for_operating_profit(self) -> None:
        """Verify that when only EBITDA is reported (not EBIT/operating profit),
        the system preserves null for operating_profit."""
        response = {
            "company_name": "EBITDA Corp",
            "annual_reports": [{
                "year": 2024,
                "currency": "USD",
                "unit": "millions",
                "income_statement": {
                    "revenue": {"value": 50000.0, "source_page": 3, "is_unavailable": False},
                    "cost_of_goods_sold": {"value": None, "source_page": None, "is_unavailable": True},
                    "operating_expenses": {"value": None, "source_page": None, "is_unavailable": True},
                    "operating_profit": {"value": None, "source_page": None, "is_unavailable": True},
                    "net_profit": {"value": 8000.0, "source_page": 3, "is_unavailable": False},
                },
                "balance_sheet": {},
                "cash_flow_statement": {},
                "source_pages": [3],
            }],
        }
        extractor, _ = self._make_extractor_with_response(response)
        page = ParsedPage(page_number=3, text="Revenue 50000 EBITDA 12000 Net profit 8000", has_text=True)
        result = extractor.extract_section([page])

        ann = result.annual_reports[0]
        self.assertIsNone(ann.income_statement.operating_profit.value)
        self.assertTrue(ann.income_statement.operating_profit.is_unavailable)

    # -------------------------------------------------------------------------
    # Regression 3: total_debt must not be set from total_liabilities
    # -------------------------------------------------------------------------
    def test_total_liabilities_must_not_populate_total_debt(self) -> None:
        """Verify that total_debt remains null when only total_liabilities is reported."""
        response = {
            "company_name": "Liabilities Corp",
            "annual_reports": [{
                "year": 2024,
                "currency": "INR",
                "unit": "crores",
                "income_statement": {},
                "balance_sheet": {
                    "total_assets": {"value": 500000.0, "source_page": 8, "is_unavailable": False},
                    "total_liabilities": {"value": 300000.0, "source_page": 8, "is_unavailable": False},
                    "total_debt": {"value": None, "source_page": None, "is_unavailable": True},
                    "equity": {"value": 200000.0, "source_page": 8, "is_unavailable": False},
                },
                "cash_flow_statement": {},
                "source_pages": [8],
            }],
        }
        extractor, _ = self._make_extractor_with_response(response)
        page = ParsedPage(page_number=8, text="Total Assets 500000 Total Liabilities 300000 Equity 200000", has_text=True)
        result = extractor.extract_section([page])

        ann = result.annual_reports[0]
        self.assertIsNone(ann.balance_sheet.total_debt.value)
        self.assertTrue(ann.balance_sheet.total_debt.is_unavailable)
        self.assertEqual(ann.balance_sheet.total_liabilities.value, 300000.0)
        self.assertEqual(ann.balance_sheet.total_assets.value, 500000.0)

    # -------------------------------------------------------------------------
    # Regression 4: equity should use owners' equity, not NCI-inclusive total equity
    # -------------------------------------------------------------------------
    def test_equity_prefers_owners_equity_over_nci_total(self) -> None:
        """Verify that for consolidated statements, equity attributable to owners is preferred."""
        response = {
            "company_name": "Consolidated Corp",
            "annual_reports": [{
                "year": 2024,
                "currency": "INR",
                "unit": "crores",
                "income_statement": {},
                "balance_sheet": {
                    "total_assets": {"value": 800000.0, "source_page": 9, "is_unavailable": False},
                    "total_liabilities": {"value": 500000.0, "source_page": 9, "is_unavailable": False},
                    "total_debt": {"value": 120000.0, "source_page": 9, "is_unavailable": False},
                    "equity": {"value": 280000.0, "source_page": 9, "is_unavailable": False},
                },
                "cash_flow_statement": {},
                "source_pages": [9],
            }],
        }
        extractor, _ = self._make_extractor_with_response(response)
        page = ParsedPage(
            page_number=9,
            text="Total Assets 800000 Equity attributable to owners 280000 Non-controlling interest 30000 Total equity 310000",
            has_text=True,
        )
        result = extractor.extract_section([page])

        ann = result.annual_reports[0]
        self.assertEqual(ann.balance_sheet.equity.value, 280000.0)
        self.assertNotEqual(ann.balance_sheet.equity.value, 310000.0)

    # -------------------------------------------------------------------------
    # Regression 5: Prior-year comparative columns must not be dropped
    # -------------------------------------------------------------------------
    def test_prior_year_comparative_column_extracted(self) -> None:
        """Verify that comparative prior-year figures are extracted into separate annual_reports entries."""
        response = {
            "company_name": "Multi-Year Corp",
            "annual_reports": [
                {
                    "year": 2024,
                    "currency": "INR",
                    "unit": "crores",
                    "income_statement": {
                        "revenue": {"value": 900000.0, "source_page": 5, "is_unavailable": False},
                        "net_profit": {"value": 70000.0, "source_page": 5, "is_unavailable": False},
                    },
                    "balance_sheet": {},
                    "cash_flow_statement": {},
                    "source_pages": [5],
                },
                {
                    "year": 2023,
                    "currency": "INR",
                    "unit": "crores",
                    "income_statement": {
                        "revenue": {"value": 800000.0, "source_page": 5, "is_unavailable": False},
                        "net_profit": {"value": 60000.0, "source_page": 5, "is_unavailable": False},
                    },
                    "balance_sheet": {},
                    "cash_flow_statement": {},
                    "source_pages": [5],
                },
            ],
        }
        extractor, _ = self._make_extractor_with_response(response)
        page = ParsedPage(
            page_number=5,
            text="Statement of Profit & Loss FY2024 FY2023 Revenue 900000 800000 Net Profit 70000 60000",
            has_text=True,
        )
        result = extractor.extract_section([page])

        years = {ann.year for ann in result.annual_reports}
        self.assertIn(2024, years, "FY2024 must be extracted")
        self.assertIn(2023, years, "FY2023 comparative column must NOT be dropped")

        fy2023 = next(a for a in result.annual_reports if a.year == 2023)
        fy2024 = next(a for a in result.annual_reports if a.year == 2024)
        self.assertEqual(fy2023.income_statement.revenue.value, 800000.0)
        self.assertEqual(fy2024.income_statement.revenue.value, 900000.0)

    # -------------------------------------------------------------------------
    # Regression 6: Indian-format keywords must be detected for page selection
    # -------------------------------------------------------------------------
    def test_indian_statement_keywords_detected_as_primary(self) -> None:
        """Verify that Indian Companies Act statement headings trigger primary page selection."""
        pages = [
            ParsedPage(page_number=1, text="Letter to Shareholders and Corporate Overview", has_text=True),
            ParsedPage(page_number=2, text="Statement of Profit & Loss for the year ended March 2024 Revenue 900000", has_text=True),
            ParsedPage(page_number=3, text="Standalone Balance Sheet as at 31 March 2024 Total Assets 500000", has_text=True),
            ParsedPage(page_number=4, text="Notes to Financial Statements accounting policies", has_text=True),
            ParsedPage(page_number=5, text="Directors Report corporate governance discussion", has_text=True),
        ]
        parsed_pdf = ParsedPDF(total_pages=5, pages=pages)
        extractor = FinancialExtractor(api_key="test_key", max_pages_per_section=4)
        sections = extractor.partition_pages(parsed_pdf)

        all_page_nums = [p.page_number for sec in sections for p in sec]
        self.assertIn(2, all_page_nums, "Statement of Profit & Loss page must be selected")
        self.assertIn(3, all_page_nums, "Standalone Balance Sheet page must be selected")
        self.assertNotIn(5, all_page_nums, "Directors Report narrative page should be excluded")

    # -------------------------------------------------------------------------
    # Regression 7: Borrowings keyword must trigger page inclusion (Indian balance sheets)
    # -------------------------------------------------------------------------
    def test_borrowings_keyword_included_in_core_metric_detection(self) -> None:
        """Verify that pages with 'borrowings' (Indian balance sheet term) are included."""
        self.assertIn(
            "borrowings", CORE_METRIC_KEYWORDS,
            "CORE_METRIC_KEYWORDS must include 'borrowings' for Indian balance sheet detection"
        )

    # -------------------------------------------------------------------------
    # Regression 8: Missing FY2023 - prior-year figures preserved via multi-year extraction
    # -------------------------------------------------------------------------
    def test_fy2023_from_comparative_column_merged_correctly(self) -> None:
        """Verify FY2023 extracted from comparative column merges with any separately extracted data."""
        mock_client = MagicMock()

        resp_sec1 = MagicMock()
        resp_sec1.text = json.dumps({
            "company_name": "Test Ltd",
            "annual_reports": [
                {
                    "year": 2024,
                    "currency": "INR",
                    "unit": "crores",
                    "income_statement": {
                        "revenue": {"value": 1000000.0, "source_page": 10, "is_unavailable": False},
                        "net_profit": {"value": 80000.0, "source_page": 10, "is_unavailable": False},
                    },
                    "balance_sheet": {},
                    "cash_flow_statement": {},
                    "source_pages": [10],
                },
                {
                    "year": 2023,
                    "currency": "INR",
                    "unit": "crores",
                    "income_statement": {
                        "revenue": {"value": 880000.0, "source_page": 10, "is_unavailable": False},
                        "net_profit": {"value": 73000.0, "source_page": 10, "is_unavailable": False},
                    },
                    "balance_sheet": {},
                    "cash_flow_statement": {},
                    "source_pages": [10],
                },
            ],
        })

        mock_client.models.generate_content.return_value = resp_sec1

        pages = [ParsedPage(page_number=10, text="Statement of Profit & Loss FY2024 FY2023", has_text=True)]
        parsed_pdf = ParsedPDF(total_pages=10, pages=pages)

        extractor = FinancialExtractor(client=mock_client, max_pages_per_section=4)
        report = extractor.extract(parsed_pdf)

        years = {ann.year for ann in report.annual_reports}
        self.assertIn(2024, years)
        self.assertIn(2023, years, "FY2023 from comparative column must be preserved in final report")

        fy2023 = next(a for a in report.annual_reports if a.year == 2023)
        self.assertEqual(fy2023.income_statement.revenue.value, 880000.0)


class TestModelConfigurationAndFallback(unittest.TestCase):
    """Unit tests for primary and fallback model configuration and fallback transitions."""

    def setUp(self) -> None:
        clear_section_cache()

    def tearDown(self) -> None:
        clear_section_cache()

    def test_extractor_initialization_with_and_without_fallback_model(self) -> None:
        """Verify that FinancialExtractor initializes correctly with and without a fallback model."""
        # Initialized with explicit fallback model
        extractor_with_fallback = FinancialExtractor(
            api_key="test_key",
            model_name="gemini-3.8-flash",
            fallback_model_name="gemini-3.5-flash-lite",
        )
        self.assertEqual(extractor_with_fallback.model_name, "gemini-3.8-flash")
        self.assertEqual(extractor_with_fallback.fallback_model_name, "gemini-3.5-flash-lite")

        # Initialized without fallback model (explicit fallback_model_name=None)
        extractor_without_fallback = FinancialExtractor(
            api_key="test_key",
            model_name="gemini-3.8-flash",
            fallback_model_name=None,
        )
        self.assertEqual(extractor_without_fallback.model_name, "gemini-3.8-flash")
        self.assertIsNone(extractor_without_fallback.fallback_model_name)

        # Initialized with default arguments (omitting fallback_model_name)
        extractor_default = FinancialExtractor(
            api_key="test_key",
            model_name="gemini-3.8-flash",
        )
        self.assertEqual(extractor_default.model_name, "gemini-3.8-flash")
        self.assertIsNone(extractor_default.fallback_model_name)

    @patch("time.sleep", return_value=None)
    def test_primary_model_works_without_fallback_model(self, mock_sleep: MagicMock) -> None:
        """Verify primary model extraction functions normally when no fallback model is configured."""
        mock_client = MagicMock()
        resp = MagicMock()
        resp.text = json.dumps({
            "company_name": "Standalone Corp",
            "annual_reports": [{
                "year": 2024,
                "income_statement": {"revenue": {"value": 75000.0, "source_page": 1, "is_unavailable": False}},
            }],
            "extraction_notes": [],
        })
        mock_client.models.generate_content.return_value = resp

        page = ParsedPage(page_number=1, text="Income Statement Revenue 75000", has_text=True)
        extractor = FinancialExtractor(
            client=mock_client,
            model_name="gemini-3.8-flash",
            fallback_model_name=None,
        )

        report = extractor.extract_section([page])

        self.assertEqual(report.company_name, "Standalone Corp")
        self.assertEqual(mock_client.models.generate_content.call_count, 1)
        call_kwargs = mock_client.models.generate_content.call_args[1]
        self.assertEqual(call_kwargs.get("model"), "gemini-3.8-flash")
        mock_sleep.assert_not_called()

    @patch("time.sleep", return_value=None)
    def test_primary_model_success(self, mock_sleep: MagicMock) -> None:
        """Verify that when primary model succeeds, fallback model is not invoked."""
        mock_client = MagicMock()
        resp = MagicMock()
        resp.text = json.dumps({
            "company_name": "Primary Corp",
            "annual_reports": [{
                "year": 2024,
                "income_statement": {"revenue": {"value": 12000.0, "source_page": 1, "is_unavailable": False}},
            }],
            "extraction_notes": [],
        })
        mock_client.models.generate_content.return_value = resp

        page = ParsedPage(page_number=1, text="Income Statement Revenue 12000", has_text=True)
        extractor = FinancialExtractor(
            client=mock_client,
            model_name="gemini-3.8-flash",
            fallback_model_name="gemini-3.5-flash-lite",
            max_retries=2,
        )

        report = extractor.extract_section([page])

        self.assertEqual(report.company_name, "Primary Corp")
        self.assertEqual(mock_client.models.generate_content.call_count, 1)
        call_kwargs = mock_client.models.generate_content.call_args[1]
        self.assertEqual(call_kwargs.get("model"), "gemini-3.8-flash")
        mock_sleep.assert_not_called()

    @patch("time.sleep", return_value=None)
    def test_fallback_model_success(self, mock_sleep: MagicMock) -> None:
        """Verify that when primary model fails transiently (503), it retries with backoff and switches to fallback."""
        mock_client = MagicMock()
        resp_fallback = MagicMock()
        resp_fallback.text = json.dumps({
            "company_name": "Fallback Corp",
            "annual_reports": [{
                "year": 2024,
                "income_statement": {"revenue": {"value": 15000.0, "source_page": 1, "is_unavailable": False}},
            }],
            "extraction_notes": [],
        })
        err_503 = RuntimeError("503 UNAVAILABLE: Model is overloaded")
        # Primary fails twice (1 initial + 1 retry = 2 attempts), then fallback succeeds
        mock_client.models.generate_content.side_effect = [err_503, err_503, resp_fallback]

        page = ParsedPage(page_number=1, text="Income Statement Revenue 15000", has_text=True)
        extractor = FinancialExtractor(
            client=mock_client,
            model_name="gemini-3.8-flash",
            fallback_model_name="gemini-3.5-flash-lite",
            max_retries=1,
            base_delay=0.1,
        )

        report = extractor.extract_section([page])

        self.assertEqual(report.company_name, "Fallback Corp")
        self.assertEqual(mock_client.models.generate_content.call_count, 3)
        calls = mock_client.models.generate_content.call_args_list
        self.assertEqual(calls[0][1].get("model"), "gemini-3.8-flash")
        self.assertEqual(calls[1][1].get("model"), "gemini-3.8-flash")
        self.assertEqual(calls[2][1].get("model"), "gemini-3.5-flash-lite")
        self.assertTrue(any("fallback model 'gemini-3.5-flash-lite'" in n for n in report.extraction_notes))
        mock_sleep.assert_called()

    @patch("time.sleep", return_value=None)
    def test_both_models_unavailable(self, mock_sleep: MagicMock) -> None:
        """Verify that when both primary and fallback fail on transient errors, an ExtractionError is raised."""
        mock_client = MagicMock()
        err_503_primary = RuntimeError("503 UNAVAILABLE: Primary overloaded")
        err_503_fallback = RuntimeError("503 UNAVAILABLE: Fallback overloaded")
        # max_retries=1: 2 primary attempts + 2 fallback attempts = 4 calls total
        mock_client.models.generate_content.side_effect = [
            err_503_primary,
            err_503_primary,
            err_503_fallback,
            err_503_fallback,
        ]

        page = ParsedPage(page_number=1, text="Income Statement Revenue 20000", has_text=True)
        extractor = FinancialExtractor(
            client=mock_client,
            model_name="gemini-3.8-flash",
            fallback_model_name="gemini-3.5-flash-lite",
            max_retries=1,
        )

        with self.assertRaises(ExtractionError) as ctx:
            extractor.extract_section([page])

        err_str = str(ctx.exception)
        self.assertIn("gemini-3.8-flash", err_str)
        self.assertIn("gemini-3.5-flash-lite", err_str)
        self.assertEqual(mock_client.models.generate_content.call_count, 4)

    @patch("time.sleep", return_value=None)
    def test_quota_exhaustion_switches_to_fallback(self, mock_sleep: MagicMock) -> None:
        """Verify that HTTP 429 quota exhaustion on primary model triggers retries with backoff and switches to fallback."""
        mock_client = MagicMock()
        resp_fallback = MagicMock()
        resp_fallback.text = json.dumps({
            "company_name": "QuotaRecovery Corp",
            "annual_reports": [{
                "year": 2024,
                "income_statement": {"revenue": {"value": 25000.0, "source_page": 1, "is_unavailable": False}},
            }],
            "extraction_notes": [],
        })
        err_429 = RuntimeError("429 RESOURCE_EXHAUSTED: Rate limit or daily quota exceeded for gemini-3.8-flash")
        # Primary fails twice with 429, then fallback succeeds on first attempt
        mock_client.models.generate_content.side_effect = [err_429, err_429, resp_fallback]

        page = ParsedPage(page_number=1, text="Income Statement Revenue 25000", has_text=True)
        extractor = FinancialExtractor(
            client=mock_client,
            model_name="gemini-3.8-flash",
            fallback_model_name="gemini-3.5-flash-lite",
            max_retries=1,
            base_delay=0.1,
        )

        report = extractor.extract_section([page])

        self.assertEqual(report.company_name, "QuotaRecovery Corp")
        self.assertEqual(mock_client.models.generate_content.call_count, 3)
        calls = mock_client.models.generate_content.call_args_list
        self.assertEqual(calls[0][1].get("model"), "gemini-3.8-flash")
        self.assertEqual(calls[1][1].get("model"), "gemini-3.8-flash")
        self.assertEqual(calls[2][1].get("model"), "gemini-3.5-flash-lite")
        self.assertTrue(any("fallback model 'gemini-3.5-flash-lite'" in n for n in report.extraction_notes))

    def test_permanent_error_never_switches_to_fallback(self) -> None:
        """Verify that permanent errors (e.g. 401 unauthenticated) never invoke the fallback model."""
        mock_client = MagicMock()
        err_401 = RuntimeError("401 UNAUTHENTICATED: Invalid API key")
        mock_client.models.generate_content.side_effect = err_401

        page = ParsedPage(page_number=1, text="Income Statement", has_text=True)
        extractor = FinancialExtractor(
            client=mock_client,
            model_name="gemini-3.8-flash",
            fallback_model_name="gemini-3.5-flash-lite",
            max_retries=2,
        )

        with self.assertRaises(ExtractionError) as ctx:
            extractor.extract_section([page])

        self.assertIn("permanent error", str(ctx.exception).lower())
        self.assertEqual(mock_client.models.generate_content.call_count, 1)
        self.assertEqual(mock_client.models.generate_content.call_args[1].get("model"), "gemini-3.8-flash")

    def test_switching_models_does_not_duplicate_extracted_sections(self) -> None:
        """Verify that switching models does not duplicate or re-extract sections."""
        mock_client = MagicMock()

        # Section 1 succeeds with primary
        resp_sec1 = MagicMock()
        resp_sec1.text = json.dumps({
            "company_name": "MultiSection Corp",
            "annual_reports": [{
                "year": 2024,
                "income_statement": {"revenue": {"value": 50000.0, "source_page": 1, "is_unavailable": False}},
                "source_pages": [1],
            }],
        })
        # Section 2 primary fails with 503, fallback succeeds
        err_503 = RuntimeError("503 UNAVAILABLE: Model overloaded")
        resp_sec2_fallback = MagicMock()
        resp_sec2_fallback.text = json.dumps({
            "company_name": "MultiSection Corp",
            "annual_reports": [{
                "year": 2024,
                "balance_sheet": {"total_assets": {"value": 100000.0, "source_page": 2, "is_unavailable": False}},
                "source_pages": [2],
            }],
        })

        mock_client.models.generate_content.side_effect = [
            resp_sec1,          # Sec 1 (primary)
            err_503,            # Sec 2 (primary try 1)
            err_503,            # Sec 2 (primary try 2)
            resp_sec2_fallback, # Sec 2 (fallback)
        ]

        pages = [
            ParsedPage(page_number=1, text="Income Statement Revenue 50000", has_text=True),
            ParsedPage(page_number=2, text="Balance Sheet Total Assets 100000", has_text=True),
        ]
        parsed_pdf = ParsedPDF(total_pages=2, pages=pages)

        extractor = FinancialExtractor(
            client=mock_client,
            model_name="gemini-3.8-flash",
            fallback_model_name="gemini-3.5-flash-lite",
            max_pages_per_section=1,
            max_retries=1,
        )

        report = extractor.extract(parsed_pdf)

        self.assertFalse(report.is_incomplete)
        self.assertEqual(len(report.annual_reports), 1)
        ann = report.annual_reports[0]
        self.assertEqual(ann.income_statement.revenue.value, 50000.0)
        self.assertEqual(ann.balance_sheet.total_assets.value, 100000.0)
        self.assertEqual(ann.source_pages, [1, 2])
        self.assertEqual(mock_client.models.generate_content.call_count, 4)


if __name__ == "__main__":
    unittest.main()
