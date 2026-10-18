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
from extraction.extractor import FinancialExtractor, ExtractionError
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
        val_balanced = validate_financial_report(report_balanced)
        self.assertFalse(any("Balance Sheet equation discrepancy" in issue.message for issue in val_balanced.warnings))


class TestFinancialExtractor(unittest.TestCase):
    """Test extractor partitioning, Gemini API integration, and section merging."""

    def test_partition_pages_respects_limits_and_keywords(self) -> None:
        """Ensure long documents are partitioned into bounded sections."""
        pages = [
            ParsedPage(page_number=1, text="Introduction and Letter to Shareholders", has_text=True),
            ParsedPage(page_number=2, text="Corporate Governance", has_text=True),
            ParsedPage(page_number=3, text="Consolidated Balance Sheet: Total Assets $500M", has_text=True),
            ParsedPage(page_number=4, text="Consolidated Statement of Operations / Income Statement", has_text=True),
            ParsedPage(page_number=5, text="Statement of Cash Flows and Investing Activities", has_text=True),
            ParsedPage(page_number=6, text="Note 1 Summary of Accounting Policies", has_text=True),
            ParsedPage(page_number=7, text="Note 2 Revenue Recognition", has_text=True),
            ParsedPage(page_number=8, text="Note 3 Commitments and Contingencies", has_text=True),
        ]
        parsed_pdf = ParsedPDF(total_pages=8, pages=pages)

        extractor = FinancialExtractor(api_key="test_key", max_pages_per_section=3)
        sections = extractor.partition_pages(parsed_pdf)

        self.assertGreater(len(sections), 1)
        for section in sections:
            self.assertLessEqual(len(section), 3)

    def test_extract_section_success_with_mocked_gemini(self) -> None:
        """Verify Gemini call and parsing of structured output."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "company_name": "Global Tech Inc",
            "annual_reports": [
                {
                    "year": 2023,
                    "currency": "USD",
                    "unit": "millions",
                    "income_statement": {
                        "revenue": {"value": 85000.0, "source_page": 4, "is_unavailable": False},
                        "cost_of_goods_sold": {"value": 50000.0, "source_page": 4, "is_unavailable": False},
                        "operating_expenses": {"value": 15000.0, "source_page": 4, "is_unavailable": False},
                        "operating_profit": {"value": 20000.0, "source_page": 4, "is_unavailable": False},
                        "net_profit": {"value": 16000.0, "source_page": 4, "is_unavailable": False}
                    },
                    "balance_sheet": {
                        "total_assets": {"value": 150000.0, "source_page": 3, "is_unavailable": False},
                        "total_liabilities": {"value": 80000.0, "source_page": 3, "is_unavailable": False},
                        "total_debt": {"value": 35000.0, "source_page": 3, "is_unavailable": False},
                        "equity": {"value": 70000.0, "source_page": 3, "is_unavailable": False}
                    },
                    "cash_flow_statement": {
                        "operating_cash_flow": {"value": 32000.0, "source_page": 5, "is_unavailable": False},
                        "capital_expenditure": {"value": 8000.0, "source_page": 5, "is_unavailable": False}
                    },
                    "source_pages": [3, 4, 5],
                    "missing_fields": []
                }
            ],
            "extraction_notes": []
        })
        mock_client.models.generate_content.return_value = mock_response

        extractor = FinancialExtractor(client=mock_client)
        section = [
            ParsedPage(page_number=3, text="Balance Sheet", has_text=True),
            ParsedPage(page_number=4, text="Income Statement", has_text=True),
        ]
        result = extractor.extract_section(section)

        self.assertEqual(result.company_name, "Global Tech Inc")
        self.assertEqual(len(result.annual_reports), 1)
        annual = result.annual_reports[0]
        self.assertEqual(annual.year, 2023)
        self.assertEqual(annual.income_statement.revenue.value, 85000.0)
        self.assertEqual(annual.income_statement.cost_of_goods_sold.value, 50000.0)
        self.assertEqual(annual.income_statement.revenue.source_page, 4)
        self.assertEqual(annual.balance_sheet.total_assets.value, 150000.0)
        self.assertEqual(annual.balance_sheet.total_debt.value, 35000.0)
        self.assertEqual(annual.cash_flow_statement.operating_cash_flow.value, 32000.0)

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
                    "cost_of_goods_sold": {"value": 700.0, "source_page": 2, "is_unavailable": False}
                },
                "source_pages": [2]
            }]
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
                    "total_debt": {"value": 600.0, "source_page": 4, "is_unavailable": False}
                },
                "source_pages": [4]
            }]
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

    def test_api_failure_handled_gracefully(self) -> None:
        """Verify API error does not crash overall execution and is noted."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("Rate limit exceeded")

        pages = [ParsedPage(page_number=1, text="Income Statement", has_text=True)]
        parsed_pdf = ParsedPDF(total_pages=1, pages=pages)

        extractor = FinancialExtractor(client=mock_client)
        report = extractor.extract(parsed_pdf)

        self.assertEqual(len(report.annual_reports), 0)
        self.assertTrue(any("extraction note" in note for note in report.extraction_notes))

    def test_missing_api_key_raises_extraction_error(self) -> None:
        """Verify missing API key raises an informative ExtractionError."""
        with patch.object(config, "GEMINI_API_KEY", None):
            extractor = FinancialExtractor(api_key=None, client=None)
            with self.assertRaises(ExtractionError):
                _ = extractor.client


if __name__ == "__main__":
    unittest.main()
